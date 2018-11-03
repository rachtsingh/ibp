import torch
import numpy as np
from torch import nn
from torch import digamma
from torch.distributions import MultivariateNormal as MVN
from torch.distributions import Bernoulli as Bern

from .utils import register_hooks, visualize_A
from .data import generate_gg_blocks, generate_gg_blocks_dataset, gg_blocks

import sys

LOG_2PI = 1.8378770664093453
EPS = 1e-16

class InfiniteIBP(nn.Module):
    """
    Infinite/Non-Truncated Indian Buffet Process
    with a mean-field variational posterior approximation.
    Section 5 in http://mlg.eng.cam.ac.uk/pub/pdf/DosMilVanTeh09b.pdf

    Generative Model:
    v_k ~ Beta(alpha,1)               for k in {1,...,inf}
    pi_k = product_{i=1}^k v_i        for k in {1,...,inf}
    z_nk ~ Bernoulli(pi_k)            for k in {1,...,inf}, n in {1,...,N}
    A_k ~ Normal(0, sigma_a^2 I)      for k in {1,...,inf}
    X_n ~ Normal(Z_n A, sigma_n^2 I)  for n in {1,...,N}

    q(v_k) = Beta(v_k;tau_k1,tau_k2 (now over v rather than pi)
    q(A_k) = Normal(A_k;phi_k,phi_var_k)
    q(z_nk) = Bernoulli(z_nk; nu_nk)

    In truncated stick breaking for the infinite model,
    pi_k = product_{i=1}^k v_i for k <= K and zero otherwise.

    NOTE: must call init_z with a given N to start.
    """
    def __init__(self, alpha, K, sigma_a, sigma_n, D):
        super(InfiniteIBP, self).__init__()

        # idempotent - all are constant and have requires_grad=True
        self.alpha = torch.tensor(alpha)
        self.K = torch.tensor(K)
        self.sigma_a = torch.tensor(sigma_a)
        self.sigma_n = torch.tensor(sigma_n)
        self.D = torch.tensor(D)
        # we don't know N, but we'll still initialize everything else
        self.init_variables()

    def init_variables(self, N=100):
        # NOTE: tau must be positive, so we use the @property below
        self._tau = nn.Parameter(torch.rand(self.K, 2))
        self.phi = nn.Parameter(torch.randn(self.K, self.D) * self.sigma_a)
        self._phi_var = nn.Parameter(torch.zeros(self.K, self.D) - 2.)

    # For Variational Tempering
    def init_r_and_T(self,N,M):
        self.M = M
        self._r = nn.Parameter(torch.rand(N,self.M))
        self.T = torch.arange(1,self.M+1.0)
        print("T IS",self.T)

    def init_z(self, N=100):
        self._nu = nn.Parameter(torch.rand(N, self.K))

    """
    Note we use the following trick for sweeping the constraint parametrization
    'under the rug' so to speak - whenever we access self.tau, we get the
    constrained-to-be-positive version
    """

    @property
    def r(self):
        return nn.Softmax(dim=1)(self._r)

    @property
    def tau(self):
        return 0.5   + nn.Softplus()(self._tau)

    @property
    def nu(self):
        return nn.Sigmoid()(self._nu)

    @property
    def phi_var(self):
        return nn.Softplus()(self._phi_var)

    def elbo(self, X):
        """
        This is the evidence lower bound evaluated at X, when X is of shape (N, D)
        i.e. log p_K(X | theta) geq ELBO
        """
        a = self._1_feature_prob(self.tau).sum()
        b = self._2_feature_assign(self.nu, self.tau).sum()
        c = self._3_feature_prob(self.phi_var, self.phi).sum()
        d = self._4_likelihood(X, self.nu, self.phi_var, self.phi).sum()
        e = self._5_entropy(self.tau, self.phi_var, self.nu).sum()
        return a + b + c + d + e
    

    def elbo_tempered(self, X):
        """
        This is the evidence lower bound evaluated at X, when X is of shape (N, D)
        i.e. log p_K(X | theta) geq ELBO
        """
        probs = torch.ones(self.M) / self.M
        E_q_logp_y =  -1.0*torch.mul(self.r,probs.log()).sum()
        q_y = torch.distributions.Categorical(self.r)

        a = self._1_feature_prob(self.tau).sum()
        b = self._2_feature_assign(self.nu, self.tau).sum()
        c = self._3_feature_prob(self.phi_var, self.phi).sum()
        e = self._5_entropy(self.tau, self.phi_var, self.nu).sum()
        lik = self._4_likelihood_tempered(X,self.nu,self.phi_var,self.phi).sum()
        return a + b + c + lik + e + E_q_logp_y + q_y.entropy().sum()

    def elbo_annealed(self, X, T):
        a = self._1_feature_prob(self.tau).sum()
        b = self._2_feature_assign(self.nu, self.tau).sum()
        c = self._3_feature_prob(self.phi_var, self.phi).sum()
        d = self._4_likelihood(X, self.nu, self.phi_var, self.phi).sum()
        e = self._5_entropy(self.tau, self.phi_var, self.nu).sum()
        return a + b + c + d/T + e

    def _1_feature_prob(self, tau):
        """
        @param tau: (K, 2)
        @return: (K,)

        Computes Cross Entropy: E_q(v) [logp(v_k|alpha)]
        """
        return self.alpha.log() + \
            ((self.alpha - 1) * (digamma(tau[:, 0]) - digamma(tau.sum(dim=1))))

    @staticmethod
    def _E_log_stick(tau, K):
        """
        @param tau: (K, 2)
        @return: ((K,), (K, K))

        where the first return value is E_log_stick, and the second is q
        """
        # we use the same indexing as in eq. (10)
        q = torch.zeros(K, K)

        # working in log space until the last step
        first_term = digamma(tau[:, 1])
        second_term = digamma(tau[:, 0]).cumsum(0) - digamma(tau[:, 0])
        third_term = digamma(tau.sum(1)).cumsum(0)
        q += (first_term + second_term - third_term).view(1, -1)
        q = torch.tril(q.exp())
        q = torch.nn.functional.normalize(q, p=1, dim=1)
        # TODO: should we detach q? what does that do to the ADVI?

        assert (q.sum(1) - torch.ones(K)).abs().max().item() < 1e-6, "WTF normalize didn't work"

        # each vector should be size (K,)
        # let's do this nonvectorized to start
        torch_e_logstick = torch.zeros(K)

        # this is really slow and can be vectorized
        for k in range(K):
            torch_e_logstick[k] += (digamma(tau[:, 1]) * q[k]).sum()
            torch_e_logstick[k] += ((1 - q[k].cumsum(0)) * digamma(tau[:, 0]))[:k].sum()
            for m in range(k + 1): # m needs to be
                torch_e_logstick[k] -= q[k][m:].sum() * digamma(tau.sum(1))[m]
            torch_e_logstick[k] -= (q[k, :k+1] * (q[k, :k+1] + EPS).log()).sum()
            # first = (q * digamma(tau[:, 1]).view(1, -1)).sum(1)
            # second = ((1 - q.cumsum(1)) * tau[:, 0]).sum(1)
            # third = ((1 - q.cumsum(1) - q) * tau.sum(1)).sum(1)
            # temp_q = q.clone() # TODO: why clone?
            # temp_q[q == 0] = 1. # since half of q is 0, log(1) is now a mask
            # fourth = (temp_q * (temp_q + EPS).log()).sum(1)
            # torch_e_logstick = first + second + third + fourth

        return torch_e_logstick, q

    def _2_feature_assign(self, nu, tau):
        """
        @param nu: (N, K)
        @param tau: (K, 2)
        @return: (N, K)

        Computes Cross Entropy: E_q(v),q(Z) [logp(z_nk|v)]
        """
        return (nu * (digamma(tau[:,1]) - digamma(tau.sum(dim=1))).cumsum(0) + \
            (1. - nu) * self._E_log_stick(tau, self.K)[0])

    def _3_feature_prob(self, phi_var, phi):
        """
        @param phi_var: (K, D)
        @param phi: (K, D)
        @return: ()

        NOTE: must return () because torch.trace doesn't allow specifying axes

        Computes Cross Entropy: E_q(A) [logp(A_k|sigma_a^2 I)]
        Same as Finite Approach
        """
        ret = 0
        constant = -0.5 * self.D * (2 * self.sigma_a.log() + LOG_2PI)
        for k in range(self.K):
            other_term = (-0.5 / (self.sigma_a**2)) * \
                (phi_var[k].sum() + phi[k].pow(2).sum())
            ret += constant + other_term
        return ret

    def _4_likelihood(self, X, nu, phi_var, phi):
        """
        @param X: (N, D)
        @param nu: (N, K)
        @param phi_var: (K, D)
        @param phi: (K, D)
        @return: ()

        Computes Likelihood: E_q(Z),q(A) [logp(X_n|Z_n,A,sigma_n^2 I)]
        Same as Finite Approach
        """
        N, _ = X.shape
        K, D = self.K, self.D # for notational simplicity
        ret = 0
        constant = -0.5 * D * (self.sigma_n.log() + LOG_2PI)

        first_term = X.pow(2).sum()
        second_term = (-2 * (nu.view(N, K, 1) * phi.view(1, K, D)) * X.view(N, 1, D)).sum()
        third_term = 2 * torch.triu((phi @ phi.transpose(0, 1)) * \
                (nu.transpose(0, 1) @ nu), diagonal=1).sum()

        # have to loop because of torch.trace again
        fourth_term = 0
        for k in range(K):
            fourth_term += (nu[:, k] * (phi_var[k].sum() + phi[k].pow(2).sum())).sum()

        nonconstant = (-0.5/(self.sigma_n**2)) * \
            (first_term + second_term + third_term + fourth_term)

        return constant + nonconstant



    def _4_likelihood_tempered(self, X, nu, phi_var, phi):
        """
        @param X: (N, D)
        @param nu: (N, K)
        @param phi_var: (K, D)
        @param phi: (K, D)
        @return: ()

        Computes Likelihood: E_q(Z),q(A) [logp(X_n|Z_n,A,sigma_n^2 I)]
        Same as Finite Approach
        """
        N, _ = X.shape
        K, D = self.K, self.D # for notational simplicity
        ret = 0
        constant = -0.5 * D * (self.sigma_n.log() + LOG_2PI)

        temps = self.r@self.T

        first_term = X.pow(2).sum()
        second_term = (-2 * (nu.view(N, K, 1) * phi.view(1, K, D)) * X.view(N, 1, D))
        second_term = second_term.sum(dim=2)
        second_term = second_term.sum(dim=1) 
        second_term = torch.mul(temps,second_term).sum()

        third_term = 2 * torch.triu((phi @ phi.transpose(0, 1)) * \
                (nu.transpose(0, 1) @ nu), diagonal=1).sum()

        # have to loop because of torch.trace again
        fourth_term = 0
        for k in range(K):
            fourth_term += (nu[:, k] * (phi_var[k].sum() + phi[k].pow(2).sum())).sum()

        nonconstant = (-0.5/(self.sigma_n**2)) * \
            (first_term + second_term + third_term + fourth_term)

        return constant + nonconstant



    @staticmethod
    def _entropy_q_v(tau):
        return ((tau.lgamma().sum(1) - tau.sum(1).lgamma()) - \
            ((tau - 1.) * digamma(tau)) .sum(1) + \
            ((tau.sum(1) - 2.) * digamma(tau.sum(1)))).sum()

    @staticmethod
    def _entropy_q_A(phi_var):
        K, D = phi_var.shape
        entropy_q_A = 0
        for k in range(K):
            entropy_q_A += 0.5 * (D * (1 + LOG_2PI) + phi_var[k].log().sum()).sum()
        return entropy_q_A

    @staticmethod
    def _entropy_q_z(nu):
        return -(nu * (nu + EPS).log() + (1 - nu) * (1 - nu + EPS).log()).sum()

    def _5_entropy(self, tau, phi_var, nu):
        """
        @param tau: (K, 2)
        @param phi_var: (K, D, D)
        @param nu: (N, K)
        @return: ()

        Computes Entropy H[q] for all variational distributions q
        Same as Finite Approach, just rename entropy_q_pi to entropy_q_v.
        """
        return self._entropy_q_v(tau) + \
               self._entropy_q_A(phi_var) + \
               self._entropy_q_z(nu)

    def cavi(self, X):
        """
        TODO: should this function have arguments?
        There are so many dumb terms in this
        """
        N, K, D = X.shape[0], self.K, self.D
        for k in range(K):
            precision = (1./(self.sigma_a ** 2) + self.nu[:, k].sum()/(self.sigma_n**2))
            self.phi_var[k] = torch.ones(self.D) / precision
            s = (self.nu[:, k].view(N, 1) * (X - (self.nu @ self.phi - torch.ger(self.nu[:, k], self.phi[k])))).sum(0)
            self.phi[k] = s/((self.sigma_n ** 2) * precision)

        # update q(z)
        # we shouldn't vectorize this I think (TODO: discuss)
        # Marks comment: update for nu_nk depends on nu_nl forall l != k
        # so perhaps it can be parallelized across n, but not k
        for k in range(K):
            for n in range(N):
                first_term = (digamma(self.tau[:k+1, 1]) - digamma(self.tau.sum(1)[:k+1])).sum() - \
                    self._E_log_stick(self.tau, self.K)[0][k]
                # this line is really slow
                other_prod = (self.nu @ self.phi - self.nu[:, k].view((N, 1)) * self.phi[k].view((1, D)))[n]
                second_term = -0.5 / (self.sigma_n ** 2) * (self.phi_var[k].sum() + self.phi[k].pow(2).sum()) + \
                    (self.phi[k] @ (X[n] - other_prod))/ (self.sigma_n ** 2)
                self.nu[n][k] = 1./(1. + (-(first_term + second_term)).exp())

        # update q(pi)
        for k in range(K):
            q = self._E_log_stick(self.tau, self.K)[1]
            self.tau[k][0] = self.alpha + self.nu[:, k:].sum() + \
                ((N - self.nu.sum(0)) * q[:, k+1:].sum(1))[k+1:].sum()
            self.tau[k][1] = 1 + ((N - self.nu.sum(0)) * q[:, k])[k:].sum()

"""
Inference runners
"""

def fit_infinite_to_ggblocks_cavi():
    from tests.test_vi import test_elbo_components, test_q_E_logstick

    N = 500
    X = generate_gg_blocks_dataset(N, 0.05)

    model = InfiniteIBP(4., 6, 0.05, 0.05, 36)
    model.init_z(N)

    for i in range(10):
        model.cavi(X)
        print("[Epoch {:<3}] ELBO = {:.3f}".format(i + 1, model.elbo(X).item()))

    visualize_A(model.phi.detach().numpy())

def fit_infinite_to_ggblocks_advi_exact():
    # used to debug infs
    from tests.test_vi import test_elbo_components, test_q_E_logstick

    SCALE = 1.

    N = 500
    X = generate_gg_blocks_dataset(N, 0.05)

    # for i in range(10):
    model = InfiniteIBP(1.5, 6, 0.1, 0.5, 36)
    model.phi.data[:4] = SCALE * gg_blocks()
    visualize_A(model.phi.detach().numpy())
    model.init_z(N)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), 0.01)

    plots = np.zeros((1000, 6, 36))

    for i in range(1000):
        optimizer.zero_grad()
        loss = -model.elbo(X)
        print("[Epoch {:<3}] ELBO = {:.3f}".format(i + 1, -loss.item()))
        loss.backward()
        optimizer.step()

        # test_elbo_components((model, X))
        # test_q_E_logstick((model.tau.detach(), model.K))
        plots[i] = model.phi.detach().numpy().reshape((6, 36))
        assert loss.item() != np.inf, "loss is inf"

    np.save('features.npy', plots)
    # visualize_A(model.phi.detach().numpy())
    # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    """
    python src/vi.py will just check that the model works on a ggblocks dataset
    """
    # fit_infinite_to_ggblocks_cavi()
    fit_infinite_to_ggblocks_advi_exact()