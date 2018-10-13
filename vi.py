import torch
from torch import nn
from torch import digamma

LOG_2PI = 1.8378770664093453

class FiniteVI(nn.Module):
    """
    This model implements mean-field VI via coordinate ascent
    (elsewhere referred to as CAVI) using a finite truncation
    (i.e., Appendix C in http://mlg.eng.cam.ac.uk/pub/pdf/DosMilVanTeh09b.pdf)

    NOTE: must call init_z with a given N to start.
    """
    def __init__(self, alpha, K, sigma_a, sigma_n, D):
        # idempotent - all are constant and have requires_grad=True
        self.alpha = torch.tensor(alpha)
        self.K = torch.tensor(K)
        self.sigma_a = torch.tensor(sigma_a)
        self.sigma_n = torch.tensor(sigma_n)
        self.D = torch.tensor(D)

        # we don't know N, but we'll still initialize everything else
        self.init_variables()

    def init_variables(N=100):
        self.tau = nn.Parameter(torch.rand(self.K, 2))
        self.phi = nn.Parameter(torch.randn(self.K, self.D))
        self.phi_var = nn.Parameter(torch.zeros(self.K, self.D, self.D))
        for k in self.K:
            self.phi_var[k] = torch.eye(self.D)

    def init_z(N=100):
        self.nu = nn.Parameter(torch.rand(N, self.K))

    def elbo(X):
        """
        This is the evidence lower bound evaluated at X, when X is of shape (N, D)
        i.e. log p_K(X | theta) \geq ELBO
        """
        return self._1_feature_prob(self.tau).sum() + \
               self._2_feature_assign(self.nu, self.tau).sum() + \
               self._3_feature_prob(self.phi_var, self.phi).sum() + \
               self._4_likelihood(X, self.)

    def _1_feature_prob(self, tau):
        """
        @param tau: (K, 2)
        @return: (K,)
        """
        return self.alpha.log() - self.K.log() + (self.alpha/self.K - 1) * \
            (digamma(tau[:, 0]) - digamma(tau.sum(dim=1)))

    def _2_feature_assign(self, nu, tau):
        """
        @param nu: (N, K)
        @param tau: (K, 2)
        @return: (N, K)
        """
        return nu * digamma(tau[:, 0]) + (1. - nu) * digamma(tau[:, 1]) - digamma(tau.sum(dim=1))

    def _3_feature_prob(self, phi_var, phi):
        """
        @param phi_var: (K, D, D)
        @param phi: (K, D)
        @return: ()

        NOTE: must return () because torch.trace doesn't allow specifying axes
        """
        ret = 0
        constant = -0.5 * self.D * (self.sigma_a.log() + LOG_2PI)
        for k in range(self.K):
            other_term = (-0.5 / (self.sigma_a**2)) * \
                (torch.trace(phi_var[k]) + phi[k].pow(2).sum())
            ret += constant + other_term

    def _4_likelihood(self, X, nu, phi_var, phi):
        """
        @param X: (N, D)
        @param nu: (N, K)
        @param phi_var: (K, D, D)
        @param phi: (K, D)
        @return: ()
        """
        N, _ = X.shape
        K, D = self.K, self.D # for notational simplicity
        ret = 0
        constant = -0.5 * D * (self.sigma_n.log() + LOG_2PI)

        first_term = X.pow(2).sum()
        second_term = (-2 * (nu.view(N, K, 1) * phi.view(1, K, D)) * X.view(1, K, D)).sum()
        third_term = 2 * torch.triu((phi @ phi.transpose(0, 1)) * (nu.transpose(0, 1) @ nu), diagonal=1).sum()

        # have to loop because of torch.trace again
        fourth_term = 0
        for k in range(K):
            fourth_term += (nu[:, k] * (torch.trace(phi_var[k]) + phi[k].pow(2).sum())).sum()
        
        nonconstant = (-0.5/(self.sigma_n**2)) * \
            (first_term + second_term + third_term + fourth_term)

        return constant + nonconstant

    def _5_entropy(self, tau, phi_var, nu):
        """
        @param tau: (K, 2)
        @param phi_var: (K, D, D)
        @param nu: (N, K)
        @return: ()
        """
        entropy_q_pi = (tau.lgamma().sum(1) - tau.sum(1).lgamma() - \
            (tau[:, 0] - 1) * digamma(tau[:, 0]) - \
            (tau[:, 1] - 1) * digamma(tau[:, 1]) + \
            (tau.sum(1) - 2.) * digamma(tau.sum(1))).sum()
        entropy_q_A = 0
        for k in self.K:
            entropy_q_A += 0.5 * (self.D * (1 + LOG_2PI) + torch.logdet(phi_var[k])).sum()
        entropy_q_z = -(nu * nu.log() + (1 - nu) * (1 - nu).log()).sum()