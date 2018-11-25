import torch
import numpy as np
from torch import nn
from torch import digamma
from torch.distributions import MultivariateNormal as MVN
from torch.distributions import RelaxedBernoulli

from .utils import register_hooks, visualize_A, inverse_softplus
from .data import generate_gg_blocks, generate_gg_blocks_dataset, gg_blocks
from .vi import InfiniteIBP

# relative path import hack
import sys, os
sys.path.insert(0, os.path.abspath('..'))

LOG_2PI = 1.8378770664093453
EPS = 1e-16

class InfiniteIBPSampled(InfiniteIBP):
    """
    This is the same IBP, except that the ELBO isn't an exact estimate - it's
    sampled
    """
    def __init__(self, alpha, K, sigma_a, sigma_n, D, T):
        super(InfiniteIBPSampled, self).__init__(alpha, K, sigma_a, sigma_n, D)
        self.T = T

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

        # we use the Concrete / Gumbel-softmax approximation
        Z = RelaxedBernoulli(temperature=self.T, probs=nu).rsample()

        # these terms are essentially the same, nu gets replaced by Z
        first_term = X.pow(2).sum()
        second_term = (-2 * (Z.view(N, K, 1) * phi.view(1, K, D)) * X.view(N, 1, D)).sum()

        # this is Z^TE[A^TA]Z
        third_term = torch.diag(Z @ \
                                (phi @ phi.transpose(0, 1) + (phi_var.sum(1) * torch.eye(K))) @ \
                                Z.transpose(0, 1)).sum()

        nonconstant = (-0.5/(self.sigma_n**2)) * \
            (first_term + second_term + third_term)

        return constant + nonconstant

    def parent_elbo(self, X):
        return super(InfiniteIBPSampled, self).elbo(X)

def fit_infinite_to_ggblocks_advi_sampled():
    # used to debug infs
    from tests.test_vi import test_elbo_components, test_q_E_logstick

    N = 250
    X = generate_gg_blocks_dataset(N, 0.5)

    # for i in range(10):
    model = InfiniteIBPSampled(1.5, 6, 0.5, 0.5, 36, 0.1)
    # model.phi.data[:4] = SCALE * gg_blocks()
    # visualize_A(model.phi.detach().numpy())
    model.init_z(N)

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), 0.01)

    T = 5000

    elbos = np.zeros(T)
    plots = np.zeros((T, 6, 36))

    for i in range(T):
        optimizer.zero_grad()
        loss = -model.elbo(X)

        print("[Epoch {:<3}] ELBO (not a bound) = {:.3f}".format(i + 1, -loss.item()))
        loss.backward()
        optimizer.step()

        # test_elbo_components((model, X))
        # test_q_E_logstick((model.tau.detach(), model.K))
        plots[i] = model.phi.detach().numpy().reshape((6, 36))
        assert loss.item() != np.inf, "loss is inf"
        elbos[i] = model.parent_elbo(X).item() # use the real elbo so we can compare plots

    np.save('features_advi_sampled.npy', plots)
    np.save('elbo_advi_sampled.npy', elbos)

if __name__ == '__main__':
    """
    python src/vi.py will just check that the model works on a ggblocks dataset
    """
    fit_infinite_to_ggblocks_advi_sampled()
