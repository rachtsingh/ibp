import torch
import numpy as np
from torch import nn
from torch import digamma
from torch.distributions import MultivariateNormal as MVN
from torch.distributions import Bernoulli as Bern
from torch import optim
from torch.nn import functional as F
from .utils import register_hooks, visualize_A
from .data import generate_gg_blocks, generate_gg_blocks_dataset, gg_blocks
from .vi import InfiniteIBP

import sys

LOG_2PI = 1.8378770664093453
EPS = 1e-16

class InfiniteIBP_VAE(InfiniteIBP):
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
    q(z_nk) = Bernoulli(z_nk; nu_nk(x))

    In truncated stick breaking for the infinite model,
    pi_k = product_{i=1}^k v_i for k <= K and zero otherwise.

    NOTE: must call init_z with a given N to start.
    """
    def init_variables(self):
        super(InfiniteIBP_VAE, self).init_variables()
        self._nu = nn.Sequential(
            nn.Linear(self.D, 400),
            nn.ReLU(),
            nn.Linear(400, self.K.item()),
            nn.Sigmoid()
        )

    @property
    def nu(self):
        return self._nu

    def train(self):
        self._tau.requires_grad = True
        self.phi.requires_grad = True
        self._phi_var.requires_grad = True
        for buffer in self.nu.parameters():
            buffer.requires_grad = True
    
    def eval(self):
        self._tau.requires_grad = False
        self.phi.requires_grad = False
        self._phi_var.requires_grad = False
        for buffer in self.nu.parameters():
            buffer.requires_grad = False
    
    def parameters(self):
        if not self._tau.requires_grad:
            print("WARNING: calling .parameters() but no grad required! Watch out (maybe call .train())")
        ret = [buffer for buffer in self.nu.parameters()]
        return ret + [self._tau, self.phi, self._phi_var]

    def elbo(self, X):
        """
        This is the evidence lower bound evaluated at X, when X is of shape (N, D)
        i.e. log p_K(X | theta) geq ELBO
        """
        nu = self.nu(X)
        self.most_recent_nu = nu
        a = self._1_feature_prob(self.tau).sum()
        b = self._2_feature_assign(nu, self.tau).sum()
        c = self._3_feature_prob(self.phi_var, self.phi).sum()
        d = self._4_likelihood(X, nu, self.phi_var, self.phi).sum()
        e = self._5_entropy(self.tau, self.phi_var, nu).sum()
        return a + b + c + d + e

def fit_infinite_to_ggblocks_vae_exact():
    N = 500
    X = generate_gg_blocks_dataset(N, 0.1)

    model = InfiniteIBP(1.5, 6, 0.5, 0.5, 36)
    model.init_z(N)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), 0.01)

    T = 5000

    elbos = np.zeros(T)
    plots = np.zeros((T, 6, 36))

    for i in range(T):
        optimizer.zero_grad()
        loss = -model.elbo(X)

        print("[Epoch {:<3}] ELBO = {:.3f}".format(i + 1, -loss.item()))
        loss.backward()
        optimizer.step()

        plots[i] = model.phi.detach().numpy().reshape((6, 36))
        elbos[i] = -loss.item()

        assert loss.item() != np.inf, "loss is inf"

    np.save('features_vae_exact.npy', plots)
    np.save('elbo_vae_exact.npy', elbos)

if __name__ == '__main__':
    """
    python -m src.vae will just check that the model works on a ggblocks dataset
    """
    fit_infinite_to_ggblocks_vae_exact()
