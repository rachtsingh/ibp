import torch
from torch import nn
from torch import digamma

LOG_2PI = 1.8378770664093453

class FiniteVI(nn.Module):
    """
    This model implements mean-field VI via coordinate ascent
    (elsewhere referred to as CAVI) using a finite truncation
    (i.e., Appendix C in http://mlg.eng.cam.ac.uk/pub/pdf/DosMilVanTeh09b.pdf)
    
    
    Generative Model:
    pi_k ~ Beta(alpha/K,1)          for k in {1,...,K}
    z_nk ~ Bernoulli(pi_k)          for k in {1,...,K}, n in {1,...,N}
    A_k ~ Normal(0, sigma_a^2 I)    for k in {1,...,K}
    X_n ~ Normal(Z_n.A, sigma_n^2 I)for n in {1,...,N}

    Variational Distributions:
    q(pi_k) = Beta(pi_k;tau_k1,tau_k2)
    q(A_k) = Normal(A_k;phi,phi_var)
    q(z_nk) = Bernoulli(z_nk;nu_nk)
    
    """
    def __init__(self, alpha, K, sigma_a, sigma_n, D):
        # idempotent - all are constant and have requires_grad=True
        self.alpha = torch.tensor(alpha)
        self.K = torch.tensor(K)
        self.sigma_a = torch.tensor(sigma_a)
        self.sigma_n = torch.tensor(sigma_n)
        self.D = torch.tensor(D)

        # init something?

    def _1_feature_prob(tau):
        """
        @param tau: (K, 2)
        @return: (K,)
        """
        return self.alpha.log() - self.K.log() + (self.alpha/self.K - 1) * \
            (digamma(tau[:, 0]) - digamma(tau.sum(dim=1)))

    def _2_feature_assign(nu, tau):
        """
        @param nu: (N, K)
        @param tau: (K, 2)
        @return: (N, K)
        """
        return nu * digamma(tau[:, 0]) + (1. - nu) * digamma(tau[:, 1]) - digamma(tau.sum(dim=1))

    def _3_feature_prob(phi_var, phi):
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

    def _4_likelihood(X, nu, phi_var, phi):
        """
        @param X: (N, D)
        @param nu: (N, K)
        @param phi_var: (K, D, D)
        @param phi: (K, D)
        @return: 
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

    def _5_entropy(tau, phi_var, nu):
        
