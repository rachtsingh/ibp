"""
This script is just to test that some of my math is correct
"""
import torch
from torch.distributions import Normal

def main():
    K = 4
    D = 6
    N = 100000
    A_mean = torch.rand(K, D)
    A_var = torch.nn.Softplus()(torch.randn(K, D)) / 2
    # we have a matrix of individual Gaussians, which can also be interpreted Gaussians with diagonal covariances

    dist = Normal(loc=A_mean, scale=A_var.sqrt())

    analytic_expectation = A_mean @ A_mean.transpose(0, 1) + (A_var.sum(1) * torch.eye(K))

    print(analytic_expectation)

    samples = dist.sample((N,))
    f_samples = torch.zeros(N, K, K)
    s = 0
    for i in range(N):
        f_samples[i] = samples[i] @ samples[i].transpose(0, 1)
    print(f_samples.mean(0))

if __name__ == '__main__':
    main()