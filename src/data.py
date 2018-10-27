"""
Datasets for experiments
"""
import torch
from torch.distributions import Beta, Bernoulli, Normal

def gg_blocks():
    A = torch.zeros(4, 36)
    A[0, [1, 6, 7, 8, 13]] = 1
    A[1, [3, 4, 5, 9, 11, 15, 16, 17]] = 1
    A[2, [18, 24, 25, 30, 31, 32]] = 1
    A[3, [21, 22, 23, 28, 34]] = 1
    return A

def generate_ibp(N, alpha):
    """
    Generate N datapoints from an Indian Buffet Process with parameter alpha

    TODO: unfinished
    """
    beta = Beta(torch.tensor(alpha), 1.).sample()


def generate_gg_blocks(N):
    """
    Generate 'clean' data points using the ggblocks feature set

    @param N: the number of data points to generate

    This function doesn't sample points from the IBP: it uses a Bernoulli
    distribution for Z
    """

    # we have to make sure there's at least one feature in each sample
    Z = Bernoulli(0.25).sample((N, 4))
    while (Z.sum(1) == 0).any():
        msk = (Z.sum(1) == 0)
        Z[msk] = Bernoulli(0.25).sample((msk.sum().item(), 4))
    A = gg_blocks()
    return Z @ A

def generate_gg_blocks_dataset(N, sigma_n):
    """
    Generate a dataset based on the ggblocks features (with noise)

    @param N: number of data points to generate
    @param sigma_n: the noise stdev
    """
    clean = generate_gg_blocks(N)
    return clean + Normal(0, sigma_n).sample(clean.size())
