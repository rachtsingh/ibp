import torch
import numpy as np
from torch import nn
from torch import digamma
from torch.distributions import MultivariateNormal as MVN
from torch.distributions import Bernoulli as Bern

from matplotlib import pyplot as plt

# relative path import hack
import sys, os
sys.path.insert(0, os.path.abspath('..'))


# inside-package imports below here
from src.vae import InfiniteIBP_VAE
from src.utils import register_hooks, visualize_A, visualize_A_save, visualize_nu_save
from src.data import generate_gg_blocks, generate_gg_blocks_dataset, gg_blocks

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        m.bias.data.fill_(0.01)
        print('encoder weight init')

def vae_test():

    N = 1500
    X = generate_gg_blocks_dataset(N, 0.05)

    model = InfiniteIBP_VAE(1.5, 6, 0.1, 0.05, 36)
    model.train()

    nu = model.nu(X)
    visualize_A_save(model.phi.detach().numpy(), 0)
    visualize_nu_save(nu.detach().numpy(), 0)

    optimizer = torch.optim.Adam([{'params': [model._tau]},
                                  {'params': model.nu.parameters()},
                                  {'params': [model._phi_var, model.phi], 'lr': 0.003}], lr=0.1)

    elbo_array = []
    iter_count = 0
    for j in range(15):
        for i in range(1000):
            optimizer.zero_grad()
            loss = -model.elbo(X)

            print("[Epoch {:<3}] ELBO = {:.3f}".format(i + 1, -loss.item()))
            loss.backward()

            optimizer.step()

            iter_count += 1
            assert loss.item() != np.inf, "loss is inf"
            elbo_array.append(-loss.item())

        visualize_A_save(model.phi.detach().numpy(), iter_count)
        visualize_nu_save(model.most_recent_nu.detach().numpy(), iter_count)
        model.nu.apply(init_weights)

    plt.plot(np.arange(len(elbo_array)), np.array(elbo_array))
    plt.show()
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    vae_test()
