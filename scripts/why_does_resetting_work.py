import torch
import numpy as np
from torch import nn
from torch import digamma
from torch.distributions import MultivariateNormal as MVN
from torch.distributions import Bernoulli as Bern

# relative path import hack
import sys, os
sys.path.insert(0, os.path.abspath('..'))

# inside-package imports below here
from src.vi import InfiniteIBP
from src.utils import register_hooks, visualize_A, visualize_A_save, visualize_nu_save
from src.data import generate_gg_blocks, generate_gg_blocks_dataset, gg_blocks

def check_gradient_of_tau_after_some_updates():
    from matplotlib import pyplot as plt

    SCALE = 1.

    N = 500
    X = generate_gg_blocks_dataset(N, 0.05)

    model = InfiniteIBP(1.5, 6, 0.1, 0.05, 36)
    model.init_z(N)
    model.train()

    optimizer = torch.optim.Adam([{'params': [model._nu, model._tau]},
                                  {'params': [model._phi_var, model.phi], 'lr': 0.003}], lr=0.1)

    elbo_array = []
    iter_count = 0
    # values = np.zeros((6 * 3000, 12))
    for j in range(6):
        for i in range(1000):
            optimizer.zero_grad()
            loss = -model.elbo(X)
            print("[Epoch {:<3}] ELBO = {:.3f}".format(i + 1, -loss.item()))
            loss.backward()

            optimizer.step()

            assert loss.item() != np.inf, "loss is inf"
            elbo_array.append(-loss.item())
            # values[iter_count] = model.tau.detach().numpy().reshape((-1,))
            iter_count += 1

        # plt.figure()
        # for i in range(12):
        #     plt.plot(np.arange(3000), values[j*3000:(j + 1)*3000, i])
        # plt.savefig('tau_set_{}.png'.format(j))

        visualize_A_save(model.phi.detach().numpy(), iter_count)
        visualize_nu_save(model.nu.detach().numpy(), iter_count)
        model._nu.data = torch.randn(model._nu.shape)
        # model._tau.data = torch.randn(model._tau.shape)

if __name__ == '__main__':
    check_gradient_of_tau_after_some_updates()