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

def find_a_better_scheme():
    SCALE = 1.

    N = 500
    X = generate_gg_blocks_dataset(N, 0.05)

    model = InfiniteIBP(1.5, 6, 0.1, 0.05, 36)
    model.init_z(N)
    model.train()

    visualize_A_save(model.phi.detach().numpy(), 0)
    visualize_nu_save(model.nu.detach().numpy(), 0)

    optimizer = torch.optim.Adam([{'params': [model._nu, model._tau]},
                                  {'params': [model._phi_var, model.phi], 'lr': 0.003}], lr=0.1)

    elbo_array = []
    iter_count = 0
    for j in range(6):
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
        visualize_nu_save(model.nu.detach().numpy(), iter_count)
        model._nu.data = torch.randn(model._nu.shape)

    plt.plot(np.arange(len(elbo_array)), np.array(elbo_array))
    plt.show()
    import ipdb; ipdb.set_trace()

def freeze_A_to_solution_and_fit():
    # used to debug infs
    # from tests.test_vi import test_elbo_components, test_q_E_logstick

    SCALE = 1.

    N = 500
    X = generate_gg_blocks_dataset(N, 0.05)

    model = InfiniteIBP(1.5, 6, 0.1, 0.05, 36)
    model.phi.data[:4] = SCALE * gg_blocks()
    model.init_z(N)
    model.train()

    visualize_A_save(model.phi.detach().numpy(), 0)
    visualize_nu_save(model.nu.detach().numpy(), 0)

    optimizer = torch.optim.Adam(model.parameters(), 0.1)

    for i in range(20):
        model.cavi(X)
        print("[Epoch {:<3}] ELBO = {:.3f}".format(i + 1, model.elbo(X).item()))

    print("CHANGE OF REGIME")

    visualize_A_save(model.phi.detach().numpy(), 20)
    visualize_nu_save(model.nu.detach().numpy(), 20)
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    freeze_A_to_solution_and_fit()