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

    optimizer = torch.optim.Adam(model.parameters(), 0.03)

    # plots = np.zeros((1000, 6, 36))

    for i in range(1000):
        optimizer.zero_grad()
        loss = -model.elbo(X)
        print("[Epoch {:<3}] ELBO = {:.3f}".format(i + 1, -loss.item()))
        loss.backward()

        # zero out the grad on phi, phi_var
        model.phi.grad.zero_()
        model._phi_var.grad.zero_()

        optimizer.step()

        # test_elbo_components((model, X))
        # test_q_E_logstick((model.tau.detach(), model.K))
        # plots[i] = model.phi.detach().numpy().reshape((6, 36))
        assert loss.item() != np.inf, "loss is inf"

    print("CHANGE OF REGIME")

    visualize_A_save(model.phi.detach().numpy(), 1000)
    visualize_nu_save(model.nu.detach().numpy(), 1000)

    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.001

    for i in range(500):
        optimizer.zero_grad()
        loss = -model.elbo(X)
        print("[Epoch {:<3}] ELBO = {:.3f}".format(i + 1, -loss.item()))
        loss.backward()

        optimizer.step()

        # test_elbo_components((model, X))
        # test_q_E_logstick((model.tau.detach(), model.K))
        # plots[i] = model.phi.detach().numpy().reshape((6, 36))
        assert loss.item() != np.inf, "loss is inf"

    # np.save('features.npy', plots)
    visualize_A_save(model.phi.detach().numpy(), 1500)
    visualize_nu_save(model.nu.detach().numpy(), 1500)
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    freeze_A_to_solution_and_fit()