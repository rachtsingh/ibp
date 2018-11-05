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

def show_that_ADVI_init_doesnt_matter():
    SCALE = 1.

    N = 500
    X = generate_gg_blocks_dataset(N, 0.05)

    model = InfiniteIBP(1.5, 6, 0.1, 0.05, 36)
    model.phi.data[:4] = SCALE * gg_blocks()
    model.init_z(N)
    model.train()

    visualize_A_save(model.phi.detach().numpy(), 0)
    visualize_nu_save(model.nu.detach().numpy(), 0)

    optimizer = torch.optim.Adam(model.parameters(), 0.003)

    for i in range(1000):
        optimizer.zero_grad()
        loss = -model.elbo(X)
        print("[Epoch {:<3}] ELBO = {:.3f}".format(i + 1, -loss.item()))
        loss.backward()

        optimizer.step()

        assert loss.item() != np.inf, "loss is inf"

    visualize_A_save(model.phi.detach().numpy(), 1000)
    visualize_nu_save(model.nu.detach().numpy(), 1000)

def find_a_better_scheme(nu_resets=False,tempering=False):
    from matplotlib import pyplot as plt

    print("")
    print("")
    print("")
    print("NU RESETS:",nu_resets)        
    print("TEMPERING:",tempering)

    N = 500
    sigma_n = 0.05
    print("DATA SIZE:",N)
    print("DATA NOISE:",sigma_n)
    X = generate_gg_blocks_dataset(N, sigma_n)

    alpha=1.5
    K = 6
    D = 36
    sigma_a = 0.1
    print("ALPHA:",alpha)
    print("K:",K)
    print("SIGMA N:",sigma_n)
    print("SIGMA A:",sigma_a)

    model = InfiniteIBP(alpha, K, sigma_a, sigma_n, D)
    model.init_z(N)

    if tempering:
        print("INIT TEMPERING PARAMS")
        M = 10
        print("NUM TEMPERATURES:",M)
        model.init_r_and_T(N,M)

    model.train()

    visualize_A_save(model.phi.detach().numpy(), 0)
    visualize_nu_save(model.nu.detach().numpy(), 0)

    if tempering:
        print("Initing optimizer with tempering params included")
        optimizer = torch.optim.Adam([{'params': [model._nu, model._tau, model._r]},
                                  {'params': [model._phi_var, model.phi], 'lr': 0.003}], lr=0.1)
    else:
        optimizer = torch.optim.Adam([{'params': [model._nu, model._tau]},
                                  {'params': [model._phi_var, model.phi], 'lr': 0.003}], lr=0.1)

    elbo_array = []
    iter_count = 0
    for j in range(6):
        for i in range(1000):
            optimizer.zero_grad()

            elbo = model.elbo(X)
            loss = -elbo
            if tempering:
                loss = -model.elbo_tempered(X)

            print("[Epoch {:<3}] ELBO = {:.3f}".format(i + 1, elbo.item()))
            loss.backward()

            optimizer.step()

            iter_count += 1
            assert loss.item() != np.inf, "loss is inf"
            elbo_array.append(elbo.item())

        visualize_A_save(model.phi.detach().numpy(), iter_count)
        visualize_nu_save(model.nu.detach().numpy(), iter_count)
        
        if nu_resets:
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

    for i in range(1000):
        optimizer.zero_grad()
        loss = -model.elbo(X)
        print("[Epoch {:<3}] ELBO = {:.3f}".format(i + 1, -loss.item()))
        loss.backward()

        # zero out the grad on phi, phi_var
        model.phi.grad.zero_()
        model._phi_var.grad.zero_()

        optimizer.step()
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
        assert loss.item() != np.inf, "loss is inf"

    visualize_A_save(model.phi.detach().numpy(), 1500)
    visualize_nu_save(model.nu.detach().numpy(), 1500)
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    
    TEMPERING = False
    NU_RESETS = False
    find_a_better_scheme(nu_resets=NU_RESETS,tempering=TEMPERING) 
   
