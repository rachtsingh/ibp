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

def alternate_optimization():
    # used to debug infs
    # from tests.test_vi import test_elbo_components, test_q_E_logstick

    N = 500
    X = generate_gg_blocks_dataset(N, 0.05)

    model = InfiniteIBP(1.5, 6, 0.1, 0.05, 36)
    model.init_z(N)
    model.train()
    
    SCALE = 1. 
    #model.phi.data[:4] = SCALE * gg_blocks()
    visualize_A_save(model.phi.detach().numpy(), 0)
    visualize_nu_save(model.nu.detach().numpy(), 0)
    optimizer = torch.optim.Adam(model.parameters(), 0.03)
    iter_count = 0


    for i in range(20):
        # local
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.03
        
        for j in range(1000):
            optimizer.zero_grad()
            loss = -model.elbo(X)
            print("[Epoch {:<3}] ELBO = {:.3f}".format(iter_count, -loss.item()))
            loss.backward()
            # zero out the grad on phi, phi_var
            model.phi.grad.zero_()
            model._phi_var.grad.zero_()
            optimizer.step()
            assert loss.item() != np.inf, "loss is inf"
            iter_count += 1

        # local + global
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
        
        for j in range(500):
            optimizer.zero_grad()
            loss = -model.elbo(X)
            print("[Epoch {:<3}] ELBO = {:.3f}".format(iter_count, -loss.item()))
            loss.backward()
            optimizer.step()
            assert loss.item() != np.inf, "loss is inf"
            iter_count += 1

        visualize_A_save(model.phi.detach().numpy(), iter_count)
        visualize_nu_save(model.nu.detach().numpy(), iter_count)
    

        if iter_count % 3000 == 0:
            model._nu.data = torch.randn(model._nu.size())

    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    alternate_optimization()
