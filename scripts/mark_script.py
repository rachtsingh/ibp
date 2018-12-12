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



def run(nu_resets=False,tempering=False,proximity=False):
    from matplotlib import pyplot as plt

    print("")
    print("NU RESETS:",nu_resets)        
    print("TEMPERING:",tempering)
    print("PROXIMITY:",proximity)

    N = 500
    sigma_n = 0.05
    X = generate_gg_blocks_dataset(N, sigma_n)
    alpha=1.5
    K = 6
    D = 36
    sigma_a = 0.1
    model = InfiniteIBP(alpha, K, sigma_a, sigma_n, D)
    model.init_z(N)

    if proximity:
        global_nu = torch.zeros_like(model._nu)
        counter=0

    if tempering:
        M = 10
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

            # ENTROPY ################################
            if proximity:

                if counter == 0:
                    global_nu = model._nu.detach()
                else:
                    global_nu.data = .8*model._nu.data + .2*global_nu.data
                counter += 1

                optimizer.zero_grad()
                #model_nu_copy = torch.zeros_like(model._nu)
                #model_nu_copy.data = model._nu.data
                model_nu_copy = model._nu.detach()
                model_nu_copy.requires_grad=True
                optimizer_2 = torch.optim.Adam([model_nu_copy],lr=0.001)
                optimizer_2.zero_grad()
                global_nu.data = global_nu.data
                curr_params = nn.Sigmoid()(model_nu_copy)
                curr_bern = torch.distributions.Bernoulli(curr_params)
                global_bern = torch.distributions.Bernoulli(global_nu)
                curr_entropy = curr_bern.entropy()
                global_entropy = global_bern.entropy()
                entropy_diff = (curr_entropy - global_entropy).pow(2).sum()
                entropy_diff.backward()
                d_grad = model_nu_copy.grad
            
                model_nu_copy = model_nu_copy.detach()
                model_nu_copy.requires_grad=True
                optimizer_3 = torch.optim.Adam([model_nu_copy],lr=0.001)
                optimizer_3.zero_grad()
                curr_params = nn.Sigmoid()(model_nu_copy)
                curr_bern = torch.distributions.Bernoulli(curr_params)
                curr_entropy = curr_bern.entropy().sum()
                curr_entropy.backward()
                f_grad = model_nu_copy.grad

                extra_terms = torch.mul(d_grad,f_grad)
                k=10.0
                model._nu.data = model._nu.data + k*extra_terms.data
            ##############################################

            iter_count += 1
            assert loss.item() != np.inf, "loss is inf"
            elbo_array.append(elbo.item())

        visualize_A_save(model.phi.detach().numpy(), iter_count)
        visualize_nu_save(model.nu.detach().numpy(), iter_count)
        
        if nu_resets and j<5:
            model._nu.data = torch.randn(model._nu.shape)

    plt.plot(np.arange(len(elbo_array)), np.array(elbo_array))
    plt.show()
    import ipdb; ipdb.set_trace()



if __name__ == '__main__':
    
    TEMPERING = False
    NU_RESETS = False
    PROXIMITY = True
    run(nu_resets=NU_RESETS,tempering=TEMPERING,proximity=PROXIMITY) 
   
