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
from src.vi import InfiniteIBP
from src.utils import register_hooks, visualize_A, visualize_A_save, visualize_nu_save
from src.data import generate_gg_blocks, generate_gg_blocks_dataset, gg_blocks






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
for j in range(10):
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
    if j < 8:
        model._nu.data = torch.randn(model._nu.shape)


plt.plot(np.arange(len(elbo_array)), np.array(elbo_array))
plt.show()


from src.vae import InfiniteIBP_VAE

vae = InfiniteIBP_VAE(1.5, 6, 0.1, 0.05, 36)
vae.phi.data = model.phi.data
vae._phi_var.data = model._phi_var.data 
vae._tau.data = model._tau
vae.train()
optimizer = torch.optim.Adam([{'params': vae.nu.parameters()}], lr=0.01)
vae_loss = []
vae_iter_count = 0


for j in range(15):
    for i in range(1000):
        optimizer.zero_grad()


        nu = vae.nu(X)
        loss = (model.nu - nu).pow(2).sum()
        print("[Epoch {:<3}] sq diff nu = {:.3f}".format(i + 1, loss.item()))
        loss.backward()
        optimizer.step()

        vae_iter_count += 1
        assert loss.item() != np.inf, "loss is inf"
        vae_loss.append(-loss.item())



    nu = vae.nu(X)



    visualize_A_save(vae.phi.detach().numpy(), iter_count)
    visualize_nu_save(nu.detach().numpy(), iter_count)
    plt.plot(np.arange(len(vae_loss)), np.array(vae_loss))
    plt.show()

    elbo = vae.elbo(X)
    print("final elbo is",elbo)
    import ipdb; ipdb.set_trace()













