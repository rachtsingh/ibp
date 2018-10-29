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

import librosa
import numpy as np


def infer_music():

    y,sr = librosa.load('bach.wav')
    true_spec = librosa.stft(y,n_fft=2048)
    true_mag = np.abs(true_spec)
    true_mag = true_mag.T
    true_mag = true_mag[:500,:]
    true_phase = np.angle(true_spec)
    true_phase = true_phase.T[:500,:].T
    X=torch.from_numpy(true_mag)
    N,D = X.size()
    print("{} timepoints, {} frequencies".format(N,D))
    model = InfiniteIBP(1.5, 30, 0.1, 0.05, D)
    model.init_z(N)
    model.train()

    optimizer = torch.optim.Adam([{'params': [model._nu, model._tau]},
                                  {'params': [model._phi_var, model.phi], 'lr': 0.003}], lr=0.1)

    elbo_array = []
    iter_count = 0
    for j in range(8):
        for i in range(100):
            optimizer.zero_grad()
            loss = -model.elbo(X)
            print("[Epoch {:<3}] ELBO = {:.3f}".format(i + 1, -loss.item()))
            loss.backward()

            optimizer.step()

            iter_count += 1
            assert loss.item() != np.inf, "loss is inf"
            elbo_array.append(-loss.item())

        #visualize_A_save(model.phi.detach().numpy(), iter_count)
        #visualize_nu_save(model.nu.detach().numpy(), iter_count)
        if j < 5:
            model._nu.data = torch.randn(model._nu.shape)

    plt.plot(np.arange(len(elbo_array)), np.array(elbo_array))
    plt.show()
    
    
    p_Z = Bern(model.nu.clone()) 
    Z = p_Z.sample()
    A = model.phi.clone().detach()
    mag = (Z@A).numpy().clip(min=0.0).T
    real = np.multiply(mag,np.cos(true_phase))
    imag = np.multiply(mag,np.sin(true_phase))
    spec = real + imag*np.complex(0,1)
    y_reconstruct = librosa.istft(spec)
    librosa.output.write_wav('bach_reconstruct.wav',y_reconstruct,sr) 
    
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    infer_music()
