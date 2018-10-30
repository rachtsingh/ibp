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

    # pip install librosa
    # Load the audio data
    # Compute the Spectrogram. N is time points. D is frequencies.
    # It is DxN by default, so we transpose.
    # We will do factorization for the magnitudes np.abs(spectrogram)
    # We will need the phase np.angle(spectrogram) to resynthesize to audio
    # Only keep first 500 time windows for now. To increase data size, keep all timesteps
    y,sr = librosa.load('wav/bach.wav')
    true_spec = librosa.stft(y,n_fft=2048)
    true_mag = np.abs(true_spec)
    true_mag = true_mag.T
    true_mag = true_mag[:500,:]
    true_phase = np.angle(true_spec)
    true_phase = true_phase.T[:500,:].T
    X=torch.from_numpy(true_mag)
    N,D = X.size()
    print("{} timepoints, {} frequencies".format(N,D))
    
    # Don't know how many latents to do.
    # Should be one for each frequency we hope to find,
    # but can be less to find groups of frequencies.
    # Using 30 for now
    model = InfiniteIBP(1.5, 30, 0.1, 0.05, D)
    model.init_z(N)
    model.train()

    optimizer = torch.optim.Adam([{'params': [model._nu, model._tau]},
                                  {'params': [model._phi_var, model.phi], 'lr': 0.003}], lr=0.1)

    elbo_array = []
    iter_count = 0
    
    # Should increase j range and i range
    # Should increase the number of times nu is reset,
    # but make sure not to reset it to close to when you stop doing inference
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

        #visualize_A_save(model.phi.detach().numpy(), iter_count)
        #visualize_nu_save(model.nu.detach().numpy(), iter_count)
        if j < 8:
            model._nu.data = torch.randn(model._nu.shape)

    plt.plot(np.arange(len(elbo_array)), np.array(elbo_array))
    plt.show()

    # Sample some assignments
    # Take the mean of the features
    # Recover the data, clip numbers below 0 because magnitude can't be negative
    # Recover real and imaginary part of spectrogram from recovered magnitude and true phase
    # Recover spectrogram from recovered real and imaginary parts.
    # Resynthesize audio with istft
    # When done, listen to bach_reconstruct.wav
    p_Z = Bern(model.nu.clone()) 
    Z = p_Z.sample()
    A = model.phi.clone().detach()
    mag = (Z@A).numpy().clip(min=0.0).T
    real = np.multiply(mag,np.cos(true_phase))
    imag = np.multiply(mag,np.sin(true_phase))
    spec = real + imag*np.complex(0,1)
    y_reconstruct = librosa.istft(spec)
    librosa.output.write_wav('wav/bach_reconstruct.wav',y_reconstruct,sr) 
    
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    infer_music()
