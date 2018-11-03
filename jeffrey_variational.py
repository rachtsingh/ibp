# This is a very basic file for variational inference, taken from the
# tech report.  
import numpy as np
import numpy.random as npr 
from numpy.random import beta
import scipy.special as sps
from scipy.special import gamma, digamma, gammaln
from numpy.linalg import slogdet
from scipy.misc import logsumexp
import pdb 
import math
import matplotlib.pyplot as plt
import time
import argparse
import sys
import torch
import os

from utils import *
from make_toy_data import GGData

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log-epoch', type=int, default=1, metavar='N',
                    help='wait every epochs')
parser.add_argument('--load-data', type=str, default=None,
                    help='load dataset')
parser.add_argument('--savefile', type=str, required=True, 
                    help='savefile name')
parser.add_argument('--dataset', type=str, default='infinite-random',
                    help='mnist, gg, infinite-random')
parser.add_argument('--truncation', type=int, default=10,
                    help='number of sticks')
parser.add_argument('--alpha0', type=float, default=2.,
                    help='prior alpha for stick breaking Betas')
parser.add_argument('--uuid', type=str, default=gen_id(), help='(somewhat) unique identifier for the model/job')

global args
args = parser.parse_args()

# data_set is n x d
# phi is k x d
# Phi is k x d x d 
# nu is n x k 
# tau is a k x 2 matrix

EPS = 1e-50

# Compute the elbo 
def compute_elbo( data_set , alpha , sigma_a , sigma_n , phi , Phi , nu , tau ):
    N, D = data_set.shape
    K = len(phi)

    digamma1 = np.array([sps.digamma(tau[0][k]) for k in range(K)])
    digamma2 = np.array([sps.digamma(tau[1][k]) for k in range(K)])
    digamma12 = np.array([sps.digamma(tau[0][k] + tau[1][k]) for k in range(K)])

    vterm = K*np.log(alpha) + (alpha - 1) * (sum(digamma1) - sum(digamma12))

    digamma_partial = [digamma2[0] - digamma12[0]]
    for k in range(1,K):
        digamma_partial.append(digamma_partial[-1] + digamma2[k] - digamma12[k])

    hard_terms = []
    for k in range(K):
        _, Elogstick = compute_q_Elogstick( tau, k )
        hard_terms.append(Elogstick)
    hard_terms = np.array(hard_terms)

    zterm = np.sum(nu * np.tile(digamma_partial, (N, 1))) + np.sum((1 - nu) * np.tile(hard_terms, (N, 1)))

    Aterm = -D*K*0.5*np.log(2*np.pi*sigma_a**2) - 0.5 / (sigma_a**2) * sum([np.trace(Phi[k]) + np.dot(phi[k], phi[k]) for k in range(K)])

    innerXterm = np.sum(data_set * data_set)
    innerXterm += - 2*np.sum(nu * np.dot(data_set, phi.T))
    phi_prod = np.dot(phi, phi.T) # K x K
    innerXterm += 2 * ( sum([np.sum(np.outer(nu[n], nu[n]) * phi_prod) for n in range(N)]) \
            - np.sum( nu * np.tile(np.diagonal(phi_prod), (N,1)) ) ) # correction terms
    t = np.array([np.trace(Phi[k]) + np.dot(phi[k], phi[k]) for k in range(K)])
    innerXterm += np.sum(nu * np.tile(t, (N,1)))
    Xterm = -0.5*N*D*np.log(2*np.pi*sigma_n**2) - 0.5 / (sigma_n**2) * innerXterm

    tauterm = sum([sps.gammaln(tau[0][k]) + sps.gammaln(tau[1][k]) - sps.gammaln(tau[0][k] + tau[1][k]) for k in range(K)])
    tauterm += sum([-(tau[0][k] - 1) * digamma1[k] - (tau[1][k] - 1) * digamma2[k] + (tau[0][k] + tau[1][k] - 2) * digamma12[k] for k in range(K)])

    Phiterm = 0.5 * D*K * (np.log(2*np.pi) + 1)
    for k in range(K):
        _, det = np.linalg.slogdet(Phi[k])
        Phiterm += 0.5 * det

    nuterm = -np.sum(nu * np.log(nu + EPS) + (1 - nu) * np.log(1 - nu + EPS))

    elbo = vterm + zterm + Aterm + Xterm + tauterm + Phiterm + nuterm
    return elbo

# Compute q and Elogstick variables for a given feature index
def compute_q_Elogstick( tau , k ):
    k = k + 1 # length
    digamma1 = np.array([sps.digamma(tau[0][i]) for i in range(k)])
    digamma2 = np.array([sps.digamma(tau[1][i]) for i in range(k)])
    digamma12 = np.array([sps.digamma(tau[0][i] + tau[1][i]) for i in range(k)])

    # initialize q distribution
    lqs = np.array([digamma2[i] + sum(digamma1[:i-1]) - sum(digamma12[:i]) for i in range(k)])
    lqs[0] = digamma2[0] - digamma12[0] # dumb indexing fix
    q_partition = logsumexp(lqs)
    qs = np.exp(lqs - q_partition)
    q_tails = [qs[-1]]
    for m in range(k-2, -1, -1):
        q_tails.append(qs[m] + q_tails[-1])
    q_tails.reverse()

    Elogstick = np.sum(qs * digamma2)
    Elogstick += sum([digamma1[m] * q_tails[m+1] for m in range(k-1)])
    Elogstick += -sum([digamma12[m] * q_tails[m] for m in range(k)])
    Elogstick += - np.sum(qs * np.log(qs))

    # return
    return qs , Elogstick

# Run the VI  
def run_vi(data_set, alpha, holdout, sigma_a=0.1, sigma_n=0.5, iter_count=50, truncation=15, init_Phi=1.0):
    data_count = data_set.shape[0]
    dim_count = data_set.shape[1] 
    elbo_set = np.zeros( [ iter_count ] )
    nu_set = list()   # nu are the varitional parameters on Z 
    phi_set = list()  # phi mean param of A 
    Phi_set = list()  # Phi cov param of A, per feat -> same for all dims 
    tau_set = list()  # tau are the variational parameters on the stick betas
    iter_times = list()

    N, D = data_set.shape
    K = truncation
    feature_count = truncation

    # Initialize objects 
    Z = npr.binomial(1, 0.5, [data_count, feature_count])
    nu = npr.uniform(0, 1, [data_count, feature_count])
    phi = np.zeros((feature_count, dim_count))
    Phi = [init_Phi * np.eye(dim_count) for k in range(feature_count)]
    tau = [np.ones(feature_count), np.ones(feature_count)]

    # Optimization loop 
    t_start = time.clock()

    try:
        for vi_iter in range( iter_count ):

            # Update Phi and phi
            for k in range(feature_count):
                coeff = 1 / (1 / (sigma_a**2) + np.sum(nu[:,k]) / (sigma_n ** 2))
                Phi[k] = coeff * np.eye(dim_count)

                phi_sums = np.dot(nu, phi)
                phi_sums_cur = phi_sums - np.outer(nu[:,k], phi[k])
                phi[k] = coeff * (1 / (sigma_n**2) * np.dot(nu[:,k], (data_set - phi_sums_cur)))
                assert len(phi[k]) == dim_count

            # Get the intermediate variables
            qks = []
            Elogsticks = []
            for k in range(feature_count):
                qk, Elogstick = compute_q_Elogstick( tau, k )
                qks.append(qk); Elogsticks.append(Elogstick)

            # Update tau, nu
            for k in range( int( feature_count ) ):
     
                # update nu_k
                theta = np.sum([sps.digamma(tau[0][i]) - sps.digamma(tau[0][i] + tau[1][i]) for i in range(k)]) - Elogsticks[k]
                theta += -0.5 / (sigma_n**2) * (np.trace(Phi[k]) + np.dot(phi[k], phi[k]))
                phi_sums = np.dot(nu, phi) # recompute for each nu_k
                phi_sums_cur = phi_sums - np.outer(nu[:,k], phi[k])
                theta += 1 / (sigma_n**2) * np.dot(phi[k], (data_set - phi_sums_cur).T)

                nu[:, k] = 1 / (1 + np.exp(-theta))

                # update tau
                tau[0][k] = alpha + np.sum(nu[:,k:]) + sum([(data_count - np.sum(nu[:,m])) * np.sum(qks[m][k+1:m]) for m in range(k+1, feature_count)])
                tau[1][k] = 1 + sum( [ (data_count - np.sum(nu[:,m])) * qks[m][k] for m in range(k, feature_count) ] )

            # Compute the ELBO
            elbo = compute_elbo(data_set, alpha, sigma_a, sigma_n, phi, Phi, nu, tau)

            H = holdout.shape[0]

            num_Z_samples = 50
            num_A_samples = 5
            num_pi_samples = 10
            total_loss = 0.0

            # initialize memory
            sampled_z_counts = np.zeros(K)
            mses = np.zeros((num_pi_samples, num_Z_samples, num_A_samples, H))
            
            for pk in range(num_pi_samples):
                vs = np.zeros(K)
                for k in range(K):
                    vs[k] = np.random.beta(tau[0][k], tau[1][k])
                pi = np.cumprod(vs)
                Z_new = np.zeros((H, K))
                for zi in range(num_Z_samples):
                    Z_new = np.random.binomial(1, pi, (H, K))
                    sampled_z_counts += Z_new.sum(0)
                    A = np.zeros((K, D))
                    for ai in range(num_A_samples):
                        for k in range(K):
                            A[k] = phi[k].copy() + np.random.normal(0, Phi[k][0][0], D)
                        X_pred = Z_new.dot(A)
                        diff = (-0.5 * (np.square(X_pred - holdout))/sigma_n**2 - 0.5 * np.log(2 * np.pi) - np.log(sigma_n))
                        mses[pk, zi, ai] += diff.sum(axis=1)
            total_loss = mses.mean()
            lse = (logsumexp(mses, axis=(0, 1, 2)) - np.log(num_pi_samples * num_A_samples * num_Z_samples)).mean()
            sampled_z_counts /= float(num_Z_samples * H * num_pi_samples)
            # print("z: {}".format(sampled_z_counts))
            # Store things and report  
            elbo_set[ vi_iter ] = elbo 
            nu_set.append( nu )
            phi_set.append( phi )
            Phi_set.append( Phi ) 
            tau_set.append( tau )
            iter_times.append( time.clock() - t_start )
            print("[Epoch: {:<3}]: ELBO: {:<10} | Test Loss: {:<10} | MSE (LSE): {:<10}".format(vi_iter, elbo/(float(N)), -1. * total_loss, -1 * lse))
    except KeyboardInterrupt:
        pass

    return nu_set , phi_set , Phi_set , tau_set , elbo_set , iter_times

def main():
    # load data 
    if args.load_data:
        if os.path.isfile(args.load_data):
            print("=> loading data '{}'".format(args.load_data))
            checkpoint = torch.load(args.load_data)
            train_loader = checkpoint['train_loader']
            test_loader = checkpoint['test_loader']
            print("=> loaded data '{}'".format(args.load_data))
        else:
            print("=> no load data found at '{}'".format(args.load_data))
            sys.exit(1)
        train_data = train_loader.dataset
        test_data = test_loader.dataset
    else:
        train_data = GGData(train=True, data_count=1000, data_type=args.dataset, save_data=True, uid=args.uuid)
        test_data = GGData(train=False, data_count=100, data_type=args.dataset, save_data=True, uid=args.uuid, A=train_data.A)

    train_X = train_data.train_data
    test_X = test_data.test_data

    nu_set, phi_set, Phi_set, _, _, _ = run_vi(train_X.numpy(), args.alpha0, test_X.numpy(), sigma_a=0.1, sigma_n=0.7071, iter_count=args.epochs, truncation=args.truncation)

    num_to_show = 10
    for k in range(num_to_show):
        plt.subplot(2, num_to_show, k + 1)
        plt.imshow(phi_set[-1][k].reshape(20, 25), cmap='gray', interpolation='none')
        plt.subplot(2, num_to_show, num_to_show + k + 1)
        plt.imshow(train_data.A[k].reshape(20, 25), cmap='gray', interpolation='none')
    plt.show(block=False)
    pdb.set_trace()

if __name__ == '__main__':
    main()
