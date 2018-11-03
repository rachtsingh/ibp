import torch
import numpy as np
from torch import nn
from torch import digamma
from torch.distributions import MultivariateNormal as MVN
from torch.distributions import Bernoulli as Bern
from torch.distributions import Beta

# relative path import hack
import sys, os
sys.path.insert(0, os.path.abspath('..'))

# inside-package imports below here
from src.vi import InfiniteIBP

np.set_printoptions(precision=4, suppress=True)
EPS = 1e-16

def test_q_E_logstick(inputs=None):
    """
    Test the computation of the multinomial bound in Finale's tech report
    by computing it by hand for k=2 and k=3

    PASSES
    """

    from scipy.special import digamma as dgm

    if inputs is None:
        K = 6
        tau = 0.01 + nn.Softplus()(torch.randn(K, 2))
    else:
        tau, K = inputs

    # # compute, by hand, the distribution q_2
    hand_q = np.zeros(2)
    taud = tau.clone().detach().numpy()
    hand_q[0] = dgm(taud[0, 1]) - dgm(taud[0, 1] + taud[0, 0])
    hand_q[1] = dgm(taud[1, 1]) + dgm(taud[0, 0]) - \
        (dgm(taud[0, 0] + taud[0, 1]) + dgm(taud[1, 0] + taud[1, 1]))
    hand_q = np.exp(hand_q)
    hand_q /= hand_q.sum()

    # let's check E_log_stick for k = 2 (i.e. index 1)
    hand_E_logstick = 0
    a = (hand_q[0] * dgm(taud[0, 1]) + hand_q[1] * dgm(taud[1, 1]))
    b = (hand_q[1] * dgm(taud[0, 0]))
    ca = dgm(taud[0, 0] + taud[0, 1]) * (hand_q[0] + hand_q[1])
    cb = dgm(taud[1, 0] + taud[1, 1]) * (hand_q[1])
    d = (hand_q[0] * np.log(hand_q[0]) + hand_q[1] * np.log(hand_q[1]))

    hand_E_logstick = a + b - ca - cb - d
    # print(a, b, ca + cb, d)

    # for q_3
    hand_q_3 = np.zeros(3)
    hand_q_3[0] = dgm(taud[0, 1]) - dgm(taud[0, 1] + taud[0, 0])
    hand_q_3[1] = dgm(taud[1, 1]) + dgm(taud[0, 0]) - \
        (dgm(taud[0, 0] + taud[0, 1]) + dgm(taud[1, 0] + taud[1, 1]))
    hand_q_3[2] = dgm(taud[2, 1]) + (dgm(taud[0, 0]) + dgm(taud[1, 0])) - \
        (dgm(taud[0, 0] + taud[0, 1]) + dgm(taud[1, 0] + taud[1, 1]) + dgm(taud[2, 0] + taud[2, 1]))
    hand_q_3 = np.exp(hand_q_3)
    hand_q_3 /= hand_q_3.sum()

    # let's check E_log_stick for k = 3 (i.e. index 2)
    hand_E_logstick_3 = 0
    hand_E_logstick_3 += (hand_q_3[0] * dgm(taud[0, 1]) + hand_q_3[1] * dgm(taud[1, 1]) + hand_q_3[2] * dgm(taud[2, 1]))
    hand_E_logstick_3 += ((hand_q_3[1] + hand_q_3[2]) * dgm(taud[0, 0])) + \
                         ((hand_q_3[2]) * dgm(taud[1, 0]))
    hand_E_logstick_3 -= (dgm(taud[0, 0] + taud[0, 1]) * (hand_q_3[0] + hand_q_3[1] + hand_q_3[2]) + \
                          dgm(taud[1, 0] + taud[1, 1]) * (hand_q_3[1] + hand_q_3[2]) + \
                          dgm(taud[2, 0] + taud[2, 1]) * (hand_q_3[2]))
    hand_E_logstick_3 -= (hand_q_3[0] * np.log(hand_q_3[0]) + \
                          hand_q_3[1] * np.log(hand_q_3[1]) + \
                          hand_q_3[2] * np.log(hand_q_3[2]))

    # since the errors seem to appear later down inthe term, let's check the value of _E_log_stick[4] but use the (hopefully correct) q
    E_logstick, q = InfiniteIBP._E_log_stick(tau, K)

    k = 3 # this is the index, so k = 4 in the equations (i.e. 1, 2, 3, 4), so q is a 4-dim Cat
    hand_E_logstick_4 = 0
    for m in range(k + 1):
        hand_E_logstick_4 += q[k][m] * digamma(tau[m, 1])
    for m in range(k):
        q_sum = 0
        for n in range(m + 1, k + 1):
            q_sum += q[k][n]
        hand_E_logstick_4 += q_sum * digamma(tau[m, 0])
    for m in range(k + 1):
        q_sum = 0
        for n in range(m, k + 1):
            q_sum += q[k][n]
        hand_E_logstick_4 -= q_sum * digamma(tau[m, 0] + tau[m, 1])
    for m in range(k + 1):
        hand_E_logstick_4 -= q[k][m] * (q[k][m] + EPS).log()

    # test that the computed q is equal to the hand-computed q
    assert np.abs((q[1, :2].numpy() - hand_q)).max() < 1e-6, "_E_log_stick doesn't compute q_2 correctly"
    assert np.abs((q[2, :3].numpy() - hand_q_3)).max() < 1e-6, "_E_log_stick doesn't compute q_3 correctly"

    # test that the hand-computed e logstick is equal to the computed e logstick
    assert np.abs(E_logstick[1].item() - hand_E_logstick).max() < 1e-5, "_E_logstick_2 isn't computed correctly"
    assert np.abs(E_logstick[2].item() - hand_E_logstick_3).max() < 1e-5, "_E_logstick_3 isn't computed correctly"
    assert np.abs(E_logstick[3].item() - hand_E_logstick_4.item()).max() < 1e-5, "_E_logstick_4 isn't computed correctly"

def test_elbo_components(inputs=None):
    """
    Test that various KL divergences are positive, and in the case of the
    approximate posterior q(v), compute it exactly in two ways and check
    that both give the same result.
    """
    if inputs is None:
        model = InfiniteIBP(4., 6, 0.1, 0.5, 36)
        model.init_z(10)
        model.train()

        X = torch.randn(10, 36)
    else:
        model, X = inputs

    a = model._1_feature_prob(model.tau).sum()
    b = model._2_feature_assign(model.nu, model.tau).sum()
    c = model._3_feature_prob(model.phi_var, model.phi).sum()
    d = model._4_likelihood(X, model.nu, model.phi_var, model.phi).sum()
    e = model._5_entropy(model.tau, model.phi_var, model.nu).sum()

    entropy_q_v = InfiniteIBP._entropy_q_v(model.tau)
    entropy_q_A = InfiniteIBP._entropy_q_A(model.phi_var)
    entropy_q_z = InfiniteIBP._entropy_q_z(model.nu)

    try:
        assert (a + b + c + d + e).item() not in (np.inf, -np.inf), "ELBO is inf"
    except AssertionError:
        print("a: ", a)
        print("b: ", b)
        print("c: ", c)
        print("d: ", d)
        print("e: ", e)
        print("entropy_q_v: ", entropy_q_v)
        print("entropy_q_A: ", entropy_q_A)
        print("entropy_q_z: ", entropy_q_z)
        raise

    # check the sign of the various KL divergences (summed, so less powerful than it could be)
    assert (a + entropy_q_v).item() <= 0, "KL(q(pi) || p(pi)) is negative"
    # assert (b + entropy_q_z).item() <= 10, "KL(q(z) || p(z)) is negative" # we give this one some tolerance
    assert (c + entropy_q_A).item() <= 0, "KL(q(A) || p(A)) is negative"
    assert (a + b + c + e).item() <= 0, "KL divergence between q(...) || p(...) is negative"

    # check the empirical value of the component KL divergences (this is a very strong test)
    from torch.distributions import Beta, kl_divergence
    p_pi = Beta(model.alpha, 1.)
    q_pi = Beta(model.tau[:, 0], model.tau[:, 1])

    try:
        assert (kl_divergence(q_pi, p_pi).sum() + (a + entropy_q_v)).abs() < 1e-3, "KL(q(pi) || p(pi)) is incorrect"
    except:
        import ipdb; ipdb.set_trace()

def test_cavi_updates_are_correct(inputs=None):
    """
    DOES NOT PASS: for nu, works for phi
    """

    # after doing the CAVI update for a single variable, we would expect that dELBO/dv is about 0.
    if inputs is None:
        model = InfiniteIBP(4., 6, 0.1, 0.5, 36)
        model.init_z(10)
        model.train()
        X = torch.randn(10, 36)
        N, D = 10, 36
    else:
        model, X = inputs
        N, D = X.shape[0]

    optimizer = torch.optim.SGD(model.parameters(), 0.01)

    # test that the updates for phi[2] is about right:
    # k = 2

    # model.eval()
    # model.cavi_phi(k, X)

    # model.train()
    # optimizer.zero_grad()
    # loss = model.elbo(X)
    # loss.backward()

    # assert model.phi.grad[k].abs().max().item() < 1e-4, "CAVI update for phi is wrong"

    # CAVI update for nu
    k = 2
    n = 1
    # model.eval()
    log_stick, _ = model._E_log_stick(model.tau, model.K)
    model.cavi_nu(n, k, X, log_stick)

    # # compute the ELBO
    model.train()
    optimizer.zero_grad()
    loss = model.elbo(X)
    loss.backward()

    print(model._nu.grad)
    # assert model._nu.grad[n][k].abs().max().item() < 1e-4, "CAVI update for nu is wrong"

def compute_q_Elogstick( tau , k ):
    import numpy as np
    import numpy.random as npr
    from numpy.random import beta
    import scipy.special as sps
    from scipy.special import gamma, digamma, gammaln
    from numpy.linalg import slogdet
    from scipy.misc import logsumexp

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
    return qs, Elogstick

def test_e_log_stick():
    """
    This test DOES NOT PASS
    """
    model = InfiniteIBP(4., 10, 0.1, 0.5, 36)
    model.init_z(10)

    K = model.K

    # take a lot of samples to get something working
    dist = Beta(model.tau.detach()[:, 0], model.tau.detach()[:, 1])
    samples = dist.sample((100000,))
    f = (1. - samples.cumprod(1)).log().mean(0)
    log_stick, q = model._E_log_stick(model.tau, model.K)

    jeffrey_q = np.zeros((K, K))
    jeffrey_log_stick = np.zeros((K,))
    for k in range(K):
        a, b = compute_q_Elogstick(model.tau.detach().numpy().T, k)
        jeffrey_q[k, :k+1] = a
        jeffrey_log_stick[k] = b

    print("old:     {}".format(jeffrey_log_stick))
    print("new:     {}".format(log_stick.detach().numpy()))
    print("samples: {}".format(f.detach().numpy()))

    import ipdb; ipdb.set_trace()

def test_vectorized_cavi():
    """
    OUT OF DATE (unnecessary?)
    """
    model = InfiniteIBP(4., 6, 0.1, 0.05, 36)
    model.init_z(10)

    X = torch.randn(10, 36)

    N, K, D = X.shape[0], model.K, model.D

    slow_phi_var = torch.zeros(K, D)
    slow_phi = torch.zeros(K, D)

    for k in range(K):
        precision = (1./(model.sigma_a ** 2) + model.nu[:, k].sum()/(model.sigma_n**2))
        slow_phi_var[k] = torch.ones(model.D) / precision
        s = 0
        # this can definitely be parallelized - it's probably really slow right now
        for n in range(N):
            s += model.nu[n][k] * (X[n] - (model.nu[n] @ model.phi - model.phi[k]))
        slow_phi[k] = s/((model.sigma_n ** 2) * precision)

    fast_phi_var = torch.zeros(K, D)
    fast_phi = torch.zeros(K, D)
    for k in range(K):
        precision = (1./(model.sigma_a ** 2) + model.nu[:, k].sum()/(model.sigma_n**2))
        fast_phi_var[k] = torch.ones(model.D) / precision
        s = (model.nu[:, k].view(N, 1) * (X - (model.nu @ model.phi - torch.ger(model.nu[:, k], model.phi[k])))).sum(0)
        fast_phi[k] = s/((model.sigma_n ** 2) * precision)
    print(slow_phi.detach().numpy())
    print(fast_phi.detach().numpy())
    assert (slow_phi - fast_phi).abs().max() < 1e-5, "Fast phi is wrong"

if __name__ == '__main__':
    test_cavi_updates_are_correct()