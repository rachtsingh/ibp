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

def test_q_E_logstick(inputs=None):
    """
    Test the computation of the multinomial bound in Finale's tech report
    by computing it by hand for k=2 and k=3
    """

    from scipy.special import digamma as dgm

    if inputs is None:
        K = 6
        tau = 0.01 + nn.Softplus()(torch.randn(K, 2))
    else:
        tau, K = inputs

    # compute, by hand, the distribution q_2
    hand_q = np.zeros(2)
    taud = tau.clone().detach().numpy()
    hand_q[0] = dgm(taud[0, 1]) - dgm(taud[0, 1] + taud[0, 0])
    hand_q[1] = dgm(taud[1, 1]) + dgm(taud[0, 0]) - \
        (dgm(taud[0, 0] + taud[0, 1]) + dgm(taud[1, 0] + taud[1, 1]))
    hand_q = np.exp(hand_q)
    hand_q /= hand_q.sum()

    # let's check E_log_stick for k=2 (i.e. index 1)
    hand_E_logstick = 0
    hand_E_logstick += (hand_q[0] * dgm(taud[0, 1]) + hand_q[1] * dgm(taud[1, 1]))
    hand_E_logstick += (hand_q[1] * dgm(taud[0, 0]))
    hand_E_logstick -= (dgm(taud[0, 0] + taud[0, 1]) * (hand_q[0] + hand_q[1]) + \
                        dgm(taud[1, 0] + taud[1, 1]) * (hand_q[1]))
    hand_E_logstick -= (hand_q[0] * np.log(hand_q[0]) + hand_q[1] * np.log(hand_q[1]))

    # for q_3
    hand_q_3 = np.zeros(3)
    hand_q_3[0] = dgm(taud[0, 1]) - dgm(taud[0, 1] + taud[0, 0])
    hand_q_3[1] = dgm(taud[1, 1]) + dgm(taud[0, 0]) - \
        (dgm(taud[0, 0] + taud[0, 1]) + dgm(taud[1, 0] + taud[1, 1]))
    hand_q_3[2] = dgm(taud[2, 1]) + (dgm(taud[0, 0]) + dgm(taud[1, 0])) - \
        (dgm(taud[0, 0] + taud[0, 1]) + dgm(taud[1, 0] + taud[1, 1]) + dgm(taud[2, 0] + taud[2, 1]))
    hand_q_3 = np.exp(hand_q_3)
    hand_q_3 /= hand_q_3.sum()

    # let's check E_log_stick for k=2 (i.e. index 1)
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

    E_logstick, q = InfiniteIBP._E_log_stick(tau, K)
    assert np.abs((q[1, :2].numpy() - hand_q)).max() < 1e-6, "_E_log_stick doesn't compute q_2 correctly"
    assert np.abs((q[2, :3].numpy() - hand_q_3)).max() < 1e-6, "_E_log_stick doesn't compute q_3 correctly"

    try:
        assert np.abs(E_logstick[1] - hand_E_logstick).max() < 1e-5, "_E_logstick_2 isn't computed correctly"
        assert np.abs(E_logstick[2] - hand_E_logstick_3).max() < 1e-5, "_E_logstick_3 isn't computed correctly"
    except AssertionError:
        print(E_logstick[2].item())
        print(hand_E_logstick)
        print(hand_E_logstick_3)
        raise

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

    # # test that the updates for phi[2] is about right:
    # k = 2

    # # CAVI update for phi[2]
    # model.eval()
    # precision = (1./(model.sigma_a ** 2) + model.nu[:, k].sum()/(model.sigma_n**2))
    # model.phi_var[k] = torch.ones(model.D) / precision
    # s = (model.nu[:, k].view(N, 1) * (X - (model.nu @ model.phi - torch.ger(model.nu[:, k], model.phi[k])))).sum(0)
    # model.phi[k] = s/((model.sigma_n ** 2) * precision)

    # # compute the ELBO
    # model.train()
    # optimizer.zero_grad()
    # loss = model.elbo(X)
    # loss.backward()

    # assert model.phi.grad[k].abs().max().item() < 1e-4, "CAVI update for phi is wrong"

    # CAVI update for nu
    k = 2
    n = 1
    model.eval()
    first_term = (digamma(model.tau[:k+1, 0]) - digamma(model.tau.sum(1)[:k+1])).sum() - \
        model._E_log_stick(model.tau, model.K)[0][k]

    # this line is really slow
    other_prod = 0
    for l in range(model.K):
        if k != l:
            other_prod += model.nu[n][l] * model.phi[l]
    # other_prod = (model.nu[n] @ model.phi - model.nu[n, k] * model.phi[k])
    second_term = ((-0.5 / (model.sigma_n ** 2)) * (model.phi_var[k].sum() + model.phi[k].pow(2).sum())) + \
        (model.phi[k] @ (X[n] - other_prod)) / (model.sigma_n ** 2)
    model.nu[n][k] = nn.Sigmoid()(first_term + second_term)
    # model.nu[n][k] = 1./(1. + (-1 * (first_term + second_term)).exp())

    # compute the ELBO
    model.train()
    optimizer.zero_grad()
    loss = model.elbo(X)
    loss.backward()

    print(model._nu.grad)

    assert model._nu.grad[n][k].abs().max().item() < 1e-4, "CAVI update for nu is wrong"

def test_e_log_stick():
    model = InfiniteIBP(4., 6, 0.1, 0.5, 36)
    model.init_z(10)

    # take a lot of samples to get something working
    dist = Beta(model.tau[:, 0], model.tau[:, 1])
    samples = dist.sample((1000000,))
    f = (1. - samples.cumprod(1)).log().mean(0)
    log_stick = model._E_log_stick(model.tau, model.K)[0]

    import ipdb; ipdb.set_trace()

def test_vectorized_cavi():
    model = InfiniteIBP(4., 6, 0.1, 0.5, 36)
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
    test_e_log_stick()