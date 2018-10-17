import torch
from torch import nn
from torch import digamma
from torch.distributions import MultivariateNormal as MVN
from torch.distributions import Bernoulli as Bern
from torch.distributions import Poisson as Pois
from torch.distributions import Categorical as Categorical

class UncollapsedGibbsIBP(nn.Module):
    ################################################
    ########### UNCOLLAPSED GIBBS SAMPLER ##########
    ################################################
    ### Depends on a few self parameters but could##
    ### be made a standalone script if need be #####
    ###############################################
    def __init__(self, alpha, K, sigma_a, sigma_n, D):
        super(InfiniteIBP, self).__init__()

        # idempotent - all are constant and have requires_grad=True
        self.alpha = torch.tensor(alpha)
        self.K = torch.tensor(K)
        self.sigma_a = torch.tensor(sigma_a)
        self.sigma_n = torch.tensor(sigma_n)
        self.D = torch.tensor(D)

    def likelihood_given_ZA(X,Z,A):
        '''
        p(X|Z,A) = 1/([2*pi*sigma_n^2]^(ND/2)) *
               exp([-1/(2*sigma_n^2)] tr((X-ZA)^T(X-ZA)))

        Used in Zik resample
        '''
        N = X.size()[0]
        D = X.size()[1]
        pi = math.pi()
        sig_n2 = self.sigma_n.pow(2)
        first_term = (1./(2*pi*sig_n2)).pow(N*D/2.) 
        second_term = ((-1./(2*sig_n2)) * \
            torch.trace((X-Z@A).transpose(0,1)@(X-Z@A))).exp()
        return first_term * second_term

    def resample_Zik(X,Z,A,i,k):
        '''
        m = number of observations not including
            Z_ik containing feature k
        p(z_nk=1) = m / (N-1)
        p(z_nk=1|Z_-nk,A,X) propto p(z_nk=1)p(X|Z,A)
        '''
        # Prior on Z[i][k]
        Z_k = Z[:,k]
        m = Z_k.sum() - Z_k[i]

        # If Z_nk were 0
        Z_if_0 = Z.clone()
        Z_if_0[i,k] = 0
        prior_if_0 = 1 - (m/(N-1))
        likelihood_if_0 = likelihood_given_ZA(X,Z_if_0,A)
        score_if_0 = prior_if_0*likelihood_if_0

        # If Z_nk were 1
        Z_if_1 = Z.clone()
        Z_if_1[i,k]=1
        prior_if_1 = (m/(N-1))
        likelihood_if_1 = likelihood_given_ZA(X,Z_if_1,A)
        score_if_1 = prior_if_1*likelihood_if_1

        # Normalize and Sample new Z[i][k]
        denominator = score_if_0 + score_if_1
        p_znk = Bern(torch.tensor([score_if_1 / denominator]))
        return p_znk.sample()


    def renormalize_log_probs(log_probs):
        log_probs = log_probs - log_probs.max()
        likelihoods = log_probs.exp()
        return likelihoods / likelihoods.sum()

    def k_new(X,Z,A,i,truncation):
        log_probs = toch.zeros(truncation)
        poisson_probs = torch.zeros(truncation)
        N,K = Z.size()
        D = X.size()[1]
        X_minus_ZA = X - Z@A
        X_minus_ZA_T = X_minus_ZA.transpose(0,1)
        sig_n = self.sigma_n
        sig_a = self.sigma_a
        p_k_new = Pois(torch.tensor([self.alpha/N]))
        for j in range(truncation):
            if j==0:
                log_probs[j]=0.0
            else:
                w = torch.ones(j,j) + (sig_n/sig_a).pow(2)*torch.eye(j)
                # alternative: torch.potrf(a).diag().prod()
                w_numpy = w.numpy()
                sign,log_det = np.linalg.slogdet(w_numpy)
                log_det = torch.tensor([log_det])
                
                # Note this is in log space
                first_term = j*D*(sig_n/sig_a).log() - ((D/2)*log_det)
                 
                second_term = 0.5*
                        torch.trace( \
                        X_minus_ZA_T @ \
                        Z[:,-j:] @ \
                        w.inverse() @ \
                        Z[:,-j:].transpose(0,1) @ \
                        X_minus_ZA) / \
                        sig_n.pow(2)
                log_probs[j] = first_term + second_term
            poisson_probs[j] = p_k_new.log_prob(j).exp()
            Z = torch.cat((Z,torch.zeros(N).transpose(0,1)),0)
            Z[i][-1]=1
        probs = renormalize_log_probs(log_probs)
        poisson_probs = poisson_probs / poisson_probs.sum()
        sample_probs = torch.multiply(probs,poisson_probs)
        sample_probs = sample_probs / sample_probs.sum()
        Z = Z[:,:-truncation]
        assert Z.size()[1] == K
        posterior_k_new = Categorical(sample_probs)
        return posterior_k_new.sample()

    def resample_Z(X,Z,A):
        N = X.size()[0]
        K = A.size()[0]
        for i in range(N):
            for k in range(K):
                Z[i,k] = resample_Zik(X,Z,A,i,k)
            truncation=100
            k_new = k_new(X,Z,A,i,truncation)
            if k_new > 0:
                Z = torch.cat((Z,torch.zeros(N,k_new)),0)
                for j in range(k_new):
                    Z[i][-(j+1)] = 1
                Anew = A_new(X,k_new,Z,A)
                A = torch.cat((A,Anew),0)
        return Z,A

    def resample_A(X,Z):
        '''
        mu = (Z^T Z + (sigma_n^2 / sigma_A^2) I )^{-1} Z^T  X
        Cov = sigma_n^2 (Z^T Z + (sigma_n^2/sigma_A^2) I)^{-1}
        p(A|X,Z) = N(mu,cov)
        '''
        N,D = X.size()
        K = Z.size()[0]
        ZTZ = Z.transpose(0,1)@Z
        I = torch.eye(K)
        sig_n = self.sigma_n
        sig_a = self.sigma_a
        mu = (ZTZ + (sig_n/sig_a).pow(2)*I).inverse()@Z@X
        cov = sig_n.pow(2)*(ZTZ + (sig_n/sig_a).pow(2)*I).inverse()
        A = torch.zeros(K,D)
        for d in range(D):
            p_A = MVN(mu[:,d],cov)
            A[:,d] = p_A.sample()
        return A

    def A_new(X,k_new,Z,A):
        N,D = X.size()
        K = Z.size()[1]
        assert K == A.size()[0]+k_new
        ones = torch.ones(k_new,k_new)
        I = torch.eye(k_new)
        sig_n = self.sigma_n
        sig_a = self.sigma_a
        Z_new = Z[:,-k_new:]
        Z_old = Z[:,:-k_new]
        Z_new_T = Z_new.transpose(0,1)
        # mu is k_new x D
        mu = (ones + (sig_n/sig_a).pow(2)*I).inverse() @ \
            Z_new_T @ (X - Z_old@A)
        # cov is k_new x k_new
        cov = sig_n.pow(2) * (ones + (sig_n/sig_a).pow(2)*I).inverse()
        A_new = torch.zeros(k_new,D)
        for d in range(D):
            p_A = MVN(mu[:,d],cov)
            A_new[:,d] = p_A.sample()
        return A_new

    def init_A(K,D):
        # Sample from prior p(A_k)
        Ak_mean = torch.zeros(D)
        Ak_cov = self.sigma_a.pow(2)*torch.eye(D)
        p_Ak = MVN(Ak_mean, Ak_cov)
        A = torch.zeros(K,D)
        for k in range(K):
            A[k] = p_Ak.sample()
        return A

    def left_order_form(Z):
        Z_numpy = Z.clone()
        twos = np.ones(Z_numpy.shape[0])*2.0
        twos[0] = 1.0
        powers = np.cumprod(twos)[::-1]
        values = np.dot(powers,Z_numpy)
        idx = values.argsort()[::-1]
        return torch.tensor(np.take(Z_numpy,idx,axis=1))
        
    def init_Z(N=20,alpha=2.0,K=1000):
        Z = torch.zeros(N,K)
        total_dishes_sampled = 0
        for i in range(N):
            selected = random(total_dishes_sampled) < Z[:,:total_dishes_sampled].sum() / (i+1.)
            Z[i][:total_dishes_sampled][selected]=1.0
            p_new_dishes = Pois(torch.tensor([self.alpha(i+1)]))
            new_dishes = p_new_dishes.sample()
            if total_dishes_sampled + new_dishes >= K:
                new_dishes = K - total_dishes_sampled
            Z[i][total_dishes_sampled:total_dishes_sampled+new_dishes]=1.0
            total_dishes_sampled += new_dishes
        return left_order_form(Z)

    def trim_ZA(Z,A):
        sums = Z.torch.sum(axis=0)
        to_keep = torch.where(sums !=0)[0]
        return Z[:,to_keep],A[to_keep,:]

    def gibbs(self, X, iters):
        N = X.size()[0]
        D = X.size()[1]
        K = self.K
        Z = torch.zeros(N, K)
        Z = init_Z(N,self.alpha, K)
        A = init_A(K,D) 
        for iteration in range(iters):
            A = resample_A(X,Z)
            Z,A = resample_Z(X,Z,A)
            Z,A = trim_ZA(Z,A)
        # TODO we'll need to return the full chain in general,
        # but for now just return the last sample
        return A

def fit_ugibbs_to_ggblocks():
    from data import generate_gg_blocks, generate_gg_blocks_dataset
    N = 100
    X = generate_gg_blocks_dataset(N, 0.5)

    model = UncollapsedGibbsIBP(4., 6, 0.1, 0.5, 36) # these are bad parameters
    model.train()

    last_sample_A = model.gibbs(X, iters=100)
    visualize_A(last_sample_A.view(-1, 6, 6).detach().numpy()[0])

if __name__ == '__main__':
    """
    python src/mcmc.py will just check that the model works on a ggblocks dataset
    """
    fit_ugibbs_to_ggblocks()