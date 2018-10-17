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

    def _gibbs_likelihood_given_ZA(X,Z,A):
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

    def _gibbs_resample_Zik(X,Z,A,i,k):
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
        likelihood_if_0 = _gibbs_likelihood_given_ZA(X,Z_if_0,A)
        score_if_0 = prior_if_0*likelihood_if_0

        # If Z_nk were 1
        Z_if_1 = Z.clone()
        Z_if_1[i,k]=1
        prior_if_1 = (m/(N-1))
        likelihood_if_1 = _gibbs_likelihood_given_ZA(X,Z_if_1,A)
        score_if_1 = prior_if_1*likelihood_if_1

        # Normalize and Sample new Z[i][k]
        denominator = score_if_0 + score_if_1
        p_znk = Bern(torch.tensor([score_if_1 / denominator]))
        return p_znk.sample()


    def _gibbs_renormalize_log_probs(log_probs):
        log_probs = log_probs - log_probs.max()
        likelihoods = log_probs.exp()
        return likelihoods / likelihoods.sum()

    def _gibbs_k_new(X,Z,A,i,truncation):
        log_probs = toch.zeros(truncation)
        poisson_probs = torch.zeros(truncation)
        N,K = Z.size()
        D = X.size()[1]
        X_minus_ZA = X - Z@A
        X_minus_ZA_T = X_minus_ZA.transpose(0,1)
        sig_n = self.sigma_n
        sig_a = self.sigma_a
        for j in range(truncation):
            if j==0:
                log_probs[j]=0.0
            else:
                w = torch.ones(j,j) + (sig_n/sig_a).pow(2)*torch.eye(j)
                sign,ldet = np.linalg.slogdet(w)
                first_term = j*D*(sig_n/sig_a).log() - ldet
                second_term = 0.5*torch.trace( \
                        X_minus_ZA_T @ \
                        Z[:,-j:] @ \
                        w.inverse() @ \
                        Z[:.-j:] @ \
                        X_minus_ZA) / \
                        sig_n.pow(2)
                log_probs[j] = first_term + second_term
            p_k_new = Pois(torch.tensor([self.alpha/N]))
            poisson_probs[j] = p_k_new.log_prob(j).exp()

            Z = np.hstack((Z,torch.zeros(N).transpose(0,1)))
            Z[i][-1]=1

        probs = _gibbs_renormalize_log_probs(log_probs)
        poisson_probs = poisson_probs / poisson_probs.sum()
        sample_probs = torch.multiply(probs,poisson_probs)
        sample_probs = sammple_probs / sample_probs.sum()
        Z = Z[:,:-truncation]
        return np.random.choice(truncation,p=sample_probs)

    # TODO Finish Implementation
    def _gibbs_resample_Z(X,Z,A):
        N = X.size()[0]
        K = A.size()[0]
        for i in range(N):
            for k in range(K):
                Z[i,k] = _gibbs_resample_Zik(X,Z,A,i,k)
            truncation=100
            k_new = _gibbs_k_new(X,Z,A,i,truncation)
            if k_new > 0:
                Z = np.hstack((Z,toch.zeros(N,k_new)))
                for j in range(k_new):
                    Z[i][-(j+1)] = 1
                Anew = _gibbs_A_new(X,k_new,Z,A)
                A = np.concatenate((A,Anew))
        return Z

    def _gibbs_resample_A(X,Z):
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

    # TODO: Debug
    def _gibbs_A_new(X,k_new,Z,A):
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

        mu = (ones + (sig_n/sig_a).pow(2)*I).inverse() @ \
            Z_new_T @ (X - Z_old@A)

        cov = sig_n.pow(2) * (ones + (sig_n/sig_a).pow(2)*I).inverse()
        A_new = torch.zeros(K,D)
        for d in range(D):
            p_A = MVN(mu[:,d],cov)
            A_new[:,d] = p_A.sample()
        return A_new


    def _gibbs_init_A(K,D):
        # Sample from p(A_k)
        Ak_mean = torch.zeros(D)
        Ak_cov = self.sigma_a.pow(2)*torch.eye(D)
        p_Ak = MVN(Ak_mean, Ak_cov)
        A = torch.zeros(K,D)
        for k in range(K):
            A[k] = p_Ak.sample()
        return A

    # TODO: Debug
    def gibbs(self, X, iters):
        N = X.size()[0]
        D = X.size()[1]
        K = self.K
        Z = torch.zeros(N, self.K)
        A = _gibbs_init_A(K,D) 
        for iteration in range(iters):
            A = _gibbs_resample_A(X,Z)
            Z = _gibbs_resample_Z(X,Z,A)

        # we'll need to return the full chain in general,
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