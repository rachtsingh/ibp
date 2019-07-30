### Experimental research code

This repository attempts to implement several kinds of inference for generative models with an Indian Buffet Process prior (e.g. nonparametric VAEs, etc.). There's a strong focus on correct implementations and testing, since it is extremely easy to make a mistake in this area.

Inference methods implemented:

- Gibbs sampling
  - collapsed Gibbs (i.e. marginalize out A) (Griffiths & Ghahramani (2005), though the derivation is in Doshi-Velez (2009))
  - uncollapsed Gibbs
- Slice sampling (unimplemented)
- Variational inference (VI)
  - Coordinate-ascent VI (CAVI) - the "older" method of VI (derived in Doshi-Velez (2009))
  - Stochastic VI (SVI) - see Hoffman (2013), essentially subsampling data
  - Autograd on the exact ELBO (ADVI-exact): from Doshi-Velez (2009) we have an exact way to compute the ELBO (without sampling from q), we can do essentially gradient-based maximum likelihood estimation here. This is extremely fast and stable (this method is implemented in Pyro as well)
  - Autograd on sampled ELBO - as a comparison, we sample some variables from q
  - Amortized VI (AVI or VAE) - we also fit a map from x to q(z; \lambda(x)), which helps scale obviously (number of parameters is fixed rather than O(n)) but also allows data sharing.
 
A very good summary of various methods
[Zhang - Advances in Variational Inference, 2017](https://arxiv.org/pdf/1711.05597.pdf)  

[Liu - Stein Variational Gradient Descent: A General
Purpose Bayesian Inference Algorithm, 2016](https://arxiv.org/pdf/1608.04471.pdf)  

[Ranganath,Altosaar,Tran,Blei - Operator Variational Inference, 2016](https://arxiv.org/pdf/1610.09033.pdf)  

[Altosaar, Ranganath, Blei - Proximity Variational Inference, 2017](https://arxiv.org/pdf/1705.08931.pdf)  

[Vikram, Hoffman, Johnson - LORACs prior for VAE, 2018](https://arxiv.org/pdf/1810.06891.pdf)
