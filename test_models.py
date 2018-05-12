import numpy as np
import torch
import torch.nn as nn


class ToyPrior:
    def __init__(self):
        self.prior = torch.distributions.Normal(torch.zeros(1), torch.ones(1))
    
    def log_likelihood_global(self, beta):
        return self.prior.log_prob(beta)
    
    def log_likelihood_joint(self, x, z, beta):
        cond = torch.distributions.Normal(beta, torch.ones(1))
        return cond.log_prob(x)
    
    def log_likelihood_cond(self, x, z, beta):
        cond = torch.distributions.Normal(beta, torch.ones(1))
        return cond.log_prob(x)


class ToyVariationalDistribution:
    def __init__(self):
        self.mu = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.sigma = nn.Parameter(torch.ones(1), requires_grad=True)
        self.distr = torch.distributions.Normal(self.mu, self.sigma)
        self.parameters = [self.mu, self.sigma]
    
    def sample_global(self):
        return self.distr.rsample()
    
    def sample_local(self, beta, idx):
        return None
    
    def entropy(self):
        return self.distr.entropy()

    
def gen_data(mu = 2.5, num_samples = 1000, seed = 42):

    data = torch.Tensor(np.random.normal(mu, 1., size=num_samples))
    
    return data 