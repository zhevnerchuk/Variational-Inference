import torch
import numpy as np


class SVI():
    '''Class for black box stochastic variational inference
        https://arxiv.org/abs/1401.0118
    
    '''
    
    def __init__(self, data, prior_distr, var_distr, opt):
        '''Initialization
        
        Args:
            data: oserved data
            prior_distr: class for prior probabilistic model
                requred methods: log_likelihood_global(beta)
                                 log_likelihood_local(z, beta)
                                 log_likelihood_joint(x, z, beta)
            var_distr: class for variational distribution
                required methods: log_likelihood_global(beta)
                                  log_likelihood_local(z, idx)
                                  sample_global()
                                  sample_local(idx)
            opt: optimizer
        
        '''
        
        self.data = data
        self.prior_distr = prior_distr
        self.var_distr = var_distr
        self.opt = opt
        
    def bb_loss_(self, num_samples, batch_indices):
        '''Computing loss of BB SVI
        
        Args:
            num_samples: number of samples used for approximation
            batch_indices: indices of batch
        
        Returns:
            loss: Black Box loss function
        
        '''
        
        loss_global = torch.zeros(1, requires_grad=True)
        loss_local = torch.zeros(1, requires_grad=True)
        
        for i in range(num_samples):
            loss_global_sample = torch.zeros(1, requires_grad=True)
            loss_local_sample = torch.zeros(1, requires_grad=True)
            beta = self.var_distr.sample_global()
            for idx in batch_indices:
                z = self.var_distr.sample_local(beta, idx)
                x = self.data[idx]
                loss_global_sample = loss_global_sample - (self.prior_distr.log_likelihood_local(z, beta) + \
                                      self.prior_distr.log_likelihood_joint(x, z, beta))
                loss_local = loss_local + self.var_distr.log_likelihood_local(z, idx) * \
                              self.data.shape[0] * \
                              (self.prior_distr.log_likelihood_local(z, beta) + \
                               self.prior_distr.log_likelihood_joint(x, z, beta) - \
                               self.var_distr.log_likelihood_local(z, idx))
                    
            loss_global_sample = loss_global_sample * self.data.shape[0] / batch_indices.shape[0]
            loss_global_sample = loss_global_sample + self.prior_distr.log_likelihood_global(beta) - \
                                  self.var_distr.log_likelihood_global(beta)
            loss_global_sample = loss_global_sample * self.var_distr.log_likelihood_global(beta)
            loss_global = loss_global + loss_global_sample
            
        loss = -(loss_global + loss_local) / num_samples
        
        return loss
    
    def make_inference(self, num_steps=100, num_samples=10, batch_size=10, shuffle=False, print_progress=True):
        '''Making SVI
        
        Args:
            num_steps: int, number of epoches
            num_samples: int, number of samples used for ELBO approximation
            batch_size: int, size of one batch
            shuffle: boolean, if batch is shuffled every epoch or not
            print_progress: boolean, if True then progrss bar is printed
            
        '''
        
        for step in range(num_steps):
            
            if shuffle:
                indices = np.random.choice(self.data.shape[0], self.data.shape[0], False)
            else:
                indices = np.arange(self.data.shape[0])
                
            indices = np.split(indices, np.arange(batch_size, self.data.shape[0], batch_size))
                
            for batch_indices in indices:
                self.opt.zero_grad()
                loss = self.bb_loss_(num_samples, batch_indices)
                loss.backward()
                self.opt.step()
                
            if print_progress:
                if (int(25 * step / num_steps) != int(25 * (step - 1) / num_steps)):
                    print('.', end='')

