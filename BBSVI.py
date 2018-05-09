import torch
import numpy as np


class SVI():
    '''Class for black box stochastic variational inference
        https://arxiv.org/abs/1401.0118
    
    '''
    
    def __init__(self, data, prior_distr, var_distr, opt, scheduler=None):
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
            scheduler: scheduler for an optimizer
        
        '''
        
        self.data = data
        self.prior_distr = prior_distr
        self.var_distr = var_distr
        self.opt = opt
        self.scheduler = scheduler      

    def bb1_loss_(self, num_samples, batch_indices):
        '''Computing loss of BB SVI 1
        
        Args:
            num_samples: number of samples used for approximation
            batch_indices: indices of batch
        
        Returns:
            loss: Black Box loss function
        
        '''
        
        global_loss = torch.zeros(1, requires_grad=True)
        local_loss = torch.zeros(1, requires_grad=True)
        
        for i in range(num_samples):
            
            beta = self.var_distr.sample_global()
            global_const_term = torch.zeros(1, requires_grad=False)
            
            for idx in batch_indices:
                x = self.data[idx]
                z = self.var_distr.sample_local(beta, idx)
                
                local_const_term = self.prior_distr.log_likelihood_local(z, beta) + \
                                   self.prior_distr.log_likelihood_joint(x, z, beta) - \
                                   self.var_distr.log_likelihood_local(z, idx)
                
                local_var_term = self.var_distr.log_likelihood_local(z, idx)
                
                local_loss = local_loss + local_var_term * local_const_term.data
                
                global_const_term += self.prior_distr.log_likelihood_local(z, beta) + \
                                     self.prior_distr.log_likelihood_joint(x, z, beta)
            
            global_const_term *= self.data.shape[0] / batch_indices.shape[0]
            global_const_term += self.prior_distr.log_likelihood_global(beta) - \
                                 self.var_distr.log_likelihood_global(beta)
            
            global_var_term = self.var_distr.log_likelihood_global(beta)
            global_loss = global_loss + global_var_term * global_const_term.data
                    
        loss = -(global_loss + local_loss) / num_samples
        
        return loss
    
    def bb2_loss_(self, num_samples, batch_indices):
        '''Computing loss of BB SVI 2, which has lower variance compare to BB SVI 1
        
        Args:
            num_samples: number of samples used for approximation
            batch_indices: indices of batch
        
        Returns:
            loss: Black Box loss function
        
        '''
        
        loss_global = torch.zeros(1, requires_grad=True)
        loss_local = torch.zeros(1, requires_grad=True)
        
        global_samples = [self.var_distr.sample_global() for _ in range(num_samples)]
        a = torch.autograd.grad(self.var_distr.log_likelihood_global(global_samples[0]), 
                                self.var_distr.parameters, 
                                allow_unused=True)
        a = [torch.zeros(1) if x is None else x for x in a]
        
        b = torch.autograd.grad(self.var_distr.log_likelihood_global(global_samples[1]), 
                                self.var_distr.parameters, 
                                allow_unused=True)
        b = [torch.zeros(1) if x is None else x for x in b]
        c = tuple(map(operator.add, a, b))
        print(a)
        print(b)
        print(c)
        raise Exception
        for idx in batch_indices:
            pass

    def make_inference(self, num_steps=100, num_samples=10, batch_size=10, shuffle=False, print_progress=True):
        '''Making SVI
        
        Args:
            num_steps: int, maximum number of epoches
            tol: required tolerance
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
                loss = self.bb1_loss_(num_samples, batch_indices)
                loss.backward()
                self.opt.step()

            if print_progress:
                if (int(25 * step / num_steps) != int(25 * (step - 1) / num_steps)):
                    print('.', end='')
        
        if print_progress:
            print()

