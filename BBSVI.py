import torch
import operator
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
            var_distr: class for variational distribution
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
        
        prior_distr requred methods: 
            log_likelihood_global(beta)
            log_likelihood_local(z, beta)
            log_likelihood_joint(x, z, beta)
        
        var_distr required methods: 
            log_likelihood_global(beta)
            log_likelihood_local(z, idx)
            sample_global()
            sample_local(idx)
             
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
                local_const_term *= self.data.shape[0]
                
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
        
        prior_distr requred methods: 
            log_likelihood_global(beta)
            log_likelihood_local(z, beta)
            log_likelihood_joint(x, z, beta)
        
        var_distr required methods: 
            log_likelihood_global(beta)
            log_likelihood_local(z, idx)
            sample_global(num_samples)
            sample_local(idx, num_samples)
        
        var_distr required attributes:
            global_parameters: list of global parameters
            local_parameters: list of local parameters, 
                i-th entry corresponds to i-th latent variable
        
        Args:
            num_samples: number of samples used for approximation
            batch_indices: indices of batch
        
        Returns:
            loss: Black Box loss function
        
        '''
        
        
        # TODO: add checking that num_samples > 1
        
        global_samples = [self.var_distr.sample_global() for _ in range(num_samples)]
        
        global_h = [torch.autograd.grad(self.var_distr.log_likelihood_global(global_samples[s]), 
                                        self.var_distr.global_parameters,
                                        retain_graph=True) for s in range(num_samples)]
        
        local_samples = []
        for idx in batch_indices:
            local_samples.append([self.var_distr.sample_local(global_samples[s], idx) for s in range(num_samples)])
        
        global_f = []
        local_f = [[] for idx in batch_indices]
        local_h = [[] for idx in batch_indices]
        for s, beta in enumerate(global_samples):
            multiplier = torch.zeros(1, requires_grad=False)
            for i, idx in enumerate(batch_indices):
                multiplier += self.prior_distr.log_likelihood_local(local_samples[i][s], beta) + \
                              self.prior_distr.log_likelihood_joint(self.data[idx], local_samples[i][s], beta)
                
                local_h[i].append(torch.autograd.grad(self.var_distr.log_likelihood_local(local_samples[i][s], idx), 
                                                      self.var_distr.local_parameters[idx], retain_graph=True))
                local_multiplier = self.prior_distr.log_likelihood_local(local_samples[i][s], beta) + \
                                   self.prior_distr.log_likelihood_joint(self.data[idx], local_samples[i][s], beta) - \
                                   self.var_distr.log_likelihood_local(local_samples[i][s], idx)
                local_f[i].append(tuple(local_multiplier.data * grad for grad in local_h[i][s]))
                
            multiplier *= self.data.shape[0] / batch_indices.shape[0]
            multiplier += self.prior_distr.log_likelihood_global(beta) - \
                          self.var_distr.log_likelihood_global(beta)
            global_f.append(tuple(multiplier.data * grad for grad in global_h[s]))
            
        global_a = self.count_a_(global_h, global_f)
        local_a = [self.count_a_(local_h[i], local_f[i]) for i, idx in enumerate(batch_indices)]
        
        global_loss = torch.zeros(1, requires_grad=True)
        local_loss = torch.zeros(1, requires_grad=True)
        
        for s, beta in enumerate(global_samples):
            
            global_const_term = torch.zeros(1, requires_grad=False)
            
            for i, idx in enumerate(batch_indices):
                x = self.data[idx]
                z = local_samples[i][s]
                
                local_const_term = self.prior_distr.log_likelihood_local(z, beta) + \
                                   self.prior_distr.log_likelihood_joint(x, z, beta) - \
                                   self.var_distr.log_likelihood_local(z, idx)
                local_const_term *= self.data.shape[0]
                local_const_term -= local_a[i]
                
                local_var_term = self.var_distr.log_likelihood_local(z, idx)
                
                local_loss = local_loss + local_var_term * local_const_term.data
                
                global_const_term += self.prior_distr.log_likelihood_local(z, beta) + \
                                     self.prior_distr.log_likelihood_joint(x, z, beta)
            
            global_const_term *= self.data.shape[0] / batch_indices.shape[0]
            global_const_term += self.prior_distr.log_likelihood_global(beta) - \
                                 self.var_distr.log_likelihood_global(beta) - \
                                 global_a
            
            global_var_term = self.var_distr.log_likelihood_global(beta)
            global_loss = global_loss + global_var_term * global_const_term.data
                    
        loss = -(global_loss + local_loss) / num_samples
        
        return loss
        
        
    def count_a_(self, h, f):
        '''Given f and h from BB SVI II, computes a* based on an unbiased
           estimators of its components covariances
           
        Args:
            h: h computed for S samples
            f: f computed for S samples
            
        Returns:
            a: a* from BB SVI II, given by sum(cov(f_i, h_i)) / sum(var(h_i))
        
        '''

        num_samples = len(h)
        
        h_means = tuple(torch.zeros(grad.shape, requires_grad=False) for grad in h[0])
        for sample_grads in h:
            h_means = tuple(map(operator.add, h_means, sample_grads))
        h_means = tuple(grads / num_samples for grads in h_means)
        
        f_means = tuple(torch.zeros(grad.shape, requires_grad=False) for grad in f[0])
        for sample_grads in f:
            f_means = tuple(map(operator.add, f_means, sample_grads))
        f_means = tuple(grads / num_samples for grads in f_means)
    
        h_var = tuple(torch.zeros(grad.shape, requires_grad=False) for grad in h[0])
        f_h_cov = tuple(torch.zeros(grad.shape, requires_grad=False) for grad in h[0])

        for s in range(num_samples):
            
            h_term = tuple(map(operator.sub, h[s], h_means))
            f_term = tuple(map(operator.sub, f[s], f_means))
            h_var = tuple(map(operator.add, h_var, tuple(map(operator.mul, h_term, h_term))))
            f_h_cov = tuple(map(operator.add, f_h_cov, tuple(map(operator.mul, h_term, f_term))))
        h_var = tuple(var / (num_samples - 1) for var in h_var)
        f_h_cov = tuple(cov / (num_samples - 1) for cov in f_h_cov)
        
        var_term = sum(torch.sum(var) for var in h_var)
        cov_term = sum(torch.sum(cov) for cov in f_h_cov)
        
        a = cov_term / var_term
        
        return a
        
        

    def make_inference(self, num_steps=100, num_samples=10, batch_size=10, 
                       loss='bb1', shuffle=False, print_progress=True):
        '''Making SVI
        
        Args:
            num_steps: int, maximum number of epoches
            tol: required tolerance
            num_samples: int, number of samples used for ELBO approximation
            batch_size: int, size of one batch
            loss: string, loss function, currently avaliable bb1 or bb2
            shuffle: boolean, if batch is shuffled every epoch or not
            print_progress: boolean, if True then progrss bar is printed
            
        '''
        
        if loss == 'bb1':
            loss_func = self.bb1_loss_
        elif loss == 'bb2':
            loss_func = self.bb2_loss_
        else:
            raise Exception('No such loss avaliable')
        
        for step in range(num_steps):
            
            if shuffle:
                indices = np.random.choice(self.data.shape[0], self.data.shape[0], False)
            else:
                indices = np.arange(self.data.shape[0])
                
            indices = np.split(indices, np.arange(batch_size, self.data.shape[0], batch_size))
                
            for batch_indices in indices:
                self.opt.zero_grad()
                loss = loss_func(num_samples, batch_indices)
                loss.backward()
                self.opt.step()

            if print_progress:
                if (int(100 * step / num_steps) != int(100 * (step - 1) / num_steps)):
                    print('.', end='')
        
        if print_progress:
            print()

