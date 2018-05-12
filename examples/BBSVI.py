import torch
import operator
import numpy as np
from collections import defaultdict


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


    def check_methods(self, loss):

        """Check if provided model supports loss
        
        Args:
            loss: string, name of loss

        Returns:
            flag: boolean, if model supports loss
            message: string, error source if model does not support loss
            methods_to_implement: dict, methods to implement if model does not support loss

        """

        methods = { 
            'bb1': {'prior_distr' : ['log_likelihood_global', 'log_likelihood_local', 'log_likelihood_joint'], 
                    'var_distr' : ['log_likelihood_global', 'log_likelihood_local', 'sample_global', 'sample_local']},

            'bb2': {'prior_distr' : ['log_likelihood_global', 'log_likelihood_local', 'log_likelihood_joint'], 
                    'var_distr' : ['log_likelihood_global', 'log_likelihood_local', 'sample_global', 'sample_local', 
                                  'global_parameters', 'local_parameters']},

            'entropy': {'prior_distr' : ['log_likelihood_global', 'log_likelihood_joint'], 
                        'var_distr' : ['entropy', 'sample_global', 'sample_local']},

            'kl': {'prior_distr' : ['log_likelihood_cond'],
                    'var_distr' : ['sample_global', 'sample_local']}
        }

        flag = True
        message = "OK"
        methods_to_implement = defaultdict(list)

        try:

            cur_loss = methods[loss]

            for prior_method in cur_loss['prior_distr']:
                if not hasattr(self.prior_distr, prior_method):
                    methods_to_implement['prior_distr'].append(prior_method)

            for var_method in cur_loss['var_distr']:
                if not hasattr(self.var_distr, var_method):
                    methods_to_implement['var_distr'].append(var_method)

        except KeyError:
            flag = False
            message = "We do not support this loss: {0}".format(loss)

        if methods_to_implement:
            flag = False

            message = "The following methods should be implemented:"

        return flag, message, methods_to_implement


    def make_inference(self, num_steps=100, num_samples=10, batch_size=10, 
                       loss='bb1', discounter_schedule=None, kl=None,
                       shuffle=False, print_progress=True, callback=None,
                       retain_graph=False):
        '''Making SVI
        
        Args:
            num_steps: int, maximum number of epoches
            tol: required tolerance
            num_samples: int, number of samples used for ELBO approximation
            batch_size: int, size of one batch
            loss: string, loss function, currently avaliable bb1 or bb2
            discounter_schedule: used only for 'entropy' loss, None or torch tensor
                of size num_steps, discounter_schedule[i] is a discounter for an
                analytically-computed term at step i 
            kl: None or callable, compute KL divergency between variational and prior distributions,
                required only for 'kl' loss
            shuffle: boolean, if batch is shuffled every epoch or not
            print_progress: boolean, if True then progrss bar is printed
            callback: None or callable, if not None, applied to loss after every iteration
            retain_graph: boolean, passed to loss.backward()

        '''

        flag, message, methods_to_implement = self.check_methods(loss)
        if not flag:
        	raise Exception(message + '\n' + \
        		 '\n'.join([key + ':\t' + \
        		 ', '.join(methods_to_implement[key]) for key in methods_to_implement.keys()]))
        
        kwargs = lambda x: {}
        if loss == 'bb1':
            loss_func = self.bb1_loss_
        elif loss == 'bb2':
            loss_func = self.bb2_loss_
        elif loss == 'entropy':
        	loss_func = self.entropy_form_loss_
        	if discounter_schedule is not None:
        		kwargs = lambda x: {'discounter': discounter_schedule[x]}
        elif loss == 'kl':
        	loss_func = self.kl_form_loss_
        	if kl is not None:
        		if discounter_schedule is not None:
        		     kwargs = lambda x: {'kl': kl, 'discounter': discounter_schedule[x]}
        		else:
        			kwargs = lambda x: {'kl': kl}
        	else:
        		raise Exception('You should provide a function computing KL-divergency to use \'kl\' loss')
        

        for step in range(num_steps):
            
            if shuffle:
                indices = np.random.choice(self.data.shape[0], self.data.shape[0], False)
            else:
                indices = np.arange(self.data.shape[0])
                
            indices = np.split(indices, np.arange(batch_size, self.data.shape[0], batch_size))
                
            for batch_indices in indices:
                self.opt.zero_grad()
                loss = loss_func(num_samples, batch_indices, **kwargs(step))
                if callback is not None:
                	callback(loss)
                loss.backward(retain_graph=retain_graph)
                self.opt.step()

            if self.scheduler is not None:
            	self.scheduler.step()


            if print_progress:
                if (int(100 * step / num_steps) != int(100 * (step - 1) / num_steps)):
                    print('.', end='')
        
        if print_progress:
            print()



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
            sample_local(beta, idx)
             
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
            global_const_term = torch.zeros(1, requires_grad=True)
            
            for idx in batch_indices:
                x = self.data[idx]
                z = self.var_distr.sample_local(beta, idx)
                
                local_const_term = self.prior_distr.log_likelihood_local(z, beta) + \
                                   self.prior_distr.log_likelihood_joint(x, z, beta) - \
                                   self.var_distr.log_likelihood_local(z, idx).data
                local_const_term = local_const_term * self.data.shape[0]
                
                local_var_term = self.var_distr.log_likelihood_local(z, idx)
                
                local_loss = local_loss + local_var_term * local_const_term
                
                global_const_term = global_const_term + \
                					self.prior_distr.log_likelihood_local(z, beta) + \
                                    self.prior_distr.log_likelihood_joint(x, z, beta)
            
            global_const_term = global_const_term * self.data.shape[0] / batch_indices.shape[0]
            global_const_term = global_const_term + \
            					self.prior_distr.log_likelihood_global(beta) - \
                                self.var_distr.log_likelihood_global(beta).data
            
            global_var_term = self.var_distr.log_likelihood_global(beta)
            global_loss = global_loss + global_var_term * global_const_term
                    
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
            sample_global()
            sample_local(beta, idx)
        
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
        
        
        if num_samples == 1:
        	raise Exception('BB2 loss is avaliable only with num_samples > 1')
        
        global_samples = [self.var_distr.sample_global() for _ in range(num_samples)]

        global_h = [handle_nones(torch.autograd.grad(self.var_distr.log_likelihood_global(global_samples[s]), 
                                                     self.var_distr.global_parameters,
                                                     retain_graph=True,
                                                     allow_unused=True)) for s in range(num_samples)]
        

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
                
                
                local_h[i].append(handle_nones(torch.autograd.grad(self.var_distr.log_likelihood_local(local_samples[i][s], idx), 
                                                                   self.var_distr.local_parameters[idx], 
                                                                   retain_graph=True,
                                                                   allow_unused=True)))
                
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
                                   self.var_distr.log_likelihood_local(z, idx).data
                local_const_term = local_const_term * self.data.shape[0]
                local_const_term = local_const_term - local_a[i]
                
                local_var_term = self.var_distr.log_likelihood_local(z, idx)
                
                local_loss = local_loss + local_var_term * local_const_term
                
                global_const_term = global_const_term + \
                					self.prior_distr.log_likelihood_local(z, beta) + \
                                    self.prior_distr.log_likelihood_joint(x, z, beta)
            
            global_const_term = global_const_term * self.data.shape[0] / batch_indices.shape[0]
            global_const_term = global_const_term + \
            					self.prior_distr.log_likelihood_global(beta) - \
                                self.var_distr.log_likelihood_global(beta).data - \
                                global_a
            
            global_var_term = self.var_distr.log_likelihood_global(beta)
            global_loss = global_loss + global_var_term * global_const_term
                    
        loss = -(global_loss + local_loss) / num_samples
        
        return loss


    def entropy_form_loss_(self, num_samples, batch_indices, discounter=1):
        '''Computing ELBO estimator in entropy form
        
        prior_distr requred methods: 
            log_likelihood_global(beta)
            log_likelihood_joint(x, z, beta)
        
        var_distr required methods: 
            entropy(batch_indices)
            sample_global()
            sample_local(beta, idx)
        
        Args:
            num_samples: number of samples used for approximation
            batch_indices: indices of batch
            discounter: coefficient of entropy term
        
        Returns:
            loss: ELBO in entropy form estimator
        
        '''

        mc_term = torch.zeros(1, requires_grad=True)
        
        for _ in range(num_samples):
            beta = self.var_distr.sample_global()
            sample_log_likelihood = torch.zeros(1, requires_grad=True)
            for idx in batch_indices:
                z = self.var_distr.sample_local(beta, idx)
                sample_log_likelihood = sample_log_likelihood + \
                                        self.prior_distr.log_likelihood_joint(self.data[idx], z, beta)
            sample_log_likelihood = sample_log_likelihood * self.data.shape[0] / batch_indices.size
            sample_log_likelihood = sample_log_likelihood + self.prior_distr.log_likelihood_global(beta)

            mc_term = mc_term + sample_log_likelihood

        mc_term = mc_term / num_samples
        entropy_term = self.var_distr.entropy(batch_indices)

        loss = -mc_term - discounter * entropy_term

        return loss


    def kl_form_loss_(self, num_samples, batch_indices, kl, discounter=1):
        '''Computing ELBO estimator in Kullback–Leibler divergence form
        
        prior_distr requred methods: 
            log_likelihood_cond(x, z, beta)
        
        var_distr required methods: 
            sample_global()
            sample_local(beta, idx)
        
        Args:
            num_samples: number of samples used for approximation
            batch_indices: indices of batch
            kl: callable, function which computes KL divergence beetween
                variational and prior based on based indices
            discounter: coefficient of KL term
        
        Returns:
            loss: ELBO in KL form estimator
        
        '''

        mc_term = torch.zeros(1, requires_grad=True)
        
        for _ in range(num_samples):
            beta = self.var_distr.sample_global()
            sample_log_likelihood = torch.zeros(1, requires_grad=True)
            for idx in batch_indices:
                z = self.var_distr.sample_local(beta, idx)
                sample_log_likelihood = sample_log_likelihood + \
                                        self.prior_distr.log_likelihood_cond(self.data[idx], z, beta)
            sample_log_likelihood = sample_log_likelihood * self.data.shape[0] / batch_indices.size
            
            mc_term = mc_term + sample_log_likelihood

        mc_term = mc_term / num_samples
        kl_term = kl(self.var_distr, self.prior_distr, batch_indices)

        loss = -mc_term + discounter * kl_term

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
        
        if var_term == 0:
        	a = 0
        else:
        	a = cov_term / var_term
        
        return a
        
        
def handle_nones(container):
    '''Replace all Nones with torch tensors containing one zero

    Args:
        container: tuple

    Returns:
        handled_container: container with torch tensor containing 
            one zero instead of Nones

    '''

    handled_container = tuple(item if item is not None else torch.zeros(1) for item in container)
    return handled_container

class HistoryCollector():
    '''A simple class which is helpful to collect loss history of SVI
    
    Args:
        data_size: int, number of data points
        batch_size: int, batch_size for SVI
    
    '''
    
    def __init__(self, data_size, batch_size=10):
        
        self.data_size = data_size        
        self.batch_size = batch_size        
        self.size = self.data_size // self.batch_size + bool(self.data_size % self.batch_size)
        self.counter = 0
        self.history = []
        self.history_per_epoch = []
            
    def collect_history(self, loss):
        '''Saving loss, can be passed as callback arg to SVI.make_inference
        
        Args:
            loss: torch.tensor, loss
        
        '''
        
        
        self.history_per_epoch.append(float(loss))
        self.counter += 1
        if self.counter == self.size:
            self.counter = 0
            self.history.append(np.mean(self.history_per_epoch))
            self.history_per_epoch = []    
        pass
