import cma
from cma import CMAEvolutionStrategy
import torch
import numpy as np
from algorithms.optimization_algorithm_bridge import OptimizationAlgorithmBridge
import logging
logger = logging.getLogger("cmaes")
from icecream import ic

class CMAESRunner(OptimizationAlgorithmBridge):
    def __init__(self, experiment_id,  replication, dim, batch_size, bounds, sigma0,  num_init=-1, use_case_runner=None, device="cpu", dtype=torch.double) -> None:

        constraints = None
        super().__init__(experiment_id,  replication, dim, batch_size, constraints,num_init, device, dtype)
        self.use_case_runner = use_case_runner
        self.bounds = self.tensor_to_list(bounds)
        self.sigma0 = float(sigma0) if sigma0 != -1 else 0.5 #0.5 is default in tutorial https://pypi.org/project/cma/
        self.X_next_l = None
        opts = cma.CMAOptions()
        self.X = list()
        self.Y = list()
        if self.trial_size <= 1:
            logger.info(f"trial_size {self.trial_size} is too small, setting to 2")
            self.trial_size = 2
        opts.set('bounds', self.bounds)
        opts.set('popsize', self.trial_size)
        # opts.set('seed', self.seed)
        # NOTE: lt. Link oben benötigt cmaes 100xdim candidates für befriedigende Ergebnisse
        self.es = cma.CMAEvolutionStrategy(np.random.uniform(0, 1,size=self.dim), sigma0=self.sigma0,inopts=opts) # was self.dim*[0]
        self.nh = cma.optimization_tools.NoiseHandler(self.dim, maxevals=4, aggregate=np.mean) if use_case_runner is not None else None
    def tensor_to_list(self,t):
        if torch.is_tensor(t):
            t = t.tolist()
        return t
    def list_to_tensor(self, l):
        if not torch.is_tensor(l):
            l = torch.tensor(np.array(l))
        return l
    def suggest_initial(self):
        logger.info(f"Suggest Initial skipped, because cmaes doesnt require random initialization. Suggest {self.trial_size} non-random candidates instead")
        return self.suggest()

    def suggest(self):
        xx = self.es.ask()
        
        self.X_next = xx
        xx = self.list_to_tensor(xx)
        return xx

    def eval_wrapper(self, x):
        y = self.use_case_runner.eval(x)
        return self.tensor_to_list(y)
    
    def complete(self,yy, yvar=None):
        
        self.X.append(self.X_next)
        #self.es.logger.add()
        self.Y_next = yy
        # cma-es ist default minimize (BO is default maximize)

        yy = yy*-1 if self.minimize else yy
        if self.nh is not None:
            _X_next = self.tensor_to_list(self.X_next)
            _Y_next = self.tensor_to_list(yy)
            self.es.sigma *= self.nh(_X_next, _Y_next, self.eval_wrapper, self.es.ask)
        self.identity_best_in_trial()
        #NOTE: yvar not needed atm
        yy = self.tensor_to_list(yy)
        self.Y.append(yy)
        self.es.tell(self.X_next, yy)
