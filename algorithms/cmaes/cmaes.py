import cma
from cma import CMAEvolutionStrategy
import torch
import numpy as np
from algorithms.AlgorithmRunner import AlgorithmRunner
import logging
logger = logging.getLogger("cmaes")
from icecream import ic

class CMAESRunner(AlgorithmRunner):
    def __init__(self, experiment_id,  replication, dim, batch_size, bounds, sigma0, num_init=-1, device="cpu", dtype=torch.double) -> None:
        super().__init__(experiment_id,  replication, dim, batch_size, num_init, device, dtype)
        self.bounds = self.tensor_to_list(bounds)
        self.sigma0 = sigma0 if sigma0 != -1 else 0.5 #0.5 is default in tutorial https://pypi.org/project/cma/
        self.X_next_l = None
        opts = cma.CMAOptions()
        self.X = list()
        self.Y = list()
        opts.set('bounds', self.bounds)
        opts.set('popsize', self.batch_size)
        opts.set('seed', 12345)
        # NOTE: lt. Link oben benötigt cmaes 100xdim candidates für befriedigende Ergebnisse
        self.es = cma.CMAEvolutionStrategy(self.dim*[0], sigma0=self.sigma0,inopts=opts)
    def tensor_to_list(self,t):
        if torch.is_tensor(t):
            t = t.tolist()
        return t
    def list_to_tensor(self, l):
        if not torch.is_tensor(l):
            l = torch.tensor(np.array(l))
        return l
    def suggest_initial(self):
        logger.info(f"Suggest Initial skipped, because cmaes doesnt require random initialization. Suggest {self.batch_size} non-random candidates instead")
        return self.suggest()

    def suggest(self):
        xx = self.es.ask()
        self.X_next = xx
        xx = self.list_to_tensor(xx)
        return xx

    def complete(self,yy, yvar=None):

        
        self.X.append(self.X_next)
        #self.es.logger.add()
        self.Y_next = yy
        # cma-es ist default minimize (BO is default maximize)
        yy = yy*-1 if self.minimize else yy
        self.identity_best_in_trial()
        #NOTE: yvar not needed atm
        yy = self.tensor_to_list(yy)
        self.Y.append(yy)
        self.es.tell(self.X_next, yy)
