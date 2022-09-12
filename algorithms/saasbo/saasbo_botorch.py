import logging
logger = logging.getLogger("saasbo")

import torch

from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition import qExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize, normalize
from torch.quasirandom import SobolEngine

from pathlib import Path
import pickle


# def get_initial_points(dim, n_pts, seed=0):
#     sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
#     X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
#     return X_init

class SaasboRunner:
    def __init__(self, experiment_id, dim, num_init, batch_size, warmup_steps,num_samples,thinning, device, dtype):
        
        self.experiment_id = experiment_id,
        self.dim: int = dim
        self.num_init: int= num_init
        self.batch_size: int = batch_size
        self.warmup_steps: int = warmup_steps
        self.num_samples: int = num_samples

        self.thinning: int = thinning

        self.acq_values = None
        self.X = None
        self.Y = None
        self.Yvar_next = None
        self.Yvar = None
        self.median_lengthscales = None
        self.minimize= True
        self.total_runtime = 0
        self.batch_runtimes = list()
        self.eval_budget = 1000
        self.eval_runtimes_second = list()
        self.device = device
        self.dtype = dtype
        
        logger.info(f"Running on device: {device}")



    
    def suggest_initial(self):
   
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=0)
        self.X_next = sobol.draw(n=self.num_init).to(dtype=self.dtype, device=self.device)

        logger.debug(f"Initial SOBOL candidates: {self.X_next}")
        return self.X_next 
    
    def suggest(self):
        gp = SaasFullyBayesianSingleTaskGP(
            train_X=self.X, train_Y=self.Y, train_Yvar=torch.full_like(self.Y, 1e-6), outcome_transform=Standardize(m=1)
        )
        fit_fully_bayesian_model_nuts(
            gp, warmup_steps=self.warmup_steps, num_samples=self.num_samples, thinning=self.thinning, disable_progbar=True
        )
        self.median_lengthscales = gp.median_lengthscale if self.median_lengthscales == None else torch.cat((self.median_lengthscales, gp.median_lengthscale), dim=0)

        EI = qExpectedImprovement(model=gp, best_f=self.Y.max())
        self.X_next, acq_values = optimize_acqf(
            EI,
            bounds=torch.cat((torch.zeros(1, self.dim), torch.ones(1, self.dim))).to(dtype=self.dtype, device=self.device),
            q=self.batch_size,
            num_restarts=10,
            raw_samples=1024,
        )
       
        logger.debug(f"Next suggested candidate(s) (SAASBO): {self.X_next}")
        return self.X_next

    def complete(self, y, yvar = None):
        self.Y_next  = torch.tensor(y, dtype=self.dtype, device=self.device).unsqueeze(-1)
        if yvar is not None:
            self.Yvar_next = torch.tensor(yvar, dtype=self.dtype, device=self.device).unsqueeze(-1)
            self.Yvar = torch.cat((self.Yvar, self.Yvar_next),  dim=0) if self.Yvar is not None else self.Yvar_next
        self.X = torch.cat((self.X, self.X_next), dim=0) if self.X is not None else self.X_next
        self.Y = torch.cat((self.Y, self.Y_next), dim=0) if self.Y is not None else self.Y_next

     
    def terminate_experiment(self):
        best_value = max(self.Y).item() * -1 if self.minimize else max(self.Y).item() 
        logger.info(f"Experiment terminated. Best value:  {best_value}")
       
        path = "data/experiment_" + str(self.experiment_id)
        Path(path).mkdir(parents=True, exist_ok=True)
        with open((path + "/" + str(self.experiment_id) + "_saasbo_runner.pkl"), "wb") as fo:
            pickle.dump(self, fo)