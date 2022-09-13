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

from algorithms.AlgorithmRunner import AlgorithmRunner
from pathlib import Path
import pickle


# def get_initial_points(dim, n_pts, seed=0):
#     sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
#     X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
#     return X_init

class SaasboRunner(AlgorithmRunner):
    def __init__(self, experiment_id,  replication, dim, num_init, batch_size, warmup_steps,num_samples,thinning, device, dtype):
        super().__init__(experiment_id, replication, dim, batch_size, num_init, device, dtype)
        self.sm = "saasgp" # TODO: make configuable later
        self.acqf = "qEI" # TODO: make configuable later
        self.warmup_steps: int = warmup_steps
        self.num_samples: int = num_samples
        self.thinning: int = thinning
        self.median_lengthscales = None
        logger.info(f"Running on device: {device}")


    
    def suggest(self):
        # gp = SaasFullyBayesianSingleTaskGP(
        #     train_X=self.X, train_Y=self.Y, train_Yvar=torch.full_like(self.Y, 1e-6), outcome_transform=Standardize(m=1)
        # )
        gp = SaasFullyBayesianSingleTaskGP(
            train_X=self.X, train_Y=self.Y, train_Yvar=self.Yvar, outcome_transform=Standardize(m=1)
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

