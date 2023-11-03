import logging
logger = logging.getLogger("saasbo")

import torch
import time
from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition import qExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize, normalize
from torch.quasirandom import SobolEngine

from backend.algorithms.optimization_algorithm_bridge import OptimizationAlgorithmBridge
from icecream import ic

class SaasboRunner(OptimizationAlgorithmBridge):
    def __init__(self, experiment_id,  replication, dim, num_init, batch_size,constraints, warmup_steps,num_samples,thinning, device, dtype):
        super().__init__(experiment_id, replication, dim, batch_size, constraints, num_init, device, dtype)
        self.sm = "saasgp" # TODO: make configuable later
        self.acqf = "qEI" # TODO: make configuable later
        self.warmup_steps: int = int(warmup_steps)
        self.num_samples: int = int(num_samples)
        self.thinning: int = int(thinning)

        self.logger.info(f"Running on device: {device}")


    
    def suggest(self):
        fit_start = time.time()
        # gp = SaasFullyBayesianSingleTaskGP(
        #     train_X=self.X, train_Y=self.Y, train_Yvar=torch.full_like(self.Y, 1e-6), outcome_transform=Standardize(m=1)
        # )
        model = SaasFullyBayesianSingleTaskGP(
            train_X=self.X, train_Y=self.Y, train_Yvar=self.Yvar, outcome_transform=Standardize(m=1)
        )
        fit_fully_bayesian_model_nuts(
            model, warmup_steps=self.warmup_steps, num_samples=self.num_samples, thinning=self.thinning, disable_progbar=True
        )
        self.fit_runtimes.append(time.time() - fit_start)
        # self.lengthscales = torch.cat((self.lengthscales, model.median_lengthscale), dim=0)if self.lengthscales is not None else model.median_lengthscale 
        self.lengthscales.append(model.median_lengthscale.tolist()[0]) if self.lengthscales is not None else model.covar_modulemedian_lengthscale.tolist()
        gen_start = time.time()
        EI = qExpectedImprovement(model=model, best_f=self.Y.max())

        self.X_next, acq_values = optimize_acqf(
            EI,
            bounds=torch.cat((torch.zeros(1, self.dim), torch.ones(1, self.dim))).to(dtype=self.dtype, device=self.device),
            q=self.trial_size,
            num_restarts=10,
            raw_samples=1024,
            sequential=True,
            inequality_constraints=self.constraints
        )
        self.gen_runtimes.append(time.time() - gen_start)
        acq_values=torch.unsqueeze(acq_values, dim=0) if acq_values.ndim==0 else acq_values
  
        self.acq_values = torch.cat((self.acq_values, acq_values), dim=0) if self.acq_values is not None else acq_values
       
        self.logger.debug(f"Next suggested candidate(s) (SAASBO): {self.X_next}")
        del model
        return self.X_next

