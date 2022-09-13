import logging
logger = logging.getLogger("gpei")

import torch
from botorch import fit_gpytorch_model
from botorch.models import FixedNoiseGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from torch.quasirandom import SobolEngine
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition import UpperConfidenceBound
from botorch.utils.transforms import unnormalize, normalize, standardize
from algorithms.AlgorithmRunner import AlgorithmRunner
from pathlib import Path
import pickle

from icecream import ic
class GPEIRunner(AlgorithmRunner):
    def __init__(self, experiment_id,  replication, dim, batch_size, num_init, device, dtype):
        super().__init__(experiment_id, replication, dim, batch_size, num_init, device, dtype)

        self.sm = "fngp" # TODO: make configuable later
        self.acqf = "qEI" # TODO: make configuable later
        logger.info(f"Running on device: {self.device}")



    def suggest(self):
        train_Y = standardize(self.Y) #standardize because botorch says it works better
        train_Yvar = standardize(self.Yvar)
        train_Yvar = torch.abs(train_Yvar) # variance must not be negative
        #gp = FixedNoiseGP(self.X, self.Y, train_Yvar=torch.full_like(self.Y, 1e-6))
        gp = FixedNoiseGP(self.X, train_Y, train_Yvar=train_Yvar)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll=mll)
        if self.acqf == "qEI":
            acqf = qExpectedImprovement(model=gp, best_f=self.Y.max())
        if self.acqf == "UCB":
            acqf = UpperConfidenceBound(gp, beta=0.1)

        self.X_next, _ = optimize_acqf(
            acq_function=acqf,
            bounds=torch.cat((torch.zeros(1, self.dim), torch.ones(1, self.dim))).to(dtype=self.dtype, device=self.device),
            q=self.batch_size,
            num_restarts=10,
            raw_samples=1024,
        )
       
        logger.debug(f"Next suggested candidate(s) (GP/{self.acqf}): {self.X_next}")
        return self.X_next


         


