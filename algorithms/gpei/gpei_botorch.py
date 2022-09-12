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
from botorch.utils.transforms import unnormalize, normalize
from pathlib import Path
import pickle

from icecream import ic
class GPEIRunner():
    def __init__(self, experiment_id, dim, batch_size, num_init, device, dtype):
        self.experiment_id = experiment_id
        self.dim = dim
        self.batch_size = batch_size
        self.num_init = num_init


        self.X = None
        self.Y = None
        self.X_next = None
        self.Y_next = None
        self.Yvar = None
        self.Yvar_next = None
        self.minimize = True
        self.batch_runtimes = list()
        self.eval_budget = 1000
        self.eval_runtimes_second = list()
        self.device = device
        self.dtype = dtype

        logger.info(f"Running on device: {self.device}")


    def suggest_initial(self, seed=0):
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=seed)
        self.X_next = sobol.draw(n=self.num_init).to(dtype=self.dtype, device=self.device)
   
        logger.debug(f"Initial SOBOL candidates: {self.X_next}")
        
        return self.X_next

    def suggest(self, _acqf = "qEI"):

        gp = FixedNoiseGP(self.X, self.Y, train_Yvar=torch.full_like(self.Y, 1e-6))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll=mll)
        if _acqf == "qEI":
            acqf = qExpectedImprovement(model=gp, best_f=self.Y.max())
        if _acqf == "UCB":
            acqf = UpperConfidenceBound(gp, beta=0.1)

        self.X_next, _ = optimize_acqf(
            acq_function=acqf,
            bounds=torch.cat((torch.zeros(1, self.dim), torch.ones(1, self.dim))).to(dtype=self.dtype, device=self.device),
            q=self.batch_size,
            num_restarts=10,
            raw_samples=1024,
        )
       
        logger.debug(f"Next suggested candidate(s) (GP/{_acqf}): {self.X_next}")
        return self.X_next

    def complete(self, y, yvar = None):
        # TODO: Make X and X_Next SAME AS SAASBO, same for Y, also rm hasattr if and make online
        self.Y_next  = torch.tensor(y, dtype=self.dtype, device=self.device).unsqueeze(-1)
        if yvar is not None:
            self.Yvar_next = torch.tensor(yvar, dtype=self.dtype, device=self.device).unsqueeze(-1)
            self.Yvar = torch.cat((self.Yvar, self.Yvar_next),  dim=0) if self.Yvar is not None else self.Yvar_next
        self.X = torch.cat((self.X, self.X_next), dim=0) if self.X is not None else self.X_next
        self.Y = torch.cat((self.Y, self.Y_next), dim=0) if self.Y is not None else self.Y_next
         

    def terminate_experiment(self):
        best_value = max(self.Y).item() * -1 if self.minimize else max(self.Y).item() 
        logger.info(f"Experiment terminated. Best value:  {best_value}")
        path = f"data/experiment_{str(self.experiment_id)}"
        Path(path).mkdir(parents=True, exist_ok=True)
        with open((path + "/" + str(self.experiment_id) + "_gpei_runner.pkl"), "wb") as fo:
            pickle.dump(self, fo)
