import logging
logger = logging.getLogger("gpei")

import torch
from torch import tensor
from botorch import fit_gpytorch_model
from botorch.models import FixedNoiseGP, ModelListGP, HeteroskedasticSingleTaskGP, SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from torch.quasirandom import SobolEngine
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition import UpperConfidenceBound

from botorch.utils.transforms import unnormalize, normalize, standardize
from algorithms.optimization_algorithm_bridge import OptimizationAlgorithmBridge
from icecream import ic
class GPEIRunner(OptimizationAlgorithmBridge):
    def __init__(self, experiment_id,  replication, dim, batch_size, constraints, num_init, device, dtype, sm= "hsgp"):
        super().__init__(experiment_id, replication, dim, batch_size, constraints, num_init, device, dtype)

        self.sm = sm
        self.acqf = "qEI" # TODO: make configuable later
        logger.info(f"Running on device: {self.device}")



    def suggest(self):
        # Standarize Y and normalize Noise as said here: https://botorch.org/api/models.html#botorch.models.gp_regression.SingleTaskGP
        # This performs better. Testet fngp and hsgp / 2022-10-04 PZ
        train_Y = standardize(self.Y) #standardize because botorch says it works better

       
        if self.sm in ["hsgp", "hsstgp", 'heteroskedastic GP']:
            train_Yvar = normalize(self.Yvar,bounds=tensor((min(self.Y).item(), max(self.Y).item())))
            model = HeteroskedasticSingleTaskGP(self.X, train_Y, train_Yvar=train_Yvar)
        # model = FixedNoiseGP(self.X, train_Y, train_Yvar=train_Yvar)
        elif self.sm in ["fngp", "fgp","fixed noise GP"]:
            train_Yvar = normalize(self.Yvar,bounds=tensor((min(self.Y).item(), max(self.Y).item())))
            model = FixedNoiseGP(self.X, train_Y, train_Yvar=train_Yvar)
        else:
            model = SingleTaskGP(self.X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll=mll)
        self.lengthscales = torch.cat((self.lengthscales, model.covar_module.base_kernel.lengthscale), dim=0) if not self.lengthscales == None else model.covar_module.base_kernel.lengthscale 

        if self.acqf == "qEI":
            acqf = qExpectedImprovement(model=model, best_f=self.Y.max())
   
        if self.acqf == "UCB":
            acqf = UpperConfidenceBound(model, beta=0.1)
 
        self.X_next, acq_values = optimize_acqf(
            acq_function=acqf,
            bounds=torch.cat((torch.zeros(1, self.dim), torch.ones(1, self.dim))).to(dtype=self.dtype, device=self.device),
            q=self.trial_size,
            num_restarts=10,
            raw_samples=1024,
            sequential=True,
            return_best_only=True,
            inequality_constraints=self.constraints

        )

        acq_values=torch.unsqueeze(acq_values, dim=0) if acq_values.ndim==0 else acq_values
        self.acq_values = torch.cat((self.acq_values, acq_values), dim=0) if self.acq_values is not None else acq_values
   
        logger.debug(f"Next suggested candidate(s) (GP/{self.acqf}): {self.X_next}")
        return self.X_next


         


