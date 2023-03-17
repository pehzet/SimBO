import logging
logger = logging.getLogger("turbo")

import pickle
import math
from icecream import ic
from dataclasses import dataclass

import torch
from torch import tensor
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP, FixedNoiseGP, HeteroskedasticSingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize, normalize, standardize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from pathlib import Path


from algorithms.optimization_algorithm_bridge import OptimizationAlgorithmBridge

# TODO: Add constraints with ConstrainedMaxPosteriorSampling like at https://botorch.org/tutorials/scalable_constrained_bo


@dataclass
class TurboState():
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 3 # was ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 3 #float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    # def __post_init__(self):
    #     self.failure_tolerance = math.ceil(
    #         max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
    #     )
    
    def update_state(self, Y_next):
        if max(Y_next) > self.best_value + 1e-3 * math.fabs(self.best_value):
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:  # Expand trust region
            logger.info("Expand trust region")
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:  # Shrink trust region
            logger.info("Shrink trust region")
            self.length /= 2.0
            self.failure_counter = 0

        self.best_value = max(self.best_value, max(Y_next).item())

        if self.length < self.length_min:
            self.restart_triggered = True
        return self

@dataclass
class TurboRunner(OptimizationAlgorithmBridge):

    def __init__(self, experiment_id, replication, dim, trial_size, constraints, num_init, device, dtype, sm="fngp"):
        super().__init__(experiment_id,  replication, dim, trial_size, constraints, num_init, device, dtype)
        self.state = TurboState(dim=self.dim, batch_size=self.trial_size)
        logger.info(f"Running on device: {self.device} and dtype: {self.dtype}")

        self.sm = sm 
        self.acqf = "ts" # TODO: make configuable later
        self.X_all = None
    


    def restart_state(self):                
        logger.info(f"{self.num_restarts}. start of TR triggered")
        self.num_restarts +=1
        self.terminate_experiment()
        self.X_all = torch.cat((self.X_all, self.X), dim=0) if self.X_all is not None else self.X
        self.state = TurboState(dim=self.dim, batch_size=self.trial_size)
        self.X = None
        self.Y = None
        self.X_next = None
        self.Y_next = None
        self.Yvar = None
        self.Yvar_next = None
        
    def suggest(self):
        self.is_init = False
        if self.state.restart_triggered:
            self.restart_state()
            self.X_next = self.suggest_initial()
            return self.X_next
        # Standarize Y and normalize Noise as said here: https://botorch.org/api/models.html#botorch.models.gp_regression.SingleTaskGP
        # This performs better. Testet fngp and hsgp / 2022-10-04 PZ
        train_Y = standardize(self.Y) #standardize because botorch says it works better
        # train_Yvar = normalize(self.Yvar,bounds=tensor((min(self.Y).item(), max(self.Y).item())))
        #likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=self.dim, lengthscale_constraint=Interval(0.005, 4.0))
        )
        # NOTE: Check if standardize Y and Yvar perform better. stdardize = (Y-Y.mean)/Y.std


        # if self.sm in ["hsgp", "hsstgp"]:
        #     # NOTE: Its not possible to pass a covar module to the HSGP. So maybe we should use fixed Noise here to keep the lengthscale priors.
        #     # I will take a look at it at the time /PZM 2022-10-04
        #     # https://github.com/pytorch/botorch/issues/180
        #     model = HeteroskedasticSingleTaskGP(self.X, train_Y, train_Yvar=train_Yvar)
        # # model = FixedNoiseGP(self.X, train_Y, train_Yvar=train_Yvar)
        # elif self.sm in ["fngp", "fgp"]:
        #     model = FixedNoiseGP(self.X, train_Y, train_Yvar=train_Yvar, covar_module=covar_module)
        # else:
        model = SingleTaskGP(self.X, self.Y, covar_module=covar_module)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        with gpytorch.settings.max_cholesky_size(float("inf")):
                # Fit the model
            fit_gpytorch_model(mll)
            self.lengthscales = torch.cat((self.lengthscales, model.covar_module.base_kernel.lengthscale), dim=0) if not self.lengthscales == None else model.covar_module.base_kernel.lengthscale 
            self.X_next = self.generate_batch(
                model=model,
                Y=train_Y,
                acqf=self.acqf,
                n_candidates=None,
                num_restarts=10,
                raw_samples=512
            )

        return self.X_next

    def complete(self, y, yvar = None):
        self.Y_next  = torch.tensor(y, dtype=self.dtype, device=self.device).unsqueeze(-1)
        
        self.identity_best_in_trial()
        
        if yvar is not None:
            self.Yvar_next = torch.tensor(yvar, dtype=self.dtype, device=self.device).unsqueeze(-1)
            self.Yvar = torch.cat((self.Yvar, self.Yvar_next),  dim=0) if self.Yvar is not None else self.Yvar_next
        
        # do this step to avoid updating state after init trials
        if not self.is_init:
            self.state = self.state.update_state(Y_next=self.Y_next)

        self.X = torch.cat((self.X, self.X_next), dim=0) if self.X is not None else self.X_next
        self.Y = torch.cat((self.Y, self.Y_next), dim=0) if self.Y is not None else self.Y_next
    
    def generate_batch(self, model, Y, acqf, n_candidates = None, num_restarts=10, raw_samples=512):
        assert acqf in ("ts", "ei")
        assert self.X.min() >= 0.0 and self.X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * self.X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = self.X[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * self.state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * self.state.length / 2.0, 0.0, 1.0)

        if acqf == "ts":
            dim = self.X.shape[-1]
            sobol = SobolEngine(dim, scramble=True)
            pert = sobol.draw(n_candidates).to(dtype=self.dtype, device=self.device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # Create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = (
                torch.rand(n_candidates, dim, dtype=self.dtype, device=self.device)
                <= prob_perturb
            )
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=self.device)] = 1

            # Create candidate points from the perturbations and the mask        
            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]

            # Sample on the candidate points
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():  # We don't need gradients when using TS
                X_next = thompson_sampling(X_cand, num_samples=self.trial_size)

        elif acqf == "ei":
            ei = qExpectedImprovement(model, self.Y.max(), maximize=True)
            X_next, acq_value = optimize_acqf(
                ei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=self.trial_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                inequality_constraints=self.constraints
            )

        return X_next



    def terminate_experiment(self):
        if self.state.restart_triggered:
            best_value = self.state.best_value * -1 if self.minimize else self.state.best_value
            logger.info(f"Best Value at current State: {best_value:.2f}")
            #print(f"Best Value at current State: {best_value:.2f}")

    
        _name_state = "_turbostate_" + str(self.num_restarts +1) + ".pkl"
        _name_runner = "_turborunner_" + str(self.num_restarts +1) + ".pkl"
        path = f"data/experiment_{self.experiment_id}"
        Path(path).mkdir(parents=True, exist_ok=True)
        with open((path +"/" + str(self.experiment_id) + _name_state), "wb") as fo:
            pickle.dump(self.state, fo)
        with open((path +"/" + str(self.experiment_id) + _name_runner), "wb") as fo:
            pickle.dump(self, fo)      
