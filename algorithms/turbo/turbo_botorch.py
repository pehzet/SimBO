import logging
logger = logging.getLogger("turbo")

import pickle
import math

from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize, normalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior
from pathlib import Path


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

# TODO: Move to TurboState?!
def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

# TODO: Move to TurboRunner?!
def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
    device="cpu",
    dtype=torch.double
):
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(n_candidates, dim, dtype=dtype, device=device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask        
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        ei = qExpectedImprovement(model, Y.max(), maximize=True)
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next


class TurboRunner:

    def __init__(self, experiment_id, dim, batch_size, num_init, param_meta=None, device="cpu", dtype=torch.double):
        self.experiment_id = experiment_id
        self.dim = dim
        self.batch_size = batch_size
        self.state = TurboState(dim=self.dim, batch_size=self.batch_size)
        self.param_meta = param_meta
        self.bounds = self.get_bounds_from_param_meta()
        self.num_init = num_init
        self.X = None
        self.Y = None
        self.X_next = None
        self.Y_next = None
        self.minimize = True
        self.total_runtime = 0
        self.batch_runtimes = list()
        self.num_restarts = 0
        self.budget_at_restart = list()
        self.eval_budget = 10000 # TODO: CHANGE
        self.eval_runtimes_second = list()
        self.seed = 0
        self.device=device
        self.dtype=dtype

        logger.info(f"Running on device: {self.device} and dtype: {self.dtype}")
    
    def format_x_for_mrp(self, xx):
        assert self.param_meta is not None
        assert self.bounds is not None
        xx = unnormalize(xx, bounds=self.bounds)
        xx_mrp = []
        for x in xx:
            x_mrp = []
            for i,pm in enumerate(self.param_meta):
       
                x_mrp.append(
                    {   
                    "id" : pm.get("name").split("_",1)[0],
                    "name" : pm.get("name").split("_",1)[1],
                    "value" : int(round(x[i].numpy().item()))
                    }
                )
            xx_mrp.append(x_mrp)
        return xx_mrp

    def format_y_from_mrp(self, y_mrp):

        yy = torch.tensor([list(y.values())[0] for y in y_mrp])
        if self.minimize:
            yy = -1 * yy
        return yy

    def get_bounds_from_param_meta(self):
        '''
        expects list of dicts with key lower_bound: int and upper_bound: int or bounds: (int, int)
        returns bounds as tuple tensor
        '''
        
        lb = [pm.get("lower_bound") for pm in self.param_meta]
        ub = [pm.get("upper_bound") for pm in self.param_meta]
        #bounds = [(pm.get("lower_bound",pm.get("bounds")[0]) , pm.get("upper_bound",pm.get("bounds")[1])) for pm in self.param_meta]
        bounds = torch.tensor([lb, ub])
        return bounds

    def restart_state(self):                
        logger.info(f"{self.num_restarts}. of TR triggered")
        self.num_restarts +=1
        self.budget_at_restart.append(self.eval_budget)
        self.terminate_experiment()
        self.state = TurboState(dim=self.dim, batch_size=self.batch_size)
        self.X = None
        self.Y = None
        self.X_next = None
        self.Y_next = None

    def suggest(self, acqf="ts"):
        if self.state.restart_triggered:
            self.restart_state()
            self.X_next = self.suggest_initial()
            return self.X_next
        train_Y = (self.Y_turbo - self.Y_turbo.mean()) / self.Y_turbo.std()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=self.dim, lengthscale_constraint=Interval(0.005, 4.0))
        )
  
        model = SingleTaskGP(self.X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        with gpytorch.settings.max_cholesky_size(float("inf")):
                # Fit the model
            fit_gpytorch_model(mll)
        
            # Create a batch
            self.X_next = generate_batch(
                state=self.state,
                model=model,
                X=self.X_turbo,
                Y=train_Y,
                batch_size=self.batch_size,
                n_candidates=None,
                num_restarts=10,
                raw_samples=512,
                acqf=acqf,
                device=self.device,
                dtype=self.dtype
            )
        return self.X_next

    def suggest_initial(self):
        logger.debug("Suggesting >{self.num_init} points.")
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.seed)
        self.X_next = sobol.draw(n=self.num_init).to(dtype=self.dtype, device=self.device)
        return self.X_next

    def complete(self, y):
        # TODO: Make X and X_Next SAME AS SAASBO, same for Y, also rm hasattr if and make online
        self.Y_next  = torch.tensor(y, dtype=self.dtype, device=self.device).unsqueeze(-1)
        if not hasattr(self,"X_turbo"):
            self.X_turbo = self.X_next
        else:
            self.X_turbo = torch.cat((self.X_turbo, self.X_next), dim=0)
        if not hasattr(self,"Y_turbo"):
            self.Y_turbo = self.Y_next 
        else:
            self.Y_turbo = torch.cat((self.Y_turbo, self.Y_next), dim=0)
        self.state = update_state(state=self.state, Y_next=self.Y_next)

    def terminate_experiment(self):
        if self.state.restart_triggered:
            best_value = self.state.best_value * -1 if self.minimize else self.state.best_value
            logger.info(f"Best Value at current State: {best_value:.2f}")
            #print(f"Best Value at current State: {best_value:.2f}")

        best_value = self.state.best_value * -1 if self.minimize else self.state.best_value
        logger.info(f"Best Value found: {best_value:.2f}")
    
        _name_state = "_turbo_state_" + str(self.num_restarts +1) + ".pkl"
        _name_runner = "_turbo_runner_" + str(self.num_restarts +1) + ".pkl"
        path = f"data/experiment_{self.experiment_id}"
        Path(path).mkdir(parents=True, exist_ok=True)
        with open((path +"/" + str(self.experiment_id) + _name_state), "wb") as fo:
            pickle.dump(self.state, fo)
        with open((path +"/" + str(self.experiment_id) + _name_runner), "wb") as fo:
            pickle.dump(self, fo)      
