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

# FOR DEBUG - DELETE LATER
import sys
from icecream import ic

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#logger.info(f"Running on device: {device}")

tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

def get_initial_points(dim, n_pts, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init


class SaasboRunner:
    def __init__(self, experiment_id, dim, num_init, batch_size, warmup_steps,num_samples,thinning, param_meta=None):
        
        self.experiment_id = experiment_id,
        self.dim: int = dim
        self.num_init: int= num_init
        self.batch_size: int = batch_size
        self.warmup_steps: int = warmup_steps
        self.num_samples: int = num_samples
        self.param_meta = param_meta
        self.thinning: int = thinning
        self.bounds = self.get_bounds_from_param_meta()
        self.acq_values = None
        self.X = None
        self.Y = None
        self.median_lengthscales = None
        self.minimize= True
        self.total_runtime = 0
        self.batch_runtimes = list()
        self.eval_budget = 1000
        self.eval_runtimes_second = list()


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
    
    def suggest_initial(self):
   
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=0)
        self.X_next = sobol.draw(n=self.num_init).to(dtype=dtype, device=device)
        self.X = self.X_next if self.X == None else self.X
        return self.X_next 
    
    def suggest(self):
        #train_Y = -1 * Y  # Flip the sign since we want to minimize f(x)
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
            bounds=torch.cat((torch.zeros(1, self.dim), torch.ones(1, self.dim))).to(**tkwargs),
            q=self.batch_size,
            num_restarts=10,
            raw_samples=1024,
        )
        # ic(acq_values)
        # TODO: PROBLEMS WITH TENSOR
        # self.acq_values = acq_values if self.acq_values is None else torch.cat((self.acq_values, acq_values), dim=0) # TODO: Check/Test if dim=0 oder dim=1 -> dim=0 is correct
        self.X = torch.cat((self.X, self.X_next))
        return self.X_next

    def complete(self, y):
        self.Y_next  = torch.tensor(y,dtype=dtype, device=device).unsqueeze(-1)
        self.Y = self.Y_next if self.Y == None else torch.cat((self.Y, self.Y_next))
     
    def terminate_experiment(self):
        logger.info(f"Best Value found:  {max(self.Y).item()}")
        #print(str(self.experiment_id))
        path = "data/experiment_" + str(self.experiment_id)
        #print(path)
        Path(path).mkdir(parents=True, exist_ok=True)
        with open((path + "/" + str(self.experiment_id) + "_saasbo_runner.pkl"), "wb") as fo:
            pickle.dump(self, fo)