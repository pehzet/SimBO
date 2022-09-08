import os
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

class GPEIRunner():
    def __init__(self, dim, batch_size, num_init, param_meta):
        self.dim = dim
        self.batch_size = batch_size
        self.num_init = num_init
        self.param_meta = param_meta
        self.bounds = self.get_bounds_from_param_meta()
        self.X = None
        self.Y = None
        self.minimize = True
        self.batch_runtimes = list()
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
    def suggest_initial(self, seed=0):
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=seed)
        self.X_next = sobol.draw(n=self.num_init).to(dtype=dtype, device=device)
        self.X = self.X_next if self.X == None else self.X
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
            bounds=torch.cat((torch.zeros(1, self.dim), torch.ones(1, self.dim))).to(**tkwargs),
            q=self.batch_size,
            num_restarts=10,
            raw_samples=1024,
        )
        self.X = torch.cat((self.X, self.X_next))
        return self.X_next
    

    def complete(self, y):
        self.Y_next  = torch.tensor(y,dtype=dtype, device=device).unsqueeze(-1)
        self.Y = self.Y_next if self.Y == None else torch.cat((self.Y, self.Y_next))

    def terminate_experiment(self, experiment_id):
        print(f"Best Value found:  {max(self.Y).item()}")
        path = f"data/experiment_{str(experiment_id)}"
        Path(path).mkdir(parents=True, exist_ok=True)
        with open((path + "/" + str(experiment_id) + "_gpei_runner.pkl"), "wb") as fo:
            pickle.dump(self, fo)
