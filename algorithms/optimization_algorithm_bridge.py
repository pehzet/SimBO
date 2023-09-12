import torch
import logging
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch.quasirandom import SobolEngine
from pathlib import Path
import pickle
from dataclasses import dataclass
from icecream import ic
from torch import tensor
@dataclass
class OptimizationAlgorithmBridge:

    def __init__(self, experiment_id, replication, dim, trial_size, constraints, num_init=-1, device=torch.device('cuda'), dtype=torch.double) -> None:
        self.experiment_id = experiment_id
        self.replication = replication
        self.dim = dim
        self.is_ea = False
        self.num_init = dim * 2 if num_init == -1 else num_init
        self.trial_size = trial_size
        self.logger = logging.getLogger("algorithm")
        try:
            self.logger.addHandler(self.tkwargs["logging_fh"])
        except:
            pass
        self.X = None
        self.Y = None
        self.X_next = None
        self.Y_next = None
        self.Yvar = None
        self.Yvar_next = None
        self.Y_pred = None
        self.Y_current_best = None
        self.acq_values = None
        self.minimize = True
        self.is_moo = False
        self.total_runtime = 0
        self.batch_runtimes = list()
        self.num_restarts = 0
        self.eval_runtimes_second = list()
        self.seed = None
        self.eval_budget = None
        self.device=device
        self.dtype=dtype
        # NOTE: only inequality constraints implemented
        self.constraints = constraints #self.constraints_to_tensor(constraints)
        self.lengthscales = None
        self.is_init = True
    
    def constraints_to_tensor(self, constraints):
        if constraints is not None:
            _constraints = []
            for c in constraints["ieq"]:
                ic(c)
                _constraints.append((tensor(c[0], dtype=torch.int64), tensor(c[1], dtype=self.dtype)))
            return _constraints
        return None
    def suggest_initial(self, num_trials = None):
        self.is_init = True
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.seed)
        self.logger.info(f"Running SOBOL on device: {self.device}")
        if num_trials == None:
            self.X_next = sobol.draw(n=self.num_init).to(dtype=self.dtype, device=self.device)
        else:
            self.X_next = sobol.draw(n=int(num_trials)).to(dtype=self.dtype, device=self.device)
        self.logger.debug(f"Initial SOBOL candidates: {self.X_next}")
        return self.X_next 
    
    def identity_best_in_trial(self):

        best_in_trial = max(self.Y_next).item() # TODO: Think about general way to handle min and max
        if self.Y_current_best == None:
            self.Y_current_best = best_in_trial
            self.logger.info(f"New best Y found: {self.Y_current_best*-1}")
        else:
            # is_better = self.Y_current_best < best_in_trial if self.minimize else self.Y_current_best > best_in_trial
            if self.Y_current_best < best_in_trial if self.minimize else self.Y_current_best > best_in_trial:
                self.Y_current_best = best_in_trial
                self.logger.info(f"New best Y found: {self.Y_current_best*-1 if self.minimize else self.Y_current_best}")

    def complete(self, y, yvar = None):
        # if self.__class__.__name__.lower() == "morborunner":
        #     self.Y_next = y
        # else:
        #     self.Y_next  = torch.tensor(y, dtype=self.dtype, device=self.device).unsqueeze(-1)

        self.Y_next  = torch.tensor(y, dtype=self.dtype, device=self.device).unsqueeze(-1) if not torch.is_tensor(y) else y
        self.X_next = torch.tensor(self.X_next, dtype=self.dtype, device=self.device) if not torch.is_tensor(self.X_next) else self.X_next

        if not self.is_moo:
            self.identity_best_in_trial()
    
        if yvar is not None:
            self.Yvar_next = torch.tensor(yvar, dtype=self.dtype, device=self.device).unsqueeze(-1)
            self.Yvar = torch.cat((self.Yvar, self.Yvar_next),  dim=0) if self.Yvar is not None else self.Yvar_next
        self.X = torch.cat((self.X, self.X_next), dim=0) if self.X is not None else self.X_next
        self.Y = torch.cat((self.Y, self.Y_next), dim=0) if self.Y is not None else self.Y_next
    def get_tr(self):
        if self.get_name() == "turborunner":
            return self.num_restarts + 1
        else:
            return "na"
    def get_name(self):
        return  self.__class__.__name__.lower()
    def get_technical_specs(self):
        sm = self.sm if hasattr(self, "sm") else "na"
        acqf = self.acqf if hasattr(self, "acqf") else "na"
        ts = {
            "sm" : sm,
            "acqf":acqf
        }
        return ts
    def terminate_experiment(self):
        if self.is_moo:
            # STRUGGLE WITH MORBO; Can't pickle local object 'get_outcome_constraint_transforms; TODO: FIX 
            return
        path = "data/experiment_" + str(self.experiment_id)
        Path(path).mkdir(parents=True, exist_ok=True)
        with open((path + "/" + str(self.experiment_id) + "_" + str(self.replication) + "_" + self.get_name() +  ".pkl"), "wb") as fo:
            pickle.dump(self, fo)
        torch.cuda.empty_cache()
    def get_y_pred(self, index=-1):
        """
        NOTE: y-pred not implemented. Returning None
        """
        return None
    


    def get_acq_value(self, index=-1):
        if self.acq_values == None:
            return "na"
        if self.acq_values.ndim == 0:
            return self.acq_values.item()
        return self.acq_values[index].item()
    def get_feature_importance(self, all=False, index=-1):

        if self.lengthscales == None:
            return "na"
        if all==False:
            lengthscale = self.lengthscales[index] if self.lengthscales.ndim > 1 else self.lengthscales
        else:
            lengthscale = self.lengthscales
        return torch.reciprocal(lengthscale)
    
    def get_hypervolume(self):
        if self.Y.ndim == 1:
            return "na"
        return self.hv.compute(self.Y)
    def get_hypervolume_improvement(self):
        if self.Y.ndim == 1:
            return "na"
        return self.hv.compute(self.Y) - self.hv.compute(self.Y_current_best)
    def check_outcome_constraints_bool(self, value, bound, constraint_type="ieq"):
        return value <= bound if constraint_type == "ieq" else value == bound
    def check_outcome_constraints_numerical(self, value, bound, constraint_type):
        '''
        returns -1 if constraint is satisfied, 1 if not because botorch implies feasibility with negative values
        '''
        # negative values imply feasibility in botorch -> https://botorch.org/tutorials/constrained_multi_objective_bo
        return value - bound if constraint_type == "ieq" else torch.where(value==bound, torch.tensor(1), torch.tensor(-1)) 
            
      



