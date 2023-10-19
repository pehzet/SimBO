import torch
import logging
from botorch.models.gp_regression import SingleTaskGP, FixedNoiseGP, HeteroskedasticSingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch.quasirandom import SobolEngine
from pathlib import Path
import pickle
from backend.databases.sql import SQLManager
from dataclasses import dataclass
from icecream import ic
from torch import tensor
import numpy as np
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
        self.sql_database = SQLManager()
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
        self.batch_runtimes = []
        self.num_restarts = 0
        self.eval_runtimes = []
        self.fit_runtimes = []
        self.gen_runtimes = []

        self.seed = None
        self.eval_budget = None
        self.tkwargs = {"dtype": dtype, "device": device}
        self.device=device
        self.dtype=dtype
        # NOTE: only inequality constraints implemented
        self.constraints = constraints #self.constraints_to_tensor(constraints)
        self.lengthscales = []
        self.acq_values = []
        self.is_init = True
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the sqlite_db attribute before pickling
        del state['sql_database']
        return state

    def __setstate__(self, state):
        # Restore the sqlite_db attribute after unpickling
        # (You can set it to a default value or recreate it as needed)
        self.__dict__.update(state)
        self.sql_database = SQLManager()
    def constraints_to_tensor(self, constraints):
        if constraints is not None:
            _constraints = []
            for c in constraints["ieq"]:
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
    
    # def identity_best_in_trial(self):
    #     new_best_found = False
    #     if self.minimize:
    #         best_in_trial = min(self.Y_next).item()
    #         best_in_trial_idx = torch.argmin(self.Y_next).item()
    #     else:
    #         best_in_trial = max(self.Y_next).item() # TODO: Think about general way to handle min and max
    #         best_in_trial_idx = torch.argmax(self.Y_next).item()
    #     if self.Y_current_best == None:
    #         self.Y_current_best = best_in_trial
    #         new_best_found = True
    #         self.logger.info(f"New best Y found: {self.Y_current_best*-1}")
    #     else:
    #         # is_better = self.Y_current_best < best_in_trial if self.minimize else self.Y_current_best > best_in_trial
    #         if self.Y_current_best < best_in_trial if self.minimize else self.Y_current_best > best_in_trial:
    #             self.Y_current_best = best_in_trial
    #             new_best_found = True
    #             self.logger.info(f"New best Y found: {self.Y_current_best*-1 if self.minimize else self.Y_current_best}")
    #     return new_best_found, best_in_trial_idx

    def complete(self, y, yvar = None):

        self.Y_next  = torch.tensor(y, dtype=self.dtype, device=self.device).unsqueeze(-1) if not torch.is_tensor(y) else y
        self.X_next = torch.tensor(self.X_next, dtype=self.dtype, device=self.device) if not torch.is_tensor(self.X_next) else self.X_next

    
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
        if not Path(path).exists():
            Path(path).mkdir(parents=True, exist_ok=True)
        with open((path + "/" + str(self.experiment_id) + "_" + str(self.replication) + "_" + self.get_name() +  ".pkl"), "wb") as fo:
            pickle.dump(self, fo)
        torch.cuda.empty_cache()
    def get_y_pred(self, index=-1):
        """
        NOTE: y-pred not implemented. Returning None
        """
        return None
    
    def get_lengthscale_botorch(self, model):
        if isinstance(model, (SingleTaskGP, FixedNoiseGP, HeteroskedasticSingleTaskGP, SaasFullyBayesianSingleTaskGP)):
            ls = model.covar_module.base_kernel.lengthscale.clone().detach().numpy()
        elif isinstance(model, ModelListGP):
            ls = []
            for m in model.models:
                ls.append(m.covar_module.base_kernel.lengthscale.clone().detach().numpy())
            ls = np.array(ls)
        else:
            ls = np.array([])
        self.lengthscales.append(ls)
        return ls
    def append_lengthscale_mo(self, model_list: ModelListGP|list):
        ls_arr = []
        if isinstance(model_list, ModelListGP):
            for m in model_list.models:
                ls_arr.append(m.covar_module.base_kernel.lengthscale.clone().detach().tolist())
        else:
            # Mainly for MorBo wich has a ModelListGP for every TR
            for mlgp in model_list:
                inner_ls_arr = []
                for m in mlgp.models:
                    inner_ls_arr.append(m.covar_module.base_kernel.lengthscale.clone().detach().tolist())
                ls_arr.append(inner_ls_arr)


        self.lengthscales.append(ls_arr)
    def get_acq_value(self, index=-1):
        if self.acq_values == None:
            return "na"
        if self.acq_values.ndim == 0:
            return self.acq_values.item()
        return self.acq_values[index].item()
    def get_feature_importance(self, all=False, index=-1):

        if len(self.lengthscales) == 0:
            return "na"
        if all:
            lengthscale = self.lengthscales
        else:
            lengthscale = self.lengthscales[index] 
            
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
            
      



