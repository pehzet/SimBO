'''
Based on the paper
Bayesian Optimization over High-Dimensional Combinatorial Spaces via Dictionary-based Embeddings
'''
import warnings
import torch
import time
from torch import Tensor
from torch.quasirandom import SobolEngine
from botorch import fit_gpytorch_model
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.fully_bayesian import MIN_INFERRED_NOISE_LEVEL
from botorch.models.model import ModelList
from botorch.models.transforms import Normalize, Standardize
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.utils.warnings import NumericalWarning

from backend.algorithms.optimization_algorithm_bridge import OptimizationAlgorithmBridge
from .categorical_dictionary_kernel import DictionaryKernel as CatDictionaryKernel
from .dictionary_kernel import DictionaryKernel
from .optimize import optimize_acq_function_mixed_alternating, optimize_acqf_binary_local_search

class BodiRunner(OptimizationAlgorithmBridge):
    def __init__(self, 
                experiment_id,
                replication,
                dim,
                trial_size,
                objectives,
                ref_point,
                binary_inds,
                categorical_inds,
                constraints=None,
                eval_budget=9999,
                num_init=-1,
                device=...,
                dtype=torch.double) -> None:
        super().__init__(experiment_id, replication, dim, trial_size, constraints, num_init, device, dtype)
        self.is_moo = True

        self.batch_size = trial_size
        self.eval_budget = eval_budget
        self.objectives = objectives
        self.n_objectives = len(objectives)
        self.n_constraints = len(constraints) if constraints is not None else 0
        self.ref_point = ref_point
        self.hvs = []
        self.pareto_x = None
        self.pareto_y = None
        self.binary_inds = list(range(self.binary_inds)) if binary_inds is not None else None
        self.categorical_inds = list(range(self.categorical_inds)) if categorical_inds is not None else None
        self.n_binary = len(self.binary_inds) if self.binary_inds is not None else 0
        self.n_categorical = len(self.categorical_inds) if self.categorical_inds is not None else 0
        self.n_continuous = self.dim - self.n_binary - self.n_categorical
        self.n_prototype_vectors: int = 10,

        self.afo_config = {
                "n_initial_candts": 2000,
                "n_restarts": 20,
                "afo_init_design": "random",
                "n_alternate_steps": 50,
                "num_cmaes_steps": 50,
                "num_ls_steps": 50,
                "n_spray_points": 200,
                "verbose": False,
                "add_spray_points": True,
                "n_binary": self.n_binary,
                "n_cont": self.n_continuous,
            }


    def initialize_model(self, train_x, train_y, train_Y_var, train_con=None, sm="stgp"):
        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(torch.tensor(0.9, **self.tkwargs), torch.tensor(10.0, **self.tkwargs)),
            noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
        )
        if self.n_binary == 0 and self.n_categorical > 0:

            dictionary_kernel = CatDictionaryKernel(
                num_basis_vectors=self.n_prototype_vectors,

                categorical_dims = self.categorical_inds,
                num_dims=self.dim,
                similarity=True,
            )
        elif self.n_binary > 0 and self.n_categorical == 0:
            dictionary_kernel = DictionaryKernel(
                num_basis_vectors=self.n_prototype_vectors,
                binary_dims=self.binary_inds,

                num_dims=self.dim,
                similarity=True,
            )
        else:
            raise NotImplementedError("3-mixed (bin, cat, cont) not implemented. For pure continuous problems use other Algorithm")
        covar_module = ScaleKernel(
            base_kernel=dictionary_kernel,
            outputscale_prior=GammaPrior(torch.tensor(2.0, **self.tkwargs), torch.tensor(0.15, **self.tkwargs)),
            outputscale_constraint=GreaterThan(1e-6)
        )
        if train_con:
            train_y = torch.cat([train_y, train_con], dim=-1)
        model_list = []

        for i in range(train_y.shape[-1]):
            if sm  == "stgp":
                m = SingleTaskGP(train_x, train_y[..., i : i + 1], 
                                covar_module=covar_module,
                                input_transform=Normalize(d=self.X.shape[-1]),
                                likelihood=likelihood,)
            elif sm == "fngp":
                m = FixedNoiseGP(train_x, 
                                train_y[..., i : i + 1],
                                train_Y_var[..., i : i + 1], 
                                covar_module=covar_module,
                                input_transform=Normalize(d=self.X.shape[-1]),
                                likelihood=likelihood,)
            model_list.append(m)

        model = ModelList(*model_list)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        return mll, model

    def suggest_initial(self, num_trials=None):
        return super().suggest_initial(num_trials)
    



    def suggest(self, num_trials=None):
        fit_start = time.time()
        mll, model = self.initialize_model(self.X, self.Y, self.Yvar)

        fit_gpytorch_model(mll)
        self.append_lengthscale_mo(model)
        self.fit_runtimes.append(time.time() - fit_start)
        gen_start = time.time()
        sampler = SobolQMCNormalSampler(num_samples=128, collapse_batch_dims=True)
        acqf =  qNoisyExpectedHypervolumeImprovement(
                X_baseline=self.X,
                model=model,
                ref_point=self.ref_point,
                sampler=sampler,
            )
        pareto_points = self.pareto_x.clone()
        with warnings.catch_warnings():  # Filter jitter warnings
            warnings.filterwarnings("ignore", category=NumericalWarning)
            if self.n_binary > 0 and self.n_continuous == 0:
                self.X_next, acq_val = optimize_acqf_binary_local_search(
                    acqf, afo_config=self.afo_config, pareto_points=pareto_points, q=self.batch_size
                )
            elif self.n_binary > 0 and self.n_continuous > 0:  # mixed search space
                cont_dims = torch.arange(self.n_binary, self.n_binary + self.n_continuous, device=self.device)
                self.X_next, acq_val = optimize_acq_function_mixed_alternating(
                    acqf, cont_dims=cont_dims, pareto_points=pareto_points, q=self.batch_size, afo_config=self.afo_config,
                )
        self.gen_runtimes.append(time.time() - gen_start)
        self.acq_values.append(acq_val)

        return self.X_next
    def complete(self, y_next, yvar=None):
        self.Y_next = y_next
        self.Yvar_next = yvar
        

        # compute pareto front
        if self.constraints is not None:
            is_feas = self.get_feasable_idx(self.Y_next)
            feas_train_obj = y_next[is_feas]
            x_next_feas = self.X_next[is_feas]
        else:
            feas_train_obj = y_next
            x_next_feas = self.X_next
        if feas_train_obj.shape[0] > 0:
            self.Y_feas = feas_train_obj if self.Y_feas is None else torch.cat([self.Y_feas, feas_train_obj], dim=0)
            self.X_feas = x_next_feas if self.X_feas is None else torch.cat([self.X_feas, x_next_feas], dim=0)
            pareto_mask = is_non_dominated(self.Y_feas)
            pareto_y = self.Y_feas[pareto_mask]
            pareto_x = self.X_feas[pareto_mask]
            self.pareto_y = pareto_y if self.pareto_y is None else torch.cat([self.pareto_y, pareto_y], dim=0)
            self.pareto_x = pareto_x if self.pareto_x is None else torch.cat([self.pareto_x, pareto_x], dim=0)
            # compute feasible hypervolume
            volume = self.hv.compute(pareto_y)
            print(f"new Hypervolume: {volume}")
        else:
            volume = 0.0
        self.hvs.append(volume)