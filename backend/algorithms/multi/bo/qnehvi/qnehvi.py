# NOT READY /PZm 2023-07-10
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
import torch
import time
from botorch import fit_gpytorch_model
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.models import FixedNoiseGP, ModelListGP, HeteroskedasticSingleTaskGP, SingleTaskGP
from backend.algorithms.optimization_algorithm_bridge import OptimizationAlgorithmBridge
from botorch.utils.transforms import unnormalize, normalize, standardize
# TODO: MAKE PARAMETERS CONFIGUABLE
REF_POINT = torch.tensor([-85546, 0.8]) # later set with domain knowledge; this one only works for minimization
MC_SAMPLES = 128
from icecream import ic
class QNEHVIRunner(OptimizationAlgorithmBridge):
    def __init__(self, experiment_id,  replication, dim, batch_size, objectives, ref_point, constraints, num_init, device, dtype, sm= "hsgp"):
        super().__init__(experiment_id, replication, dim, batch_size, constraints, num_init, device, dtype)

        self.sm = sm
        self.batch_size = batch_size
        self.Y_feas = None
        self.X_feas = None
        self.is_moo = True
        # self.acqf = "qNEHVI" # TODO: make configuable later
        self.logger.info(f"Running on device: {self.device}")
        self.ref_point = ref_point #ref_point if not REF_POINT else REF_POINT # for debugging
        self.hv = Hypervolume(ref_point=self.ref_point)
        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        self.hvs = []
        self.objectives = objectives
        self.con_next = None
        self.standard_bounds = torch.zeros(2, self.dim, device=self.device, dtype=self.dtype)
        self.standard_bounds[1] = 1
        self.pareto_X = None
        self.pareto_Y = None
    def initialize_model(self, train_x, train_y, train_con=None, sm="stgp"):
        if train_con:
            train_y = torch.cat([train_y, train_con], dim=-1)
        model_list = []
        for i in range(train_y.shape[-1]):
            if sm in ["hsgp", "hsstgp", 'heteroskedastic GP']:
                model_list.append(HeteroskedasticSingleTaskGP(train_x, train_y[..., i : i + 1], outcome_transform=Standardize(m=1)))
            elif sm in ["fngp", "fgp","fixed noise GP"]:
                model_list.append(FixedNoiseGP(train_x, train_y[..., i : i + 1], outcome_transform=Standardize(m=1)))
            elif sm in ["stgp", "single task GP"]:
                model_list.append(SingleTaskGP(train_x, train_y[..., i : i + 1], outcome_transform=Standardize(m=1)))
            else:
                raise NotImplementedError(f"sm {sm} not implemented")
        model = ModelListGP(*model_list)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model
        
    def optimize_qnehvi(self, model):



        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.ref_point    ,
            X_baseline=self.X,
            sampler=self.sampler,
            prune_baseline=True,
            # define an objective that specifies which outcomes are the objectives
            objective=IdentityMCMultiOutputObjective(outcomes=[i for i in range(len(self.objectives))]),
            # specify that the constraint(s) are the last column(s) of the outputs
            constraints=[lambda Z,index=index: Z[..., (-(index+1))] for index in range(len(self.constraints[1]))] if self.constraints else None,
        )
        # optimize
        X_next, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.standard_bounds,
            q=self.batch_size,
            num_restarts=10,
            raw_samples=512,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        # observe new values
        # X_next = unnormalize(candidates.detach(), bounds=problem.bounds)
        X_next = X_next.detach()
        return X_next


    def suggest(self):
        # Standarize Y and normalize Noise as said here: https://botorch.org/api/models.html#botorch.models.gp_regression.SingleTaskGP
        # This performs better. Testet fngp and hsgp / 2022-10-04 PZ
        # self.X = torch.cat([self.X, self.X_next], dim=-2) if self.X_next is not None else self.X
        # self.Y = torch.cat([self.Y, self.Y_next], dim=-1) if self.Y_next is not None else self.Y
        fit_start = time.time()
        train_Y = standardize(self.Y_feas) #standardize because botorch says it works better

        mll, model = self.initialize_model(self.X_feas, train_Y)
        fit_gpytorch_model(mll=mll)
        self.fit_runtimes.append(time.time() - fit_start)
        gen_start = time.time()
        self.X_next = self.optimize_qnehvi(model)
        self.gen_runtimes.append(time.time() - gen_start)
        self.append_lengthscale_mo(model)
        return self.X_next

    def get_feasable_idx(self, Y):
        row_indices_to_keep = (self.constraints[0] == 1).nonzero(as_tuple=True)[1]
        selected_values = Y[:, row_indices_to_keep]
        result = selected_values < self.constraints[1]
        is_feas = result.all(dim=1)
        return is_feas

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
            self.pareto_Y_next = self.Y_feas[pareto_mask]
            self.pareto_X_next = self.X_feas[pareto_mask]
            self.pareto_Y = self.pareto_Y_next if self.pareto_Y is None else torch.cat([self.pareto_Y, self.pareto_Y_next], dim=0)
            self.pareto_X = self.pareto_X_next if self.pareto_X is None else torch.cat([self.pareto_X, self.pareto_X_next], dim=0)
            # compute feasible hypervolume
            volume = self.hv.compute(self.pareto_Y_next)
            print(f"new Hypervolume: {volume}")
        else:
            volume = 0.0
        self.hvs.append(volume)



        super().complete(y_next, yvar)

