# NOT TESTED! /PZm 2023-07-10
from typing import Callable, Dict, List, Optional, Union
import time
import torch
import numpy as np
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective

from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.sampling import draw_sobol_samples
from .gen import (
    TS_select_batch_MORBO,
)
from botorch.models import ModelListGP
from .state import TRBOState
from .trust_region import TurboHParams
from torch import Tensor
from backend.algorithms.optimization_algorithm_bridge import OptimizationAlgorithmBridge
from icecream import ic
from copy import copy

REF_POINT = torch.tensor([-0.1, -0.1]) # later set with domain knowledge; this one only works for minimization
class MorboRunner(OptimizationAlgorithmBridge):
    def __init__(self, experiment_id, replication, dim, trial_size, n_trust_regions, objectives,ref_point, constraints=None, eval_budget=9999, num_init=-1, device=..., dtype=torch.double) -> None:
        super().__init__(experiment_id, replication, dim, trial_size, constraints, num_init, device, dtype)
        self.is_moo = True
        self.n_trust_regions = n_trust_regions
        self.batch_size = trial_size
        self.eval_budget = eval_budget
        self.objectives = objectives
        self.n_objectives = len(objectives)
        self.n_constraints = len(constraints) if constraints is not None else 0

        tr_hparams = TurboHParams(
            length_init=TurboHParams.length_init,
            length_min=TurboHParams.length_min,
            length_max=TurboHParams.length_max,
            batch_size=self.batch_size,
            success_streak=TurboHParams.success_streak,
            failure_streak=TurboHParams.failure_streak,
            max_tr_size=TurboHParams.max_tr_size,
            min_tr_size=((dim * 2 if num_init == -1 else num_init) / n_trust_regions),
            trim_trace=TurboHParams.trim_trace,
            n_trust_regions=n_trust_regions,
            verbose=TurboHParams.verbose,
            qmc=TurboHParams.qmc,
            use_ard=TurboHParams.use_ard,
            max_cholesky_size=TurboHParams.max_cholesky_size,
            sample_subset_d=TurboHParams.sample_subset_d,
            track_history=TurboHParams.track_history,
            fixed_scalarization=TurboHParams.fixed_scalarization,
            n_initial_points= dim * 2 if num_init == -1 else num_init,
            n_restart_points=TurboHParams.n_restart_points,
            raw_samples=TurboHParams.raw_samples,
            max_reference_point=TurboHParams.max_reference_point,
            hypervolume=TurboHParams.hypervolume,
            winsor_pct=TurboHParams.winsor_pct,
            trunc_normal_perturb=TurboHParams.trunc_normal_perturb,
            decay_restart_length_alpha=TurboHParams.decay_restart_length_alpha,
            switch_strategy_freq=TurboHParams.switch_strategy_freq,
            tabu_tenure=TurboHParams.tabu_tenure,
            fill_strategy=TurboHParams.fill_strategy,
            use_noisy_trbo=TurboHParams.use_noisy_trbo,
            use_simple_rff=TurboHParams.use_simple_rff,
            batch_limit=TurboHParams.batch_limit,
            use_approximate_hv_computations=TurboHParams.use_approximate_hv_computations,
            approximate_hv_alpha=TurboHParams.approximate_hv_alpha,
            pred_batch_limit=TurboHParams.pred_batch_limit,
            infer_reference_point=TurboHParams.infer_reference_point,
            fit_gpytorch_options = TurboHParams.fit_gpytorch_options,
            restart_hv_scalarizations=TurboHParams.restart_hv_scalarizations,
        )
 

        # TurboHParams.n_initial_points = dim * 2 if num_init == -1 else num_init
        # TurboHParams.min_tr_size = ((dim * 2 if num_init == -1 else num_init) / n_trust_regions)
        lb = [0.0] * dim
        ub = [1.0] * dim
        self.trbo_state = TRBOState(
        dim=dim,
        max_evals=eval_budget,
        num_outputs=len(objectives), # TODO: CHANGE LATER IF OUTCOME IS NOT OBJECTIVE BUT CONSTRAINT+ len(constraints) if constraints is not None else len(objectives),
        num_objectives=len(objectives),
        bounds=torch.tensor([lb, ub], dtype=dtype, device=device),
        tr_hparams=tr_hparams,
        constraints=constraints,
        # infer_reference_point=True,
        #objective=IdentityMCMultiOutputObjective([i + 1 for i in range(len(objectives) + len(constraints) if constraints is not None else len(objectives))]),
        )
        # For saving outputs

        self.is_init = True
        self.n_evals = []
        self.hvs = []
        self.pareto_X = []
        self.pareto_Y = []
        self.n_points_in_tr = [[] for _ in range(n_trust_regions)]
        self.n_points_in_tr_collected_by_other = [[] for _ in range(n_trust_regions)]
        self.n_points_in_tr_collected_by_sobol = [[] for _ in range(n_trust_regions)]
        self.tr_sizes = [[] for _ in range(n_trust_regions)]
        self.tr_centers = [[] for _ in range(n_trust_regions)]
        self.tr_restarts = [[] for _ in range(n_trust_regions)]
        self.fit_times = []
        self.gen_times = []
        self.true_ref_point = torch.tensor(ref_point, dtype=dtype, device=device)
        self.X_next = None
        self.Y_next = None
        self.Yvar_next = None
        self.all_tr_indices = [-1] * self.trial_size
        self.is_restart_trial = False
        self.should_restart_trs = [False for _ in range(n_trust_regions)]

    def restart_trust_regions(self):
        all_X_centers = []
        for i in range(self.trbo_state.tr_hparams.n_trust_regions):
            if self.should_restart_trs[i]:
                # n_points = min(self.n_restart_points, self.max_evals - self.trbo_state.n_evals)
                # if n_points <= 0:
                #     break  # out of budget

                self.trbo_state.TR_index_history[self.trbo_state.TR_index_history == i] = -1
                if self.trbo_state.tr_hparams.restart_hv_scalarizations:
                    # generate new point
                    X_center = self.trbo_state.gen_new_restart_design()
                    all_X_centers.append(X_center)
        if len(all_X_centers) == 0:
            return self.suggest_initial()
        return torch.cat(all_X_centers, dim=0)


    def suggest(self, num_trials = None):
        # NOTE: Check what switch strategy does, so we will not implement it yet.
        # switch_strategy = self.trbo_state.check_switch_strategy()
        # if switch_strategy:
        #     self.should_restart_trs = [True for _ in self.should_restart_trs]
        gen_start = time.time()
        if any(self.should_restart_trs):
            self.is_restart_trial = True
            self.X_next = self.restart_trust_regions()
            return self.X_next
  
        selection_output = TS_select_batch_MORBO(trbo_state=self.trbo_state)

        all_tr_indices = [-1] * self.batch_size if num_trials is None else [-1] * num_trials
        self.X_next = selection_output.X_cand
        self.tr_indices = selection_output.tr_indices
        all_tr_indices.extend(self.tr_indices.tolist())
        self.gen_runtimes.append(time.time() - gen_start)
        self.trbo_state.tabu_set.log_iteration()
        self.append_lengthscale_mo(self.trbo_state.models)
        return self.X_next

    def complete_restart_trial(self, y, yvar=None):
        for i in range(self.trbo_state.tr_hparams.n_trust_regions):
            init_kwargs = {}
            init_kwargs["X_init"] = self.X_next
            init_kwargs["Y_init"] = y
            init_kwargs["X_center"] = self.X_next
            self.trbo_state.update(
                X=self.X_next,
                Y=y,
                new_ind=torch.tensor(
                    [i], dtype=torch.long, device=self.X_next.device
                ),
            )
            self.trbo_state.log_restart_points(X=self.X_next, Y=y)
            switch_strategy = self.trbo_state.check_switch_strategy()
            self.trbo_state.initialize_standard(
                tr_idx=i,
                restart=True,
                switch_strategy=switch_strategy,
                **init_kwargs,
            )
            if self.trbo_state.tr_hparams.restart_hv_scalarizations:
                # we initialized the TR with one data point.
                # this passes historical information to that new TR
                self.trbo_state.update_data_across_trs()
            self.tr_restarts[i].append(
                self.trbo_state.n_evals.item()
            )  # Where it restarted
        self.is_restart_trial = False

    def complete_init(self, y, yvar=None):
        self.trbo_state.update(X=self.X_next, Y=y, new_ind=torch.full(
        (self.X_next.shape[0],), 0, dtype=torch.long, device=self.X_next.device),)
        for i in range(self.n_trust_regions):
            self.trbo_state.initialize_standard(
                tr_idx=i,
                restart=False,
                switch_strategy=False,
                X_init=self.X_next,
                Y_init=y,
                )
        self.trbo_state.update_data_across_trs()
        self.trbo_state.TR_index_history.fill_(-2)
        
        self.is_init = False

    def complete(self, y, yvar=None):

        if self.is_init:
            self.complete_init(y, yvar)
        elif self.is_restart_trial:
            self.complete_restart_trial(y, yvar)

        else:
            start_fit = time.time()
            self.all_tr_indices = [-1] * self.X_next.size(dim=0)
            self.trbo_state.update(X=self.X_next, Y=y, new_ind=self.tr_indices)
            self.should_restart_trs = self.trbo_state.update_trust_regions_and_log(
                X_cand=self.X_next,
                Y_cand=y,
                tr_indices=self.tr_indices,
                batch_size=self.trial_size,
                verbose=False,
            )
            # self.fit_times.append(time.time() - start_fit)
            self.fit_runtimes.append(time.time() - start_fit)


        # Hypervolume
        if self.trbo_state.hv is not None:

            partitioning = DominatedPartitioning(
                ref_point=self.true_ref_point, Y=self.trbo_state.pareto_Y)
            
            hv = partitioning.compute_hypervolume().item()

            self.pareto_X_next = self.trbo_state.pareto_X.tolist()
            self.pareto_Y_next = self.trbo_state.pareto_Y.tolist()

            self.hvs.append(hv)
        else:
            self.pareto_X_next = []
            self.pareto_Y_next = []
            self.hvs.append(0.0)
        self.pareto_X.append(self.pareto_X_next)
        self.pareto_Y.append(self.pareto_Y_next)
        # LOGGING
        for i, tr in enumerate(self.trbo_state.trust_regions):
            inds = torch.cat(
                [torch.where((x == self.trbo_state.X_history).all(dim=-1))[0] for x in tr.X]
            )
            tr_inds = self.trbo_state.TR_index_history[inds]
            assert len(tr_inds) == len(tr.X)
            self.n_points_in_tr[i].append(len(tr_inds))
            self.n_points_in_tr_collected_by_sobol[i].append(sum(tr_inds == -2).cpu().item())
            self.n_points_in_tr_collected_by_other[i].append(
                sum((tr_inds != i) & (tr_inds != -2)).cpu().item()
            )
            self.tr_sizes[i].append(tr.length.item())
            self.tr_centers[i].append(tr.X_center.cpu().squeeze().tolist())
        super().complete(y, yvar)


