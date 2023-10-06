# NOT READY /PZm 2023-07-10
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.evaluator import Evaluator
import numpy as np
import torch
import time
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem
from backend.algorithms.constraints import create_callable_function
from backend.algorithms.optimization_algorithm_bridge import OptimizationAlgorithmBridge
from icecream import ic
class NSGA2Runner(OptimizationAlgorithmBridge):
    def __init__(self, experiment_id, replication, dim, trial_size, objectives, constraints=None, param_names=None, eval_budget=9999, num_init=-1, device=..., dtype=torch.double) -> None:
        super().__init__(experiment_id, replication, dim, trial_size, constraints, num_init, device, dtype)
        n_constraints = len(constraints) if constraints is not None else 0 # eq constraints are not supported yet
        self.is_moo = True
        self.is_ea = True
        termination = NoTermination()
        self.constraints = constraints
        if self.constraints is not None:
            self.constraints = []
            for c in constraints:
                self.constraints.append(create_callable_function(c, param_names))
        else:
            self.constraints = None
        self.problem = Problem(n_var=dim, n_obj=len(objectives), n_ieq_constr=n_constraints , xl=np.zeros(dim), xu=np.ones(dim))

        self.nsga = NSGA2(pop_size=trial_size)
        self.nsga.setup(problem=self.problem, termination=termination, )
    
    def suggest_initial(self, num_trials=None):
        return self.suggest(num_trials)


    def suggest(self, num_trials = None):
        gen_start = time.time()
        self.pop_next = self.nsga.ask()
        self.X_next = self.pop_next.get("X")
        self.gen_runtimes.append(time.time() - gen_start)
        self.fit_runtimes.append(0)
        return self.X_next
    def complete(self, y, yvar=None):
        y_with_con = None
        # for Parameter Constraints
        # TODO: Outcome Constraints
        if self.constraints is not None:
            for c in self.constraints:
                y.append([c(x) for x in self.X_next])
            y_with_con = np.column_stack(y)
   
        static = StaticProblem(self.problem, F=y_with_con if y_with_con is not None else y )
        Evaluator().eval(static, self.pop_next)
        self.nsga.tell(infills=self.pop_next)
        return super().complete(y, yvar)
        