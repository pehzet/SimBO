'''
Useful for many objectives (>=3) and many trials (>=1000)
Characteristics: An extension of NSGA-II designed specifically for many-objective optimization (problems with more than three objectives). Uses reference points to maintain diversity among solutions.
Recommended number of objectives: Works well for 3 and more objectives. Specifically designed for many-objective optimization.
Estimated trials: Similar to NSGA-II, a population size of at least 10 times the number of decision variables can be a starting point. However, for many-objective problems, you might need a larger population to adequately cover the objective space.
'''
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.core.evaluator import Evaluator
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem
from pymoo.factory import get_reference_directions
import numpy as np
import torch
import time
from constraints import create_callable_function
from optimization_algorithm_bridge import OptimizationAlgorithmBridge

class NSGA3Runner(OptimizationAlgorithmBridge):
    def __init__(self, experiment_id, replication, dim, trial_size, objectives, constraints=None, param_names=None, eval_budget=9999, num_init=-1, device=..., dtype=torch.double) -> None:
        super().__init__(experiment_id, replication, dim, trial_size, constraints, num_init, device, dtype)
        
        n_constraints = len(constraints) if constraints is not None else 0
        
        termination = NoTermination()
        self.constraints = constraints
        if self.constraints is not None:
            self.constraints = []
            for c in constraints:
                self.constraints.append(create_callable_function(c, param_names))
        else:
            self.constraints = None

        self.problem = Problem(n_var=dim, n_obj=len(objectives), n_ieq_constr=n_constraints, xl=np.zeros(dim), xu=np.ones(dim))

        # Create reference directions
        ref_dirs = get_reference_directions("das-dennis", len(objectives), n_points=trial_size)

        # Initialize the NSGA-III algorithm
        self.nsga3 = NSGA3(ref_dirs=ref_dirs)
        self.nsga3.setup(problem=self.problem, termination=termination)

    def suggest(self, num_trials=None):
        gen_start = time.time()
        self.pop_next = self.nsga3.ask()
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

        static = StaticProblem(self.problem, F=y_with_con if y_with_con is not None else y)
        Evaluator().eval(static, self.pop_next)
        self.nsga3.tell(infills=self.pop_next)
        
        return super().complete(y, yvar)
