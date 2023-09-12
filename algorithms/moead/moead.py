'''
PZM: 2023-09-12
TRIAL SIZE / POP SIZE NOT WORKING -> init is 10 then 1 
Workaround: for loop with ask working neither
Error: Popsize has no Attribute F
  File "C:\code\SimBO\simbo-py11-env\Lib\site-packages\pymoo\algorithms\moo\moead.py", line 112, in _next
    self.ideal = np.min(np.vstack([self.ideal, off.F]), axis=0)
AttributeError: 'Population' object has no attribute 'F'

Dont know what to do. Working on other things first
'''
'''
Characteristics: Decomposes a multi-objective problem into several sub-tasks and solves them simultaneously. Each sub-task focuses on optimizing a certain region of the Pareto front.
Recommended number of objectives: Works well for 2-3 objectives but can be adapted for many-objective optimization using different decomposition and reference direction strategies.
Estimated trials: The number of trials (or subproblems) is usually set equal to the population size. For dimensionality, the rule of 10 times the number of dimensions can be used as a starting point, but the true number might vary depending on the specific problem.
'''

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.problem import Problem
from pymoo.core.evaluator import Evaluator
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem
from pymoo.factory import get_reference_directions
from pymoo.core.population import Population
import numpy as np
import torch
from icecream import ic
from ..constraints import create_callable_function
from ..optimization_algorithm_bridge import OptimizationAlgorithmBridge

class MOEADRunner(OptimizationAlgorithmBridge):
    def __init__(self, experiment_id, replication, dim, trial_size, objectives, constraints=None, param_names=None, eval_budget=9999, num_init=-1, device=..., dtype=torch.double) -> None:
        super().__init__(experiment_id, replication, dim, trial_size, constraints, num_init, device, dtype)
        self.is_moo = True
        n_constraints = len(constraints) if constraints is not None else 0 # eq constraints are not supported yet
        
        termination = NoTermination()
        self.constraints = constraints
        if self.constraints is not None:
            self.constraints = []
            for c in constraints:
                self.constraints.append(create_callable_function(c, param_names))
        else:
            self.constraints = None

        self.problem = Problem(n_var=dim, n_obj=len(objectives), n_ieq_constr=n_constraints, xl=np.zeros(dim), xu=np.ones(dim))

        # Create reference directions for decomposition
        # Asked ChatGPT for the reference direction method
        if len(objectives) == 2:
            ref_dir_method = "uniform"
        elif len(objectives) == 3:
            ref_dir_method = "das-dennis"
        else:
            ref_dir_method = "energy"
        ref_dirs = get_reference_directions(ref_dir_method, len(objectives), n_points=trial_size, )

        # Initialize the MOEA/D algorithm
        self.moead = MOEAD(ref_dirs, n_neighbors=20, decomposition="pbi", prob_neighbor_mating=0.9)
        self.moead.setup(problem=self.problem, termination=termination)
        self.moead.pop_size = trial_size
        ic(self.moead.__dict__)
    def suggest_initial(self, num_trials=None):
        return self.suggest(num_trials)

    def suggest(self, num_trials=None):

        self.pop_next = self.moead.ask()
        self.X_next = self.pop_next.get("X")

        if self.X_next.ndim == 1:
  
            _pop_next = [self.pop_next]
            self.X_next = [self.X_next]
            for i in range(self.trial_size - 1):
                pop = self.moead.ask()
                x = pop.get("X")
                _pop_next.append(pop)
                self.X_next.append(x)
            self.pop_next = Population.merge(*_pop_next)
        ic(self.X_next)
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
        ic(type(self.pop_next))
        Evaluator().eval(static, self.pop_next)
        self.moead.tell(infills=self.pop_next)
        
        return super().complete(y, yvar)
