from algorithms.optimization_algorithm_bridge import OptimizationAlgorithmBridge
import logging
from torch import tensor
from itertools import product
logger = logging.getLogger("brute_force")
import numpy as np
from icecream import ic
import sys
from botorch.utils.transforms import unnormalize, normalize
class BruteForceRunner(OptimizationAlgorithmBridge):
    def __init__(self, experiment_id, replication, dim, batch_size, bounds, num_init=-1, device="cpu", dtype=...) -> None:
        super().__init__(experiment_id, replication, dim, batch_size, num_init, device, dtype)
        self.bounds = bounds

        logger.info("Note: Brute Force is no optimization algorithm. Its just a evaluation of all permuations of the search space.")
        #self.suggest_initial = types.MethodType(self._suggest_initial, self)
    def suggest_initial(self):
        return self.suggest()

    def format_bounds_brute_force(self):
        return [(self.bounds[0][i], self.bounds[1][i])for i in range(len(self.bounds[0]))]

    def get_max_param_range(self):
        return max([float(self.bounds[1][i] - self.bounds[0][i] + 1) for i in range(len(self.bounds[0]))])

    def calc_num_permutations(self):
        return int(np.prod([float(self.bounds[1][i] - self.bounds[0][i] + 1) for i in range(len(self.bounds[0]))]))
    
    def suggest(self):
        if self.calc_num_permutations() > 1e+6:
            logger.error(f"Number of permuations >{self.calc_num_permutations()}< is too large (max: 1e+6). going to exit.")
            sys.exit()
        logger.warning(f"Going to generate permuations {self.calc_num_permutations()} of {len(self.bounds[0])} parameters (max range: {self.get_max_param_range()}). There will be no log while generating. So dont get stressed :).")
        xx = product(*[list(range(b[0],b[1]+1)) for b in self.format_bounds_brute_force()])
        xx = tensor([x for x in xx])
        xx = normalize(xx,self.bounds)
        return xx