from torch.quasirandom import SobolEngine
import torch
from algorithms.optimization_algorithm_bridge import OptimizationAlgorithmBridge
import logging
import types
logger = logging.getLogger("sobol")
class SobolRunner(OptimizationAlgorithmBridge):

    def __init__(self, experiment_id, replication, dim, batch_size, num_init=-1, device="cpu", dtype=torch.double) -> None:
        super().__init__(experiment_id, replication, dim, batch_size, constraints = None, num_init=num_init, device=device, dtype=dtype)
        self.logger.info(f"Running on device: {self.device} and dtype: {self.dtype}")
        #self.suggest_initial = types.MethodType(self._suggest_initial, self)
    def suggest_initial(self):
        return self.suggest()
    def suggest(self):
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.seed) # set seed to None so it would be random.
        self.X_next = sobol.draw(n=self.eval_budget).to(dtype=self.dtype, device=self.device)
        self.logger.debug(f"SOBOL candidates: {self.X_next}")
        return self.X_next 