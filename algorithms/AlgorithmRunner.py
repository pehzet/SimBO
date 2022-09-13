import torch
import logging
logger = logging.getLogger("algorithm")
from torch.quasirandom import SobolEngine
from pathlib import Path
import pickle
from dataclasses import dataclass

@dataclass
class AlgorithmRunner:

    def __init__(self, experiment_id, replication, dim, batch_size, num_init=-1, device="cpu", dtype=torch.double) -> None:
        self.experiment_id = experiment_id
        self.replication = replication
        self.dim = dim
        self.num_init = dim * 2 if num_init == -1 else num_init
        self.batch_size = batch_size
        
        self.X = None
        self.Y = None
        self.X_next = None
        self.Y_next = None
        self.Yvar = None
        self.Yvar_next = None

        self.minimize = True
        self.minimize = True
        self.total_runtime = 0
        self.batch_runtimes = list()
        self.num_restarts = 0
        self.eval_runtimes_second = list()
        self.seed = 0

        self.device=device
        self.dtype=dtype


    
    def suggest_initial(self):
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=0)
        self.X_next = sobol.draw(n=self.num_init).to(dtype=self.dtype, device=self.device)
        logger.debug(f"Initial SOBOL candidates: {self.X_next}")
        return self.X_next 
    

    def complete(self, y, yvar = None):
        self.Y_next  = torch.tensor(y, dtype=self.dtype, device=self.device).unsqueeze(-1)
        if yvar is not None:
            self.Yvar_next = torch.tensor(yvar, dtype=self.dtype, device=self.device).unsqueeze(-1)
            self.Yvar = torch.cat((self.Yvar, self.Yvar_next),  dim=0) if self.Yvar is not None else self.Yvar_next
        self.X = torch.cat((self.X, self.X_next), dim=0) if self.X is not None else self.X_next
        self.Y = torch.cat((self.Y, self.Y_next), dim=0) if self.Y is not None else self.Y_next

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

        path = "data/experiment_" + str(self.experiment_id)
        Path(path).mkdir(parents=True, exist_ok=True)
        with open((path + "/" + str(self.experiment_id) + "_" + str(self.replication) + "_" + self.get_name() +  ".pkl"), "wb") as fo:
            pickle.dump(self, fo)


