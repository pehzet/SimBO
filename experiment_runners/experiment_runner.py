# Logging and warnings

import warnings
import logging
from numpy import NaN

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s: %(levelname)s: %(name)s: %(message)s'
    )

logger = logging.getLogger("runner")

# Surpress PyTorch warning
warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is") 
warnings.filterwarnings("ignore", message="Could not import matplotlib.pyplot")
warnings.filterwarnings("ignore", message="torch.triangular_solve is deprecated") 



import json

import os

from pathlib import Path

from algorithms.turbo.turbo_botorch import TurboRunner
from algorithms.saasbo.saasbo_botorch import SaasboRunner
from algorithms.gpei.gpei_botorch import GPEIRunner
from algorithms.cmaes.cmaes import CMAESRunner
from algorithms.sobol.sobol_botorch import SobolRunner
from algorithms.brute_force.brute_force import BruteForceRunner
from use_cases.mrp.mrp_runner import MRPRunner
from use_cases.pfp.pfp_runner import PfpRunner


import torch

tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

from icecream import ic
class ExperimentRunner():

    def __init__(self, experiment, replication, tkwargs) -> None:
        self.experiment_id = experiment.get("experiment_id")
        self.replication = replication
        self.tkwargs = tkwargs
        self.algorithm = None
        self.minimize = True # get from config later

        self.total_duration_seconds = None
        self.experiment_start_dts = None
        self.experiment_end_dts = None
        self.trial_runtimes_second = list()
        self.eval_runtimes_second = list()
      
        self.feature_importances = list()
        self.candidates = list()
        self.best_candidat = None

        self.first_log_done = False
        
        self.current_candidat = 0
        self.current_trial = 0
        self.current_arm = 0
        self.current_x = None
        self.current_y = None
        self.config = experiment
    
        self.use_case_runner = self.get_use_case_runner()
        self.algorithm_runner = self.get_algorithm_runner()
        self.algorithm_runner.minimize = self.minimize
    
    def get_experiment_config(self):
        fpath = "configs/config" + str(self.experiment_id) +".json"
        with open(fpath, 'r') as file:
            config = json.load(file)
        
        logger.info(f"Configuration for experiment >{self.experiment_id}< successfully loaded.")
        return config

    def get_algorithm_runner(self):
        algorithm_config = self.config.get("algorithm_config")
        dim = len(self.use_case_runner.param_meta) 
    
        constraints = self.use_case_runner.constraints 
        objectives = self.use_case_runner.objectives
        self.is_moo = True if len(objectives) > 1 else False
        
        self.algorithm = algorithm_config.get("strategy", algorithm_config.get("algorithm")).lower()
        self.eval_budget = int(self.config.get("budget", self.config.get("evaluation_budget")))
        # num_init = algorithm_config.get("n_init", algorithm_config.get("num_init", algorithm_config.get("init_arms", 1)))
        num_init = int(self.config.get("init_arms", 1))
        batch_size = int(self.config.get("batch_size"))
        if self.algorithm == "turbo":

    
            self.num_trials = algorithm_config.get("num_trials")
            sm = algorithm_config.get("sm") if algorithm_config.get("sm") not in ["None", None, "default", "Default", "nan", NaN] else "fngp"
            return TurboRunner(self.experiment_id, self.replication, dim,batch_size,constraints, num_init=num_init, device=self.tkwargs["device"], dtype=self.tkwargs["dtype"],sm=sm)
        
        if self.algorithm == "gpei":
            self.num_trials = algorithm_config.get("num_trials")
            sm = algorithm_config.get("sm") if algorithm_config.get("sm") not in ["None", None, "default", "Default","nan", NaN] else "hsgp"
            return GPEIRunner(self.experiment_id, self.replication, dim,batch_size, constraints, num_init, device=self.tkwargs["device"], dtype=self.tkwargs["dtype"],sm=sm)
        
        if self.algorithm == "saasbo":
            warmup_steps = algorithm_config.get("warmup_steps", 512)
            num_samples = algorithm_config.get("num_samples", 256)
            thinning = algorithm_config.get("thinning", 16)
            return SaasboRunner(self.experiment_id, self.replication, dim, num_init=num_init, batch_size=batch_size, constraints=constraints, warmup_steps=warmup_steps,num_samples=num_samples, thinning=thinning, device=self.tkwargs["device"], dtype=self.tkwargs["dtype"])
        
        if self.algorithm == "cmaes" or self.algorithm == "cma-es":
            
            sigma0 = algorithm_config.get("sigma", 0.5)

            return CMAESRunner(self.experiment_id, self.replication, dim, batch_size,self.use_case_runner.bounds,sigma0, num_init=num_init)
        
        if self.algorithm == "sobol":
            return SobolRunner(self.experiment_id, self.replication,dim,batch_size=1, num_init=1, device=self.tkwargs["device"], dtype=self.tkwargs["dtype"])
        
        if self.algorithm in ["brute_force", "bruteforce"]:

            return BruteForceRunner(self.experiment_id, self.replication, dim, batch_size=1, bounds = self.use_case_runner.bounds, num_init=1,device=self.tkwargs["device"], dtype=self.tkwargs["dtype"])
    
    def get_use_case_runner(self):
        use_case_config = self.config.get("use_case_config")
        if use_case_config.get("use_case").lower() == "mrp":
            return MRPRunner(use_case_config.get("bom_id"), use_case_config.get("num_sim_runs"), use_case_config.get("stochastic_method"))
        if use_case_config.get("use_case").lower() == "pfp":
            return PfpRunner()
    def save_experiment_json(self):
        fi = self.use_case_runner.format_feature_importance(self.feature_importances)

        obj = {
            "experiment_id": self.experiment_id,
            "replication" : self.replication,
            "algorithm" : self.algorithm,
            "bom_id" :  -1, #self.bom_id,
            "num_trials" : self.current_trial,
            "num_candidates" : len(self.candidates),
            "total_duration_seconds": self.total_duration_seconds,
            "experiment_start" : self.experiment_start_dts,
            "experiment_end" : self.experiment_end_dts,
            "trial_runtimes" : self.trial_runtimes_second if self.algorithm != "brute_force" else "na",
            "eval_runtimes" : self.eval_runtimes_second if self.algorithm != "brute_force" else "na",
            "best_candidate" : self.best_candidat,
            "candidates": self.candidates if self.algorithm != "brute_force" else "na",
            "final_feature_importances" : fi[-1] if fi != "na" else "na",
            "feature_importances" : fi if self.algorithm != "brute_force" else "na"
        }
        ffolder = "data/" + "experiment_" + str(self.experiment_id)
        fpath = ffolder +"/" + "experiment_" + str(self.experiment_id) +"_"+str(self.replication) + ".json"

        if not os.path.exists(ffolder):
            Path(ffolder).mkdir(parents=True, exist_ok=True)
        with open(fpath, 'w+') as fo:
            json.dump(obj, fo)
        logger.info(f"Experiment data saved to >{fpath}<")
        self.results = obj
        return obj