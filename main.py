# Main File for Cluster Experiments

# Logging and warnings
import warnings
import logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s: %(levelname)s: %(name)s: %(message)s'
    )

logger = logging.getLogger("main")

# Surpress PyTorch warning
warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is") 
warnings.filterwarnings("ignore", message="Could not import matplotlib.pyplot") 

from locale import normalize
import sys
import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from algorithms.turbo.turbo_botorch import TurboRunner
from algorithms.saasbo.saasbo_botorch import SaasboRunner
from algorithms.gpei.gpei_botorch import GPEIRunner
from algorithms.cmaes.cmaes import CMAESRunner
from algorithms.sobol.sobol_botorch import SobolRunner
from algorithms.brute_force.brute_force import BruteForceRunner
from use_cases.mrp.mrp_runner import MRPRunner

from utils.gsheet_utils import get_configs_from_gsheet
import time
import torch
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

from icecream import ic


class ExperimentRunner:
    def __init__(self, experiment_id, replication):
        self.experiment_id = experiment_id
        self.replication = replication
        self.algorithm = None
        self.minimize = True # get from config later
        self.bom_id = None
        self.total_duration_seconds = None
        self.experiment_start_dts = None
        self.experiment_end_dts = None
        self.trial_runtimes_second = list()
        self.eval_runtimes_second = list()
      
        self.feature_importances = list()
        self.candidates = list()
        self.best_candidat = None

        self.current_candidat = 0
        self.current_trial = 0
        self.current_x = None
        self.current_y = None
        self.config = self.get_experiment_config()
        self.use_case_runner = self.get_use_case_runner(self.config.get("use_case_config"))
        self.algorithm_runner = self.get_algorithm_runner(self.config.get("algorithm_config"), len(self.use_case_runner.param_meta))


    def get_experiment_config(self):
        fpath = "configs/config" + str(self.experiment_id) +".json"
        with open(fpath, 'r') as file:
            config = json.load(file)
        
        logger.info(f"Configuration for experiment >{self.experiment_id}< successfully loaded.")
        return config

    def get_algorithm_runner(self, algorithm_config:dict, dim : int):
        self.algorithm = algorithm_config.get("strategy", algorithm_config.get("algorithm")).lower()
        self.eval_budget = algorithm_config.get("evaluation_budget")
    
        
        if self.algorithm == "turbo":
            num_init = algorithm_config.get("n_init", algorithm_config.get("num_init"))
            batch_size = algorithm_config.get("batch_size")
            self.num_batches = algorithm_config.get("num_batches")
            return TurboRunner(self.experiment_id, self.replication, dim,batch_size, num_init=num_init, device=tkwargs["device"], dtype=tkwargs["dtype"])
        
        if self.algorithm == "gpei":
            num_init = algorithm_config.get("n_init", algorithm_config.get("num_init"))
            batch_size = algorithm_config.get("batch_size")
            self.num_batches = algorithm_config.get("num_batches")
            return GPEIRunner(self.experiment_id, self.replication, dim,batch_size, num_init, device=tkwargs["device"], dtype=tkwargs["dtype"])
        
        if self.algorithm == "saasbo":
            self.num_batches = algorithm_config.get("num_batches")
            warmup_steps = algorithm_config.get("warmup_steps", 512)
            num_samples = algorithm_config.get("num_samples", 256)
            thinning = algorithm_config.get("thinning", 16)
            batch_size = algorithm_config.get("batch_size")
            num_init = algorithm_config.get("n_init", algorithm_config.get("num_init"))
            return SaasboRunner(self.experiment_id, self.replication, dim, num_init=num_init, batch_size=batch_size, warmup_steps=warmup_steps,num_samples=num_samples, thinning=thinning, device=tkwargs["device"], dtype=tkwargs["dtype"])
        if self.algorithm == "cmaes":
            batch_size = algorithm_config.get("batch_size")
            sigma0 = algorithm_config.get("sigma0", 0.5)
            num_init = algorithm_config.get("n_init", algorithm_config.get("num_init"))
            return CMAESRunner(self.experiment_id, self.replication, dim, batch_size,self.use_case_runner.bounds,sigma0,num_init, device=tkwargs["device"], dtype=tkwargs["dtype"])
        if self.algorithm == "sobol":
            return SobolRunner(self.experiment_id, self.replication,dim,batch_size=1, num_init=1, device=tkwargs["device"], dtype=tkwargs["dtype"])
        
        if self.algorithm == "brute_force" or self.algorithm == "bruteforce":
 
            return BruteForceRunner(self.experiment_id, self.replication, dim, batch_size=1, bounds = self.use_case_runner.bounds, num_init=1,device=tkwargs["device"], dtype=tkwargs["dtype"])
    def get_use_case_runner(self, use_case_config : dict):
        if use_case_config.get("use_case").lower() == "mrp":
            return MRPRunner(use_case_config.get("bom_id"), use_case_config.get("num_solver_runs"), use_case_config.get("stochastic_method"))
     
    def save_experiment_json(self):
        obj = {
            "experiment_id": self.experiment_id,
            "replication" : self.replication,
            "algorithm" : self.algorithm,
            "bom_id" : self.bom_id,
            "num_trials" : self.current_trial,
            "num_candidates" : len(self.candidates),
            "total_duration_seconds": self.total_duration_seconds,
            "experiment_start" : self.experiment_start_dts,
            "experiment_end" : self.experiment_end_dts,
            "trial_runtimes" : self.trial_runtimes_second,
            "eval_runtimes" : self.eval_runtimes_second,
            "best_candidat" : self.best_candidat,
            "candidates": self.candidates,
            "final_feature_importances" : "na",
            "feature_importances" : self.use_case_runner.format_feature_importance(self.feature_importances)
        }
        ffolder = "data/" + "experiment_" + str(self.experiment_id)
        fpath = ffolder +"/" + "experiment_" + str(self.experiment_id) +"_"+str(self.replication) + ".json"

        if not os.path.exists(ffolder):
            Path(ffolder).mkdir(parents=True, exist_ok=True)
        with open(fpath, 'w+') as fo:
            json.dump(obj, fo)
        logger.info(f"Experiment data saved to >{fpath}<")
    

    def append_candidate_to_candidates_list(self, xx,yy):

        assert len(xx) == len(yy)
        for i, x in enumerate(xx):
            self.current_candidat +=1
            y = yy[i]
            ts = self.algorithm_runner.get_technical_specs()
            self.candidates.append({
                "id" : self.current_candidat,
                "sm" : ts.get("sm", "na"),
                "acqf" : ts.get("acqf", "na"),
                "tr" : self.algorithm_runner.get_tr(),
                "x" : self.use_case_runner.format_x_for_candidate(x),
                "y" : self.use_case_runner.format_y_for_candidate(y),
                "fi" : self.use_case_runner.format_feature_importance(self.algorithm_runner.get_feature_importance()),
                "ei" : "na",
                "y_pred" : "na",
                "acq_value" : self.algorithm_runner.get_acq_value((self.current_candidat - self.algorithm_runner.num_init - 1))
            })
    




    def get_best_candidate(self):
        # TODO: make oneline?
        ys= list()
        for c in self.candidates: 
            ys.append([_y for _y in c.get("y").values()][0])
        # NOTE: is minimize
        best = self.candidates[pd.DataFrame(ys).idxmin()[0]]
        logger.info(f"Best candidate found:\n {json.dumps(best, indent=2)}")
        return best

 
        
    def run_optimization_loop(self):
        logger.info(f"Starting optimization run with evaluation budget of >{self.eval_budget}<")
        self.algorithm_runner.eval_budget = self.eval_budget
        _start = time.monotonic()
        self.experiment_start_dts = datetime.now().isoformat()
        _start_trial = time.monotonic()

        x = self.algorithm_runner.suggest_initial()
        _end_trial = time.monotonic()
        self.trial_runtimes_second.append((_end_trial- _start_trial))
        _y = list()
        for xx in x:
            _eval_start_seconds = time.monotonic()
            _y.append(self.use_case_runner.eval(xx))
            _eval_end_seconds = time.monotonic()
            self.eval_runtimes_second.append(_eval_end_seconds - _eval_start_seconds)
        self.append_candidate_to_candidates_list(x,_y)
        y, ysem = self.use_case_runner.transform_y_to_tensors_mean_sem(_y)
        self.algorithm_runner.complete(y, yvar = ysem)
        self.eval_budget -= len(x)
        self.current_trial +=1
        while self.eval_budget > 0:
            _start_trial = time.monotonic()
            x = self.algorithm_runner.suggest()
            assert len(x) > 0
            _y = list()

            for xx in x:
                _eval_start_seconds = time.monotonic()
                _y.append(self.use_case_runner.eval(xx))
                _eval_end_seconds = time.monotonic()
                self.eval_runtimes_second.append(_eval_end_seconds - _eval_start_seconds)


            self.append_candidate_to_candidates_list(x,_y)
            y, ysem = self.use_case_runner.transform_y_to_tensors_mean_sem(_y)
            self.algorithm_runner.complete(y, yvar = ysem)
            _end_trial = time.monotonic()
            self.trial_runtimes_second.append((_end_trial- _start_trial))

            self.eval_budget -= len(x)
            self.current_trial +=1
        _end = time.monotonic()
        self.experiment_end_dts = datetime.now().isoformat()
        self.total_duration_seconds =  _end -_start
        
        self.best_candidat = self.get_best_candidate()
        self.feature_importances = self.algorithm_runner.get_feature_importance(all=True)
        
        self.algorithm_runner.terminate_experiment()
        self.save_experiment_json()

def check_sysargs():
    if "load" in sys.argv:
        get_configs_from_gsheet(from_main=True)
        if len(sys.argv) <= 2 :
            print("No experiment ID detected. Re-loaded only config files. Going to exit")
            sys.exit()
        sys.argv.remove("load")
    if len(sys.argv) < 2 :
        print("Please provide experiment ID")
        sys.exit()
    else:
        experiment_id = sys.argv[1]
        replication = sys.argv[2] if len(sys.argv) >= 3 else 0
    return experiment_id, replication

if __name__ == "__main__":
    experiment_id, replication = check_sysargs()
    ExperimentRunner(experiment_id, replication).run_optimization_loop()


