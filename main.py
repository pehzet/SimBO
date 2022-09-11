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
from use_cases.mrp.mrp_runner import get_param_meta_from_materials, run_solver, init_sheets, get_param_meta, init_mrp_runner
from use_cases.mrp.mrp_sim import mrp_simulation, init_mrp_sim #run_simulation

import time

import torch
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

if len(sys.argv) < 1 :
    print("Please provide experiment ID")
    sys.exit()
else:
    experiment_id = sys.argv[1]

class ExperimentRunner:
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id

        self.algorithm = None
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

    def get_experiment_config(self):
        fpath = "configs/config" + str(self.experiment_id) +".json"
        with open(fpath, 'r') as file:
            config = json.load(file)
        #print("Config loaded")
        logger.info(f"Configuration for experiment >{self.experiment_id}< successfully loaded.")
        return config

    def get_algorithm(self, algorithm_config:dict, param_meta:dict):
        self.algorithm = algorithm_config.get("strategy", algorithm_config.get("algorithm")).lower()
        self.eval_budget = algorithm_config.get("evaluation_budget")
        
        if self.algorithm == "turbo":
            num_init = algorithm_config.get("n_init", algorithm_config.get("num_init"))
            batch_size = algorithm_config.get("batch_size")
            self.num_batches = algorithm_config.get("num_batches")
            return TurboRunner(experiment_id, len(param_meta),batch_size, num_init, param_meta=param_meta, device=tkwargs["device"], dtype=tkwargs["dtype"])
        
        if self.algorithm == "gpei":
            num_init = algorithm_config.get("n_init", algorithm_config.get("num_init"))
            batch_size = algorithm_config.get("batch_size")
            self.num_batches = algorithm_config.get("num_batches")
            return GPEIRunner(experiment_id, len(param_meta),batch_size, num_init, param_meta=param_meta, device=tkwargs["device"], dtype=tkwargs["dtype"])
        
        if self.algorithm == "saasbo":
            self.num_batches = algorithm_config.get("num_batches")
            warmup_steps = algorithm_config.get("warmup_steps", 512)
            num_samples = algorithm_config.get("num_samples", 256)
            thinning = algorithm_config.get("thinning", 16)
            batch_size = algorithm_config.get("batch_size")
            num_init = algorithm_config.get("n_init", algorithm_config.get("num_init"))
            return SaasboRunner(self.experiment_id, len(param_meta), num_init=num_init, batch_size=batch_size, warmup_steps=warmup_steps,num_samples=num_samples, thinning=thinning, param_meta=param_meta, device=tkwargs["device"], dtype=tkwargs["dtype"])

    def init_mrp_experiment(self, use_case_config):
        self.bom_id = use_case_config.get("bom_id")

        bom, materials, orders, inventory = init_sheets(self.bom_id)
        init_mrp_runner(self.bom_id, bom, materials, orders, inventory)
        init_mrp_sim(bom, materials, orders)
        self.param_meta = get_param_meta_from_materials(materials) # param meta is min, max and value type. Last one is constant in our case
        

    def save_experiment_json(self):
        obj = {
            "experiment_id": self.experiment_id,
            "algorithm" : self.algorithm,
            "bom_id" : self.bom_id,
            "total_duration_seconds": self.total_duration_seconds,
            "experiment_start" : self.experiment_start_dts,
            "experiment_end" : self.experiment_end_dts,
            "trial_runtimes" : self.trial_runtimes_second,
            "eval_runtimes" : self.eval_runtimes_second,
            "best_candidat" : self.best_candidat,
            "candidates": self.candidates,
            "final_feature_importances" : "na",
            "feature_importances" : self.feature_importances
        }
        ffolder = "data/" + "experiment_" + str(experiment_id)
        fpath = ffolder +"/" + "experiment_" + str(experiment_id) + ".json"

        if not os.path.exists(ffolder):
            Path(ffolder).mkdir(parents=True, exist_ok=True)
        with open(fpath, 'w+') as fo:
            json.dump(obj, fo)
        logger.info(f"Experiment data saved to >{fpath}<")
    

    def append_candidate_to_candidates_list(self, x,y):
        assert len(x) == len(y)
        for i, xx in enumerate(x):
            self.current_candidat +=1
            y_i = y[i]
            y_i["service_level"] = -1
            x_json = dict()
            for x_d in xx:
                x_json[str(x_d.get("id","material") + "_" + x_d.get("name","param"))] = x_d.get("value")
            self.candidates.append({
                "id" : self.current_candidat,
                "sm" : "na",
                "acqf" : "na",
                "tr" : self.get_tr_from_runner(),
                "x" : x_json,
                "y" : y_i,
                "ei" : "na",
                "y_pred" : "na"
            })
    
    def format_lengthscale_to_feature_importance():
        pass

    def get_tr_from_runner(self):
        if self.algorithm == "turbo":
            return self.runner.num_restarts + 1
        else:
            return None

    def get_best_candidate(self):
        # TODO: make oneline?
        ys= list()
        for c in self.candidates:
            ys.append([_y for _y in c.get("y").values()][0])
        # NOTE: is minimize
        best = self.candidates[pd.DataFrame(ys).idxmin()[0]]
        return best

    def run_optimization_loop(self, ):
        _start = time.monotonic()
        self.experiment_start_dts = datetime.now().isoformat()
        config = self.get_experiment_config()
        self.init_mrp_experiment(config.get("use_case_config"))
    
        self.runner = self.get_algorithm(config.get("algorithm_config"), self.param_meta)
        _start_trial = time.monotonic()
        _x = self.runner.suggest_initial()
        _end_trial = time.monotonic()
        self.trial_runtimes_second.append((_end_trial- _start_trial))
        x = self.runner.format_x_for_mrp(_x)
      
        _y = list()

        # TODO: Rename xx to something meaninful
        for xx in x:
            release = run_solver(xx)
            _eval_start_seconds = time.monotonic()
            _y.append(mrp_simulation().run_simulation(release))
            _eval_end_seconds = time.monotonic()
            self.eval_runtimes_second.append(_eval_end_seconds - _eval_start_seconds)
        self.append_candidate_to_candidates_list(x,_y)
        y = self.runner.format_y_from_mrp(_y)
   
        self.runner.complete(y)

        for _ in range(self.num_batches):
            _start_trial = time.monotonic()
            _x = self.runner.suggest()
            x = self.runner.format_x_for_mrp(_x)
            assert len(x) > 0
            if len(x[-1]) > 1:
                _y = list()
                for xx in x:
                    _eval_start_seconds = time.monotonic()
                    release = run_solver(xx)
                    _eval_start_seconds = time.monotonic()
                    _y.append(mrp_simulation().run_simulation(release))
                    _eval_end_seconds = time.monotonic()
                    self.eval_runtimes_second.append(_eval_end_seconds - _eval_start_seconds)
            
            # NOTE: maybe remove this part and wrap single candidate in list, so "if" would work as well
            else:
                _eval_start_seconds = time.monotonic()
                releases = run_solver(x[0])
                _y = mrp_simulation().run_simulation(releases)
                _eval_end_seconds = time.monotonic()
                self.eval_runtimes_second.append(_eval_end_seconds - _eval_start_seconds)
            self.append_candidate_to_candidates_list(x,_y)
            y = self.runner.format_y_from_mrp(_y)
            self.runner.complete(y)
            _end_trial = time.monotonic()
            self.trial_runtimes_second.append((_end_trial- _start_trial))
        _end = time.monotonic()
        self.experiment_end_dts = datetime.now().isoformat()
        self.total_duration_seconds =  _end -_start
        
        self.best_candidat = self.get_best_candidate()
        self.runner.terminate_experiment()
        self.save_experiment_json()
    
if __name__ == "__main__":
    ExperimentRunner(experiment_id).run_optimization_loop()


