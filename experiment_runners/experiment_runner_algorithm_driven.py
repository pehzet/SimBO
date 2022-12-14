# Main File for Cluster Experiments

# Logging and warnings
from genericpath import isfile
import warnings
import logging
from numpy import NaN

from sqlalchemy import false
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s: %(levelname)s: %(name)s: %(message)s'
    )

logger = logging.getLogger("main")

# Surpress PyTorch warning
warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is") 
warnings.filterwarnings("ignore", message="Could not import matplotlib.pyplot")
warnings.filterwarnings("ignore", message="torch.triangular_solve is deprecated") 


import sys
import json
import csv
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


class ExperimentRunnerAlgorithmDriven:
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

        self.first_log_done = False
        
        self.current_candidat = 0
        self.current_trial = 0
        self.current_x = None
        self.current_y = None
        self.config = self.get_experiment_config()
        self.use_case_runner = self.get_use_case_runner(self.config.get("use_case_config"))
        self.algorithm_runner = self.get_algorithm_runner(self.config.get("algorithm_config"), len(self.use_case_runner.param_meta))
        self.algorithm_runner.minimize = self.minimize


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
            self.num_trials = algorithm_config.get("num_trials")
            sm = algorithm_config.get("sm") if algorithm_config.get("sm") not in ["None", None, "default", "Default", "nan", NaN] else "fngp"
            return TurboRunner(self.experiment_id, self.replication, dim,batch_size, num_init=num_init, device=tkwargs["device"], dtype=tkwargs["dtype"],sm=sm)
        
        if self.algorithm == "gpei":
            num_init = algorithm_config.get("n_init", algorithm_config.get("num_init"))
            batch_size = algorithm_config.get("batch_size")
            self.num_trials = algorithm_config.get("num_trials")
            sm = algorithm_config.get("sm") if algorithm_config.get("sm") not in ["None", None, "default", "Default","nan", NaN] else "hsgp"
            return GPEIRunner(self.experiment_id, self.replication, dim,batch_size, num_init, device=tkwargs["device"], dtype=tkwargs["dtype"],sm=sm)
        
        if self.algorithm == "saasbo":
            self.num_trials = algorithm_config.get("num_trials")
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
        
        if self.algorithm in ["brute_force", "bruteforce"]:

            return BruteForceRunner(self.experiment_id, self.replication, dim, batch_size=1, bounds = self.use_case_runner.bounds, num_init=1,device=tkwargs["device"], dtype=tkwargs["dtype"])
    
    def get_use_case_runner(self, use_case_config : dict):
        if use_case_config.get("use_case").lower() == "mrp":
            return MRPRunner(use_case_config.get("bom_id"), use_case_config.get("num_sim_runs"), use_case_config.get("stochastic_method"))
     

    def log_x_and_y(self,x,y):

        ffolder = "data/" + "experiment_" + str(self.experiment_id)
        fpath = ffolder +"/" + "experiment_" + str(self.experiment_id) +"_"+str(self.replication) + "_brute_force_log" + ".csv"
        if not self.first_log_done:

            self.first_log_done = True
            self.bf_batch = 1
            data = list()

            xx = [self.use_case_runner.transform_x(_x) for _x in x[:99]]
            for i, _x in enumerate(xx):
    
                date = {}
                for xl in _x:
                    date[xl["id"] + "_" + xl["name"]] = xl["value"]
          
                date["cost_mean"]= y[i][0][0]
                date["cost_sem"]= y[i][0][1]
                date["sl_mean"]= y[i][1][0]
                date["sl_sem"]= y[i][1][1]
                data.append(date)
            header = list(data[0].keys())
       
            if not os.path.exists(ffolder):
                Path(ffolder).mkdir(parents=True, exist_ok=True)
            if os.path.isfile(fpath):
                os.remove(fpath)
            with open(fpath, "x",newline='',encoding="utf-8") as f:
                writer = csv.DictWriter(f,header)
                writer.writeheader() 
                writer.writerows(data)
            
        else:
            self.bf_batch +=1
            data = list()
            xx = [self.use_case_runner.transform_x(_x) for _x in x[(max(((self.bf_batch-1)*100-1),0)):self.bf_batch*100-1]]
            for i, _x in enumerate(xx):
                date = {}
                for xl in _x:
                    date[xl["id"] + "_" + xl["name"]] = xl["value"]
                date["cost_mean"]= y[i][0][0]
                date["cost_sem"]= y[i][0][1]
                date["sl_mean"]= y[i][1][0]
                date["sl_sem"]= y[i][1][1]
                data.append(date)
                header = list(data[0].keys())
            with open(fpath, "a",newline='',encoding="utf-8") as f:
                writer = csv.DictWriter(f,fieldnames=header)
                writer.writerows(data)

        y_cost = [y[0][0] for y in y[(max(((self.bf_batch-1)*100-1),0)):self.bf_batch*100-1]]

        best_in_trial = min(y_cost).item() 

        if self.algorithm_runner.Y_current_best == None:
            self.algorithm_runner.Y_current_best = best_in_trial
            logger.info(f"New best Y found: {self.algorithm_runner.Y_current_best}")
        else:
            # is_better = self.Y_current_best < best_in_trial if self.minimize else self.Y_current_best > best_in_trial
            if best_in_trial < self.algorithm_runner.Y_current_best if self.minimize else best_in_trial > self.algorithm_runner.Y_current_best :
                self.algorithm_runner.Y_current_best = best_in_trial
                logger.info(f"New best Y found: {self.algorithm_runner.Y_current_best}")

    def save_experiment_json(self):
        fi = self.use_case_runner.format_feature_importance(self.feature_importances)

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
    
    def append_candidate_to_candidates_list(self, xx,yy):

        assert len(xx) == len(yy)
        for i, x in enumerate(xx):
            self.current_candidat +=1
            y = yy[i]
            ts = self.algorithm_runner.get_technical_specs()
            self.candidates.append({
                "id" : self.current_candidat,
                "sm" : ts.get("sm", "na") if self.current_candidat > self.algorithm_runner.num_init else "init",
                "acqf" : ts.get("acqf", "na") if self.current_candidat > self.algorithm_runner.num_init else "init",
                "tr" : self.algorithm_runner.get_tr(),
                "x" : self.use_case_runner.format_x_for_candidate(x),
                "y" : self.use_case_runner.format_y_for_candidate(y),
                "fi" : self.use_case_runner.format_feature_importance(self.algorithm_runner.get_feature_importance()),
                "ei" : "na",
                "y_pred" : "na",
                "acq_value" : self.algorithm_runner.get_acq_value((self.current_candidat - self.algorithm_runner.num_init - 1))
            })
    

    def get_best_candidate(self):
        ys = list()
        for c in self.candidates: 
            ys.append([_y for _y in c.get("y").values()][0])
        best = self.candidates[pd.DataFrame(ys).idxmin()[0]] if self.minimize else self.candidates[pd.DataFrame(ys).idxmax()[0]]
        logger.info(f"Best candidate found:\n {json.dumps(best, indent=2)}")
        return best
        
    def run_optimization_loop(self):
        logger.info(f"Starting optimization run with evaluation budget of >{self.eval_budget}<")
        self.algorithm_runner.eval_budget = self.eval_budget
        _start = time.monotonic()
        self.experiment_start_dts = datetime.now().isoformat()
        _start_trial = time.monotonic()

        x = self.algorithm_runner.suggest_initial()
        logger.info(f"Got >{x.size()[0]}< initial points from algorithm.")
        _end_trial = time.monotonic()
        #self.trial_runtimes_second.extend([(_end_trial- _start_trial) for _ in range(len(x))]) # ASKNICOLAS: soll len(trial_runtimes) = len(eval_runtimes) sein, damit es bei der Analyse nachher einfacher ist?
        self.trial_runtimes_second.append(_end_trial- _start_trial)
        _y = list()

        eval_counter = 0
        for xx in x:
            eval_counter += 1
            _eval_start_seconds = time.monotonic()
            _y.append(self.use_case_runner.eval(xx))
            _eval_end_seconds = time.monotonic()
            self.eval_runtimes_second.append(_eval_end_seconds - _eval_start_seconds)
            if eval_counter % 100 == 0:
                logger.info(f"Number evaluations: {eval_counter}")
                if self.algorithm == "brute_force":
                    self.log_x_and_y(x,_y)

            

        self.append_candidate_to_candidates_list(x,_y)
        y, ysem = self.use_case_runner.transform_y_to_tensors_mean_sem(_y)

        self.algorithm_runner.complete(y, yvar = ysem)
        self.eval_budget -= len(x)
        self.current_trial +=1
        retries = 0
        logger.info("Initial Trial completed")
  
        while self.eval_budget > 0:
            _start_trial = time.monotonic()
            if retries == 5:
                logger.info("Max Retries reached. Going to Exit Optimization")
                self.algorithm_runner.terminate_experiment()
                self.save_experiment_json()
                sys.exit()
            try:
                x = self.algorithm_runner.suggest()
            except BaseException as e:
                retries += 1
                logger.info(f"Error at Suggest: {e}. Retry {retries} of 5")
                continue
            assert len(x) > 0
            _y = list()

            for xx in x:
                _eval_start_seconds = time.monotonic()
                _y.append(self.use_case_runner.eval(xx))
                _eval_end_seconds = time.monotonic()
                self.eval_runtimes_second.append(_eval_end_seconds - _eval_start_seconds)
                
                if eval_counter % 100 == 0:
                    logger.info(f"Number evaluations: {eval_counter}")
                eval_counter += 1

            self.append_candidate_to_candidates_list(x,_y)
            y, ysem = self.use_case_runner.transform_y_to_tensors_mean_sem(_y)
            self.algorithm_runner.complete(y, yvar = ysem)
            _end_trial = time.monotonic()
            self.trial_runtimes_second.append((_end_trial- _start_trial))
            self.eval_budget -= len(x)
            self.current_trial +=1
            logger.info(f"Trial {self.current_trial} with {len(x)} Arms completed")
        _end = time.monotonic()
        self.experiment_end_dts = datetime.now().isoformat()
        self.total_duration_seconds =  _end -_start
        
        self.best_candidat = self.get_best_candidate()
        self.feature_importances = self.algorithm_runner.get_feature_importance(all=True)
        
        self.algorithm_runner.terminate_experiment()
        self.save_experiment_json()

