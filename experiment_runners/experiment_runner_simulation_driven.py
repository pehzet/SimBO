# Main File for Cluster Experiments

# Logging and warnings
from genericpath import isfile
import warnings
import logging
from numpy import NaN
from flask import request
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
from use_cases.pfp.pfp_runner import PfpRunner

from utils.gsheet_utils import get_configs_from_gsheet
import time
import torch
from flask import Flask, Request, Response, jsonify
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

from icecream import ic

class EndpointAction(object):
    def __init__(self, function):
        self.function = function
        self.first = False
        

    def __call__(self, *args):

        data = self.function(*args)
        if not isinstance(data, dict):
            try:
                data.to_dict()
            except:
                data = {}
        if self.function.__name__ == "suggest":
            sa = []
            for a in data.get('arm_names_in_call'):
                sa.append({"parameters" : {}, "trial_name" : data.get("current_trial"), "arm_name" : str(a), "strategy" : "gpei" })
            return {"suggested_arms" : sa, "status" : "OK", "msg" : f"{self.function.__name__} completed"}, 200
        # https://stackoverflow.com/questions/45412228/sending-json-and-status-code-with-a-flask-response
        return {"data" : data, "status" : "OK", "msg" : f"{self.function.__name__} completed"}, 200

class FlaskWrapper(object):
    app = None

    def __init__(self, name):
        self.app = Flask(name) # static_url_path='', static_folder='web'
        self.app.config["DEBUG"] = False

    def run(self):
        self.app.run()

    def add_endpoint(self, endpoint=None, endpoint_name=None, function=None):
        self.app.add_url_rule(endpoint, endpoint_name, EndpointAction(function), methods = ["GET", "POST"], provide_automatic_options=True)




class ExperimentRunnerSimulationDriven:
    def __init__(self, experiment_id, replication):
        self.experiment_id = experiment_id
        self.replication = replication
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
        self.config = self.get_experiment_config()
        self.use_case_runner = self.get_use_case_runner(self.config.get("use_case_config"))
        self.algorithm_runner = self.get_algorithm_runner(self.config.get("algorithm_config"), len(self.use_case_runner.param_meta), self.use_case_runner.constraints)
        self.algorithm_runner.minimize = self.minimize
        self.is_ddo = True
        self.init_flask()


    def get_experiment_config(self):
        fpath = "configs/config" + str(self.experiment_id) +".json"
        with open(fpath, 'r') as file:
            config = json.load(file)
        
        logger.info(f"Configuration for experiment >{self.experiment_id}< successfully loaded.")
        return config

    def get_algorithm_runner(self, algorithm_config:dict, dim : int, constraints: list):
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
            return GPEIRunner(self.experiment_id, self.replication, dim,batch_size, constraints, num_init, device=tkwargs["device"], dtype=tkwargs["dtype"],sm=sm)
        
        if self.algorithm == "saasbo":
            self.num_trials = algorithm_config.get("num_trials")
            warmup_steps = algorithm_config.get("warmup_steps", 512)
            num_samples = algorithm_config.get("num_samples", 256)
            thinning = algorithm_config.get("thinning", 16)
            batch_size = algorithm_config.get("batch_size")
            num_init = algorithm_config.get("n_init", algorithm_config.get("num_init"))
            return SaasboRunner(self.experiment_id, self.replication, dim, num_init=num_init, batch_size=batch_size, constraints=constraints, warmup_steps=warmup_steps,num_samples=num_samples, thinning=thinning, device=tkwargs["device"], dtype=tkwargs["dtype"])
        
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
        if use_case_config.get("use_case").lower() == "pfp":
            return PfpRunner()

     

   

    
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
    
    def init(self):
        logger.info(f"Starting optimization run with evaluation budget of >{self.eval_budget}<")
        self.algorithm_runner.eval_budget = self.eval_budget
        self._start = time.monotonic()
        self.experiment_start_dts = datetime.now().isoformat()
        # self._start_trial = time.monotonic()
            
        return {}

    def suggest(self):
 
        self._start_trial = time.monotonic()
        if self.current_trial == 0:
            self.x = self.algorithm_runner.suggest_initial()
        else:
            self.x = self.algorithm_runner.suggest()

        assert len(self.x) > 0
        logger.info(f"Trial {self.current_trial} with {len(self.x)} Arms generated")
        
        self.current_trial +=1
        arm_names_in_call = []
        x_t = [self.use_case_runner.transform_x(_x) for _x in self.x]
        
        for xx in x_t:

            if self.is_ddo:
                self.use_case_runner.write_x_to_xlsx(xx, self.current_arm+1)
                arm_names_in_call.append(self.current_arm+1)
                self.current_arm +=1
        return {"status" : "OK", "parameters" : [], "current_trial" : self.current_trial, "arm_names_in_call" : arm_names_in_call}
        
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


    def complete(self):
        req:dict = request.get_json()
        response = []
        for a in req.get("arms"):
            response.append(a.get('responses'))

        y = self.use_case_runner.format_simulation_response(response)
        self.append_candidate_to_candidates_list(self.x,y)
        # y, ysem = self.use_case_runner.transform_y_to_tensors_mean_sem(_y)
        self.algorithm_runner.complete(y)
        _end_trial = time.monotonic()
        self.trial_runtimes_second.append((_end_trial- self._start_trial))
        logger.info(f"Trial {self.current_trial} with {len(self.x)} Arms completed")
        return {"status" : "OK"}
    def terminate(self):
        _end = time.monotonic()
        self.experiment_end_dts = datetime.now().isoformat()
        self.total_duration_seconds =  _end -self._start
        
        self.best_candidat = self.get_best_candidate()
        self.feature_importances = self.algorithm_runner.get_feature_importance(all=True)
        
        self.algorithm_runner.terminate_experiment()
        self.save_experiment_json()
        return {"status" : "OK"}

    def init_flask(self,port=5000):
        self.app = FlaskWrapper("experiment")
        self.app.add_endpoint("/initialize","init", self.init)
        self.app.add_endpoint("/suggest","suggest", self.suggest)
        self.app.add_endpoint("/complete","complete", self.complete)
        self.app.add_endpoint("/terminate","terminate", self.terminate)
        # self.app.run(port=port)
        self.app.run()
    
    

        
    




