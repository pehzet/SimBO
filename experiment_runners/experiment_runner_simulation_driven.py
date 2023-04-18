# Main File for Cluster Experiments

# Logging and warnings

import warnings
import logging

from flask import request

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s: %(levelname)s: %(name)s: %(message)s'
    )

logger = logging.getLogger("simulated_runner")

# Surpress PyTorch warning
warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is") 
warnings.filterwarnings("ignore", message="Could not import matplotlib.pyplot")
warnings.filterwarnings("ignore", message="torch.triangular_solve is deprecated") 


import sys
import json

import pandas as pd

from datetime import datetime



import time
import torch
from flask import Flask, Request, Response, jsonify
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

from icecream import ic
from .experiment_runner import ExperimentRunner
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

    def __init__(self, name) -> None:
        self.app = Flask(name) # static_url_path='', static_folder='web'
        
        self.app.config["DEBUG"] = False

    def run(self):
        self.app.run()

    def add_endpoint(self, endpoint=None, endpoint_name=None, function=None):
        self.app.add_url_rule(endpoint, endpoint_name, EndpointAction(function), methods = ["GET", "POST"], provide_automatic_options=True)

    def shutdown(self):
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()


class ExperimentRunnerSimulationDriven(ExperimentRunner):
    def __init__(self, experiment, replication, tkwargs, database):
        super().__init__(experiment, replication, tkwargs, database)
        # self.init_flask()


    def get_experiment_config(self):
        fpath = "configs/config" + str(self.experiment_id) +".json"
        with open(fpath, 'r') as file:
            config = json.load(file)
        
        logger.info(f"Configuration for experiment >{self.experiment_id}< successfully loaded.")
        return config

   


    
    def append_candidate_to_candidates_list(self, xx,yy):

        assert len(xx) == len(yy)
        for i, x in enumerate(xx):
            self.current_candidate +=1
            y = yy[i]
            ts = self.algorithm_runner.get_technical_specs()
            self.candidates.append({
                "id" : self.current_candidate,
                "sm" : ts.get("sm", "na") if self.current_candidate > self.algorithm_runner.num_init else "init",
                "acqf" : ts.get("acqf", "na") if self.current_candidate > self.algorithm_runner.num_init else "init",
                "tr" : self.algorithm_runner.get_tr(),
                "x" : self.use_case_runner.format_x_for_candidate(x),
                "y" : self.use_case_runner.format_y_for_candidate(y),
                "fi" : self.use_case_runner.format_feature_importance(self.algorithm_runner.get_feature_importance()),
                "ei" : "na",
                "y_pred" : "na",
                "acq_value" : self.algorithm_runner.get_acq_value((self.current_candidate - self.algorithm_runner.num_init - 1))
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
        try:
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
        except Exception as e:
            print(e)
            logger.error(e)
            self.terminate()
            print("going to exit")
            sys.exit()
  
        
    # def save_experiment_json(self):
    #     fi = self.use_case_runner.format_feature_importance(self.feature_importances)

    #     obj = {
    #         "experiment_id": self.experiment_id,
    #         "replication" : self.replication,
    #         "algorithm" : self.algorithm,
    #         "bom_id" :  -1, #self.bom_id,
    #         "num_trials" : self.current_trial,
    #         "num_candidates" : len(self.candidates),
    #         "total_duration_seconds": self.total_duration_seconds,
    #         "experiment_start" : self.experiment_start_dts,
    #         "experiment_end" : self.experiment_end_dts,
    #         "trial_runtimes" : self.trial_runtimes_second if self.algorithm != "brute_force" else "na",
    #         "eval_runtimes" : self.eval_runtimes_second if self.algorithm != "brute_force" else "na",
    #         "best_candidate" : self.best_candidat,
    #         "candidates": self.candidates if self.algorithm != "brute_force" else "na",
    #         "final_feature_importances" : fi[-1] if fi != "na" else "na",
    #         "feature_importances" : fi if self.algorithm != "brute_force" else "na"
    #     }
    #     ffolder = "data/" + "experiment_" + str(self.experiment_id)
    #     fpath = ffolder +"/" + "experiment_" + str(self.experiment_id) +"_"+str(self.replication) + ".json"

    #     if not os.path.exists(ffolder):
    #         Path(ffolder).mkdir(parents=True, exist_ok=True)
    #     with open(fpath, 'w+') as fo:
    #         json.dump(obj, fo)
    #     logger.info(f"Experiment data saved to >{fpath}<")
    #     return obj

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
        self.results = self.save_experiment_json()
        self.app.shutdown()
        return {"status" : "OK"}

    def init_flask(self,port=5000):
        self.app = FlaskWrapper("experiment")
        self.app.add_endpoint("/initialize","init", self.init)
        self.app.add_endpoint("/suggest","suggest", self.suggest)
        self.app.add_endpoint("/complete","complete", self.complete)
        self.app.add_endpoint("/terminate","terminate", self.terminate)
        # self.app.run(port=port)
        self.app.run()
        
        return self.results

    
    

        
    




