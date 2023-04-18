# Main File for Cluster Experiments

# Logging and warnings

import warnings
import logging

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s: %(levelname)s: %(name)s: %(message)s'
    )

# logger = logging.getLogger("algorithm_runner")

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

import time
import torch
from .experiment_runner import ExperimentRunner
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

from icecream import ic


class ExperimentRunnerAlgorithmDriven(ExperimentRunner):
    def __init__(self, experiment, replication, tkwargs):
        super().__init__(experiment, replication, tkwargs)

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
            self.logger.info(f"New best Y found: {self.algorithm_runner.Y_current_best}")
        else:
            # is_better = self.Y_current_best < best_in_trial if self.minimize else self.Y_current_best > best_in_trial
            if best_in_trial < self.algorithm_runner.Y_current_best if self.minimize else best_in_trial > self.algorithm_runner.Y_current_best :
                self.algorithm_runner.Y_current_best = best_in_trial
                self.logger.info(f"New best Y found: {self.algorithm_runner.Y_current_best}")


    
    def append_candidate_to_candidates_list(self, xx,yy):

        assert len(xx) == len(yy)
        for i, x in enumerate(xx):
            self.current_candidate +=1
            y = yy[i]
            ts = self.algorithm_runner.get_technical_specs()
            if self.algorithm.lower() in ["cmaes", "cma-es"]:
                sm = 'cmaes'
                acqf = 'cmaes'
            elif self.algorithm.lower() in ["turbo"]:
                sm = self.algorithm_runner.current_sm
                acqf = self.algorithm_runner.current_acqf
            else:
                sm = ts.get("sm", "na") if self.current_candidate > self.algorithm_runner.num_init else "sobol"
                acqf = ts.get("acqf", "na") if self.current_candidate > self.algorithm_runner.num_init else "sobol"
            self.candidates.append({
                "id" : self.current_candidate,
                "sm" : sm,
                "acqf" : acqf,
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
        self.logger.info(f"Best candidate found:\n {json.dumps(best, indent=2)}")
        return best
        
    def run_optimization_loop(self):
        self.logger.info(f"Starting optimization run with evaluation budget of >{self.eval_budget}<")
        self.algorithm_runner.eval_budget = self.eval_budget
        _start = time.monotonic()
        self.experiment_start_dts = datetime.now().isoformat()
        _start_trial = time.monotonic()

        x = self.algorithm_runner.suggest_initial()
        self.logger.info(f"Got >{x.size()[0]}< initial points from algorithm.")
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
            if eval_counter % 10 == 0:
                self.logger.info(f"Number evaluations: {eval_counter}")
                self.database.update_replication_progress(self.experiment_id, self.replication, eval_counter, self.eval_budget)
                if self.algorithm == "brute_force":
                    self.log_x_and_y(x,_y)

            

        self.append_candidate_to_candidates_list(x,_y)
        y, ysem = self.use_case_runner.transform_y_to_tensors_mean_sem(_y)

        self.algorithm_runner.complete(y, yvar = ysem)
        self.eval_budget -= len(x)
        self.current_trial +=1
        retries = 0
        self.logger.info("Initial Trial completed")
        del x
        del _y
        del y
        del ysem
        while self.eval_budget > 0:
            _start_trial = time.monotonic()
            if retries == 5:
                self.logger.info("Max Retries reached. Going to Exit Optimization")
                self.algorithm_runner.terminate_experiment()
                self.save_experiment_json()
                sys.exit()
            try:
                x = self.algorithm_runner.suggest()
            except BaseException as e:
                retries += 1
                self.logger.info(f"Error at Suggest: {e}. Retry {retries} of 5")
                continue
            assert len(x) > 0
            _y = list()

            for xx in x:
                _eval_start_seconds = time.monotonic()
                _y.append(self.use_case_runner.eval(xx))
                _eval_end_seconds = time.monotonic()
                self.eval_runtimes_second.append(_eval_end_seconds - _eval_start_seconds)
                
                if eval_counter % 10 == 0:
                    self.logger.info(f"Number evaluations: {eval_counter}")
                    self.database.update_replication_progress(self.experiment_id, self.replication, eval_counter, self.eval_budget)
                eval_counter += 1

            self.append_candidate_to_candidates_list(x,_y)
            y, ysem = self.use_case_runner.transform_y_to_tensors_mean_sem(_y)
            self.algorithm_runner.complete(y, yvar = ysem)
            _end_trial = time.monotonic()
            self.trial_runtimes_second.append((_end_trial- _start_trial))
            self.eval_budget -= len(x)
            self.current_trial +=1
            self.logger.info(f"Trial {self.current_trial} with {len(x)} Arms completed")
            del x
            del y
            del ysem
            del _y
            torch.cuda.empty_cache()
        _end = time.monotonic()
        self.experiment_end_dts = datetime.now().isoformat()
        self.total_duration_seconds =  _end -_start
        self.database.update_replication_progress(self.experiment_id, self.replication, eval_counter, self.eval_budget)
        self.best_candidate = self.get_best_candidate()
        self.feature_importances = self.algorithm_runner.get_feature_importance(all=True)
        
        self.algorithm_runner.terminate_experiment()
        results = self.save_experiment_json()
        return results

