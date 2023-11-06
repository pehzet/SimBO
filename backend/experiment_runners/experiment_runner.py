# Logging and warnings

import warnings
import logging
from numpy import NaN

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s: %(levelname)s: %(name)s: %(message)s'
    )

# logger = logging.getLogger("runner")

# Surpress PyTorch warning
warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is") 
warnings.filterwarnings("ignore", message="Could not import matplotlib.pyplot")
warnings.filterwarnings("ignore", message="torch.triangular_solve is deprecated") 
import sys



import json

import os
import re
from pathlib import Path
import importlib
import pickle
from backend.algorithms.single.bo.turbo.turbo_botorch import TurboRunner
from backend.algorithms.single.bo.saasbo.saasbo_botorch import SaasboRunner
from backend.algorithms.single.bo.gpei.gpei_botorch import GPEIRunner
from backend.algorithms.single.ea.cmaes.cmaes import CMAESRunner
from backend.algorithms.sobol.sobol_botorch import SobolRunner
from backend.algorithms.brute_force.brute_force import BruteForceRunner
from backend.algorithms.multi.bo.morbo.morbo import MorboRunner
from backend.algorithms.multi.bo.qnehvi.qnehvi import QNEHVIRunner
from backend.algorithms.multi.ea.nsga2.nsga2 import NSGA2Runner
from backend.algorithms.multi.bo.saasmo.saasmo import SAASMORunner
from backend.algorithms.multi.ea.moead.moead import MOEADRunner
from backend.use_cases.mrp.mrp_runner import MRPRunner
from backend.use_cases.pfp.pfp_runner import PfpRunner
from backend.use_cases.dummy.dummy_runner import DummyRunner
from backend.use_cases.mrp.mrp_mo_runner import MRPMORunner
from backend.utils.plot import create_convergence_plot
from backend.databases.sql import SQLManager
import torch

tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

from icecream import ic
class ExperimentRunner():

    def __init__(self, experiment, replication, tkwargs, database) -> None:
        self.experiment_id = experiment.get("experiment_id")
        self.replication = replication
        self.tkwargs = tkwargs
        self.database = database
        self.sql_database = SQLManager()
        self.logger =logging.getLogger("runner")
        try:
            self.logger.addHandler(self.tkwargs["logging_fh"])
        except:
            pass
        self.algorithm = None
        self.minimize = True # get from config later

        self.total_duration = None
        self.experiment_start_dts = None
        self.experiment_end_dts = None
        self.trial_runtimes = list()
        self.eval_runtimes = list()
      
        self.feature_importances = list()
        self.candidates = list()
        self.best_candidate = None

        self.first_log_done = False
        
        self.current_candidate = 0
        self.current_trial = 0
        self.current_arm = 0
        self.current_x = None
        self.current_y = None
        self.config = experiment
    
        self.use_case_runner = self.get_use_case_runner()
        self.algorithm_runner = self.get_algorithm_runner()
        self.algorithm_runner.minimize = self.minimize
    
    def check_botorch_version(self, required_version="default"):
        import subprocess
        if required_version == "default":
            required_version = "0.8.4"

        try:
            import botorch
            if botorch.__version__ != required_version:
                sys.modules.pop('botorch')
                print(f"Installing botorch=={required_version}")
                subprocess.check_call(["pip", "install", f"botorch=={required_version}"])
                importlib.invalidate_caches() 
                import botorch
                importlib.reload(botorch)
        except ImportError:
            subprocess.check_call(["pip", "install", f"botorch=={required_version}"])
        self.logger.info(f"Using botorch version {botorch.__version__}")

    def get_experiment_config(self):
        fpath = "configs/config" + str(self.experiment_id) +".json"
        with open(fpath, 'r') as file:
            config = json.load(file)
        
        self.logger.info(f"Configuration for experiment >{self.experiment_id}< successfully loaded.")
        return config

    def get_algorithm_runner(self):
        algorithm_config = self.config.get("algorithm_config")
        dim = len(self.use_case_runner.param_meta) 
    
        constraints = self.format_outcome_constraints(self.use_case_runner.constraints) 
        objectives = self.use_case_runner.objectives
        self.is_moo = True if len(objectives) > 1 else False
        
        self.algorithm = algorithm_config.get("strategy", algorithm_config.get("algorithm")).lower()
        self.eval_budget = int(self.config.get("budget", self.config.get("evaluation_budget")))
        self.inital_budget = self.eval_budget

        init_arms = self.config.get("init_arms", self.config.get("num_init", self.config.get("n_init", 1)))
        # num_init = int(self.config.get("init_arms", 1))
        batch_size = int(self.config.get("batch_size"))
        
        # SINGLE OBJECTIVE
        if self.algorithm == "turbo":
            # self.check_botorch_version()
            sm = algorithm_config.get("sm") if algorithm_config.get("sm") not in ["None", None, "default", "Default", "nan", NaN] else "fngp"
            return TurboRunner(self.experiment_id, self.replication, dim,batch_size,constraints, num_init=init_arms, device=self.tkwargs["device"], dtype=self.tkwargs["dtype"],sm=sm)
        
        if self.algorithm == "gpei":
            # self.check_botorch_version()
            sm = algorithm_config.get("sm") if algorithm_config.get("sm") not in ["None", None, "default", "Default","nan", NaN] else "stgp"
            return GPEIRunner(self.experiment_id, self.replication, dim,batch_size, constraints, init_arms, device=self.tkwargs["device"], dtype=self.tkwargs["dtype"],sm=sm)
        
        if self.algorithm == "saasbo":
            # self.check_botorch_version()
            warmup_steps = algorithm_config.get("warmup_steps", 512)
            num_samples = algorithm_config.get("num_samples", 256)
            thinning = algorithm_config.get("thinning", 16)
            return SaasboRunner(self.experiment_id, self.replication, dim, num_init=init_arms, batch_size=batch_size, constraints=constraints, warmup_steps=warmup_steps,num_samples=num_samples, thinning=thinning, device=self.tkwargs["device"], dtype=self.tkwargs["dtype"])
        
        if self.algorithm == "cmaes" or self.algorithm == "cma-es":
            sigma0 = algorithm_config.get("sigma", 0.5)
            # ucr = self.use_case_runner if self.use_case_runner.stochastic_method != 'deterministic' else None
            ucr = None # Problems with tensors using noise handling at cma. Will be fixed later /PZM 2023-04-14
            return CMAESRunner(self.experiment_id, self.replication, dim, batch_size,self.use_case_runner.bounds,sigma0, num_init=init_arms, use_case_runner=ucr, device=self.tkwargs["device"], dtype=self.tkwargs["dtype"])
        
        # MULTI OBJECTIVE
        if self.algorithm in ["morbo", "MORBO"]:
            # botorch_version = "0.7.0"
            # self.check_botorch_version(botorch_version)
            ref_point = self.use_case_runner.get_ref_point()
            return MorboRunner(self.experiment_id, self.replication, dim, batch_size, algorithm_config.get("n_trust_regions", algorithm_config.get("num_trs", 1)), objectives, ref_point, constraints, eval_budget=self.eval_budget, num_init=init_arms, device=self.tkwargs["device"], dtype=self.tkwargs["dtype"])
        if self.algorithm in ["saasmo", "SAASMO"]:
            # self.check_botorch_version("0.9.2")
            ref_point = self.use_case_runner.get_ref_point()
            return SAASMORunner(self.experiment_id, self.replication, dim, batch_size, objectives, ref_point, constraints, algorithm_config.get("warmup_steps", 512), algorithm_config.get("num_samples", 256), algorithm_config.get("thinning", 16),  num_init=init_arms, device=self.tkwargs["device"], dtype=self.tkwargs["dtype"])
        if self.algorithm in ["qnehvi", "qNEHVI"]:
            # self.check_botorch_version()
            ref_point = self.use_case_runner.get_ref_point()
            # ref_point = torch.tensor(ref_point) if ref_point != None and ref_point != "" else None
            return QNEHVIRunner(self.experiment_id, self.replication, dim, batch_size, objectives, ref_point, constraints, num_init=init_arms, device=self.tkwargs["device"], dtype=self.tkwargs["dtype"], sm=algorithm_config.get("sm", "hsgp"))
        if self.algorithm in ["nsga2", "NSGA2"]:
            ref_point = self.use_case_runner.get_ref_point()
            param_meta = self.use_case_runner.get_param_meta()
            param_names = [p["name"] for p in param_meta]
            return NSGA2Runner(self.experiment_id, self.replication, dim, batch_size, objectives, constraints, param_names = param_names,eval_budget=self.eval_budget, num_init=init_arms,ref_point=ref_point, device=self.tkwargs["device"], dtype=self.tkwargs["dtype"])
        if self.algorithm in ["moead", "MOEAD"]:
            param_meta = self.use_case_runner.get_param_meta()
            param_names = [p["name"] for p in param_meta]
            return MOEADRunner(self.experiment_id, self.replication, dim, batch_size, objectives, constraints, param_names = param_names,eval_budget=self.eval_budget, num_init=init_arms, device=self.tkwargs["device"], dtype=self.tkwargs["dtype"])
        
        if self.algorithm == "sobol":
            # self.check_botorch_version()
            return SobolRunner(self.experiment_id, self.replication,dim,batch_size=1, num_init=1, device=self.tkwargs["device"], dtype=self.tkwargs["dtype"])
        
        if self.algorithm in ["brute_force", "bruteforce"]:

            return BruteForceRunner(self.experiment_id, self.replication, dim, batch_size=1, bounds = self.use_case_runner.bounds, num_init=1,device=self.tkwargs["device"], dtype=self.tkwargs["dtype"])
    
    def get_use_case_runner(self):
        use_case_config = self.config.get("use_case_config")

        if use_case_config.get("use_case").lower() == "mrp":
            # return MRPRunner(use_case_config.get("bom_id"), use_case_config.get("num_sim_runs"), use_case_config.get("stochastic_method"))
            return MRPRunner(use_case_config.get("bom_id"), use_case_config.get("num_sim_runs"), use_case_config.get("stochastic_method"))
        if use_case_config.get("use_case").lower() == "pfp":
            return PfpRunner()
        if use_case_config.get("use_case").lower() == "dummy":
            return DummyRunner()
        if use_case_config.get("use_case").lower() in ["mrp_mo","mrpmo"]:
            return MRPMORunner(use_case_config.get("bom_id"), use_case_config.get("num_sim_runs"), use_case_config.get("stochastic_method"))
    
    def format_parameter_constraints(self, constraints):
       
        constraints_formatted = {"ieq": [], "eq": []}
        if len(constraints) == 0:
            return constraints_formatted

        for input_string in constraints:
            # Find all matches of the pattern

            #params
            pattern = r"([^\s+<=>-]+)"
            params = re.findall(pattern, input_string)
            for p in params:
                try: 
                    float(p)
                    params.remove(p)
                except:
                    pass
            params_idx = [self.use_case_runner.param_meta.index(p) for p in params]


            pattern = r"([^\s+]+)\s*([<==]+)\s*([^\s]+)"
            matches = re.findall(pattern, input_string)
            if matches:
            # Loop through each match to extract parameters, operator, and value
                for match in matches:

                    constraint_type = "ieq" if match[1] == "<=" else "eq"
                    value = torch.tensor(float(match[2]))
                params_idx = torch.tensor(params_idx)
                constraints_formatted[constraint_type].append(torch.cat([params_idx, value.unsqueeze(0)]))
        return constraints_formatted
    
    def format_outcome_constraints(self, constraints):
        constraints_formatted = None
        if constraints == None or len(constraints) == 0:
            return constraints_formatted

        for input_string in constraints:
            # Find all matches of the pattern

            #params
            pattern = r"([^\s+<=>-]+)"
            objectives = re.findall(pattern, input_string)
            for p in objectives:
                try: 
                    float(p)
                    objectives.remove(p)
                except:
                    pass
            obj_idx = [self.use_case_runner.objectives.index(p.replace("'", "")) for p in objectives]
            obj_list_for_botorch_constraint = []
            value_list_for_botorch_constraint = []
            pattern = r"([^\s+]+)\s*([<==]+)\s*([^\s]+)"
            matches = re.findall(pattern, input_string)
            if matches:
            # Loop through each match to extract parameters, operator, and value
                for match in matches:

                    constraint_type = "ieq" if match[1] == "<=" else "eq"
                    value = torch.tensor(float(match[2]))
                obj_idx = torch.tensor(obj_idx)
                _object_list = [0 for i in range(len(self.use_case_runner.objectives))]
                _object_list[obj_idx] = 1
                obj_list_for_botorch_constraint.append(_object_list)
                value_list_for_botorch_constraint.append(value)
        constraints_formatted = (torch.tensor(obj_list_for_botorch_constraint, dtype=torch.double), torch.tensor(value_list_for_botorch_constraint, dtype=torch.double))
        return constraints_formatted

    def log_trial_data(self, ):
        gt = self.algorithm_runner.gen_runtimes[-1] if len(self.algorithm_runner.gen_runtimes) > 0  else 0
        ft = self.algorithm_runner.fit_runtimes[-1] if len(self.algorithm_runner.fit_runtimes) > 0 else 0
        et = self.eval_runtimes[-1] if len(self.eval_runtimes) > 0 else 0
        self.sql_database.insert_runtime(self.experiment_id, self.replication, self.current_trial, gt, ft, et)
        ls = self.algorithm_runner.lengthscales[-1] if len(self.algorithm_runner.lengthscales) > 0 else None
        self.sql_database.insert_lengthscale(self.experiment_id, self.replication, self.current_trial, ls)
        x = self.algorithm_runner.X[-self.algorithm_runner.batch_size:] if len(self.algorithm_runner.X) > 0 else None
        y = self.algorithm_runner.Y[-self.algorithm_runner.batch_size:] if len(self.algorithm_runner.Y) > 0 else None
        self.sql_database.insert_x_and_y(self.experiment_id, self.replication, self.current_trial, x, y)
        # if self.algorithm_runner.is_ea:
        #     hv = self.algorithm_runner.hvs[-1] if len(self.algorithm_runner.hvs) > 0 else None
        #     self.sql_database.insert_hv(self.experiment_id, self.replication, self.current_trial, hv)
        #     return
        if self.algorithm_runner.is_moo:
            px = self.algorithm_runner.pareto_X[-1] if len(self.algorithm_runner.pareto_X) > 0 else None
            py = self.algorithm_runner.pareto_Y[-1] if len(self.algorithm_runner.pareto_Y) > 0 else None
            self.sql_database.insert_pareto(self.experiment_id, self.replication, self.current_trial, px, py)
            hv = self.algorithm_runner.hvs[-1] if len(self.algorithm_runner.hvs) > 0 else None
            self.sql_database.insert_hv(self.experiment_id, self.replication, self.current_trial, hv)
        else:

            # MO Algorithm got no acq values
            av = self.algorithm_runner.acq_values[-1] if len(self.algorithm_runner.acq_values) > 0  else None
            self.sql_database.insert_acq_values(self.experiment_id, self.replication, self.current_trial, av)
    def terminate_experiment(self):
        if self.is_moo:
            # STRUGGLE WITH MORBO; Can't pickle local object 'get_outcome_constraint_transforms; TODO: FIX 
            return
        path = "data/experiment_" + str(self.experiment_id)
        if not Path(path).exists():
            Path(path).mkdir(parents=True, exist_ok=True)
        with open((path + "/" + str(self.experiment_id) + "_" + str(self.replication) + "_" + self.algorithm_runner.get_name() +  ".pkl"), "wb") as fo:
            pickle.dump(self.algorithm_runner, fo)
        self.database.append_best_arm_to_use_case(self.config.get("use_case_config").get("use_case").lower(), self.best_candidate)
        
        torch.cuda.empty_cache()

    def save_experiment_json(self):
        fi = self.use_case_runner.format_feature_importance(self.feature_importances)
        try:
            device = str(self.tkwargs["device"].type)
        except:
            device = "na"
        use_case_informations = self.use_case_runner.get_log_informations()
        obj = {
            "experiment_id": self.experiment_id,
            "replication" : self.replication,
            "algorithm" : self.algorithm,
            # "bom_id" :  self.use_case_runner.bom_id if self.use_case_runner.bom_id != None else "na",
            "num_trials" : self.current_trial,
            "eval_budget" : self.inital_budget,
            "num_candidates" : len(self.candidates),
            "total_duration": self.total_duration,
            "experiment_start" : self.experiment_start_dts,
            "experiment_end" : self.experiment_end_dts,
            "trial_runtimes" : self.trial_runtimes if self.algorithm != "brute_force" else "na",
            "eval_runtimes" : self.eval_runtimes if self.algorithm != "brute_force" else "na",
            "best_candidate" : self.best_candidate,
            "raw_results" : self.use_case_runner.Y_raw,
            # "stochastic_method" : self.use_case_runner.stochastic_method , TODO: Make Use Case kwargs or something else
            # "num_sim_runs" : self.use_case_runner.num_sim_runs
            "device" : device,
            "candidates": self.candidates if self.algorithm != "brute_force" else "na",
            "final_feature_importances" : fi[-1] if fi != "na" else "na",
            "feature_importances" : fi if self.algorithm != "brute_force" else "na",
            "use_case_informations" : use_case_informations if self.algorithm != "brute_force" else "na"
        }
        if self.use_case_runner.__class__.__name__ == "MRPRunner":
            obj["bom_id"] =  self.use_case_runner.bom_id if self.use_case_runner.bom_id != None else "na"
        # ffolder = "data/" + "experiment_" + str(self.experiment_id)
        # fpath = ffolder +"/" + "experiment_" + str(self.experiment_id) +"_"+str(self.replication) + ".json"
        ffolder = os.path.join("data", "experiment_" + str(self.experiment_id))
        fpath = os.path.join(ffolder, "experiment_" + str(self.experiment_id) +"_"+str(self.replication) + ".json")

        if not os.path.exists(ffolder):
            Path(ffolder).mkdir(parents=True, exist_ok=True)
        with open(fpath, 'w+') as fo:
            json.dump(obj, fo)
        try:
            y = self.algorithm_runner.Y.cpu().tolist()
            yerr = self.algorithm_runner.Yvar.cpu().tolist()
        except:
            y = self.algorithm_runner.Y
            yerr = self.algorithm_runner.Yvar
        self.logger.info(f"Experiment data saved to >{fpath}<")

        self.results = obj
        return obj

    def identity_best_in_trial(self):
        if self.algorithm_runner.is_ea:
            return
        new_best_found = False
        if self.algorithm_runner.is_moo:
            if self.algorithm_runner.pareto_Y is not None:
                if len(self.algorithm_runner.pareto_Y_next) > 0:
                    x = self.algorithm_runner.pareto_X_next
                    y = self.algorithm_runner.pareto_Y_next
                    best_in_trial_idx = -999
                    self.logger.info(f"New best Y pareto found: {y}")
            else:
                self.logger.debug("No pareto front found yet")


          
        else:
            if self.minimize:
                best_in_trial = min(self.algorithm_runner.Y_next).item()
                best_in_trial_idx = torch.argmin(self.algorithm_runner.Y_next).item()
            else:
                best_in_trial = max(self.algorithm_runner.Y_next).item() # TODO: Think about general way to handle min and max
                best_in_trial_idx = torch.argmax(self.algorithm_runner.Y_next).item()
            if self.algorithm_runner.Y_current_best == None:
                self.algorithm_runner.Y_current_best = best_in_trial
                new_best_found = True
                self.logger.info(f"New best Y found: {self.algorithm_runner.Y_current_best*-1}")
            else:
                # is_better = self.Y_current_best < best_in_trial if self.minimize else self.Y_current_best > best_in_trial
                if self.algorithm_runner.Y_current_best < best_in_trial if self.minimize else self.algorithm_runner.Y_current_best > best_in_trial:
                    self.algorithm_runner.Y_current_best = best_in_trial
                    new_best_found = True
                    self.logger.info(f"New best Y found: {self.algorithm_runner.Y_current_best*-1 if self.minimize else self.algorithm_runner.Y_current_best}")

        if new_best_found:
            x = self.algorithm_runner.X_next[best_in_trial_idx].tolist()
            y = self.algorithm_runner.Y_next[best_in_trial_idx].tolist()
            self.best_candidate = {"x":x, "y":y, "trial" : self.current_trial, "idx_in_trial" : best_in_trial_idx, "experiment_id" : self.experiment_id, "replication" : self.replication}
            self.database.append_best_arm_to_use_case(self.config.get("use_case_config").get("use_case").lower(), self.best_candidate)