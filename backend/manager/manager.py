
import torch.multiprocessing as mp
try:
   mp.set_start_method('spawn', force=True)
except RuntimeError:
   pass
import os
from icecream import ic
import sys

from backend.experiment_runners.experiment_runner_simulation_driven import ExperimentRunnerSimulationDriven
from backend.experiment_runners.experiment_runner_algorithm_driven import ExperimentRunnerAlgorithmDriven
from backend.databases.firebase import FirebaseManager
from backend.databases.sql import SQLManager
import torch
from queue import Queue
import warnings
import logging
import json
import time
import copy
from collections import deque
import traceback


logging.basicConfig(
    level=logging.INFO,

    format='%(asctime)s: %(levelname)s: %(name)s: %(message)s'
)
logger = logging.getLogger("manager")


error_queue = Queue()
# Surpress PyTorch warning
warnings.filterwarnings(
    "ignore", message="To copy construct from a tensor, it is")
warnings.filterwarnings("ignore", message="Could not import matplotlib.pyplot")
warnings.filterwarnings(
    "ignore", message="torch.triangular_solve is deprecated")

def send_experiment_to_runner(experiment, replication, tkwargs, main_dir=None):
    exp_name = experiment.get("experiment_name", experiment.get("experiment_id"))
    exp_id = experiment.get("experiment_id")
    logger = logging.getLogger(f"experiment_{exp_id}")
    log_file = f"experiment_{exp_id}.log"
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s: %(name)s: %(message)s'))
    tkwargs["logging_fh"] = fh
    logger.addHandler(fh)
    results = None
    main_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) if main_dir is None else main_dir
    try:
        runner_type = experiment.get("runner_type")
        if runner_type == "simulation":
            ersd = ExperimentRunnerSimulationDriven(experiment, replication, tkwargs, FirebaseManager(main_dir))
            results = ersd.init_flask()
        elif runner_type == "algorithm":
            erad = ExperimentRunnerAlgorithmDriven(experiment, replication, tkwargs, FirebaseManager(main_dir))
            results = erad.run_optimization_loop()
        else:
            raise ValueError(f"Runner Type of experiment {exp_name} (ID: {exp_id}) not identified")
        return results
    except Exception as e:
        logger.error(f"Error in running experiment {exp_name} (ID: {exp_id}): {e}")
        traceback.print_exc()
        error_queue.put({
            'experiment': experiment,
            'replication': replication,
            'exp_name': exp_name,
            'exp_id': exp_id,
            'error_msg': str(e)
        })
        # raise e


class ExperimentManager:
    def __init__(self, manager_id, checking_interval=60, main_dir=None, db=None, num_parallel_experiments=-1):
        self.manager_id = manager_id
        self.checking_interval = int(checking_interval)
        self.experiments_queue = Queue()
        self.experiments_running = Queue()
        
        self.processes_running = []
        self.main_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) if main_dir == None else main_dir 
        self.database = db if db is not None else FirebaseManager(self.main_dir)
        self.sql_database = SQLManager()
        self.last_check = None
        self.should_listen = True
        self.processes = []
        self.dtype = torch.double

        self.logger = logging.getLogger("manager")
        fh = logging.FileHandler("manager.log")
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)
        self.num_parallel_experiments = num_parallel_experiments if num_parallel_experiments != -1 else max(torch.cuda.device_count(),1)
        self.gpus_available = [torch.cuda.device(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() and num_parallel_experiments != -1 else [i for i in range(self.num_parallel_experiments)]
        self.last_experiment_ids = deque(maxlen = self.num_parallel_experiments)
   
        self.date_format = "%Y-%m-%d %H:%M:%S"
        self.logger.info("Manager initialized.")
    def gpu_free(self):
        return len(self.gpus_available) > 0

    def handle_errors(self):
        while not error_queue.empty():
            error = error_queue.get()
            self.logger.error(f"Error in experiment {error.get('exp_name')} (ID: {error.get('exp_id')})")
            self.logger.error(error.get('error_msg'))
            self.close_experiment(error.get('experiment'), error.get('replication'), failed=True)
            # self.check_processes()
    def run_experimentation_process(self, experiment: dict):
        
        tkwargs = {"device": torch.device(f"cuda" if torch.cuda.is_available() and not experiment.get("use_cpu", False) else "cpu"), "dtype": self.dtype}
        exp_name = experiment.get("experiment_name", experiment.get("experiment_id"))
        exp_id = experiment.get("experiment_id")
        self.logger.info("Running experiment: " + str(exp_name))
        self.logger.info("Execution time is: " + str(experiment.get("execution_datetime")))
        current_replication = experiment.get("current_replication", 1)
        self.database.update_current_replication(exp_id, current_replication)
        self.experiments_running.put(experiment)
        gpu = self.gpus_available.pop(0)
        try:
            p = mp.Process(target=send_experiment_to_runner, args=(experiment, current_replication, tkwargs), kwargs={"main_dir": self.main_dir})
            p.start()
            process_dict = {
                "experiment_id": exp_id,
                "experiment_name": exp_name,
                "current_replication": int(current_replication),
                "replications": int(experiment.get("replications", 1)),
                "process": p,
                "gpu": gpu,
                "experiment" : copy.deepcopy(experiment),
            }
            self.database.set_experiment_status(exp_id, "running")
            self.processes_running.append(process_dict)
            self.logger.info(f"Process started within manager {self.manager_id}: Replication {current_replication} of experiment {exp_name} (ID: {exp_id})")
            # Processes will be closed later at "check_processes" function
      
        except Exception as e:
            self.logger.error(f"Error while running experiment {exp_name} (ID: {exp_id}) ")
            self.logger.error(e)
            self.close_experiment(experiment, current_replication, failed=True)
            self.check_processes()


    def close_experiment(self, experiment, replication=-1, failed=False, aborted=False):
        '''
        Closes the experiment and saves the results if aborted or failed. Replication must not be passed if successful.
        '''
        exp_name = experiment.get("experiment_name", experiment.get("experiment_id"))
        exp_id = experiment.get("experiment_id")

        if failed:

            _experiment = self.experiments_running.get(experiment)
            self.logger.error(f"Experiment failed: {exp_name} (ID: {exp_id}) ")
            self.database.set_experiment_status(exp_id, "failed")
            self.save_experiment_as_json(experiment, replication)
        elif aborted:

            _experiment = self.experiments_running.get(experiment)
            self.logger.error(f"Experiment aborted: {exp_name} (ID: {exp_id}) ")
            self.database.set_experiment_status(exp_id, "aborted")
            self.save_experiment_as_json(experiment, replication)
        
        else:

            _experiment = self.experiments_running.get(experiment)
            self.logger.info(f"Experiment finished: {exp_name} (ID: {exp_id}) ")
            self.database.set_experiment_status(exp_id, "done")
            # self.save_experiment_as_json(experiment, replication)
        
        # save database after every experiment
        
    def save_experiment_as_json(self, experiment, replication=-1):
        path = os.path.join(self.main_dir, 'manager', 'data','experiment_' + str(experiment.get("experiment_id")))
        if not os.path.exists(path):
            os.makedirs(path)
        fpath = os.path.join(path, str(experiment.get("experiment_id")) +"_"+str(replication) + '.json')
        with open(fpath, 'w') as outfile:
            json.dump(experiment, outfile)

    def break_experiment_listener(self):
        self.should_listen = False

    def identify_runner_type(self, experiment):
        if experiment.get("use_case", "").lower() in ["mrp", "dummy", "mrp_mo","mrpmo"]:
            rt = "algorithm"
        elif experiment.get("use_case", "").lower() in ["pfp"]:
            rt = "simulation"
        else:
            exp_id = experiment.get("experiment_id")
            exp_name = experiment.get("experiment_name")
            self.logger.error(f"Use case of Experiment {exp_name} (ID: {exp_id}) not identified. Please check the experiment creation.")

        return rt


    def check_processes(self):
        if len(self.processes_running) == 0:
            self.logger.info("No processes running. Waiting for experiments to be added to the queue...")
            return
        processes_to_rm = []
        for process in self.processes_running:
            p = process.get("process")
            if not p.is_alive():
                _completed = self.database.check_if_local_files_exist(process.get("experiment_id"), process.get("current_replication"))
                if _completed:
                    try:
                        exp_id = process.get("experiment_id")
                        replication = process.get("current_replication")
                        self.database.write_result_to_firestore(exp_id, replication)
                        self.database.write_all_files_to_storage(exp_id)
                        self.database.update_replication(exp_id, replication)
                        self.last_experiment_ids.append(exp_id)
                        if replication >= process.get("replications"):
                            experiment = process.get("experiment")
                            self.close_experiment(experiment)
                        
                        # self.processes_running.remove(process)
                        self.gpus_available.append(process.get("gpu"))
                        # self.gpu_free = True
                        processes_to_rm.append(process)
                        self.logger.info(f"Process finished within manager {self.manager_id}: Replication {replication} of experiment {process.get('experiment_name')} (ID: {exp_id})")
                    except Exception as e:
                        self.logger.error("Error while closing experiment")
                        self.logger.error(e)
                else:
                    replication = process.get("current_replication")
                    self.gpus_available.append(process.get("gpu"))
                    self.close_experiment(process.get("experiment"),replication, failed=True)
                    processes_to_rm.append(process)
                    self.logger.error(f"Process terminated unexpectedly within manager {self.manager_id}: Replication {replication} of experiment {process.get('experiment_name')} (ID: {process.get('experiment_id')})")
                # p.close()
            else:
                  self.logger.warning(f"Process still running within manager {self.manager_id}: Replication {process.get('current_replication')} of experiment {process.get('experiment_name')} (ID: {process.get('experiment_id')})")
        for rmp in processes_to_rm:
            self.processes_running.remove(rmp)
        return
 
    def check_if_terminating_manager(self, no_experiment_counter, ):
        if no_experiment_counter > 5:
            # self.checking_interval *= 1.5
            # self.logger.info(f"No experiments found. Increasing checking interval to {self.checking_interval} seconds")
            # return 0
            self.logger.info(f"No further experiments. Shutting down manager {self.manager_id}...")
            self.sql_database.send_db_file_to_storage()
            self.break_experiment_listener()
            return 6
        else:
            return no_experiment_counter + 1


    def prepare_experiments(self, experiments):
        exp_in_this_loop = []
        for exp in experiments:
            original_experiment = exp.to_dict()
            for i in range(1, int(original_experiment.get("replications")) - int(original_experiment.get("replications_fulfilled")) + 1):
                experiment = copy.deepcopy(original_experiment)
                if experiment.get("replications_fulfilled", 0) + i != experiment.get("current_replication", 0):
                    experiment["experiment_id"] = exp.id
                    experiment["current_replication"] = experiment.get("replications_fulfilled", 0) + i
                    experiment["runner_type"] = self.identify_runner_type(experiment)
                    self.sql_database.insert_experiment(experiment.get("experiment_id"), experiment.get("experiment_name"), experiment.get("replications"), experiment.get("algorithm"), experiment.get("use_case"), experiment.get("created_at"))
                    exp_in_this_loop.append(experiment)
        return exp_in_this_loop

    def run_prepared_experiments(self, exp_in_this_loop):
        for i in range(len(self.gpus_available)):
            exp = exp_in_this_loop[i]
            if not exp.get("experiment_id") in self.last_experiment_ids:
                for l_e_id in self.last_experiment_ids:
                    if not self.database.check_experiment_status(l_e_id, "done"):
                        self.database.set_experiment_status(l_e_id, "paused")
            self.run_experimentation_process(exp)

    def run(self):
        no_experiment_counter = 0

        while self.should_listen:
            self.logger.info("Checking for finished experiment replications...")
            self.check_processes()
            self.handle_errors()
            
            if self.gpu_free():
                self.logger.info("Checking for new experiments to run...")
                experiments = self.database.check_database_for_experiments(self.manager_id, len(self.gpus_available))
                if len(experiments) == 0:
                    self.logger.info(f"No experiments found. Waiting {self.checking_interval} seconds")
                    no_experiment_counter = self.check_if_terminating_manager(no_experiment_counter)
                else:
                    no_experiment_counter = 0
                    exp_in_this_loop = self.prepare_experiments(experiments)
                    
                    if exp_in_this_loop:
                        self.logger.info(f"Found {len(exp_in_this_loop)} experiment replications to run")
                        self.run_prepared_experiments(exp_in_this_loop)
            
            time.sleep(max(self.checking_interval, 10))

    
 



def log_gpu_usage():
    logger = logging.getLogger("gpu_logger")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("gpu_logger.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s: %(name)s: %(message)s'))
    logger.addHandler(fh)
    logger.info("Starting GPU logger")
    logger.propagate = False
    
    try:
        if torch.cuda.is_available():
            import subprocess
            import re
            command = 'nvidia-smi --query-gpu=name,utilization.gpu,memory.total,memory.used --format=csv,nounits,noheader'
            while True:
                p = subprocess.check_output(command, shell=True)
                lines = p.decode("utf-8").strip().split("\n")
                for line in lines:
                    gpu_name, gpu_util, memory_total, memory_used = line.split(", ")
                    log_message = f"Usage of {gpu_name} {gpu_util} % ({memory_used} MB / {memory_total} MB)"
                    logger.info(log_message)
                time.sleep(2)
        else:
            logger.info("No GPU available")
    except Exception as e:
        logger.error("Error while logging GPU usage")
        logger.error(e)
        exit(1)

