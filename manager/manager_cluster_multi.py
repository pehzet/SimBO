

# import multiprocessing_on_dill as mp
# import multiprocessing as mp
import torch.multiprocessing as mp
try:
   mp.set_start_method('spawn', force=True)
except RuntimeError:
   pass
import os
from pynvml import *

import sys
sys.path.append('../')
from experiment_runners.experiment_runner_simulation_driven import ExperimentRunnerSimulationDriven
from experiment_runners.experiment_runner_algorithm_driven import ExperimentRunnerAlgorithmDriven
from database import Database
import torch
from queue import Queue
import warnings
import logging
import json
import time
import copy
sys.path.append('../')


logging.basicConfig(
    level=logging.INFO,

    format='%(asctime)s: %(levelname)s: %(name)s: %(message)s'
)
logger = logging.getLogger("manager")



# Surpress PyTorch warning
warnings.filterwarnings(
    "ignore", message="To copy construct from a tensor, it is")
warnings.filterwarnings("ignore", message="Could not import matplotlib.pyplot")
warnings.filterwarnings(
    "ignore", message="torch.triangular_solve is deprecated")


all_results = []
def send_experiment_to_runner(experiment, replication, tkwargs):

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
    try:
        runner_type = experiment.get("runner_type")
        if runner_type == "simulation":
            ersd = ExperimentRunnerSimulationDriven(experiment, replication, tkwargs)
            results = ersd.init_flask()

        elif runner_type == "algorithm":
            erad = ExperimentRunnerAlgorithmDriven(experiment, replication, tkwargs)
            results = erad.run_optimization_loop()
            # erad.simulate_best_candidat_of_experiment_replication(exp_id, replication, experiment_config=experiment) 

        else:
            raise ValueError(f"Runner Type of experiment {exp_name} (ID: {exp_id}) not identified")

        # return results
    except Exception as e:
        logger.error(f"Error in running experiment {exp_name} (ID: {exp_id}): {e}")
        raise e


class ExperimentManager:
    def __init__(self, manager_id, checking_interval=60):
        self.manager_id = manager_id
        self.checking_interval = checking_interval
        self.experiments_queue = Queue()
        self.experiments_running = Queue()
        self.experiments_done = Queue()
        self.experiments_failed = Queue()
        self.experiments_aborted = Queue()
        self.processes_running = []
        self.main_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))  # r"C:\code\SimBO"
        self.database = Database(self.main_dir)
        self.last_check = None
        self.should_listen = True
        self.processes = []
        self.dtype = torch.double
        # self.number_of_gpus = torch.cuda.device_count()
        self.logger = logging.getLogger("manager")
        fh = logging.FileHandler("manager.log")
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)
        # for debug:
        # self.number_of_gpus = 7
        self.gpu_free = True

        self.date_format = "%Y-%m-%d %H:%M:%S"
        self.logger.info("Manager initialized.")

    def run_experimentation_process(self, experiment: dict, tkwargs: dict):
        exp_name = experiment.get("experiment_name", experiment.get("experiment_id"))
        exp_id = experiment.get("experiment_id")
        self.logger.info("Running experiment: " + str(exp_name))
        self.logger.info("Execution time is: " + str(experiment.get("execution_datetime")))
        replication = experiment.get("current_replication", 1)
        self.experiments_running.put(experiment)
        try:
            p = mp.Process(target=send_experiment_to_runner, args=(experiment, replication, tkwargs,))
            p.start()

            process_dict = {
                "experiment_id": exp_id,
                "experiment_name": exp_name,
                "current_replication": int(replication),
                "replications": int(experiment.get("replications", 1)),
                "process": p,
                "experiment" : copy.deepcopy(experiment),


            }
            self.database.set_experiment_status(exp_id, "running")
            self.processes_running.append(process_dict)
            self.logger.info(f"Process started within manager {self.manager_id}: Replication {replication} of experiment {exp_name} (ID: {exp_id})")
            self.gpu_free = False
            # Processes will be closed later at "check_processes" function
      
        except Exception as e:
            self.logger.error(f"Error while running experiment {exp_name} (ID: {exp_id}) ")
            self.logger.error(e)
            self.close_experiment(experiment, failed=True)
          

    def close_experiment(self, experiment, failed=False):
        exp_name = experiment.get("experiment_name", experiment.get("experiment_id"))
        exp_id = experiment.get("experiment_id")
        if failed:
            self.experiments_failed.put(experiment)
            _experiment = self.experiments_running.get(experiment)
            self.logger.error(f"Experiment failed: {exp_name} (ID: {exp_id}) ")
            self.database.set_experiment_status(exp_id, "failed")
            self.save_experiment_as_json(experiment)
        else:
            self.experiments_done.put(experiment)
            _experiment = self.experiments_running.get(experiment)
            self.logger.info(f"Experiment finished: {exp_name} (ID: {exp_id}) ")
            self.database.set_experiment_status(exp_id, "done")

    def save_experiment_as_json(self, experiment):

        path = os.path.join(self.main_dir, 'manager', 'data','experiment_' + str(experiment.get("experiment_id")))
        if not os.path.exists(path):
            os.makedirs(path)
        fpath = os.path.join(
            path, str(experiment.get("experiment_id")) + '.json')
        with open(fpath, 'w') as outfile:
            json.dump(experiment, outfile)

    def break_experiment_listener(self):
        self.should_listen = False

    def identify_runner_type(self, experiment):
        if experiment.get("use_case", "").lower() in ["mrp"]:
            experiment["runner_type"] = "algorithm"
        elif experiment.get("use_case", "").lower() in ["pfp"]:
            experiment["runner_type"] = "simulation"
        else:
            exp_id = experiment.get("experiment_id")
            exp_name = experiment.get("experiment_name")
            self.logger.error(f"Use case of Experiment {exp_name} (ID: {exp_id}) not identified. Please check the experiment creation.")

        return experiment

    def add_experiment_to_queue(self, experiment):
        replications = experiment.get("replications", 1)
        replications_fulfilled = experiment.get("replications_fulfilled", 0)
        if replications_fulfilled == replications:
            self.logger.info("Experiment already fulfilled. Skipping...")
            self.close_experiment(experiment)
            return
        for r in range(int(replications_fulfilled), int(replications)):
            _experiment = copy.deepcopy(experiment)
            _experiment["current_replication"] = r + 1
            _experiment = self.identify_runner_type(_experiment)
            self.experiments_queue.put(_experiment)
            self.logger.info("Experiment added to queue: " + str(_experiment.get("experiment_name", _experiment.get("experiment_id"))) + " Replication: " + str(r + 1))
      

    def check_processes(self):
        if len(self.processes_running) == 0:
            self.logger.info("No processes running. Waiting for experiments to be added to the queue...")
            return
        # Copy to remove elements while iterating
        # processes = copy.deepcopy(self.processes_running)
        processes = self.processes_running
        processes_to_rm = []
        for process in processes:
            p = process.get("process")
            if not p.is_alive():
                exp_id = process.get("experiment_id")
                replication = process.get("current_replication")
                self.database.update_replication_at_firestore(exp_id, replication)
                self.database.write_result_to_firestore(exp_id, replication)
                self.database.write_all_files_to_storage(exp_id)
                if replication >= process.get("replications"):
                    experiment = process.get("experiment")
                    self.close_experiment(experiment)
                
                # self.processes_running.remove(process)
                # self.available_gpus.put(process.get("gpu"))
                self.gpu_free = True
                processes_to_rm.append(process)
                # p.close()
        for rmp in processes_to_rm:
            self.processes_running.remove(rmp)
 


    def check_experiment_queue(self):
        while not self.experiments_queue.empty():
            if not self.gpu_free:
                self.logger.info("GPU busy. Going to wait...")
                return
            try:
                experiment_to_run = self.experiments_queue.get()
                tkwargs = {"device": torch.device(f"cuda" if torch.cuda.is_available() else "cpu"), "dtype": self.dtype}
                # logger.info("Starting experiment with ID: " + str(experiment_to_run.get("experiment_id")) + " and name: " + str(experiment_to_run.get("experiment_name")) + " at " + str(datetime.now()) + "on: " + str(gpu))
                self.logger.info("Execution time is: " + str(experiment_to_run.get("execution_datetime")))
                self.run_experimentation_process(experiment_to_run, tkwargs)
            except Exception as e:
                print(e)
            print("No more experiments to run")

    def start_firestore_listener(self):
        self.logger.info("Starting Listening to Firestore...")
        def check_changes(col_snapshot, changes, read_time):
            for change in changes:
                if change.type.name == 'ADDED':
                    experiment = change.document.to_dict()
                    experiment["experiment_id"] = change.document.id
                    self.add_experiment_to_queue(experiment)
        if self.manager_id == -1:
            col_query = self.database.db.collection(u'experiments').where(u'status',  u'in', [u'open',u'running']).order_by("created_at")
        else:
            col_query = self.database.db.collection(u'experiments').where(u'status', u'in', [u'open',u'running']).where(u'manager_id', u'==', self.manager_id).order_by("created_at")
        # TODO: make handler for failed experiments
        query_watch = col_query.on_snapshot(check_changes)
        self.logger.info("Checking initially for new experiments to run...")
        self.check_experiment_queue()
        time.sleep(5)
        while self.should_listen:
            self.logger.info("Checking for finished experiment replications...")
            self.check_processes()
            self.logger.info("Checking for new experiments to run...")
            self.check_experiment_queue()
   
            time.sleep(self.checking_interval)
        # query_watch.unsubscribe()
    

def log_gpu_usage():
    if torch.cuda.is_available():
        logger = logging.getLogger("gpu_logger")
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler("gpu_logger.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s: %(name)s: %(message)s'))
        logger.addHandler(fh)
        import subprocess
        import re
        command = 'nvidia-smi'
        while True:
            p = subprocess.check_output(command)
            memory_usage_values = re.findall(r"\|\s+\d+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\d+)MiB\s+/\s+(\d+)MiB", str(p.decode("utf-8")))

            # Print memory usage values
            for i, values in enumerate(memory_usage_values):
                used, total = values
                perc = (int(used) / int(total))*100
                logger.info(f"GPU {i}: Used {used} MiB, Total {total} MiB ({perc:.2%} %)")
            time.sleep(1)
    else:
        print("No GPU available")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 experiment_manager.py <manager_id> <checking_interval>")
        exit(1)
    manager_id = int(sys.argv[1])
    interval = int(sys.argv[2])
    mp.Process(target=log_gpu_usage).start() 
    ExperimentManager(manager_id, interval).start_firestore_listener()
