

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
from icecream import ic
from datetime import datetime
import time
import copy
sys.path.append('../')


logging.basicConfig(
    level=logging.INFO,
    filename='manager.log',
    filemode='w',
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
    print(os.getenv('CUDA_VISIBLE_DEVICES'))
    os.environ["CUDA_VISIBLE_DEVICES"] = tkwargs.get("UUID", "0")
    print(os.getenv('CUDA_VISIBLE_DEVICES'))

    nvmlInit()
    torch.cuda.init()    
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        # handle = nvmlDeviceGetHandleByIndex(i)
        handle = nvmlDeviceGetMigDeviceHandleByIndex(nvmlDeviceGetHandleByIndex(0), 0)
        logger.info(f"Running on MIG: {nvmlDeviceGetUUID(handle)}")
        logger.info(f"CUDA Device: {torch.cuda.current_device()}")
    exp_name = experiment.get("experiment_name", experiment.get("experiment_id"))
    exp_id = experiment.get("experiment_id")
    results = None
    try:
        runner_type = experiment.get("runner_type")
        if runner_type == "simulation":
            results = ExperimentRunnerSimulationDriven(experiment, replication, tkwargs).init_flask()
       
        elif runner_type == "algorithm":
            results = ExperimentRunnerAlgorithmDriven(experiment, replication, tkwargs).run_optimization_loop()

        else:
            raise ValueError(f"Runner Type of experiment {exp_name} (ID: {exp_id}) not identified")

        # return results
    except Exception as e:
        logger.error(f"Error in running experiment {exp_name} (ID: {exp_id}): {e}")
        raise e


class ExperimentManager:
    def __init__(self, checking_interval=60):

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

        # for debug:
        # self.number_of_gpus = 7
        self.available_gpus = Queue()
        uuids = ["MIG-aeca6767-99c9-5fb8-b956-723bcb25d82b", "MIG-aae631f3-f7ad-5d35-b717-ccb9ba0b92f9", "MIG-c5f6bdea-5751-5327-bdce-7086947edd84", "MIG-07510fb0-c5df-56b5-bbc2-1d94286e3553", "UUID: MIG-32191766-a3ca-5c44-bddd-d0dc6d33acbd", "MIG-4839a599-f156-51e4-a526-4592d46b179d", "MIG-4839a599-f156-51e4-a526-4592d46b179d"]
        for i, uuid in enumerate(uuids):
            self.available_gpus.put({"UUID": uuid, "device_idx" : i})

        self.used_gpus = Queue()

        self.date_format = "%Y-%m-%d %H:%M:%S"
        self.logger.info("Manager initialized.")

    def run_experimentation_process(self, experiment: dict, tkwargs: dict, gpu: str = None):
        exp_name = experiment.get("experiment_name", experiment.get("experiment_id"))
        exp_id = experiment.get("experiment_id")
        self.logger.info("Running experiment: " + str(exp_name))
        self.logger.info("Execution time is: " + str(experiment.get("execution_datetime")))
        replication = experiment.get("current_replication", 1)
        self.experiments_running.put(experiment)
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = tkwargs.get("UUID", "0")
            p = mp.Process(target=send_experiment_to_runner, args=(experiment, replication, tkwargs,))
            p.start()

            process_dict = {
                "experiment_id": exp_id,
                "experiment_name": exp_name,
                "current_replication": int(replication),
                "replications": int(experiment.get("replications", 1)),
                "process": p,
                "gpu" : gpu,
                "experiment" : copy.deepcopy(experiment),


            }
            self.processes_running.append(process_dict)
            self.logger.info(f"Process started on gpu {gpu}: Replication {replication} of experiment {exp_name} (ID: {exp_id})")
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

                self.database.write_result_to_firestore(exp_id, replication)
                self.database.write_all_files_to_storage(exp_id)
                if replication >= process.get("replications"):
                    experiment = process.get("experiment")
                    self.close_experiment(experiment)
                
                # self.processes_running.remove(process)
                self.available_gpus.put(process.get("gpu"))
                processes_to_rm.append(process)
                # p.close()
        for rmp in processes_to_rm:
            self.processes_running.remove(rmp)
 


    def check_experiment_queue(self):
        while not self.experiments_queue.empty():
            if self.available_gpus.empty():
                self.logger.info("No more GPUs available. Waiting for a GPU to be available...")
                return
            try:
                experiment_to_run = self.experiments_queue.get()
                gpu = self.available_gpus.get()
                device_idx = gpu.get("device_idx")
                tkwargs = {"device": torch.device(f"cuda" if torch.cuda.is_available() else "cpu"), "dtype": self.dtype, "UUID":gpu.get("UUID"), "device_idx" : device_idx}
                # logger.info("Starting experiment with ID: " + str(experiment_to_run.get("experiment_id")) + " and name: " + str(experiment_to_run.get("experiment_name")) + " at " + str(datetime.now()) + "on: " + str(gpu))
                self.logger.info("Execution time is: " + str(experiment_to_run.get("execution_datetime")))
                self.run_experimentation_process(experiment_to_run, tkwargs, gpu)
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

        col_query = self.database.db.collection(u'experiments').where(u'status', u'==', u'open') 
        # TODO: make handler for failed experiments
        query_watch = col_query.on_snapshot(check_changes)

        while self.should_listen:
            self.logger.info("Checking for finished experiment replications...")
            self.check_processes()
            self.logger.info("Checking for new experiments to run...")
            self.check_experiment_queue()
   
            time.sleep(self.checking_interval)
        # query_watch.unsubscribe()
    


# def wrapper_run_experiment(experiment, tkwargs):
#     run_experiment(experiment, tkwargs)

if __name__ == "__main__":
    # pool = Pool(nodes=5)
    # ExperimentManager(pool, 10).start_firestore_listener()
    
    ExperimentManager(5).start_firestore_listener()
