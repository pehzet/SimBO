
import torch.multiprocessing as mp
try:
   mp.set_start_method('spawn', force=True)
except RuntimeError:
   pass
import os
from icecream import ic
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
# sys.path.append('../')


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
    main_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    try:
        runner_type = experiment.get("runner_type")
        if runner_type == "simulation":
            ersd = ExperimentRunnerSimulationDriven(experiment, replication, tkwargs, Database(main_dir))
            results = ersd.init_flask()
        elif runner_type == "algorithm":
            erad = ExperimentRunnerAlgorithmDriven(experiment, replication, tkwargs, Database(main_dir))
            results = erad.run_optimization_loop()
        else:
            raise ValueError(f"Runner Type of experiment {exp_name} (ID: {exp_id}) not identified")
        return results
    except Exception as e:
        logger.error(f"Error in running experiment {exp_name} (ID: {exp_id}): {e}")
        raise e


class ExperimentManager:
    def __init__(self, manager_id, checking_interval=60, num_parallel_experiments=-1):
        self.manager_id = manager_id
        self.checking_interval = checking_interval
        self.experiments_queue = Queue()
        self.experiments_running = Queue()

        self.processes_running = []
        self.main_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))  # r"C:\code\SimBO"
        self.database = Database(self.main_dir)
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
        

        self.date_format = "%Y-%m-%d %H:%M:%S"
        self.logger.info("Manager initialized.")
    def gpu_free(self):
        return len(self.gpus_available) > 0
    def run_experimentation_process(self, experiment: dict):
        self.database.update_current_replication_at_firestore(experiment.get("experiment_id"), experiment.get("current_replication", 1))
        tkwargs = {"device": torch.device(f"cuda" if torch.cuda.is_available() and not experiment.get("use_cpu", False) else "cpu"), "dtype": self.dtype}
        exp_name = experiment.get("experiment_name", experiment.get("experiment_id"))
        exp_id = experiment.get("experiment_id")
        self.logger.info("Running experiment: " + str(exp_name))
        self.logger.info("Execution time is: " + str(experiment.get("execution_datetime")))
        current_replication = experiment.get("current_replication", 1)
        self.experiments_running.put(experiment)
        gpu = self.gpus_available.pop(0)
        try:
            p = mp.Process(target=send_experiment_to_runner, args=(experiment, current_replication, tkwargs, ))
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
            self.gpus_available.append(gpu)
          

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
        if experiment.get("use_case", "").lower() in ["mrp"]:
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
                        self.database.update_replication_at_firestore(exp_id, replication)
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
                    self.close_experiment(process.get("experiment"),replication, failed=True)
                # p.close()
        for rmp in processes_to_rm:
            self.processes_running.remove(rmp)
 


    
    def run(self):
        no_experiment_counter = 0
        initial_checking_interval = self.checking_interval
        try:
            exp_id = sys.argv[4]
            logger.info(f"Running experiment {exp_id} from command line")
            experiment = self.database.get_experiment_from_firestore(exp_id)
            self.run_experimentation_process(experiment)
            sys.argv.pop(4)
        except:
            pass
        while self.should_listen:
            self.logger.info("Checking for finished experiment replications...")
            self.check_processes()
            if self.gpu_free():
                self.logger.info("Checking for new experiments to run...")
                experiments = self.database.check_database_for_experiments(self.manager_id, len(self.gpus_available))
                if len(experiments) == 0:
                    self.logger.info(f"No experiments found. Waiting {self.checking_interval} seconds")
                    no_experiment_counter += 1
                    if self.checking_interval > initial_checking_interval*5:
                        self.checking_interval = initial_checking_interval*5
                        self.logger.info(f"Limiting checking interval to {self.checking_interval} seconds")
                    if no_experiment_counter > 10:
                        self.checking_interval *= 1.5
                        self.logger.info(f"No experiments found. Increasing checking interval to {self.checking_interval} seconds")
                        no_experiment_counter = 0
                exp_in_this_loop = []
                for exp in experiments:
                    experiment = exp.to_dict()
                    for i in range(int(experiment.get("replications_fulfilled")), int(experiment.get("replications"))):
                        if experiment.get("replications_fulfilled", 0) + (i+1) == experiment.get("current_replication", 0):
                            continue
                        else:
                            experiment["experiment_id"] = exp.id
                            experiment["current_replication"] = experiment.get("replications_fulfilled", 0) + (i+1)
                            experiment["runner_type"] = self.identify_runner_type(experiment)
                            exp_in_this_loop.append(experiment)
                if len(exp_in_this_loop) > 0:
                    self.logger.info(f"Found {len(exp_in_this_loop)} experiments to run")
                    self.checking_interval = initial_checking_interval
                    for i in range(len(self.gpus_available)):
                        self.run_experimentation_process(exp_in_this_loop[i])
            time.sleep(max(self.checking_interval,10))


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
            command = 'nvidia-smi'
            while True:
                p = subprocess.check_output(command, shell=True)
                logger.info(str(p.decode("utf-8")))
                time.sleep(2)
        else:
            logger.info("No GPU available")
    except Exception as e:
        logger.error("Error while logging GPU usage")
        logger.error(e)
        exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 experiment_manager.py <manager_id> <checking_interval> <opt: num_parallel_experiments> <opt: experiment_id>")
        exit(1)
    manager_id = int(sys.argv[1])
    interval = int(sys.argv[2])
    if len(sys.argv) > 3:
        num_parallel_experiments = int(sys.argv[3])
    else:
        num_parallel_experiments = -1
    mp.Process(target=log_gpu_usage).start() 
    ExperimentManager(manager_id, interval, num_parallel_experiments).run()
