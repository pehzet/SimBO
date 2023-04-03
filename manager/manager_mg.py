

import multiprocessing as mp2
import multiprocessing_on_dill as mp
import os
from pathlib import Path
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
import sys
sys.path.append('../')


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s: %(levelname)s: %(name)s: %(message)s'
)
logger = logging.getLogger("manager")
# file_handler = logging.FileHandler('pathos.log')
# formatter = logging.Formatter('%(asctime)s %(process)s %(levelname)s %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
# logger.debug("Number of processors: ", mp.cpu_count())


# set_loky_pickler('dill')


# Surpress PyTorch warning
warnings.filterwarnings(
    "ignore", message="To copy construct from a tensor, it is")
warnings.filterwarnings("ignore", message="Could not import matplotlib.pyplot")
warnings.filterwarnings(
    "ignore", message="torch.triangular_solve is deprecated")


class ExperimentManager:
    def __init__(self, checking_interval=60):

        self.checking_interval = checking_interval
        self.experiments_queue = Queue()
        self.experiments_running = Queue()
        self.experiments_done = Queue()
        self.experiments_failed = Queue()
        self.experiments_aborted = Queue()
        self.results = Queue()
        self.main_dir = os.path.abspath(os.path.join(
            os.getcwd(), os.pardir))  # r"C:\code\SimBO"
        self.database = Database(self.main_dir)
        self.last_check = None
        self.should_listen = True
        self.processes = []
        self.dtype = torch.double
        self.number_of_gpus = torch.cuda.device_count()

        # for debug:
        self.number_of_gpus = 2
        self.available_gpus = Queue()

        for i in range(self.number_of_gpus):
            self.available_gpus.put(f"cuda:{i}")

        self.used_gpus = Queue()

        self.date_format = "%Y-%m-%d %H:%M:%S"
        logger.info("Manager initialized.")

    def run_experiment(self, experiment: dict, tkwargs: dict):
        exp_name = experiment.get(
            "experiment_name", experiment.get("experiment_id"))
        exp_id = experiment.get("experiment_id")
        logger.info("Running experiment: " + str(exp_name))
        logger.info("Execution time is: " +
                    str(experiment.get("execution_datetime")))
        # start_random_loop()
        # for multiprocessing check: https://stackoverflow.com/questions/56481306/running-different-python-functions-in-separate-cpus
        try:
            runner_type = experiment.get("runner_type")
            replication = 1
            while replication <= int(experiment.get("replications")):
                logger.info(
                    f"Replication {replication} of experiment {exp_name} (ID: {exp_id})  started")
                # Here we should use multiprocessing
                if runner_type == "simulation":
                    ExperimentRunnerSimulationDriven(
                        experiment, replication, tkwargs)
                    results = None
                elif runner_type == "algorithm":

                    ExperimentRunnerAlgorithmDriven(experiment, replication, tkwargs).run_optimization_loop()

                else:
                    logger.error(
                        f"Runner Type of experiment {exp_name} (ID: {exp_id}) not identified. Maybe typo at gsheet. Going to exit")
                    sys.exit()
                try:
                    self.database.write_result_to_firestore(
                        exp_id, replication, results)
                    self.database.write_all_files_to_storage(exp_id)

                except Exception as e:
                    logger.error(
                        f"Error while writing results of experiment {exp_name} (ID: {exp_id}) to Firestore")
                    logger.error(e)
                    sys.exit()
                replication += 1

            self.experiments_done.put(experiment)
            _experiment = self.experiments_running.get(experiment)
            logger.info(f"Experiment finished: {exp_name} (ID: {exp_id}) ")
            try:
                self.database.set_experiment_status(exp_id, "done")
            except Exception as e:
                logger.error(
                    f"Error while setting experiment {exp_name} (ID: {exp_id})  to status 'done'")
            return
        except Exception as e:
            logger.error(
                f"Error while running experiment {exp_name} (ID: {exp_id}) ")
            logger.error(e)
            self.experiments_failed.put(experiment)
            _experiment = self.experiments_running.get(experiment)
            self.save_experiment_as_json(experiment)
            sys.exit()

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
            logger.error(
                f"Use case of Experiment {exp_name} (ID: {exp_id}) not identified. Please check the experiment creation.")
            sys.exit()
        return experiment

    def add_experiment_to_queue(self, experiment):
        experiment = self.identify_runner_type(experiment)
        self.experiments_running.put(experiment)
        self.experiments_queue.put(experiment)
        logger.info("Experiment added to queue: " +
                    str(experiment.get("experiment_name", experiment.get("experiment_id"))))
        logger.info("Execution time is: " +
                    str(experiment.get("execution_datetime")))

    def check_experiment_queue(self):

        tmp_list = []
        # Check if there are any available processors
        while not self.experiments_queue.empty():
            if self.available_gpus.empty():
                logger.info("No available GPUs. Waiting 60 seconds...")
                time.sleep(2)
                break
            try:
                # Get a task from the queue
                experiment_to_run = self.experiments_queue.get()
                gpu = self.available_gpus.get()
                tkwargs = {"device": torch.device(
                    gpu if torch.cuda.is_available() else "cpu"), "dtype": self.dtype}
                # Submit the task to the pool

                print("Starting experiment with ID: " + str(experiment_to_run.get("experiment_id")) + " and name: " +
                      str(experiment_to_run.get("experiment_name")) + " at " + str(datetime.now()) + "on: " + str(gpu))
                # d = joblib.delayed(wrapper_run_experiment)(experiment_to_run, tkwargs)
                # ic(d)
                # self.parallel([d])

                # self.results.put(work)

                self.run_experiment(experiment_to_run, tkwargs)
                self.available_gpus.put(gpu)
            except Exception as e:
                print(e)

        print("No more experiments to run")

    def start_firestore_listener(self):
        logger.info("Starting Listening to Firestore...")

        # handler = QueueHandler(self.experiment_queue)

        # logger.addHandler(handler)
        # Create a callback on_snapshot function to capture changes

        def check_changes(col_snapshot, changes, read_time):

            for change in changes:
                if change.type.name == 'ADDED':
                    experiment = change.document.to_dict()
                    experiment["experiment_id"] = change.document.id
                    self.add_experiment_to_queue(experiment)
                    # self.experiments_queue.put(experiment)

        # if self.last_check is None:
        #     self.last_check = datetime.now()

        col_query = self.database.db.collection(u'experiments').where(
            u'status', u'==', u'open')  # .where(u'created_at', u'>=', self.last_check)
        query_watch = col_query.on_snapshot(check_changes)
        # self.last_check = datetime.now()
        # Watch the collection query
        while self.should_listen:
            logger.info("Checking for new experiments to run...")
            self.check_experiment_queue()
            time.sleep(self.checking_interval)
            # self.last_check = datetime.now()


# def wrapper_run_experiment(experiment, tkwargs):
#     EM.run_experiment(experiment, tkwargs)

if __name__ == "__main__":
    # pool = Pool(nodes=5)
    # ExperimentManager(pool, 10).start_firestore_listener()

    ExperimentManager(1).start_firestore_listener()
