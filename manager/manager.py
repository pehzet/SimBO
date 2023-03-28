

import sys
sys.path.append('../')
import threading
import time
from datetime import datetime
from icecream import ic
import json
import sys
import logging
import warnings
import multiprocessing as mp
from database import Database
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s: %(levelname)s: %(name)s: %(message)s'
    )
logger = logging.getLogger("manager")
from experiment_runners.experiment_runner_algorithm_driven import ExperimentRunnerAlgorithmDriven
from experiment_runners.experiment_runner_simulation_driven import ExperimentRunnerSimulationDriven
logger.debug("Number of processors: ", mp.cpu_count())
from pathlib import Path
import os





# Surpress PyTorch warning
warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is") 
warnings.filterwarnings("ignore", message="Could not import matplotlib.pyplot")
warnings.filterwarnings("ignore", message="torch.triangular_solve is deprecated") 


class ExperimentManager:
    def __init__(self, checking_interval=60):
        self.checking_interval = checking_interval
        self.experiments_queue = []
        self.experiments_running = []
        self.experiments_done = []
        self.experiments_failed = []
        self.experiments_aborted = []
        self.main_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) #r"C:\code\SimBO"
        self.database = Database(self.main_dir)
        self.last_check = None
        self.should_listen = True
        self.processes = []

        self.date_format = "%Y-%m-%d %H:%M:%S"
        logger.info("Manager initialized.")
        
    
    def run_experiment(self,experiment:dict):
        exp_name = experiment.get("experiment_name", experiment.get("experiment_id"))
        exp_id = experiment.get("experiment_id")
        logger.info("Running experiment: " + str(exp_name))
        logger.info("Execution time is: " + str(experiment.get("execution_datetime")))
        # start_random_loop()
        # for multiprocessing check: https://stackoverflow.com/questions/56481306/running-different-python-functions-in-separate-cpus
        try:
            runner_type = experiment.get("runner_type")
            replication = 1
            while replication <= int(experiment.get("replications")):
                logger.info(f"Replication {replication} of experiment {exp_name} (ID: {exp_id})  started")
                # Here we should use multiprocessing
                if runner_type == "simulation":
                    ExperimentRunnerSimulationDriven(experiment, replication)
                    results = None
                elif runner_type == "algorithm":
                    results = ExperimentRunnerAlgorithmDriven(experiment, replication).run_optimization_loop()
                else:
                    logger.error(f"Runner Type of experiment {exp_name} (ID: {exp_id}) not identified. Maybe typo at gsheet. Going to exit")
                    sys.exit()
                try:
                    self.database.write_result_to_firestore(exp_id,replication,results)
                    self.database.write_all_files_to_storage(exp_id)
                except Exception as e:
                    logger.error(f"Error while writing results of experiment {exp_name} (ID: {exp_id}) to Firestore")
                    logger.error(e)
                    sys.exit()
                replication += 1
      
            self.experiments_done.append(experiment)
            self.experiments_running.remove(experiment)
            logger.info(f"Experiment finished: {exp_name} (ID: {exp_id}) ")
            try:
                self.database.set_experiment_status(exp_id, "done")
            except Exception as e:
                logger.error(f"Error while setting experiment {exp_name} (ID: {exp_id})  to status 'done'")
            return
        except Exception as e:
            logger.error(f"Error while running experiment {exp_name} (ID: {exp_id}) ")
            logger.error(e)
            self.experiments_failed.append(experiment)
            self.experiments_running.remove(experiment)
            self.save_experiment_as_json(experiment)
            sys.exit()
        

    def save_experiment_as_json(self,experiment):

        path = os.path.join(self.main_dir, 'manager', 'data','experiment_' + str(experiment.get("experiment_id")))
        if not os.path.exists(path):
            os.makedirs(path)
        fpath = os.path.join(path, str(experiment.get("experiment_id")) + '.json')
        with open(fpath, 'w') as outfile:
            json.dump(experiment, outfile)
        
    def break_experiment_listener(self):
        self.should_listen = False

    def identify_runner_type(self, experiment):
        if experiment.get("use_case","").lower() in ["mrp"]:
            experiment["runner_type"] = "algorithm"
        elif experiment.get("use_case","").lower() in ["pfp"]:
            experiment["runner_type"] = "simulation"
        else:
            exp_id = experiment.get("experiment_id")
            exp_name = experiment.get("experiment_name")
            logger.error(f"Use case of Experiment {exp_name} (ID: {exp_id}) not identified. Please check the experiment creation.")
            sys.exit()
        return experiment


    def add_experiment_to_queue(self,experiment):
        if not experiment in self.experiments_queue:
            if experiment.get("experiment_id") == "1BER5iWbxTA1sv00iGPT":
                return
            experiment = self.identify_runner_type(experiment)
            self.experiments_queue.append(experiment)
            logger.info("Experiment added to queue: " + str(experiment.get("experiment_name", experiment.get("experiment_id"))))
            logger.info("Execution time is: " + str(experiment.get("execution_datetime")))
    
    def check_experiment_queue(self):
        if len(self.experiments_queue) > 0:
            for experiment in self.experiments_queue:
                # if datetime.now().strftime(self.date_format) >= experiment.get("execution_datetime") and len(self.experiments_running) < 6:
                if len(self.experiments_running) < 6:
                    try:
                        # t = threading.Thread(target=self.run_experiment, args=[experiment])
                        # t.daemon = True
                        # t.start()
                        self.run_experiment(experiment)
                        
                        self.experiments_running.append(experiment)                    
                        self.experiments_queue.remove(experiment)
                       
                    except Exception as e:
                        logger.error("Error: unable to start process to run experiment")
                        logger.error(e)
                        sys.exit()


    def start_firestore_listener(self):
        logger.info("Starting Listening to Firestore...")
        # Create an Event for notifying main thread.
        callback_done = threading.Event()

        # Create a callback on_snapshot function to capture changes
        def check_changes(col_snapshot, changes, read_time):

            for change in changes:
                if change.type.name == 'ADDED':
                    experiment = change.document.to_dict()
                    experiment["experiment_id"] = change.document.id
                    self.add_experiment_to_queue(experiment)
                    callback_done.set()
        # if self.last_check is None:
        #     self.last_check = datetime.now()

        col_query = self.database.db.collection(u'experiments').where(u'status', u'==', u'open')#.where(u'created_at', u'>=', self.last_check)
        query_watch = col_query.on_snapshot(check_changes)
        # self.last_check = datetime.now()
        # Watch the collection query
        while self.should_listen:
            logger.info("Checking for new experiments to run...")
            self.check_experiment_queue()
            time.sleep(self.checking_interval)
            # self.last_check = datetime.now()


          
if __name__ == "__main__":
    ExperimentManager(10).start_firestore_listener()

