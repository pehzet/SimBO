import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import threading
import time
from datetime import datetime
from icecream import ic
import json
import sys
import multiprocessing as mp
from ..experiment_runners.experiment_runner_algorithm_driven import ExperimentRunnerAlgorithmDriven
from ..experiment_runners.experiment_runner_simulation_driven import ExperimentRunnerSimulationDriven
print("Number of processors: ", mp.cpu_count())

class ExperimentManager:
    def __init__(self):
        self.experiments_queue = []
        self.experiments_running = []
        self.experiments_done = []
        self.experiments_failed = []
        self.experiments_aborted = []
        self.db = self.init_firestore()
        self.last_check = None
        self.should_listen = True
        self.date_format = "%Y-%m-%d %H:%M:%S"
        print("Manager initialized. Starting Listening to Firestore...")
        self.start_firestore_listener()
    
    def run_experiment(self,experiment:dict):
        print("Running experiment: " + str(experiment.get("experiment_id")))
        print("Execution time is: " + str(experiment.get("execution_time")))
        # start_random_loop()
        # for multiprocessing check: https://stackoverflow.com/questions/56481306/running-different-python-functions-in-separate-cpus
        runner_type = experiment.get("runner_type")
        replication = 1
        while replication < experiment.get("replications")+1:
            # Here we should use multiprocessing
            if runner_type == "simulation":
                ExperimentRunnerSimulationDriven(experiment, replication)
            elif runner_type == "algorithm":
                ExperimentRunnerAlgorithmDriven(experiment, replication).run_optimization_loop()
            else:
 
                print("Runner Type of experiment " + str(experiment.get("experiment_id")) + " not identified. Maybe typo at gsheet. Going to exit")
                sys.exit()
            replication += 1
        print("Experiment finished: " + str(experiment.get("experiment_id")))
        




    def save_experiment_as_json(self,experiment):
        with open('C:\code\SimBO\manager\experiments\experiment_' + str(experiment.get("experiment_id")) + '.json', 'w') as outfile:
            json.dump(experiment, outfile)
        
    def break_experiment_listener(self):
        self.should_listen = False

    def init_firestore(self):
        cred = credentials.Certificate('C:\code\SimBO\simbo-bf62e-firebase-adminsdk-atif6-cbeac3a8e4.json')
        app = firebase_admin.initialize_app(cred)
        return firestore.client()

    def add_experiment_to_queue(self,experiment):
        if not experiment in self.experiments_queue:
            self.experiments_queue.append(experiment)
            print("Experiment added to queue: " + str(experiment.get("experiment_id")))
            print("Execution time is: " + str(experiment.get("execution_time")))
    
    def check_experiment_queue(self):
        if len(self.experiments_queue) > 0:
            for experiment in self.experiments_queue:
                if datetime.now().strftime(self.date_format) >= experiment.get("execution_time"):
                    self.experiments_running.append(experiment)                    
                    self.experiments_queue.remove(experiment)
                            
                    t = threading.Thread(target=self.run_experiment, args=(experiment))
                    t.daemon = True
                    t.start()
                    

    def start_firestore_listener(self):
        # Create an Event for notifying main thread.
        callback_done = threading.Event()

        # Create a callback on_snapshot function to capture changes
        def on_snapshot(col_snapshot, changes, read_time):
            for change in changes:
                if change.type.name == 'ADDED':
                    self.add_experiment_to_queue(change.document.to_dict())
                    callback_done.set()
        # if self.last_check is None:
        #     self.last_check = datetime.now()

        col_query = self.db.collection(u'experiments').where(u'status', u'==', u'open')#.where(u'created_at', u'>=', self.last_check)
        query_watch = col_query.on_snapshot(on_snapshot)
        # self.last_check = datetime.now()
        # Watch the collection query
        while self.should_listen:
            time.sleep(5)
            print("Checking for new experiments...")
            self.check_experiment_queue()
            # self.last_check = datetime.now()


          

ExperimentManager()

