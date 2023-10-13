import firebase_admin
from google.cloud.firestore_v1.base_query import FieldFilter
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
import sys
import backend.utils.gsheet_utils as gsheet_utils
import os
import json
import logging
import config
import traceback
from icecream import ic

logger = logging.getLogger("database")
class FirebaseManager:
    def __init__(self, main_dir=None):
        if main_dir is None:
            self.main_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        else:
            self.main_dir = main_dir
        self.fb_key_name = config.FIREBASE_CONFIG
        self.app = self.init_firebase()
        self.db = self.init_firestore()
        self.bucket = self.init_storage()

    def update_current_best_candidate(self, experiment_id, replication, best_candidate):
        doc_ref = self.db.collection(u'experiments').document(str(experiment_id))
        doc_ref.update({"best_candidate": best_candidate})
    def set_experiment_status(self, experiment_id, status, with_id = False):
        if status not in ["running", "done", "failed", "aborted"]:
            logger.error(f"Status {status} not recognized. Experiment {experiment_id} not updated.")
            raise Exception(f"Status {status} not recognized. Experiment {experiment_id} not updated.")

        doc_ref = self.db.collection(u'experiments').document(str(experiment_id))
        if not with_id:
             doc_ref.update({u'status': status})
        else:
            doc_ref.update({u'status': status, experiment_id : experiment_id})

           
    def init_firebase(self):
        cred = credentials.Certificate(self.fb_key_name)
        return firebase_admin.initialize_app(cred, {
            'storageBucket': config.BUCKET
        })
    def init_firestore(self):
        return firestore.client()

    def init_storage(self):
        return storage.bucket()

    def write_experiment_to_firestore(self, experiment):
        doc_ref = self.db.collection(u'experiments').document(str(experiment.get("experiment_id")))
        doc_ref.set(experiment)

    def get_experiment_from_firestore(self, experiment_id):
        doc_ref = self.db.collection(u'experiments').document(str(experiment_id))
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            return None

    def get_best_candidate_of_replication(self, experiment_id, replication):
        doc_ref = self.db.collection(u'experiments').document(str(experiment_id)).collection(u"results").document("Replication_" + str(replication))
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            return None

    def read_local_config(self):
        fpath = os.path.join(self.main_dir,'config.json')
        with open(fpath) as json_file:
            data = json.load(json_file)
            return data

    def get_configs_from_gsheet_and_save(self, fb=True, local=True):
        configs = gsheet_utils.get_configs_from_gsheet()
        dir_path = os.path.join(self.main_dir,'configs')

        for config in configs:
            if local:
                path = os.path.join(dir_path, str(config.get("experiment_id")) + ".json")
                with open(path, 'w') as outfile:
                    json.dump(config, outfile)
            if fb:
                self.write_experiment_to_firestore(config)
    def set_experiment_status_interrupted(self, experiment_id):
        doc_ref = self.db.collection(u'experiments').document(str(experiment_id))
        status = doc_ref.get().to_dict().get("status")
        if not status == "done":
            doc_ref.update({u'status': "interrupted"})
                
    def update_replication(self, experiment_id, replication):
        doc_ref = self.db.collection(u'experiments').document(str(experiment_id))
        doc_ref.update({u'replications_fulfilled': replication})
    def update_current_replication(self, experiment_id, replication):
        doc_ref = self.db.collection(u'experiments').document(str(experiment_id))
        doc_ref.update({u'current_replication': replication})
    def check_experiment_status(self, experiment_id, status_to_check=None):
        '''
        Returns status if status_to_check is None else returns True if status is equal to status_to_check.
        Returns None if experiment_id does not exist.
        '''
        doc_ref = self.db.collection(u'experiments').document(str(experiment_id))
        doc = doc_ref.get()
        if doc.exists:
            exp_status = doc.to_dict().get("status")
            if status_to_check is None:
                return exp_status
            return exp_status == str(status_to_check)
                 
        else:
            return None
    def write_result_to_firestore(self, experiment_id, replication, results=None):
        
        if results == None:
            # dir_path = r'C:\code\SimBO\manager\data'
            dir_path = os.path.join(self.main_dir,'data')
            exp_string = "experiment_"+str(experiment_id)
            result_path = os.path.join(dir_path, exp_string, exp_string+"_" + str(replication)+".json")
            try:
                with open(result_path, "r") as outfile:
                    results = json.load(outfile)
            except Exception as e:
                logger.error(e)
                logger.error(f"File has not been created yet for experiment {experiment_id} Replication: {replication}")

        replication_str = "Replication_" + str(replication)
        doc_ref = self.db.collection(u'experiments').document(str(experiment_id)).collection(u"results").document(replication_str)
        
      
        try:
            best_candidate = results.get("best_candidate", results.get("best_candidat", {}))
            doc_ref.set({"best_candidate": best_candidate})
            logger.info(f"Results written to firestore for experiment {experiment_id} Replication: {replication}")
        except Exception as e:
            logger.error(e)
            traceback.print_exc()
            logger.error(f"Error writing results to firestore for experiment {experiment_id} Replication: {replication}")
        # self.update_replication_at_firestore(experiment_id, replication)

    def write_file_to_storage(self,experiment_id, replication, obj_name, obj_suffix="pkl"):
        dir_path = os.path.join(self.main_dir,'data')
        exp_path = "experiment_" +str(experiment_id)
        fname = str(experiment_id)+"_" + str(obj_name) + "_" + str(replication) + "." + str(obj_suffix)
        fpath = os.path.join(dir_path, exp_path, fname)
        blob = self.bucket.blob(exp_path + "/" + fname)
        try:
            blob.upload_from_filename(fpath)
        except Exception as e:
            print(e)
            print(f"Error writing {obj_name} pickle to storage")
        
    def write_all_files_to_storage(self, exp_id):
        folder_name = "experiment_" + str(exp_id)
        folder = os.path.join(self.main_dir,  'data', folder_name)
        try:
            for file in os.listdir(folder):
                blob = self.bucket.blob(folder_name + "/" + file)
                if not blob.exists():
                    blob.upload_from_filename(os.path.join(folder, file))
            logger.info(f"Files for experiment {exp_id} written to storage")
        except Exception as e:
            logger.error(e)
            logger.error(f"Error writing files to storage for experiment {exp_id}")
    # TODO REMOVE FROM FIREBASE MANAGER
    def check_if_local_files_exist(self, experiment_id, replication):
        if None in [experiment_id, replication]:
            logger.error("Experiment id or replication is None")
            return False
        dir_path = os.path.join(self.main_dir,'data')
        exp_string = "experiment_"+str(experiment_id)
        result_path = os.path.join(dir_path, exp_string, exp_string+"_" + str(replication)+".json")
        return os.path.exists(result_path)
    
    def update_replication_progress(self, experiment_id, replication, current_arm, budget):
        doc_ref = self.db.collection(u'experiments').document(str(experiment_id))

        doc_ref.update({f'replication_progress.{str(replication)}' : current_arm })

    def check_database_for_experiments(self, manager_id=-1, limit=1):
        if manager_id == -1 or manager_id == None:
            experiments = self.db.collection(u'experiments').where(filter=FieldFilter("status", "in", ["open", "running", "paused"])).order_by("priority_value", direction=firestore.Query.DESCENDING).order_by("created_at").limit(limit).get()
        else:
            experiments = self.db.collection(u'experiments').where(filter=FieldFilter("status", "in", ["open", "running", "paused"])).where(filter=FieldFilter(u'manager_id', u'==', int(manager_id))).order_by("priority_value", direction=firestore.Query.DESCENDING).order_by("created_at").limit(limit).get()
        return experiments
    def get_experiment_manager(self, manager_id=None):
        if manager_id is None:
            manager = self.db.collection(u'managers').where(u'status', u'==', u'free').get()
            if len(manager) > 0:
                return manager[0].to_dict()
            else:
                raise Exception("No free manager found")
        else:
            manager = self.db.collection(u'managers').document(f'manager{manager_id}').get()
            if manager.exists:
                manager = manager.to_dict()
                if manager.get("status") == "free":
                    return manager
                else:
                    raise Exception(f"Manager {manager_id} is not free")
            else:
                raise Exception(f"Manager {manager_id} does not exist")
    def set_experiment_manager_experiment(self, manager_id,  exp_id):
        obj = {
                "status": "free" if exp_id == None else "busy",
                'current_experiment': exp_id
            }
        self.db.collection(u'managers').document(f'manager{manager_id}').update(obj)
    
    def append_best_arm_to_use_case(self, use_case, best_arm):
        doc_ref = self.db.collection(u'use_cases').document(use_case)
        doc_ref.update({u'best_arms': firestore.ArrayUnion([best_arm])})
    
    def get_best_arms_from_use_case(self, use_case):
        doc_ref = self.db.collection(u'use_cases').document(use_case)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict().get("best_arms")
        else:
            return []


# def get_all_files_from_folder():
#     # folder path

#     dir_path = r'C:\code\SimBO\data'
#     db = Database()


#     # Iterate directory
#     for folder in os.listdir(dir_path):
#         # check if current path is a file
#         pathes = os.path.join(dir_path, folder)
#         for path in os.listdir(pathes):
#             if os.path.isfile(os.path.join(pathes, path)):
#                 if path.endswith(".json"):
#                     with open(os.path.join(pathes, path)) as json_file:
#                         data = json.load(json_file)
#                         db.write_result_to_firestore(data.get("experiment_id"), 1, data)

