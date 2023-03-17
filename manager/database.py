import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
import sys
sys.path.insert(1, 'C:/code/SimBO/utils')
import utils.gsheet_utils as gsheet_utils
import os
import json
import logging
logger = logging.getLogger("database")
class Database:
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.fb_key_name = 'simbo-bf62e-firebase-adminsdk-atif6-cbeac3a8e4.json'
        self.app = self.init_firebase()
        self.db = self.init_firestore()
        self.bucket = self.init_storage()

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
        credit_path = os.path.join(self.main_dir,self.fb_key_name)
        cred = credentials.Certificate(credit_path)
        return firebase_admin.initialize_app(cred, {
            'storageBucket': 'simbo-bf62e.appspot.com'
        })
    def init_firestore(self):
        return firestore.client()
    
    def init_storage(self):
        return storage.bucket()

    def write_experiment_to_firestore(self, experiment):
        doc_ref = self.db.collection(u'experiments').document(str(experiment.get("experiment_id")))
        doc_ref.set(experiment)

    def read_experiment_from_firestore(self, experiment_id):
        doc_ref = self.db.collection(u'experiments').document(str(experiment_id))
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            return None

    def read_local_config(self):
        fpath = os.path.join(self.main_dir,'manager','config.json')
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
    
    def write_result_to_firestore(self, experiment_id, replication, results=None):

        if results == None:
            dir_path = r'C:\code\SimBO\data'
            exp_string = "Experiment_"+str(experiment_id)
            result_path = os.path.join(dir_path, exp_string, exp_string+"_" + str(replication)+".json")
            with open(result_path, "w") as outfile:
                results = json.load(outfile)

        replication = "Replication_" + str(replication)
        doc_ref = self.db.collection(u'experiments').document(str(experiment_id)).collection(u"results").document(replication)
        from icecream import ic
        ic(doc_ref)
        try:
            doc_ref.set(results)
            logger.info(f"Results written to firestore for experiment {experiment_id} replication {replication}")
        except Exception as e:
            logger.error(e)
            logger.error(f"Error writing results to firestore for experiment {experiment_id} replication {replication}")

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
def get_all_files_from_folder():
    # folder path

    dir_path = r'C:\code\SimBO\data'
    db = Database()


    # Iterate directory
    for folder in os.listdir(dir_path):

        # check if current path is a file
        pathes = os.path.join(dir_path, folder)
     
        for path in os.listdir(pathes):
           
            if os.path.isfile(os.path.join(pathes, path)):
          
                if path.endswith(".json"):
                 
                    with open(os.path.join(pathes, path)) as json_file:
                        data = json.load(json_file)
              
                        db.write_result_to_firestore(data.get("experiment_id"), 1, data)

