import sqlite3
import json
import datetime
import csv
import config
from google.oauth2 import service_account
from google.cloud import storage
from google.cloud import bigquery
import os
import logging
import traceback
import numpy as np
from torch import Tensor
from icecream import ic
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.GCLOUD_SERVICE_ACCOUNT
logger = logging.getLogger("database")
DB_NAME = config.DB_NAME
class SQLManager():
    def __init__(self):
        db_path = os.path.join(os.path.dirname(__file__), DB_NAME)
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self.all_table_names = ["experiments", "runtimes", "lengthscales", "acq_values"]
        # self.creds = service_account.Credentials.from_service_account_file(
        #     config.GCLOUD_SERVICE_ACCOUNT
        # )
    def create_table(self, table_name, columns):
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})")
        self.connection.commit()
    
    def insert_experiment(self, experiment_id, experiment_name, replications, algorithm, use_case, created_at):
        query = """
        INSERT OR IGNORE INTO experiments (experiment_id, experiment_name, replications, algorithm, use_case, created_at, last_updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        last_updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(query, (experiment_id, experiment_name, replications, algorithm, use_case, created_at, last_updated_at))
        self.connection.commit()

    def insert_runtime(self, experiment_id, replication, trial, runtimes_gen, runtimes_fit, runtimes_eval):
        query = """
        INSERT INTO runtimes (experiment_id, replication, trial, runtimes_gen, runtimes_fit, runtimes_eval, last_updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        runtimes_gen = json.dumps(runtimes_gen)
        runtimes_fit = json.dumps(runtimes_fit)
        runtimes_eval = json.dumps(runtimes_eval)
        last_updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(query, (experiment_id, replication, trial, runtimes_gen, runtimes_fit, runtimes_eval, last_updated_at))
        self.connection.commit()

    def insert_lengthscale(self, experiment_id, replication, trial, lengthscales):
        query = """
        INSERT INTO lengthscales (experiment_id, replication, trial, lengthscales, last_updated_at)
        VALUES (?, ?, ?, ?, ?)
        """

        if isinstance(lengthscales, np.ndarray):
            lengthscales = lengthscales.tolist()

        lengthscales = json.dumps(lengthscales)
        last_updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(query, (experiment_id, replication, trial, lengthscales, last_updated_at))
        self.connection.commit()

    
    def insert_acq_values(self, experiment_id, replication, trial, acq_values):
        query = """
        INSERT INTO acq_values (experiment_id, replication, trial, acq_values, last_updated_at)
        VALUES (?, ?, ?, ?, ?)
        """
        if isinstance(acq_values, np.ndarray):
            acq_values = acq_values.tolist()
        acq_values = json.dumps(acq_values)
        last_updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(query, (experiment_id, replication, trial, acq_values, last_updated_at))
        self.connection.commit()
    
    def insert_x_and_y(self, experiment_id, replication, trial, X, Y):
        query = """
        INSERT INTO x_and_y (experiment_id, replication, trial, x, y, last_updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        if isinstance(X, (np.ndarray,Tensor)):
            X = X.tolist()
        if isinstance(Y, (np.ndarray,Tensor)):
            Y = Y.tolist()
        X = json.dumps(X)
        Y = json.dumps(Y)
        last_updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(query, (experiment_id, replication, trial, X, Y, last_updated_at))
        self.connection.commit()
    
    def insert_pareto(self, experiment_id, replication, trial, pareto_X, pareto_Y):
        query = """
        INSERT INTO pareto (experiment_id, replication, trial, pareto_x, pareto_y, last_updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        if isinstance(pareto_X, (np.ndarray,Tensor)):
            pareto_X = pareto_X.tolist()
        if isinstance(pareto_Y, (np.ndarray,Tensor)):
            pareto_Y = pareto_Y.tolist()

        pareto_X = json.dumps(pareto_X)
        pareto_Y = json.dumps(pareto_Y)
        last_updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(query, (experiment_id, replication, trial, pareto_X, pareto_Y, last_updated_at))
        self.connection.commit()

    def insert_hv(self, experiment_id, replication, trial, hv):
        query = """
        INSERT INTO hv (experiment_id, replication, trial, hv, last_updated_at)
        VALUES (?, ?, ?, ?, ?)
        """
        last_updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(query, (experiment_id, replication, trial, hv, last_updated_at))
        self.connection.commit()

    def close_connection(self):
        self.connection.close()

    def _extract_data_to_csv(self, table_name: str, last_updated_at: str):
        query = f"""
        SELECT * FROM {table_name}
        WHERE last_updated_at > ?
        """
        self.cursor.execute(query, (last_updated_at,))
        data = self.cursor.fetchall()
        # Write data to csv
        with open(f'{table_name}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([i[0] for i in self.cursor.description])  # Write headers
            writer.writerows(data)

    def _upload_to_gcs(self, bucket_name, source_file_path, destination_blob_name):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_path)

    def _load_data_from_gcs_to_bigquery(self,dataset_id, table_id, gcs_path):
        bigquery_client = bigquery.Client()
        dataset_ref = bigquery_client.dataset(dataset_id)
        job_config = bigquery.LoadJobConfig(autodetect=True)

        job_config.source_format = bigquery.SourceFormat.CSV
        job_config.skip_leading_rows = 1
        job_config.autodetect = True

        load_job = bigquery_client.load_table_from_uri(
            gcs_path, dataset_ref.table(table_id), job_config=job_config
        )
        load_job.result()

    def send_local_database_to_bigquery(self):
        lua_path = os.path.join(os.path.dirname(__file__), 'last_updated_at.txt')
        if not os.path.exists(lua_path):
            with open(lua_path, 'w') as f:
                f.write("2020-01-01 00:00:00")
        with open(lua_path, 'r') as f:
            last_updated_at = f.read()
        try:
            bucket_name = "simbo-data"
            dataset_id = "simbo_data"
            for table_name in self.all_table_names:
                    self._extract_data_to_csv(table_name, last_updated_at)
                    # Upload csv files to GCS
                    source_file_path = f"{table_name}.csv"
                    destination_blob_name = f"{table_name}.csv"
                    self._upload_to_gcs(bucket_name, source_file_path, destination_blob_name)
                    # Load data from GCS to BigQuery
                    table_id = table_name
                    gcs_path = "gs://simbo-data/"
                    gcs_path += f"{table_name}.csv"
                    self._load_data_from_gcs_to_bigquery(dataset_id, table_id, gcs_path)
                    # Update last_updated_at.txt
            with open(lua_path, 'w') as f:
                f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            logger.info(f"Successfully sent data to Bigquery")
        except Exception as e:
            logger.error(f"Error in SQLite to Bigquery: {e}")
            traceback.print_exc()
        self.send_db_file_to_storage()
    
    def send_db_file_to_storage(self):
        try:
            bucket_name = "simbo-data"
            source_file_path = os.path.join(os.path.dirname(__file__), DB_NAME)
            destination_blob_name = DB_NAME
            self._upload_to_gcs(bucket_name, source_file_path, destination_blob_name)
            logger.info(f"Successfully sent db file to GCS")
        except Exception as e:
            logger.error(f"Error in SQLite to GCS: {e}")
            traceback.print_exc()

    def get_db_file_from_storage(self):
        try:
            bucket_name = "simbo-data"
            source_blob_name = DB_NAME
            destination_file_name = os.path.join(os.path.dirname(__file__), DB_NAME)
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_name)
            logger.info(f"Successfully downloaded db file from GCS")
        except Exception as e:
            logger.error(f"Error in GCS to SQLite: {e}")
            traceback.print_exc()
    # def send_sql_db_to_storage(self):
    #     if not db_name.endswith(".db"):
    #         db_name += ".db"
    #     bucket_name = "simbo-data"
    #     # source_file_path = os.path.join(self.main_dir, db_name)
    #     source_file_path = os.path.join(os.path.dirname(__file__), db_name)
    #     destination_blob_name = db_name
    #     self._upload_to_gcs(bucket_name, source_file_path, destination_blob_name)
    #     logger.info(f"Successfully sent db file to GCS")