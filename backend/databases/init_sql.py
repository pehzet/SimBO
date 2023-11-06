import sqlite3
import json

connection = sqlite3.connect('SimBO.db')
cursor = connection.cursor()

def create_table(table_name, columns):
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})")
    connection.commit()
    print(f"Created table {table_name}")


experiment_columns ='''
experiment_id STRING,
experiment_name STRING,
replications INTEGER,
algorithm STRING,
use_case STRING,
created_at STRING,
last_updated_at STRING,
PRIMARY KEY (experiment_id)

'''

runtime_columns = '''
ID INTEGER PRIMARY KEY AUTOINCREMENT,
experiment_id STRING,
replication INTEGER,
trial INTEGER,
runtimes_gen STRING,
runtimes_fit STRING,
runtimes_eval STRING,
last_updated_at STRING,
FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
'''

lengthscale_columns = '''
ID INTEGER PRIMARY KEY AUTOINCREMENT,
experiment_id STRING,
replication INTEGER,
trial INTEGER,
lengthscales STRING,
last_updated_at STRING,
FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
'''

acq_values_columns = '''
ID INTEGER PRIMARY KEY AUTOINCREMENT,
experiment_id STRING,
replication INTEGER,
trial INTEGER,
acq_values STRING,
last_updated_at STRING,
FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
'''

x_and_y_columns = '''
ID INTEGER PRIMARY KEY AUTOINCREMENT,
experiment_id STRING,
replication INTEGER,
trial INTEGER,
x STRING,
y STRING,
last_updated_at STRING,
FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
'''

pareto_columns = '''
ID INTEGER PRIMARY KEY AUTOINCREMENT,
experiment_id STRING,
replication INTEGER,
trial INTEGER,
pareto_x STRING,
pareto_y STRING,
last_updated_at STRING,
FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
'''

hv_columns = '''
ID INTEGER PRIMARY KEY AUTOINCREMENT,
experiment_id STRING,
replication INTEGER,
trial INTEGER,
hv STRING,
last_updated_at STRING,
FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
'''


tables_to_create = [("experiments", experiment_columns), ("runtimes", runtime_columns), ("lengthscales", lengthscale_columns), ("acq_values", acq_values_columns), ("x_and_y", x_and_y_columns), ("pareto", pareto_columns), ("hv", hv_columns)]

if __name__ == "__main__":
    for table_name, columns in tables_to_create:
        create_table(table_name, columns)