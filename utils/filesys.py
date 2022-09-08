import os
import shutil


def move_experiment_json_file(experiment_name, source="open" ,destination="closed"):

    fname = experiment_name + ".json"
    source_path = os.path.join("..","data",source,fname)
    destination_path = os.path.join("..","data",destination,fname)
    if os.path.exists(source_path):
        shutil.move(source_path, destination_path)
        print(f"Moved file from {source_path} to {destination_path}")

