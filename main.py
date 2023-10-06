
import torch.multiprocessing as mp
import sys
import os

from icecream import ic

if __name__ == "__main__":
    # Get the absolute path of the project root directory
    project_root = os.path.abspath(os.path.dirname(__file__))
    # Check if the project root is in sys.path, if not, append it
    if project_root not in sys.path:
        sys.path.append(project_root)
    from backend.manager.manager import ExperimentManager, log_gpu_usage
    from backend.databases.firebase import FirebaseManager
    database = FirebaseManager(project_root)
    exp_manager = database.get_experiment_manager()

    mp.Process(target=log_gpu_usage).start() 

    EM = ExperimentManager(exp_manager.get("id",-1), exp_manager.get("default_interval", 30), db = database)
    EM.run()