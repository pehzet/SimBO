
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
    if len(sys.argv) > 1:
        try:
            manager_id = int(sys.argv[1])
        except:
            raise Exception("First argument must be Manager ID (int)")
    else:
        manager_id = None
    exp_manager = database.get_experiment_manager(manager_id=manager_id)

    mp.Process(target=log_gpu_usage).start() 

    EM = ExperimentManager(manager_id, exp_manager.get("default_interval", 30), db = database)
    EM.run()