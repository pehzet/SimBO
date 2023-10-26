
import torch.multiprocessing as mp
import sys
import os
from backend.manager.manager import ExperimentManager, log_gpu_usage
from backend.databases.firebase import FirebaseManager
from icecream import ic
import os
import subprocess
import psutil

def get_memory():
    try:
        mem_info = psutil.virtual_memory()
        
        # Convert bytes to GB
        total_memory_gb = mem_info.total / (1024 ** 3)
        return total_memory_gb
    except Exception as e:
        print(f"Error checking memory: {e}")
        return None

def get_gpu_memory():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        total_gpu_memory = float(result.stdout.strip())
        return total_gpu_memory
    except Exception as e:
        print(f"Error checking GPU memory: {e}")
        return None


def get_gpu_type():
    try:
        # Querying GPU name using nvidia-smi
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        gpu_name = result.stdout.strip()

        # Checking if MIG mode is enabled
        mig_result = subprocess.run(["nvidia-smi", "--query-gpu=mig.mode.current", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        mig_mode = mig_result.stdout.strip()

        # If MIG mode is enabled, append it to the GPU name
        if mig_mode == "Enabled":
            gpu_name += " (MIG enabled)"

        return gpu_name

    except Exception as e:
        print(f"Error retrieving GPU type: {e}")
        return None
  

if __name__ == "__main__":
    # This is for debugging purposes
    num_cpus = os.cpu_count()
    memory_gb = get_memory()
    gpu_name = get_gpu_type()
    gpu_memory_gb = get_gpu_memory()
    
    print(f"OS Type: {os.name}")
    print(f"Number of CPUs: {num_cpus}")
    print(f"RAM (in GB): {memory_gb}")
    if gpu_name is not None:
        print(f"GPU Type: {gpu_name}")
    else:
        print("Couldn't retrieve GPU information. Ensure NVIDIA drivers are installed and nvidia-smi is accessible.")
    if gpu_memory_gb is not None:
        print(f"GPU Memory (in GB): {gpu_memory_gb:.2f}")
    else:
        print("Couldn't retrieve GPU information. Ensure NVIDIA drivers are installed and nvidia-smi is accessible.")
    # Get the absolute path of the project root directory
    project_root = os.path.abspath(os.path.dirname(__file__))
    # Check if the project root is in sys.path, if not, append it
    if project_root not in sys.path:
        sys.path.append(project_root)

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

    EM = ExperimentManager(manager_id, exp_manager.get("default_interval", 30), main_dir=project_root, db = database)
    EM.run()