# Main File for Cluster Experiments

# Logging and warnings

import warnings
import logging

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s: %(levelname)s: %(name)s: %(message)s'
    )

self.logger = logging.getLogger("main")

# Surpress PyTorch warning
warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is") 
warnings.filterwarnings("ignore", message="Could not import matplotlib.pyplot")
warnings.filterwarnings("ignore", message="torch.triangular_solve is deprecated") 


import sys
from experiment_runners.experiment_runner_algorithm_driven import ExperimentRunnerAlgorithmDriven
from experiment_runners.experiment_runner_simulation_driven import ExperimentRunnerSimulationDriven

from utils.gsheet_utils import get_configs_from_gsheet, get_experiment_runner_type
import time
import torch
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

from icecream import ic


def check_sysargs():
    if "load" in sys.argv:
        self.logger.info("Argument 'load' found. Getting info from Google Spreadsheet.")
        get_configs_from_gsheet(from_main=True)
        if len(sys.argv) <= 2 :
            print("No experiment ID detected. Re-loaded only config files. Going to exit")
            sys.exit()
        sys.argv.remove("load")
    if len(sys.argv) < 2 :
        print("Please provide experiment ID")
        sys.exit()

    experiment_id = sys.argv[1]
    replication = sys.argv[2] if len(sys.argv) >= 3 else 0
    return experiment_id, replication 

if __name__ == "__main__":
    experiment_id, replication = check_sysargs()
    try:
        runner_type = get_experiment_runner_type(experiment_id)
    except:
        get_configs_from_gsheet(from_main=True)
        try:
            runner_type = get_experiment_runner_type(experiment_id)
        except:
            print(f"Runner Type of experiment {experiment_id} not detected at configs. Going to exit")
        sys.exit()
    if runner_type == "simulation":
        ExperimentRunnerSimulationDriven(experiment_id, replication)
    elif runner_type == "algorithm":
        ExperimentRunnerAlgorithmDriven(experiment_id, replication).run_optimization_loop()
    else:
        print(f"Runner Type of experiment {experiment_id} not identified. Maybe typo at gsheet. Going to exit")
        sys.exit()


