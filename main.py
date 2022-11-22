# Main File for Cluster Experiments

# Logging and warnings
from genericpath import isfile
import warnings
import logging
from numpy import NaN

from sqlalchemy import false
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s: %(levelname)s: %(name)s: %(message)s'
    )

logger = logging.getLogger("main")

# Surpress PyTorch warning
warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is") 
warnings.filterwarnings("ignore", message="Could not import matplotlib.pyplot")
warnings.filterwarnings("ignore", message="torch.triangular_solve is deprecated") 


import sys
from experiment_runners.experiment_runner_algorithm_driven import ExperimentRunnerAlgorithmDriven
from experiment_runners.experiment_runner_simulation_driven import ExperimentRunnerSimulationDriven

from utils.gsheet_utils import get_configs_from_gsheet
import time
import torch
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

from icecream import ic


def check_sysargs():
    if "load" in sys.argv:
        logger.info("Argument 'load' found. Getting info from Google Spreadsheet.")
        get_configs_from_gsheet(from_main=True)
        if len(sys.argv) <= 2 :
            print("No experiment ID detected. Re-loaded only config files. Going to exit")
            sys.exit()
        sys.argv.remove("load")
    if len(sys.argv) < 2 :
        print("Please provide experiment ID")
        sys.exit()
    server = False
    if "server" in sys.argv:
        server = True
        sys.argv.remove("server")
    experiment_id = sys.argv[1]
    replication = sys.argv[2] if len(sys.argv) >= 3 else 0
    return experiment_id, replication, server

if __name__ == "__main__":
    experiment_id, replication, server = check_sysargs()
    if server:
        ExperimentRunnerSimulationDriven(experiment_id, replication)
    else:
        ExperimentRunnerAlgorithmDriven(experiment_id, replication).run_optimization_loop()


