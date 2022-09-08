# Main File for Cluster Experiments
from locale import normalize
import sys
import json
from algorithms.turbo.turbo_botorch import TurboRunner
from algorithms.saasbo.saasbo_botorch import SaasboRunner
from algorithms.gpei.gpei_botorch import GPEIRunner
from use_cases.mrp.mrp_runner import get_param_meta_from_materials, run_solver, init_sheets, get_param_meta, init_mrp_runner
from use_cases.mrp.mrp_sim import mrp_simulation, init_mrp_sim #run_simulation
#from utils.ax_utils import *

from icecream import ic
import time
if len(sys.argv) < 1 :
    print("Please provide experiment ID")
    sys.exit()
else:
    experiment_id = sys.argv[1]

# Dummy class for global variables
class g:
    pass

def get_experiment_config(experiment_id):
    fpath = "configs/config" + str(experiment_id) +".json"
    with open(fpath, 'r') as file:
        config = json.load(file)
    print("Config loaded")
    return config

def get_algorithm(algorithm_config:dict, param_meta:dict):
    algorithm = algorithm_config.get("strategy", algorithm_config.get("algorithm")).lower()

    if algorithm == "turbo":
        num_init = algorithm_config.get("n_init", algorithm_config.get("num_init"))
        batch_size = algorithm_config.get("batch_size")
        num_batches = algorithm_config.get("num_batches")
        g.num_batches = num_batches
        return TurboRunner(len(param_meta),batch_size, num_init, param_meta=param_meta)

    if algorithm == "gpei":
        num_init = algorithm_config.get("n_init", algorithm_config.get("num_init"))
        batch_size = algorithm_config.get("batch_size")
        num_batches = algorithm_config.get("num_batches")
        g.num_batches = num_batches
        return GPEIRunner(len(param_meta),batch_size, num_init, param_meta=param_meta)

    if algorithm == "saasbo":
        num_batches = algorithm_config.get("num_batches")
        g.num_batches = num_batches
        warmup_steps = algorithm_config.get("warmup_steps", 512)
        num_samples = algorithm_config.get("num_samples", 256)
        thinning = algorithm_config.get("thinning", 16)
        batch_size = algorithm_config.get("batch_size")
        num_init = algorithm_config.get("n_init", algorithm_config.get("num_init"))
        return SaasboRunner(len(param_meta), num_init=num_init, batch_size=batch_size, warmup_steps=warmup_steps,num_samples=num_samples, thinning=thinning, param_meta=param_meta)

def init_mrp_experiment(use_case_config):
    bom_id = use_case_config.get("bom_id")

    bom, materials, orders, inventory = init_sheets(bom_id)
    init_mrp_runner(bom_id, bom, materials, orders, inventory)
    init_mrp_sim(bom, materials, orders)
    param_meta = get_param_meta_from_materials(materials) # param meta is min, max and value type. Last one is constant in our case
    return param_meta

def run_optimization_loop(experiment_id):
    _start = time.monotonic()
  
    config = get_experiment_config(experiment_id)
    param_meta = init_mrp_experiment(config.get("use_case_config"))
  
    runner = get_algorithm(config.get("algorithm_config"),param_meta)
    runner.experiment_id = experiment_id
    x = runner.suggest_initial()

    x = runner.format_x_for_mrp(x)
    releases =[run_solver(xx) for xx in x]

    y = [mrp_simulation().run_simulation(release) for release in releases]
    y = runner.format_y_from_mrp(y)
 
    runner.complete(y)

    for _ in range(g.num_batches):
        _start_batch = time.monotonic()
        x = runner.suggest()
        x = runner.format_x_for_mrp(x)
        assert len(x) > 0
        if len(x[-1]) > 1:
            releases =[run_solver(xx) for xx in x]
            y = [mrp_simulation().run_simulation(release) for release in releases]
        else:
            releases = run_solver(x[0])
            y = mrp_simulation().run_simulation(releases)
        y = runner.format_y_from_mrp(y)
        runner.complete(y)
        _end_batch = time.monotonic()
        runner.batch_runtimes.append((_start_batch - _end_batch))
    _end = time.monotonic()
    runner.total_runtime = _start - _end
    runner.terminate_experiment(experiment_id)
    
if __name__ == "__main__":
    run_optimization_loop(experiment_id)


