

from unicodedata import name

from ax.storage.metric_registry import register_metric

from ax.storage.runner_registry import register_runner

import numpy as np
import json
import icecream
from datetime import datetime
import time

# Relativer Import der UseCases
import importlib.util
import sys

from icecream import ic
from ax import core

import copy

from ax import (
    ComparisonOp,
    ParameterType,
    RangeParameter,
    ChoiceParameter,
    FixedParameter,
    SearchSpace,
    Experiment,
    OutcomeConstraint,
    OrderConstraint,
    SumConstraint,
    OptimizationConfig,
    Objective,
    Metric,
)
from ax import Data
import pandas as pd
from ax import Runner
from ax import ObjectiveThreshold
from ax.modelbridge.registry import Models
from ax.modelbridge.factory import get_GPEI
from ax.modelbridge.cross_validation import cross_validate, compute_diagnostics
from ax.service.utils.best_point import get_best_raw_objective_point


# SaaSBO specific imports

from ax.core.metric import Metric
from ax.modelbridge.registry import Models
from ax.core.objective import Objective

from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
import torch
import pandas as pd

# relative Imports from other local files
from utils.ax_utils import load_experiment_from_json, save_experiment_as_json

def suggest_next_trial(experiment, algorithm_config:dict):
    numPoints = algorithm_config.get("batch_size")
    print("SAASBO SUGGEST TRIAL")
    
    # ic(vars(exp))
    # ic(exp.fetch_data())
    # These are hyperparameters that can be changed every batch.
    # If not specified for this trial, it will get the value of the general experiment or a default value

    num_samples = algorithm_config.get("number_samples", 256)
    warmup_steps = algorithm_config.get("warm_up_steps", 512)

    #https://github.com/pyro-ppl/pyro/issues/3018
    m = Models.FULLYBAYESIAN(
        experiment=experiment,
        data=experiment.fetch_data(),
        num_samples=num_samples,  # Increasing this may result in better model fits
        warmup_steps=warmup_steps,  # Increasing this may result in better model fits
        gp_kernel="rbf",  # "rbf" is the default in the paper, but we also support "matern"
        torch_device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"),
        torch_dtype=torch.double,
        verbose=True,  # Set to True to print stats from MCMC
        disable_progbar=False,  # Set to False to print a progress bar from MCMC
    )
    print(f"Model created. Going to Suggest {numPoints} Points")
    gr = m.gen(numPoints)
    
    if numPoints == 1:
        trial = experiment.new_trial(generator_run=gr)
    else:
        trial = experiment.new_batch_trial(generator_run=gr)
   
    return trial, experiment



def suggest_next_trial_2(experiment, algorithm_config:dict):
    numPoints = algorithm_config.get("batch_size")

    
    # ic(vars(exp))
    # ic(exp.fetch_data())
    # These are hyperparameters that can be changed every batch.
    # If not specified for this trial, it will get the value of the general experiment or a default value

    num_samples = algorithm_config.get("number_samples", 256)
    warmup_steps = algorithm_config.get("warm_up_steps", 512)

    #data = exp.fetch_data()

    # gr = generator run
    _retries = 0
    while _retries < 5:
        try:
            # on non positive error go to: C:\code\black-box-opt-simio-server\bboss-env\Lib\site-packages\ax\models\torch\botorch_defaults.py and change value of MIN_OBSERVED_NOISE_LEVEL higher (1e-3 or 1e-4)
            print("Try No.: " + str(_retries + 1))
            m = Models.FULLYBAYESIAN(
                experiment=experiment,
                data=experiment.fetch_data(),
                num_samples=num_samples,  # Increasing this may result in better model fits
                warmup_steps=warmup_steps,  # Increasing this may result in better model fits
                gp_kernel="rbf",  # "rbf" is the default in the paper, but we also support "matern"
                torch_device=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"),
                torch_dtype=torch.double,
                verbose=True,  # Set to True to print stats from MCMC
                disable_progbar=False,  # Set to False to print a progress bar from MCMC
            )
      
            gr = m.gen(numPoints)
            if numPoints == 1:
                trial = experiment.new_trial(generator_run=gr)
            else:
                trial = experiment.new_batch_trial(generator_run=gr)

            return trial, experiment
        except Exception as err:
            print(err)
            if _retries == 4:
                print(err)
                sys.exit("Error at SAASBO MODEL. Tried 5 times.")
            _retries += 1
    print("Error at SAASBO MODEL. Tried 5 times.")
    return None


def complete_trial(experiment_name, trial_index=-1):

    global responses
    exp, responses = load_experiment_from_json(experiment_name)

    if trial_index == -1:
        # get the last trial
        indices = []
        for k, v in exp.trials.items():
            indices.append(k)
        trial_index = indices[-1]

    trial = exp.get_trials_by_indices([trial_index])[0]

    trial.run()
    trial.mark_completed()
    data = exp.fetch_data()
    print(f"Completed Trial no. {trial_index}")
    save_experiment_as_json(exp, responses, experiment_name)




def get_model(experiment, algorithm_config):
 

    # ic(vars(exp))
    # ic(exp.fetch_data())
    # These are hyperparameters that can be changed every batch.
    # If not specified for this trial, it will get the value of the general experiment or a default value

    num_samples = algorithm_config.get("number_samples", 256)
    warmup_steps = algorithm_config.get("warm_up_steps", 512)
    #data = exp.fetch_data()
    # gr = generator run
    _retries = 0
    while _retries < 5:
        try:
            # on non positive error go to: C:\code\black-box-opt-simio-server\bboss-env\Lib\site-packages\ax\models\torch\botorch_defaults.py and change value of MIN_OBSERVED_NOISE_LEVEL higher (1e-3 or 1e-4)
            print("Try No.: " + str(_retries + 1))
            m = Models.FULLYBAYESIAN(
                experiment=experiment,
                data=experiment.fetch_data(),
                num_samples=num_samples,  # Increasing this may result in better model fits
                warmup_steps=warmup_steps,  # Increasing this may result in better model fits
                gp_kernel="rbf",  # "rbf" is the default in the paper, but we also support "matern"
                torch_device=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"),
                torch_dtype=torch.double,
                verbose=True,  # Set to True to print stats from MCMC
                disable_progbar=False,  # Set to False to print a progress bar from MCMC
            )

            return m
        except Exception as err:
            print(err)
            if _retries == 4:
                print(err)
                sys.exit("Error at SAASBO MODEL. Tried 5 times.")
            _retries += 1
    print("Error at SAASBO MODEL. Tried 5 times.")
    return None