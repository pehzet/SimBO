


from unicodedata import name
from urllib import response
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
import random
from icecream import ic
from ax import MultiObjective, MultiObjectiveOptimizationConfig, core

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
from ax.modelbridge.factory import get_MOO_EHVI
from ax.modelbridge.modelbridge_utils import observed_hypervolume
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
import torch
import pandas as pd

# relative Imports from other local files
from utils.storage import load_experiment_from_json, save_experiment_as_json


class Submetric(Metric):
    def fetch_trial_data(self, trial):
        records = []

        for arm_name, arm in trial.arms_by_name.items():

            mean, sem = self.get_raw_eval_data(arm_name)
            _obj = {
                "arm_name": arm_name,
                "metric_name": self.name,
                "trial_index": trial.index,
                "mean": mean,
                "sem": sem,
            }
            records.append(_obj)
        return Data(df=pd.DataFrame.from_records(records))

        # ic(pd.DataFrame.from_records(records))
    def is_available_while_running() -> bool:
        return True

    def get_raw_eval_data(self, arm_name):

        # unsure if trial_index or arm for identification
        # e = {"trial_index" : 1, "mean" : 10.1, "sem" : 0.1}

        mean = [e["mean"]
                for e in responses if int(e["arm_name"]) == int(arm_name)][0]
        sem = [e["sem"]
               for e in responses if int(e["arm_name"]) == int(arm_name)][0]

        return mean, sem


class SubRunner(Runner):
    def __init__(self, params):
        self.params = params

    def run(self, trial):
        for arm in trial.arms:
            name_to_params = {
                arm.name: arm.parameters for arm in trial.arms
            }
        run_metadata = {}
        return name_to_params
        # return run_metadata

    @property
    def staging_required(self):
        return False


# experiment_meta = {
#     "experiment_name": "aaaa",
#     "metric_name": "bbb",
#     "parameters": [{
#         "name": "p1",
#         "type": "Int",
#         "lb": 10,
#         "ub": 100,
#         "values": [],  # only if type = string
#         "fixed": False
#     }],
#     "parameter_contraints" : None # else [{}] or [str] - not sure yet
# }


def create_experiment(experiment):
    global responses
    responses = []
    ic(experiment)
    search_space, params = create_search_space_and_format_parameters(
        experiment)
    response_name = [r["name"]
                     for r in experiment["responses"] if r["is_primary"] == True][0]

    # TODO: If implemented correctly use function
    # optimization_config = OptimizationConfig(
    #     objective=Objective(metric=Submetric(
    #         name=response_name + "_Metric", lower_is_better=False), minimize=False)
    # )
    optimization_config = create_optimization_config(experiment)
    exp = Experiment(
        name=experiment["model_id"],
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SubRunner(params=params),
    )

    register_runner(SubRunner)
    register_metric(Submetric)
    save_experiment_as_json(exp, responses, experiment_name=experiment["model_id"],
                            path_präfix="")
    return exp

# TODO: Define Close Experiment Function which maybe starts the analysis
def create_search_space_and_format_parameters(experiment):
    parameters = experiment["parameters"]
    constraints = experiment["constraints"] if len(
        experiment["constraints"]) > 0 else None
    parameters_ax_dev = []
    # need to identify the Parameter Object by name later and dunno how to get a value of an unknwon Object easily, so i made an extra list
    # TODO: make better
    paramaters_maps = []
    type_map = {
        "int": ParameterType.INT,
        "integer": ParameterType.INT,
        "float": ParameterType.FLOAT,
        "string": ParameterType.STRING,
        "bool": ParameterType.BOOL,
        "continuous": ParameterType.FLOAT,
        "discrete": ParameterType.INT
    }
    for e in parameters:
        ic(e)
        if (e["type"] == "float" or e["type"] == "continuous") and e["fixed"] == False:
            _p = RangeParameter(
                name=e["name"],
                parameter_type=type_map[e["type"]],
                lower=float(e["min"]),
                upper=float(e["max"]),
            )
            parameters_ax_dev.append(_p)
            paramaters_maps.append({"name": e["name"], "parameter_object": _p})
        elif (e["type"] == "integer" or e["type"] == "int" or e["type"] == "discrete") and e["fixed"] == False:
            _p = RangeParameter(
                name=e["name"],
                parameter_type=type_map[e["type"]],
                lower=int(e["min"]),
                upper=int(e["max"]),
            )
            parameters_ax_dev.append(_p)
            paramaters_maps.append({"name": e["name"], "parameter_object": _p})
        elif e["type"] == "choice" and e["fixed"] == False:
            _p = ChoiceParameter(
                name=e["name"],
                parameter_type=type_map[e["type"]],
                values=e["values"],
            )

            parameters_ax_dev.append(_p)
            paramaters_maps.append({"name": e["name"], "parameter_object": _p})
        elif e["type"] == "bool" or e["type"] == "boolean" and e["fixed"] == False:
            _p = ChoiceParameter(
                name=e["name"],
                parameter_type=type_map[e["type"]],
                values=["true", "false"],
            )

            parameters_ax_dev.append(_p)
            paramaters_maps.append({"name": e["name"], "parameter_object": _p})
        elif e["fixed"] == True:
            _p = FixedParameter(
                name=e["name"],
                parameter_type=type_map[e["type"]],
                value=e["value"],
            )
            parameters_ax_dev.append(_p)
            paramaters_maps.append({"name": e["name"], "parameter_object": _p})
    if constraints == None:

        search_space = SearchSpace(parameters=parameters_ax_dev)
    else:

        _constraints = []
        for c in constraints:

            if c["type"] == "sum":
                _param_names = c["param_names"].replace(" ", "").split(",")
                _params = [
                    p for p in parameters_ax_dev if p.name in _param_names]

                _params = [p["parameter_object"]
                           for p in paramaters_maps if p["name"] in _param_names]

                _sum_constraint = SumConstraint(
                    parameters=_params,
                    is_upper_bound=True if c["is_upper_bound"].lower() in [
                        "true", "wahr", "1"] else False,
                    bound=float(c["bound"])
                )
                _constraints.append(_sum_constraint)
            elif c["type"] == "order":
                ic(c["upper_parameter_name"])
                ic(paramaters_maps)
                _order_constraint = OrderConstraint(
                    lower_parameter=[p["parameter_object"]
                                     for p in paramaters_maps if p["name"] == c["lower_parameter_name"]][0],
                    upper_parameter=[p["parameter_object"]
                                     for p in paramaters_maps if p["name"] == c["upper_parameter_name"]][0]
                )
                _constraints.append(_order_constraint)
            elif c["type"] == "linear":
                # TODO: Implement linear
                pass
        search_space = SearchSpace(
            parameters=parameters_ax_dev, parameter_constraints=_constraints)

    return search_space, parameters_ax_dev
def calc_dummy_threshold(experiment):
    ref_points = []
    num_metrics = len([r for r in experiment["responses"] if r["is_multi_objective"] == True])
    for i in range(num_metrics):
        _ref_points = []
        for p in experiment["parameters"]:
            if (p["type"] == "float" or p["type"] == "continuous" or p["type"] == "integer" or p["type"] == "int" or p["type"] == "discrete") and p["fixed"] == False:
                rp = random.randint(int(p["min"]), int(p["max"]))
                _ref_points.append(rp)
            elif p["fixed"] == True:
                rp = p["value"]
            elif p["type"] == "bool" or p["type"] == "boolean":
                rp = True
        ref_points.append(_ref_points)
    return ref_points


def create_optimization_config(experiment, objective_threshold = []):
    response_meta = experiment["responses"]
    # param_names = [p["name"] for p in model["parameters"]]
    _outcome_constraints = []
    _objectives = []
    if len(objective_threshold) == 0:
        _objective_threshold = calc_dummy_threshold(experiment)
    _objective_thresholds = objective_threshold 
    for r in response_meta:
        i = 0
        if r["is_multi_objective"] == True:
            _m = Submetric(
                    name=r["name"],
                    lower_is_better=r["minimize"]
                )
            _objectives.append(_m)
            # _objective_thresholds.append(ObjectiveThreshold(metric=_m, bound=r["ref_point"], relative=False))
            _objective_thresholds.append(ObjectiveThreshold(metric=_m, bound=_objective_threshold[i], relative=False))
            i +=1

            
        if r["lower_bound"] != '-Infinity':

            _outcome_constraint = OutcomeConstraint(
                metric=Submetric(
                    name=r["name"],
                ),
                op=ComparisonOp.GEQ,
                bound=r["lower_bound"],
                relative=False
            )
            _outcome_constraints.append(_outcome_constraint)
        elif r["upper_bound"] != 'Infinity':
            _outcome_constraint = OutcomeConstraint(
                metric=Submetric(
                    name=r["name"],
                ),
                op=ComparisonOp.LEQ,
                bound=r["upper_bound"],
                relative=False
            )

            _outcome_constraints.append(_outcome_constraint)
    mo = MultiObjective(
            objectives=_objectives
        )
        
    # optimizatio_config = OptimizationConfig(
    #     objective=_objective, outcome_constraints=_outcome_constraints)
    optimization_config= MultiObjectiveOptimizationConfig(objective=mo, objective_thresholds=_objective_thresholds, outcome_constraints=_outcome_constraints)
    return optimization_config


def suggest_initial_points(experiment_name, numPoints=1):
    global responses
    exp, responses = load_experiment_from_json(experiment_name)
    sobol = Models.SOBOL(search_space=exp.search_space, seed=1234)
    generator_run = sobol.gen(n=numPoints)
    trial = exp.new_trial(generator_run=generator_run)
    save_experiment_as_json(exp, responses, experiment_name)

    return trial


def suggest_next_trial(experiment_name, numPoints=1, **kwargs):
    global responses
    exp, responses = load_experiment_from_json(experiment_name)
    
    _retries = 0
    while _retries < 5:
        try:
            m = get_MOO_EHVI(experiment=exp, data=exp.fetch_data())
            gr = m.gen(numPoints)
            if numPoints == 1:
                trial = exp.new_trial(generator_run=gr)
            else:
                trial = exp.new_batch_trial(generator_run=gr)
            save_experiment_as_json(
                exp, responses, experiment_name=experiment_name)
            return trial
        except Exception as err:
            if _retries == 4:
                print(err)
                sys.exit("Error at QNEHVI MODEL. Tried 5 times.")
            _retries += 1
    #print("Error at QNEHVI MODEL. Tried 5 times.")
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
    save_experiment_as_json(exp, responses, experiment_name)
    print(f"Completed Trial no. {trial_index} ")
    save_experiment_as_json(exp, responses, experiment_name)

def init_analytics(experiment_name):
    register_metric(Submetric)
    register_runner(SubRunner)
    global responses
    exp, responses = load_experiment_from_json(experiment_name)
