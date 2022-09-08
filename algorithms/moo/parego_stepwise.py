from ax.core import optimization_config
from ax.service.ax_client import AxClient

from ax.modelbridge.factory import get_MOO_EHVI, get_MOO_PAREGO
from ax.modelbridge.modelbridge_utils import observed_hypervolume

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
from ax.benchmark.benchmark import full_benchmark_run
from ax.benchmark.benchmark_result import aggregate_problem_results, BenchmarkResult
from ax.core.metric import Metric
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner

from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.branin import BraninMetric
from ax.benchmark.benchmark_problem import BenchmarkProblem

# saasbo-qnehvi
from ax import MultiObjectiveOptimizationConfig
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from ax.service.utils.report_utils import exp_to_df


from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment

import torch
import pandas as pd


class qnehvi_saasbo:
    def __init__(self, **kwargs):
        self.type_map = {
            "int": ParameterType.INT,
            "integer": ParameterType.INT,
            "float": ParameterType.FLOAT,
            "string": ParameterType.STRING,
            "bool": ParameterType.BOOL,
        }
        # check essential args:
        essential_args = [
            "experiment_name",
            "parameters",
            "metrics_metainformations",
            "ref_point",
            "minimize",
            "optimization_model",
        ]
        for e in essential_args:
            if e not in kwargs.keys():
                sys.exit(f"Missing Argument: {e}")

        self.__dict__.update(kwargs)
        self.hv_list = []
        
        global responses
        responses = []
        if not "experiment_file_name" in self.__dict__.keys():
            current_datetime = datetime.now()
            self.experiment_file_name = current_datetime.strftime("%Y-%m-%d_%H-%M-%S") + "_qnehvi-saasbo"







    def create_search_space(self):
        self.parameters_ax_dev = []
        for e in self.parameters:
            if e["type"] == "float" and e["fixed"] == False:
                self.parameters_ax_dev.append(
                    RangeParameter(
                        name=e["name"],
                        parameter_type=self.type_map[e["type"]],
                        lower=e["lb"],
                        upper=e["ub"],
                    )
                )
            elif (e["type"] == "integer" or e["type"] == "int") and e["fixed"] == False:
                self.parameters_ax_dev.append(
                    RangeParameter(
                        name=e["name"],
                        parameter_type=self.type_map[e["type"]],
                        lower=e["lb"],
                        upper=e["ub"],
                    )
                )
            elif e["type"] == "string" and e["fixed"] == False:
                self.parameters_ax_dev.append(
                    ChoiceParameter(
                        name=e["name"],
                        parameter_type=self.type_map[e["type"]],
                        values=e["values"],
                    )
                )

            elif e["fixed"] == True:
                self.parameters_ax_dev.append(
                    FixedParameter(
                        name=e["name"],
                        parameter_type=self.type_map[e["type"]],
                        value=e["value"],
                    )
                )

        if "sum_constraint" in dir(self):
            pass
        if "order_constraint" in dir(self):
            pass

        self.search_space = SearchSpace(parameters=self.parameters_ax_dev)

        return self.search_space

    def create_optimization_config_moo(self):
        _constraints = None
        _objectives = []
        self.metric_names = []
        for m in self.metrics_metainformations:
            _metric = Submetric(m.get("name"),m.get("minimize"))
            self.metric_names.append(m.get("name"))
            _objectives.append(Objective(metric=_metric))

        mo = MultiObjective(objectives=_objectives)

        objective_thresholds = [
        ObjectiveThreshold(metric=metric, bound=val, relative=False)
        for metric, val in zip(mo.metrics, self.ref_point)
        ]
        self.optimization_config = MultiObjectiveOptimizationConfig(
        objective=mo,
        objective_thresholds=objective_thresholds,
        )

        return self.optimization_config

    def create_experiment(self):
        self.exp = Experiment(
            name=self.experiment_name,
            search_space=self.create_search_space(),
            optimization_config=self.create_optimization_config_moo(),
            runner=SubRunner(self.parameters_ax_dev),
        )
        self.save_experiment_as_json(self.experiment_file_name)
        return self.exp

    def save_experiment_as_json(self, name, präfix = "data/"):
        _name = präfix + str(name) + ".json"
        register_runner(SubRunner)
        register_metric(Submetric)
        save_experiment(self.exp, _name)

        #save responses as well
        _resp_name = präfix + str(name) + "_responses.json"
        if len(responses) == 0:
            print("Experiment saved")
            return

        try:
            with open(_resp_name, "ab") as file:
                np.save(file, np.array(responses),allow_pickle=True)
        except:
            with open(_resp_name, "wb") as file:
                np.save(file, np.array(responses),allow_pickle=True)

        print("Experiment and Responses saved")

    def load_experiment_from_json(self, name, präfix = "data/"):
        _name = präfix + str(name) + ".json"
        self.exp = load_experiment(_name)
        _resp_name = präfix + str(name) + "_responses.json"
        global responses
        if len(responses) == 0:
            print("Experiment saved")
            return self.exp
        with open(_resp_name, "rb") as file:
            _loaded_responses = np.load(file, allow_pickle=True)
        _responses = []
        for lr in _loaded_responses:

            #_lr = json.loads(lr)
            _lr = lr
            _responses.append(_lr)
        responses = _responses

        print("Experiment and Responses loaded")
        return self.exp

    def suggest_initial_points(self, numPoints=1):
        self.exp = self.load_experiment_from_json(self.experiment_file_name)
        sobol = Models.SOBOL(search_space=self.exp.search_space)
        generator_run = sobol.gen(n=numPoints)
        trial = self.exp.new_trial(generator_run=generator_run)
        return trial

    def suggest_next_batch(self, **kwargs):
        self.load_experiment_from_json(self.experiment_file_name)

        if "batch_size" in kwargs.keys():
            self.batch_size = kwargs.get("batch_size")
        else:
            self.batch_size = self.__dict__.get("batch_size", 10)
        

        #data = self.exp.fetch_data()

        #gr = generator run
        try:
            self.m = get_MOO_PAREGO(
                experiment=self.exp,
                data=self.exp.fetch_data()
            )
            gr = self.m.gen(self.batch_size)
            trials = self.exp.new_batch_trial(generator_run=gr)
        except RuntimeError as err:
            print(err)
            self.m = get_MOO_PAREGO(
                experiment=self.exp,
                data=self.exp.fetch_data()
            )
            gr = self.m.gen(self.batch_size)
            trials = self.exp.new_batch_trial(generator_run=gr)
            #https://github.com/pytorch/pytorch/issues/64818
        global current_trials
        current_trials = trials
        self.current_trials = trials
        return trials

    def complete_trial(self, trials = None):
        if trials is not None:
            _trials = trials
        else:
            #_trials = current_trials
            _trials = self.current_trials
        if isinstance(_trials["trials"], list):

            for t in _trials["trials"]:
    
                _trial = self.exp.get_trials_by_indices([t.get("index")])[0]
                _trial.run()
                _trial.mark_completed()
        else:

            _trials["trials"].run()
            _trials["trials"].mark_completed()



        #data = Data.from_multiple_data([self.exp.fetch_data(), _trials.fetch_data()])
        exp_df = exp_to_df(self.exp)
        outcomes = torch.tensor(exp_df[self.metric_names].values, **self.tkwargs)
        _ref_points = torch.tensor(self.ref_point, **self.tkwargs)
        partitioning = DominatedPartitioning(ref_point=_ref_points, Y=outcomes)
        try:
            hv = partitioning.compute_hypervolume().item()
        except:
            hv = 0
            print("Failed to compute hv")
        self.hv_list.append(hv)
        print(f"Hypervolume: {hv}")

        self.save_experiment_as_json(self.experiment_file_name)

        return True
        
    def complete_trial_moo(self, trial_index):
        pass



    def cross_validation(self):
        cv = cross_validate(self.m)
        diagnostics = compute_diagnostics(cv)

        return diagnostics

    def get_trial_data_list(self, params_as_columns=False):

        _objList = []
        for k, v in self.exp.trials.items():
            _obj = {}
            _obj["trial_index"] = k
            if params_as_columns == True:
                for name, value in v.arm.parameters.items():
                    _obj[name] = value
            else:
                _obj["parameters"] = v.arm.parameters
            for i, vv in self.exp.fetch_data().df.iterrows():
                vvd = vv.to_dict()
                if vv.trial_index == k:
                    _obj["mean"] = vvd["mean"]
                    _obj["sem"] = vvd["sem"]

            _objList.append(_obj)

        return _objList

    def get_best_point(self):
        return get_best_raw_objective_point(self.exp, self.optimization_config)

    def save_obj_as_json(self, obj, filename):
        # json.dumps(obj)
        f_name = str(filename) + ".json"
        try:
            with open(f_name, "w") as f:
                json.dump(obj, f)
        except:
            print(f"Error at saving Object to {f_name}")
            return

    def save_best_point_as_json(self, best_point, filename):
        f_name = str(filename) + ".json"

        _bp = {}
        _bp["parameters"] = {}
        _bp["response"] = {}

        for p_name, p_value in best_point[0].items():
            _bp["parameters"][p_name] = p_value

        for r_name, r_value in best_point[1].items():
            # r_name_mean = (r_name + "_mean")
            _bp["response"][(r_name + "_mean")] = r_value[0]
            _bp["response"][(r_name + "_sem")] = r_value[1]

        # make util function for adding same meta data in every function
        _bp["meta"] = {}
        _bp["meta"]["ts"] = str(datetime.now())
        _bp["meta"]["user"] = "Philipp"

        try:
            with open(f_name, "w") as f:
                json.dump(_bp, f)
        except:
            print(f"Error at saving Object to {f_name}")
            return

    def add_dummy_responses(self, i):
        data = [
        {
            "trial_index": 0,
            "arm_name": "0_0",
            "metric" : "metric_a",
            "mean" : 10.5,
            "sem" : 1.0
        },
        {
            "trial_index": 1,
            "arm_name": "1_0",
            "metric" : "metric_a",
            "mean" : 15.5,
            "sem" : 1.0
        },
                {
            "trial_index": 2,
            "arm_name": "2_0",
            "metric" : "metric_a",
            "mean" : 20.5,
            "sem" : 1.0
        },
        {
            "trial_index": 3,
            "arm_name": "3_0",
            "metric" : "metric_a",
            "mean" : 25.5,
            "sem" : 1.0
        },
        {
            "trial_index": 3,
            "arm_name": "3_1",
            "metric" : "metric_a",
            "mean" : 16.5,
            "sem" : 1.0
        },
                {
            "trial_index": 3,
            "arm_name": "3_2",
            "metric" : "metric_a",
            "mean" : 25.5,
            "sem" : 1.0
        },
                {
            "trial_index": 4,
            "arm_name": "4_0",
            "metric" : "metric_a",
            "mean" : 12.5,
            "sem" : 1.0
        },
                {
            "trial_index": 4,
            "arm_name": "4_1",
            "metric" : "metric_a",
            "mean" : 30.5,
            "sem" : 1.0
        },
                {
            "trial_index": 4,
            "arm_name": "4_2",
            "metric" : "metric_a",
            "mean" : 33.5,
            "sem" : 1.0
        },
               {
            "trial_index": 0,
            "arm_name": "0_0",
            "metric" : "metric_b",
            "mean" : 7.3,
            "sem" : 1.0
        },
        {
            "trial_index": 1,
            "arm_name": "1_0",
            "metric" : "metric_b",
            "mean" : 12.1,
            "sem" : 1.0
        },
                {
            "trial_index": 2,
            "arm_name": "2_0",
            "metric" : "metric_b",
            "mean" : 18.2,
            "sem" : 1.0
        },
        {
            "trial_index": 3,
            "arm_name": "3_0",
            "metric" : "metric_b",
            "mean" : 23.7,
            "sem" : 1.0
        },
        {
            "trial_index": 3,
            "arm_name": "3_1",
            "metric" : "metric_b",
            "mean" : 14.15,
            "sem" : 1.0
        },
                {
            "trial_index": 3,
            "arm_name": "3_2",
            "metric" : "metric_b",
            "mean" : 22.52,
            "sem" : 1.0
        },
                {
            "trial_index": 4,
            "arm_name": "4_0",
            "metric" : "metric_b",
            "mean" : 11.8,
            "sem" : 1.0
        },
                {
            "trial_index": 4,
            "arm_name": "4_1",
            "metric" : "metric_b",
            "mean" : 28.1,
            "sem" : 1.0
        },
                {
            "trial_index": 4,
            "arm_name": "4_2",
            "metric" : "metric_b",
            "mean" : 39.1,
            "sem" : 1.0
        },

        ]
        global responses
        responses = data
        return True
    def add_response(self, resp):
        responses.append(resp)
        print("Response added to global list")



class Submetric(Metric):
    def fetch_trial_data(self, trial):
        records = []

        for arm_name, arm in trial.arms_by_name.items():
  
            mean, sem = self.get_raw_eval_data(arm_name, self.name)
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
    
    def get_raw_eval_data(self, arm_name, metric_name):
        f = open("eval_data.json")
        #eval_data = json.load(f)
        eval_data = responses
        # unsure if trial_index or arm for identification
        # e = {"trial_index" : 1, "mean" : 10.1, "sem" : 0.1}

        mean = [e["mean"] for e in eval_data if int(e["arm_name"]) == int(arm_name) and e["metric"] == metric_name ][0]
        sem = [e["sem"] for e in eval_data if int(e["arm_name"]) == int(arm_name) and e["metric"] == metric_name ][0]
        f.close()
        return mean, sem

def create_dummy_eval_data():
    data = [
        {
            "trial_index": 0,
            "arm_name": "0_0",
            "metric" : "metric_a",
            "mean" : 10.5,
            "sem" : 1.0
        },
        {
            "trial_index": 1,
            "arm_name": "1_0",
            "metric" : "metric_a",
            "mean" : 15.5,
            "sem" : 1.0
        },
                {
            "trial_index": 2,
            "arm_name": "2_0",
            "metric" : "metric_a",
            "mean" : 20.5,
            "sem" : 1.0
        },
        {
            "trial_index": 3,
            "arm_name": "3_0",
            "metric" : "metric_a",
            "mean" : 25.5,
            "sem" : 1.0
        },
        {
            "trial_index": 3,
            "arm_name": "3_1",
            "metric" : "metric_a",
            "mean" : 16.5,
            "sem" : 1.0
        },
                {
            "trial_index": 3,
            "arm_name": "3_2",
            "metric" : "metric_a",
            "mean" : 25.5,
            "sem" : 1.0
        },
                {
            "trial_index": 4,
            "arm_name": "4_0",
            "metric" : "metric_a",
            "mean" : 12.5,
            "sem" : 1.0
        },
                {
            "trial_index": 4,
            "arm_name": "4_1",
            "metric" : "metric_a",
            "mean" : 30.5,
            "sem" : 1.0
        },
                {
            "trial_index": 4,
            "arm_name": "4_2",
            "metric" : "metric_a",
            "mean" : 33.5,
            "sem" : 1.0
        },
               {
            "trial_index": 0,
            "arm_name": "0_0",
            "metric" : "metric_b",
            "mean" : 7.3,
            "sem" : 1.0
        },
        {
            "trial_index": 1,
            "arm_name": "1_0",
            "metric" : "metric_b",
            "mean" : 12.1,
            "sem" : 1.0
        },
                {
            "trial_index": 2,
            "arm_name": "2_0",
            "metric" : "metric_b",
            "mean" : 18.2,
            "sem" : 1.0
        },
        {
            "trial_index": 3,
            "arm_name": "3_0",
            "metric" : "metric_b",
            "mean" : 23.7,
            "sem" : 1.0
        },
        {
            "trial_index": 3,
            "arm_name": "3_1",
            "metric" : "metric_b",
            "mean" : 14.15,
            "sem" : 1.0
        },
                {
            "trial_index": 3,
            "arm_name": "3_2",
            "metric" : "metric_b",
            "mean" : 22.52,
            "sem" : 1.0
        },
                {
            "trial_index": 4,
            "arm_name": "4_0",
            "metric" : "metric_b",
            "mean" : 11.8,
            "sem" : 1.0
        },
                {
            "trial_index": 4,
            "arm_name": "4_1",
            "metric" : "metric_b",
            "mean" : 28.1,
            "sem" : 1.0
        },
                {
            "trial_index": 4,
            "arm_name": "4_2",
            "metric" : "metric_b",
            "mean" : 39.1,
            "sem" : 1.0
        },

        ]
    with open("eval_data.json", "w") as f:
        json.dump(data, f)
        f.close()
    for d in data:
        responses.append(d)
    print("file created")
#create_dummy_eval_data()

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
        #return run_metadata

    @property
    def staging_required(self):
        return False

    '''
    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        return trial_metadata
    
    def run(self, batch_trials):

        trials = []
        for trial in batch_trials.arms:
            #trial_metadata = {"name": str(trial.index)}
            # print(trial)
            # trial.mean = trial.arm.parameters["x1"] + trial.arm.parameters["x2"]
            xA = []
            for x in trial.parameters.values():
                xA.append(x)
            # trial.sem = 0.0
            (trial.mean, trial.sem) = (10, 0.1)
            # return trial_metadata
            trials.append(trial)
        ic(batch_trials.arms)
        return batch_trials
    '''




