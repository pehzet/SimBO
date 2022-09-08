# TODO: Clear import before Production
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from ax.core.base_trial import TrialStatus
from ax import *
from ax.storage.registry_bundle import RegistryBundle
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
from ax.modelbridge.cross_validation import cross_validate, compute_diagnostics
import pandas as pd
import sys
import logging
from icecream import ic
class g:
    experiment = None
    responses = None
    experiment_name = None
class Submetric(Metric):

    def fetch_trial_data(self, trial):
    
        records = []
    

        experiment_name, metric_name, metric_suffix = self.name.split("-")

        #exp, res = load_experiment_from_json(experiment_name)
        exp = g.experiment
        res = g.responses
        # res: [{'arm_name': '0_0',
        #    'parameters': {'SafetyStock': '3', 'SafetyTime': '4'},
        #    'responses': {'Costs': [9590.0, 0.0]},
        #    'strategy': 'sobol'}]
        for arm_name, arm in trial.arms_by_name.items():
            # get raw eval data might be a problem because i dont know how to get the experiment
            metric_values = [r.get("metric_values") for r in res if r["arm_name"] == arm_name][0]

            _obj = {
                "arm_name": arm_name,
                "metric_name": self.name,
                "trial_index": trial.index,
                "mean": [metric for metric in metric_values if metric.get("name") == metric_name][0].get("mean"),
                "sem": [metric for metric in metric_values if metric.get("name") == metric_name][0].get("sem")
            }
            records.append(_obj)
        return Data(df=pd.DataFrame.from_records(records))

        # ic(pd.DataFrame.from_records(records))
    def is_available_while_running() -> bool:
        return True
register_metric(Submetric)
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
register_runner(SubRunner)

#######################################################################################################################
#                               STORAGE                                                                               #
#######################################################################################################################
from utils.ax_utils import Submetric, SubRunner
import pickle
import json
import pandas as pd
import os
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
from ax.storage.registry_bundle import RegistryBundle
bundle = RegistryBundle(
    metric_clss={Submetric: None},
    runner_clss={SubRunner: None}
)
# Crap Function might be useless
def get_raw_eval_data(arm_name: str, experiment_name : str, metric_name: str = None):
    path = os.path.join("..","data", experiment_name+"_responses")
    responses = pd.read_csv(path)
    resp_tuple = [(r["mean"], r["sem"]) for r in responses if r["arm_name"] == arm_name][0]
    return resp_tuple[0], resp_tuple[1]

def save_experiment_as_json(exp, responses, is_cmaes=False,exp_name="Exp1"):

    path_präfix = "data"
    if is_cmaes == False:  
        exp_name = exp.name
        _name = os.path.join(path_präfix, str(exp_name) + ".json")



        # path_präfix = os.path.join("..","data")

        #_name = str(name) + ".json"
        _name = os.path.join(path_präfix, str(exp_name) + ".json")
        # register_runner(SubRunner)
        # register_metric(Submetric)
        bundle = RegistryBundle(
        metric_clss={Submetric: None},
        runner_clss={SubRunner: None}
    )
        save_experiment(exp, _name, encoder_registry=bundle.encoder_registry)
    else:
        _name = os.path.join(path_präfix, str(exp_name) + ".json")
        # open(_name).write(exp.pickle_dumps())
        pickle.dump(exp, open(_name, 'w'))
    # save responses as well
    _resp_name = os.path.join(path_präfix, str(exp_name) + "_responses.json")

    if not os.path.exists(_resp_name):
        with open(_resp_name, "w") as file:
            j = json.dump({"Responses": []}, file)
        print("Experiment saved and empty Response File created")
        return

    else:
        with open(_resp_name, "w") as file:
            #np.save(file, np.array(responses),allow_pickle=True)
            j = json.dump({"Responses": responses}, file)
            # file.write(j)
    print("Experiment and Responses saved")

def load_experiment_from_json(experiment_name: str = "undefined"):
    # path_präfix = os.path.join("..","data")
    path_präfix = "data"
    _name = os.path.join(path_präfix, str(experiment_name) + ".json")
    #_name = str(name) + ".json"
    exp = load_experiment(_name, decoder_registry=bundle.decoder_registry)
    _resp_name = os.path.join(path_präfix, str(exp.name) + "_responses.json")
    global responses
    if os.path.exists(_resp_name):
        f = open(_resp_name)
        _loaded_responses = json.load(f)
        f.close()
        # with open(_resp_name, "r") as file:
        #    #_loaded_responses = np.load(file, allow_pickle=True)
        #    _loaded_responses = json.loads(str(file))#["Responses"]
        _responses = []
        for lr in _loaded_responses["Responses"]:
            #_lr = json.loads(lr)
            _responses.append(lr)
        responses = _responses
        print("Experiment and Responses loaded")
        g.experiment = exp
        g.responses = responses
    return exp, responses



def save_experiment_configs(use_case_config, algorithm_config, experiment_name: str = "undefined"):
    # path_präfix = os.path.join("..","data")
    path_präfix = "data"
    _object = {
        "experiment_name" : experiment_name,
        "use_case_config" : use_case_config,
        "algorithm_config": algorithm_config
    }
    _filename = os.path.join(path_präfix, str(experiment_name) + "_config.json")
    with open(_filename, "w") as file:
        json.dump(_object, file)
    return

def load_experiment_configs(experiment_name:str):
    # path_präfix = os.path.join("..","data")
    path_präfix = "data"
    _filename = os.path.join(path_präfix, str(experiment_name) + "_config.json")

    if os.path.exists(_filename):
        f = open(_filename)
        config = json.load(f)
        f.close()
        return config["use_case_config"], config["algorithm_config"]
    print("Path not exist")
    return {}, {}

def format_arm_data(experiment_name: str, trial_data):
    trial_data_format = {
        "trial_name" : "1",
        "experiment_name" : "124_abc",
        "arms" : [
            {
                "arm_name" : "1_0",
                "metric_values" : [{
                    "name" : "Costs",
                    "value" : 145.7
                }] 
            }
        ]
    }
    return trial_data

def append_new_arm_data_to_responses(experiment_name:str, arm_data):
    exp, res = load_experiment_from_json(experiment_name)
    arm_data = format_arm_data(experiment_name, arm_data)
    res.append(
        arm_data
    )
    save_experiment_as_json(exp, res)
    return True

def format_trial_to_arms_ax(trial, strategy="unknown"):

    suggested_arms = []
    for arm in trial.arms:
        _trial = {}
        _trial["parameters"] = arm.parameters
        _trial["strategy"] = strategy
        _trial["arm_name"] = arm.name
        _trial["trial_name"], _ = arm.name.split("_")
        suggested_arms.append(_trial)
    return suggested_arms
###############################################################################################
#                           AX EXPERIMENT STUFF                                 #

def create_ax_param_class(param : dict):
    param_ax = None # returns AX Range or Choice Param
    return param_ax

def create_search_space(params: list, constraints: list = None):
  
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
        "discrete": ParameterType.STRING
    }
    for p in params:
        if (p["type"] == "float" or p["type"] == "continuous") and p["fixed"] == False:
            _p = RangeParameter(
                name=p["name"],
                parameter_type=type_map[p["type"]],
                lower=float(p["lower_bound"]),
                upper=float(p["upper_bound"]),
            )
            parameters_ax_dev.append(_p)
            paramaters_maps.append({"name": p["name"], "parameter_object": _p})
        elif (p["type"] == "integer" or p["type"] == "int" ) and p["fixed"] == False:
            _p = RangeParameter(
                name=p["name"],
                parameter_type=type_map[p["type"]],
                lower=int(p["lower_bound"]),
                upper=int(p["upper_bound"]),
            )
            parameters_ax_dev.append(_p)
            paramaters_maps.append({"name": p["name"], "parameter_object": _p})
        elif (p["type"] == "string" or p["type"] == "discrete") and p["fixed"] == False:
            _p = ChoiceParameter(
                name=p["name"],
                parameter_type=type_map[p["type"]],
                values=p["values"],
            )

            parameters_ax_dev.append(_p)
            paramaters_maps.append({"name": p["name"], "parameter_object": _p})
        elif p["type"] == "bool" or p["type"] == "boolean" and p["fixed"] == False:
            _p = ChoiceParameter(
                name=p["name"],
                parameter_type=type_map[p["type"]],
                values=["true", "false"],
            )

            parameters_ax_dev.append(_p)
            paramaters_maps.append({"name": p["name"], "parameter_object": _p})
        elif p["fixed"] == True:
            _p = FixedParameter(
                name=p["name"],
                parameter_type=type_map[p["type"]],
                value=p["value"],
            )
            parameters_ax_dev.append(_p)
            paramaters_maps.append({"name": p["name"], "parameter_object": _p})


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

def create_optimization_config(experiment_name: str, primary_responses: list, secondary_responses: list = [], is_moo: bool = False):

    if '-' in experiment_name:
        print("Experiment Name must not contain '-'! Use '_' instead")
    for p in primary_responses:
        if not isinstance(p, dict):
            print("Response Element must be Dict")
            return
      
        _metric_name = experiment_name + "-" + p.get("name") + "-metric"
     
        _objective = Objective(
                metric=Submetric(
                    name=_metric_name,
                    lower_is_better=p["minimize"]
                ),
                minimize=p["minimize"])
     
   
    _outcome_constraints = []
    for s in secondary_responses:

        _metric_name = experiment_name + "-" + s.get("name") + "-metric"
        if s.get("lower_bound") != "-Infinity" or s.get("is_lower_constraint") == True:
            _op = ComparisonOp.GEQ
            _bound = s.get("lower_bound", s.get("bound"))
        else:
            _op = ComparisonOp.LEQ
            _bound = s.get("upper_bound",  s.get("bound"))
        _outcome_constraint = OutcomeConstraint(
            metric=Submetric(
                name=_metric_name,
            ),
            op=_op,
            bound=_bound,
            relative=False
        )
  
        _outcome_constraints.append(_outcome_constraint)
    return OptimizationConfig(objective=_objective, outcome_constraints=_outcome_constraints)



def create_experiment_ax(name, search_space, optimization_config, params):
    return Experiment(
        name=name,
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SubRunner(params=params),
    )
def suggest_initial_point(experiment_name, numPoints=1):
    global responses
    exp, responses = load_experiment_from_json(experiment_name)
    sobol = Models.SOBOL(search_space=exp.search_space)
    generator_run = sobol.gen(n=numPoints)
    trial = exp.new_trial(generator_run=generator_run)
    save_experiment_as_json(exp, responses)


    return trial, exp

def get_raw_eval_data_from_json(experiment_name: str, arm_name: str):
    exp, res = load_experiment_from_json(experiment_name)
    trial_response = [r for r in res if r["arm_name"] == arm_name]
    return trial_response["mean"], trial_response["sem"]

def complete_trial(experiment_name, trial_index=-1):

    global responses
    exp, responses = load_experiment_from_json(experiment_name)

    if trial_index == -1:
        # get the last trial
        trial_index = exp.trials.keys()[-1]

    trial = exp.get_trials_by_indices([trial_index])[0]

    if trial.status == TrialStatus.RUNNING:
        trial.mark_completed()
    else:
        trial.run()
        trial.mark_completed()
    data = Data.from_multiple_data([exp.fetch_data(), trial.fetch_data()])
    print(f"Completed Trial no. {trial_index}")
    save_experiment_as_json(exp, responses)

def get_model_from_experiment_name(experiment_name:str):
    exp, res = load_experiment_from_json(experiment_name)
    use_case_config, algorithm_config = load_experiment_configs(experiment_name)
    algorithm = algorithm_config.get("algorithm", algorithm_config.get("name",algorithm_config.get("algorithmn",None))).lower()
    
    # algorithm_config["number_samples"] = 32
    # algorithm_config["warm_up_steps"] = 32
    # ic(algorithm_config)
    if algorithm == "gpei":
        from algorithms.gpei.gpei_stepwise import get_model as gm_gpei
        return gm_gpei(exp)
    if algorithm == "saasbo":
        from algorithms.saasbo.saasbo_stepwise import get_model as gm_saasbo
        return gm_saasbo(exp, algorithm_config)
    if algorithm ==None:
        logging.error("No algorithm defined in algorithm_config. Can not run optimization. Going to exit")
        sys.exit()

def get_cross_validation_and_diagnostics(model):
    cv_results = cross_validate(model)
    diagnostics = compute_diagnostics(cv_results)
    return cv_results, diagnostics


def get_lengthscales(model, params):


    if isinstance(model.model.model, FixedNoiseGP) or isinstance(model.model.model, SingleTaskGP):
        median_lengthscales = model.model.model.covar_module.base_kernel.lengthscale.squeeze().median(axis=0)

    else:
        median_lengthscales = model.model.model.models[0].covar_module.base_kernel.lengthscale.squeeze().median(axis=0)


    param_names = [k for k in params.keys()]
    lengthscales = []
    for i, p in enumerate(param_names):
        lengthscales.append({p : median_lengthscales.values.tolist()[i]})




    return lengthscales

def get_feature_importance(model):
    feature_importances = []

    for metric_name in model.metric_names:
        fi = model.feature_importances(metric_name)
        _obj = {"metric" : metric_name, "feature_importances" : fi}
        feature_importances.append(_obj)
    return feature_importances