from ax.service.ax_client import AxClient
from ax.modelbridge.registry import Models

from colorama import Fore, Style
from colorama import init
from icecream import ic
init()


def setup_model_and_experiment(model):

    # Every model must have a unique id
    model_id = model['model_id']

    # Get the experiment parameters (controls)
    params = get_params_for_ax(model)

    # Get the primary response name and objective (minimize/maximize)
    objective_name, objective = get_objective_name_and_objective(model)

    # Get optional outcome constraints
    outcome_constraints = get_outcome_constraints(model)

    # Get parameter constrains
    parameter_constraints = get_parameter_constraints(model)
    # parameter_constraints = ["PropertyCapaServer1 + PropertyCapaServer2 + PropertyCapaServer3 + PropertyCapaServer4 + PropertyCapaServer5 + PropertyCapaServer6 + PropertyCapaServer7 + PropertyCapaServer8 + PropertyCapaServer9 + PropertyCapaServer10 <= 50"]
    # ic(parameter_constraints)
    # Initialize a new Ax client    
    ax = AxClient(verbose_logging = False)

    ax.create_experiment(
        name=model_id,
        parameters=params,
        objective_name=objective_name,
        minimize=objective,
        parameter_constraints=parameter_constraints,
        outcome_constraints=outcome_constraints
    )

    # Save model to file
    ax.save_to_json_file("models/" + model_id + ".json")

def generate(model_id, generation_strategy, number_points, number_retries_with_same_parameter_values):
    ic(generation_strategy)
    trials = []

    # Load the model from json
    ax = load_model(model_id)
    
    # Instantiate the generator models
    if generation_strategy == "sobol":
        sobol = Models.SOBOL(ax.experiment.search_space)

        for i in range(number_points):
            trial = ax.experiment.new_trial(generator_run=sobol.gen(1))
            trials.append(trial)
            trial.mark_running(no_runner_required = True)

    elif generation_strategy == "gpei":
        data = ax.experiment.fetch_data()
        gpei = Models.GPEI(experiment=ax.experiment, data=data)

        for i in range(number_points):



            generator_run = gpei.gen(1)
            ei = generator_run.gen_metadata["expected_acquisition_value"]
            print(Fore.GREEN + "Expected improvement: " + str(ei) + Style.RESET_ALL)

            # TODO: Add error handling for: RuntimeError: cholesky_cpu: For batch 0: U(107,107) is zero, singular U.
            trial = ax.experiment.new_trial(generator_run=generator_run)
            '''
            # Check whether this arm has been run in a trial before. If yes, get the response
            response_for_arm = get_response_for_arm(ax, trial.arms[0].name, trial.index)
            ic(response_for_arm)

            # Generate new trial if the trial was already run
            if response_for_arm[1] > 0:
                print(Fore.RED + "Parameters for trial >" + str(trial.index) + "< were already suggested. Giving same answer and asking for new point: >(" + str(response_for_arm[0][0])+ ", "+ str(response_for_arm[0][1]) + ")<." + Style.RESET_ALL)

                # Max retries reached for same parameter constellation? Add-in will know by looking for trial_index = -1
                if response_for_arm[1] >= number_retries_with_same_parameter_values:
                    break
    
                # TODO: Leave trial open as suggested in Ax issue?
                print("Response: " + str(response_for_arm[0]))
                trial.mark_running(no_runner_required = True)
                #ax.complete_trial(trial_index = trial.index, raw_data = response_for_arm[0])
                ax.complete_trial(trial_index = trial.index, raw_data = response_for_arm[0][0])
            else:
            '''

            print("Generated arm: " + trial.arms[0].name)
            trials.append(trial)
            trial.mark_running(no_runner_required = True)
            
    # Save model back to file
    save_model(ax, model_id)

    result = []
    for t in trials:
        point = {}
        point["strategy"] = generation_strategy
        point["trial_name"] = t.index
        point["arm_name"] = t.arm.name
        point["parameters"] = t.arm.parameters
        result.append(point)

    return { "trials" : result }

def complete(model_id, trial_responses):
    print(trial_responses)
 

    # Do nothing if not trial responses reported
    if len(trial_responses) == 0:
        return

     # Load the model from json
    ax = load_model(model_id)

    # Iterate all responses and complete trials 
    # TODO: Implement error handling  
    for response in trial_responses:
        response_values = to_tupel(response.get('responses'))

        trial_index = int(response.get('trial_name'))

        ax.complete_trial(trial_index = trial_index, raw_data = response_values)
        
    # Save the model back to file
    save_model(ax, model_id)

def get_params_for_ax(model):
    result = []

    if not 'parameters' in model:
        raise Exception("Error registering new model: No parameters found.")

    for p in model['parameters']:
        # Determine the type of the parameter
        # TODO: Add choice parameters
        if p['type'] == "discrete":
            value_type = "int"
        elif p['type'] == "continuous":
            value_type = "float"

        # Handle fixed parameters
        if 'fixed' in p and p['fixed'] == True:
            axP = {"name": p['name'],
                "type":  "fixed",
                "value": p['fixed_value'],
                "value_type": value_type}

        # Handle non-fixed parameters   
        else:
            axP = {"name": p['name'],
                "type":  "range",
                "bounds": [p['min'], p['max']],
                "value_type": value_type}

        result.append(axP)
   
    return result

def get_objective_name_and_objective(model):

    if not 'responses' in model:
        raise Exception("Error registering new model: no objective found.")

    for r in model['responses']:
        if r['is_primary']:
            return (r['name'], r['minimize'])

    raise Exception("Error registering model: list of responses is empty.")

def get_outcome_constraints(model):
    outcome_constraints = []

    for r in model['responses']:
        if 'lower_bound' in r and r['lower_bound'] != "-Infinity":
            outcome_constraints.append(r['name'] + " >= " + str(r['lower_bound']))
        if 'upper_bound' in r and r['upper_bound'] != "Infinity":
            outcome_constraints.append(r['name'] + " <= " + str(r['upper_bound']))

    return outcome_constraints

def get_parameter_constraints(model):
    constraints = []
    if "constraints" in model:
        for c in model["constraints"]:
            constraints.append(c["constraint"])
    return constraints

def get_response_for_arm(ax, arm_name, current_trial_index):

    if current_trial_index == 0:
        return [(None, None), 0]
    
    data = ax.get_trials_data_frame().sort_values('trial_index')

    data = data.loc[data["arm_name"] == arm_name]
    ic(len(data.index))
    number_repeated = len(data.index)
    
    if number_repeated == 0:
        return [(None, None), 0]
    else:
        return [(data.iloc[0][ax.experiment.optimization_config.objective.metric.name], 0.0), number_repeated]

def load_model(model_id):
    ax = AxClient.load_from_json_file("models/" + model_id + ".json")
    return ax

def save_model(ax, model_id):
    ax.save_to_json_file("models/" + model_id + ".json")

def save_data(model_id):
    ax = load_model(model_id)
    data = ax.get_trials_data_frame()


    if len(data.index) > 0:
        data = data.sort_values('trial_index')
        data.to_csv("data/" + model_id + ".csv", index = False)

def to_tupel(responses):

    '''
    if len(responses) == 1:
        return responses[list(responses.keys())[0]][0]
    '''

    response_as_tupel = {}
    for k, v in responses.items():
        response_as_tupel[k] = (v[0], v[1])
        #response_as_tupel[k] = v[0]

    
    return response_as_tupel
    


