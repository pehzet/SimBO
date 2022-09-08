from ax.modelbridge.registry import Models

def suggest_next_trial(experiment, algorithm_config):
    batch_size = algorithm_config.get("batch_size", 1)
    num_trials = 1 if batch_size == 0 else batch_size
    
    gpei = Models.BOTORCH(experiment=experiment, data=experiment.fetch_data())
    gr = gpei.gen(num_trials)
    if num_trials > 1:
        trials = experiment.new_batch_trial(generator_run=gr)
        trials.run()
        return trials, experiment
    else:
        trial = experiment.new_trial(generator_run=gr)
        trial.run()
        return trial, experiment
    

def get_model(experiment):
    m = Models.BOTORCH(experiment=experiment, data=experiment.fetch_data())
    return m