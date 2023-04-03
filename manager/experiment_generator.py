from datetime import datetime
import pandas as pd
bom_ids = ["4", "10", "20", "50", "100"]
algorithms = ["turbo", "cma-es", "sobol", "gpei", "saasbo"]
budgets = [40, 100, 200, 500, 1000]
batch_sizes = {40: 2, 100: 4, 200: 8, 500: 10, 1000: 20}
replications = 5
all_experiments = []
# make permutations of all combinations
for bom in bom_ids:
    for algorithm in algorithms:
        for budget in budgets:
            batch_size = batch_sizes[budget]
            for i in range(replications):
              experiment_name = "bom_" + str(bom) + "_" + algorithm + "_" + str(budget)
              experiment = {
                "experiment_name": experiment_name,
                "bom_id": bom,
                "algorithm": algorithm,
                "budget": budget,
                "batch_size": batch_size,
                "replication": i + 1 
              }
              all_experiments.append(experiment)
              
pd.DataFrame(all_experiments).to_excel("all_experiments.xlsx", index=True)



# experiment_object = {
#         "status": "open",
#         "execution_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

#         "created_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "experiment_name": experimentName,
#         "replications": replications,
#         "runner_type" : runner_type,
#         "abortion_criterion": abortionCriterion,
#         "abortion_value": abortionValue,
#         "budget": budget,
#         "batch_size": batchSize,
#         "init_arms": numInit,
#         "algorithm": algorithm,
#         "algorithm_config": {
#         "turbo_TR": turboTR,
#         "sm": sm,
#         "algorithm": algorithm,
#         "mc_samples": mcSamples,
#         "warm_up_steps": warmUpSteps,
#         "thinning": thinning,
#         "sigma": sigma,
#       },
#         "use_case": useCase,
#         "use_case_config": {
#         "bom_id": bom,
#         "use_case": useCase,
#         "num_sim_runs" : simRuns,
#         "stochastic_method": stochastic_method,
#         },
#       }