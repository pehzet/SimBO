###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

import numpy as np
import json
from icecream import ic

def to_unit_cube(x, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = (x - lb) / (ub - lb)
    return xx

def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx

def latin_hypercube(n_pts, dim):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X

def normalize_parameters(x, model_params):

    x = []
    for param_name in model_params:
        x.append(model_params[param_name])
    
    result = []
    for i, px in enumerate(x):
        min = model_params[i].get("min")
        max = model_params[i].get("max")
        normalized = (px - min) / (max - min)
        result.append(normalized)
    
    return result

def unnormalize_parameters(x, model_params):
    result = {}
    for i, px in enumerate(x):
        discrete = model_params[i].get("type") == "discrete"
        min = model_params[i].get("min")
        max = model_params[i].get("max")
        unnormalized = px * (max - min) + min

        if discrete == True:
            unnormalized = round(unnormalized)

        result[model_params[i].get("name")] = unnormalized

    return result
    
def get_primary_response_name(model_responses):
    for r in model_responses:
        if r.get("is_primary") == True:
            return r.get("name")

    return None

def save_data(turbo, model):
    data = turbo.data()
    model_params = model.get("parameters")

    # Translate trial's data to real dimensions
    for t in data["trials"]:

        # x-values
        t["parameters_unnormalized"] = {}
        unnormalized_params = unnormalize_parameters(t["x"], model_params)

        for p_name in unnormalized_params:
            t["parameters_unnormalized"][p_name] = unnormalized_params[p_name]

        # Lower/upper bounds
        t["trust_region_lower_bounds_unnormalized"] = {}
        t["trust_region_upper_bounds_unnormalized"] = {}

        unnormalized_lower_bounds = unnormalize_parameters(t["trust_region_lower_bounds"], model_params)
        unnormalized_upper_bounds = unnormalize_parameters(t["trust_region_upper_bounds"], model_params)

        for p_name in unnormalized_lower_bounds:
            t["trust_region_lower_bounds_unnormalized"][p_name] = unnormalized_lower_bounds[p_name]
            t["trust_region_upper_bounds_unnormalized"][p_name] = unnormalized_upper_bounds[p_name]

        # Trust region center
        # x-values
        t["trust_region_center_x_unnormalized"] = {}
        unnormalized_center_x = unnormalize_parameters(t["trust_region_center_x"], model_params)

        for p_name in unnormalized_center_x:
            t["trust_region_center_x_unnormalized"][p_name] = unnormalized_center_x[p_name]


    with open('turbo_data.json', 'w') as outfile:
        json.dump(data, outfile)
        
    