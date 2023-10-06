import numpy as np
from botorch.test_functions.multi_objective import BraninCurrin
import torch
from botorch.utils.transforms import unnormalize, normalize
from icecream import ic

class DummyRunner:
    def __init__(self):
        self.bounds=[(0,10),(0,10)]
        self.Y_raw = None
        self.param_meta = self.get_param_meta()
        self.constraints = ["'obj_3' <= 10"]
        self.objectives = ["obj_1", "obj_2","obj_3"]
        self.outcomes = ["obj_1", "obj_2","obj_3"]
        self.eval_counter = 0
    def eval(self, x):
        if torch.is_tensor(x):
            x = x.numpy()
        # x_0 = unnormalize(x[0], bounds=self.bounds[0])
        # x_1 = unnormalize(x[1], bounds=self.bounds[1])
        x_0 = x[0]
        x_1 = x[1]
        # Botorch assumes maximization
        obj_1 = x_0**2 + x_1**2 # main objective to maximize
        obj_2 = - abs(x_0 - x_1) # try to keep the values equal
        obj_3 = - (10 - (x_0 + x_1)) # best is 5/5
        self.eval_counter += 1
        if self.eval_counter % 10 == 0:
            print(f"Evaluated: {self.eval_counter}")
        result = ((obj_1, 0.1), (obj_2,0.1), (obj_3,0.1))
        self.Y_raw = result if self.Y_raw is None else np.vstack([self.Y_raw, result])
        return result
    def format_x_for_candidate(self,x):
        x_dict = dict()
        for i in range(len(x)):
            x_dict[f"x_{i}"] = x[i]
        return x_dict
    def format_y_for_candidate(self,y):
        y_dict = dict()
        for i in range(len(y)):
            y_dict[f"obj_{i}"] = y[i]
        return y_dict
    def get_param_meta(self):
        param_meta = [{
            "name" : "x1",
            "lower_bound" : 0,
            "upper_bound" : 10,
            "type" : "int",
            "fixed" : False
            },
            {
            "name" : "x2",
            "lower_bound" : 0,
            "upper_bound" : 10,
            "type" : "int",
            "fixed" : False
            }]
        return param_meta
    def transform_y_to_tensors_mean_sem(self, y):

        y_mean = [tuple(x[0] for x in item) for item in y]
        y_sem = [tuple(x[1] for x in item) for item in y]
        y_mean = torch.tensor(y_mean)
        y_sem = torch.tensor(y_sem)
        return y_mean, y_sem
    def format_feature_importance(self, fi):
        return None

    def get_log_informations(self):
        return None