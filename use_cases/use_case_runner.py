
# from utils import *
from copy import deepcopy
import os
import logging


logger = logging.getLogger("mrp")

# from main import run_solver
from pandas import DataFrame
from utils.gsheet_utils import read_gsheet, formatDF
from torch import tensor

import os
import torch
import numpy as np
import traceback
from dotenv import load_dotenv
load_dotenv()
from botorch.utils.transforms import unnormalize, normalize
from icecream import ic
class UseCaseRunner():

    def __init__(self) -> None:
        self.X = list()
        self.Y_raw = list()
        self.sheet_id = None
        self.use_case_id = None
        self.objectives = self.get_objectives_from_config()

    def get_param_informations_from_sheet(self):
        sheet_name = "parameters"
        if self.sheet_id == "env":
            self.sheet_id = os.getenv("SHEET_ID")
        param_meta = []
        raws = formatDF(read_gsheet(self.sheet_id, sheet_name))


        for r in raws: 
            lb = r.get("lower_bound",0)
            ub = r.get("upper_bound",1)
            pname = r.get("param_id","param_name")
            assert lb < ub, f"Lower bound {lb} must be smaller than upper bound {ub} for parameter {pname}"
            if r.get("use_case_id") == self.use_case_id:
                param_meta.append({
                "name" : pname,
                "lower_bound" : lb,
                "upper_bound" : ub,
                "type" : r.get("dtype","float"),
                })
        self.param_meta = param_meta
        return param_meta

    def get_response_informations_from_sheet(self):
        sheet_name = "responses"
        if sheet_id == "env":
            sheet_id = os.getenv("SHEET_ID")
        resp_meta = []
        raws = formatDF(read_gsheet(sheet_id, sheet_name))

        for r in raws:
            if r.get("use_case_id") == self.use_case_id:
                resp_meta.append({
                "name" : r.get("response_name","response_id"),
                "is_primary" : True if r.get("lower_bound") and r.get("upper_bound") else False,
                "bound" : r.get("bound"),
                "minimize" : r.get("minimize",True),
                })
        self.resp_meta = resp_meta
        return resp_meta

    def get_objectives_from_config():
        # not implemented yet
        return None
    def get_mean_and_sem_from_y(self, y_raw):
        data = list()

        for i in range(len(y_raw[0].keys())):
            values = [list(y.values())[i] for y in y_raw]
            mean = np.mean(values)
            sem = (np.std(values, ddof=1) / np.sqrt(np.size(values)))
            sem = sem if not np.isnan(sem) else 0
            data.append((mean,sem))
        return data

    def format_y_for_candidate(self, y):
        '''
        writes y to dict with name as key and y as value from resp_meta
        '''
        y_format = dict()
        for i in range(self.resp_meta):
            y_format[self.resp_meta[i].get("name")] = y[i]
        return y_format

    def format_x_for_candidate(self, x):
        return self.transform_x(x)

    def transform_x(self, x):
        assert self.param_meta is not None
        assert self.bounds is not None

        x = unnormalize(x, bounds=self.bounds)
        x_mrp = []
        for i,pm in enumerate(self.param_meta):

            x_mrp.append(
                {   
                "id" : pm.get("name").split("_",1)[0],
                "name" : pm.get("name").split("_",1)[1],
                "value" : int(round(x[i].item()))
                }
            )

        return x_mrp
    def transform_y_to_tensors_mean_sem(self, y_mrp):
        '''
        returns two tensors with mean and sem
        '''
        y_mean = torch.tensor([y[0][0] for y in y_mrp])

        y_sem = torch.tensor([y[0][1] for y in y_mrp])
        
        #yy = torch.tensor([list(y.values())[0] for y in y_mrp])
        if self.minimize:
            y_mean = y_mean * -1
            #y_sem = y_sem * -1
        return y_mean, y_sem

    def get_bounds_from_param_meta(self):
        '''
        expects list of dicts with key lower_bound:number and upper_bound: number or bounds: (number, number)
        returns bounds as tuple tensor
        '''
        
        lb = [pm.get("lower_bound") for pm in self.param_meta]
        ub = [pm.get("upper_bound") for pm in self.param_meta]
        #bounds = [(pm.get("lower_bound",pm.get("bounds")[0]) , pm.get("upper_bound",pm.get("bounds")[1])) for pm in self.param_meta]
        bounds = torch.tensor([lb, ub])
        return bounds


    def format_feature_importance(self, fi: torch.Tensor):
        if fi == "na":
            return fi
        pm_name = [pm.get("name") for pm in self.param_meta]


        if fi.ndim == 1:
            if not isinstance(fi, list):
                fi = fi.tolist()
            return [dict((k,v) for k,v in zip(pm_name, fi))]
  
        fi_per_trial = []
        for _fi in fi:
            if not isinstance(_fi, list):
                _fi = _fi.tolist()
            fi_dict = dict((k,v) for k,v in zip(pm_name, _fi))
            fi_per_trial.append(fi_dict)
        return fi_per_trial


    def eval(self):
        pass