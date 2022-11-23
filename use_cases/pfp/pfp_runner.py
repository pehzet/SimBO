
# from utils import *
from copy import deepcopy
import os
import logging


logger = logging.getLogger("mrp")

# from main import run_solver
from pandas import DataFrame
import pandas as pd
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
class PfpRunner():

    def __init__(self):


        self.minimize = True
        self.param_meta = self.get_param_meta_from_gsheet()
        self.bounds = self.get_bounds_from_param_meta()
        self.is_ddo = True
        self.X = list()
        self.Y_raw = list()
        self.constraints = self.create_constraints()
        self.x_t = []
        self.cost_factor = 4.33

    # def eval(self, x, ):
    #     x = self.transform_x(x)

    #     self.X.append(x)
    #     releases = self.run_solver(x)
    #     for _ in range(self.num_sim_runs):
    #         result = MRPSimulation(releases, stochastic_method = self.stochastic_method).run_simulation()
    #         self.Y_raw.append(result)
    #     y = self.get_mean_and_sem_from_y(self.Y_raw[-self.num_sim_runs:])
  
    #     return y
    def create_constraints(self):
        
        _indices = [i for i in range(len(self.param_meta))]
        _coefficients = [-1 for j in range(len(self.param_meta))]
        _rhs = -2.5*len(self.param_meta)
        # return [(_indices,_coefficients, _rhs)]
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
        return {"TotalWastedMaterial" : y}

    def format_x_for_candidate(self, x):
        return self.transform_x(x)

    def transform_x(self, x):

        assert self.param_meta is not None
        assert self.bounds is not None

        x = unnormalize(x, bounds=self.bounds)
  
        x_t = []
        for i,pm in enumerate(self.param_meta):

            x_t.append(
                {   
                "name" : pm.get("name"),
                "value" : int(round(x[i].item())),
                "costs" : pm.get("costs")
                }
            )
        self.x_t = x_t
        return x_t
    def write_x_to_xlsx(self, x, run_no, path=None):
        # x= [{name : _id, value : 1}]
        # read
        # read_file_name = Path(r"C:\Users\pzmijews\Documents\Studium\Projekt\Simio\Simio_ProductionModel_MRethmann\ModelData\ModelData_v2.xlsx")
        _original = pd.read_excel(r"C:\Users\pzmijews\Documents\Studium\Projekt\Simio\Simio_ProductionModel_MRethmann\ModelData\ModelData_v2.xlsx","Molds",)
        original = _original.to_dict('records')

        for _x in x:

            for o in original:
                if o.get("ID",1) == int(_x.get("name",-1)):
                    o["Quantity"] = _x.get("value")
        if path is None:
            path = r'C:\Users\pzmijews\Documents\Studium\Projekt\Simio\Simio_ProductionModel_MRethmann\ModelData'
        fname = 'Molds_' + str(run_no) + '.xlsx'
        #x = unnormalize(x, bounds=self.bounds)

        fpath = os.path.join(path,fname)
        modified = pd.DataFrame.from_records(original)
        modified.to_excel(fpath,'Molds',index=False)


    def format_simulation_response(self,resp):
        y = []
        for r in resp:
            for rv in r.values():
                material_costs = 0
                for xx in self.x_t:
                    material_costs += max((xx.get("value")-1),0) * xx.get("costs")
                c = rv[0] * self.cost_factor + material_costs
                ic(c)
                y.append(c)
        return y

    def get_bounds_from_param_meta(self):
        '''
        expects list of dicts with key lower_bound: int and upper_bound: int or bounds: (int, int)
        returns bounds as tuple tensor
        '''
        
        lb = [pm.get("lower_bound") for pm in self.param_meta]
        ub = [pm.get("upper_bound") for pm in self.param_meta]
        #bounds = [(pm.get("lower_bound",pm.get("bounds")[0]) , pm.get("upper_bound",pm.get("bounds")[1])) for pm in self.param_meta]
        bounds = torch.tensor([lb, ub])
        return bounds






   

    def get_param_meta_from_gsheet(self):
        param_meta = []
        
        sheet_id = os.getenv("SHEET_ID")
        raw = formatDF(read_gsheet(sheet_id, "pfp_molds"))

        for r in raw:

            if r.get("usedInCurrentPeriod") and r.get("productType") == "Cup":
                param_meta.append({
                "name" : r.get("id"),
                "lower_bound" : r.get("lower_bound",0),
                "upper_bound" : r.get("upper_bound",1),
                "type" : "int",
                "fixed" : False,
                "costs" : r.get("costs")
                })
           



        return param_meta



    def format_feature_importance(self, fi: torch.Tensor):
        if fi == "na":
            return fi
        pm_name = [pm.get("name") for pm in self.param_meta]

        fi = tensor(fi) 
        if fi.ndim == 1:
            fi = fi.tolist()
            return [dict((k,v) for k,v in zip(pm_name, fi))]
  
        fi_per_trial = []
        for _fi in fi:
            _fi = _fi.tolist()
            fi_dict = dict((k,v) for k,v in zip(pm_name, _fi))
            fi_per_trial.append(fi_dict)
        return fi_per_trial






