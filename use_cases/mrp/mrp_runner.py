
# from utils import *
from copy import deepcopy
import os
import logging


logger = logging.getLogger("mrp")
import config
# from main import run_solver
from pandas import DataFrame
from utils.gsheet_utils import read_gsheet, formatDF
from torch import tensor
from use_cases.mrp.mrp_solver import MRPSolver
from use_cases.mrp.mrp_sim import MRPSimulation
import os
import torch
import numpy as np
import traceback
from botorch.utils.transforms import unnormalize, normalize
from icecream import ic
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}
class MRPRunner():

    def __init__(self, bom_id, num_sim_runs=5, stochastic_method=True):
        self.bom_id = bom_id
        self.num_sim_runs = num_sim_runs
        self.stochastic_method = stochastic_method
        self.bom = None
        self.materials = None
        self.orders = None
        self.stock = None
        self.sheet_id = config.SHEET_ID
        self.init_sheets() # to get the variables above
        self.minimize = True
        self.param_meta = self.get_param_meta_from_materials()
        self.bounds = self.get_bounds_from_param_meta()
        # init_mrp_sim(self.bom, self.materials, self.orders)
        self.X = list()
        self.Y_raw = list()
        self.constraints = self.create_constraints()
        self.objectives = self.create_objectives()


    def create_objectives(self):
        return [{"name":"costs"}, {"name" : "sl"}]
    def create_constraints(self):
        return None
    def eval(self, x, ):
        x = self.transform_x(x)

        self.X.append(x)
        releases = self.run_solver(x)
        for _ in range(self.num_sim_runs):
      
            result = MRPSimulation(releases, self.materials, self.bom, self.orders, stochastic_method = self.stochastic_method).run_simulation(sim_time=200)

            self.Y_raw.append(result)
        y = self.get_mean_and_sem_from_y(self.Y_raw[-self.num_sim_runs:])
  
        return y

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
        return {"costs" : y[0], "service_level" : y[1]}

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
        y_mean = torch.tensor([y[0][0] for y in y_mrp]).to(tkwargs["device"])

        y_sem = torch.tensor([y[0][1] for y in y_mrp]).to(tkwargs["device"])
        
        #yy = torch.tensor([list(y.values())[0] for y in y_mrp])
        if self.minimize:
            y_mean = y_mean * -1
            #y_sem = y_sem * -1
        return y_mean, y_sem

    def get_bounds_from_param_meta(self):
        '''
        expects list of dicts with key lower_bound: int and upper_bound: int or bounds: (int, int)
        returns bounds as tuple tensor
        '''
        
        lb = [pm.get("lower_bound") for pm in self.param_meta]
        ub = [pm.get("upper_bound") for pm in self.param_meta]
        #bounds = [(pm.get("lower_bound",pm.get("bounds")[0]) , pm.get("upper_bound",pm.get("bounds")[1])) for pm in self.param_meta]
        bounds = torch.tensor([lb, ub]).to(tkwargs["device"])
        return bounds

    def format_params_for_mrp(self, params):
        '''
        Expects: {material_param_name : value} (one dict, no list!)
        Returns: [{id: material, name : param_name, value : value}] (list of dicts)
        '''
        _params = []
        # NOTE: Not general! Only works for safety_stock and safety_time
        for k,v in params.items():
            els = k.split("_")
            _params.append({

                'id' : els[0].lower(),
                'name' : els[1].lower() + "_" + els[2].lower(),
                'value' : v
            })
        return _params

    def filter_relevant_materials(self, data, filter_by_id=False):
        if isinstance(data, DataFrame):
            data = data.to_dict('records')
        ids = []

        for b in self.bom:
            for k,v in b.items():
                if k == 'child_id' or k == 'parent_id':
                    if int(v) not in ids:
                        ids.append(int(v))
        if filter_by_id == True:
            data = [d for d in data if int(d["id"]) in ids and int(d.get("bom_id")) == int(self.bom_id)]
        else:
            data = [d for d in data if int(d["id"]) in ids]

        return data

    def get_releases_from_results(self,mrp_results,parameters):
        write_data = []
        for r in mrp_results:
            id = r["id"]
            lead_time = [m["lead_time"] for m in self.materials if m["id"] == id][0]
            safety_time = [p["value"] for p in parameters if p["id"] == id and p["name"] == "safety_time"][0]

            for i in range(len(r["ReleasedOrders"])):
                if float(int(r["ReleasedOrders"][i]*100)/100) != 0:
                    write_data.append({
                        "period" : i+1, 
                        "material" : id, 
                        "quantity" : float(int(r["ReleasedOrders"][i]*10000)/10000), 
                        "lead_time": lead_time,
                        "period_due" :i+1+int(safety_time) + int(lead_time)
                        })


        write_data_sorted_period = sorted(write_data,key=lambda x: x["period"])

        return write_data_sorted_period

    def init_sheets(self):
        
        self.bom = [b for b in formatDF(read_gsheet(self.sheet_id, "bom")) if b["bom_id"] == int(self.bom_id)]
        if len(self.bom) == 0:
            raise Exception("No BOM with this ID found")
        self.materials = self.filter_relevant_materials(formatDF(read_gsheet(self.sheet_id , "materials")))
        self.orders = self.filter_relevant_materials(formatDF(read_gsheet(self.sheet_id , "orders")), filter_by_id=True)
        self.stock = self.filter_relevant_materials(formatDF(read_gsheet(self.sheet_id , "stock")))

        assert len(self.materials) > 0
        assert len(self.orders) > 0
        logger.info("Sheets initialized")
   


    def get_param_meta_from_materials(self,format_type="ax"):
        """
        format_types = ["ax", "lb_ub_name", "lb_ub_tuple", ]
        """
        param_meta = []
        if format_type =="ax":

            for m in self.materials:
                obj_ss ={
                "name" : m.get("id") + "_" + "safety_time",
                "lower_bound" : m.get("safety_time_min"),
                "upper_bound" : m.get("safety_time_max"),
                "type" : "int",
                "fixed" : False
                }
                obj_st = {
                "name" : m.get("id") + "_" + "safety_stock",
                "lower_bound" : m.get("safety_stock_min"),
                "upper_bound" : m.get("safety_stock_max"),
                "type" : "int",
                "fixed" : False
                }
                param_meta.append(obj_ss)
                param_meta.append(obj_st)
        if format_type =="lb_ub_name":
            for m in self.materials:
                obj_ss ={
                "name" : m.get("id") + "_" + "safety_time",
                "lower_bound" : m.get("safety_time_min"),
                "upper_bound" : m.get("safety_time_max")
                }
                obj_st = {
                "name" : m.get("id") + "_" + "safety_stock",
                "lower_bound" : m.get("safety_stock_min"),
                "upper_bound" : m.get("safety_stock_max")
                }
                param_meta.append(obj_ss)
                param_meta.append(obj_st)
        if format_type =="lb_ub_tuple":
            for m in self.materials:
                obj_ss ={
                    "name" : m.get("id") + "_" + "safety_time",
                    "bounds" : (m.get("safety_time_min"),m.get("safety_time_max"))
                    }
        
                obj_st = {
                    "name" : m.get("id") + "_" + "safety_stock",
                    "bounds" : (m.get("safety_stock_min"),m.get("safety_stock_max"))
                    }
                    
                param_meta.append(obj_ss)
                param_meta.append(obj_st)
        param_meta = sorted(param_meta, key=lambda p: p["name"].split("_",1))
        return param_meta
                

    def get_param_meta(self):
   
        bom = formatDF(read_gsheet(self.sheet_id , "bom"))

        ids = []
        for b in bom:
            for k,v in b.items():
                if k == 'child_id' or k == 'parent_id':
                    if v not in ids and b["bom_id"] == self.bom_id:
                        ids.append(v)

        # TODO GET PARAMS META FROM SPREADSHEET
        params = []
        # ids = ["1","1001","1002","1003","1004","1010","1011","1020","2","2001","2002","2003","2004","2010", "2011", "2012", "2013", "2014", "2020", "2021", "3", "4"]
        param_types = ["safety_stock", "safety_time"]
        for id in ids:

            obj = {
                "name" : id + "_" + "safety_stock",
                "lower_bound" : 0,
                "upper_bound" : 100,
                "type" : "int",
                "fixed" : False
            }
            params.append(obj)
            obj = {
                "name" : id + "_" + "safety_time",
                "lower_bound" : 0,
                "upper_bound" : 5,
                "type" : "int",
                "fixed" : False
            }
            params.append(obj)
        params = sorted(params, key=lambda p: p["name"].split("_",1))
        return params

    def format_feature_importance(self, fi: torch.Tensor):
        if fi == "na":
            return fi
        pm_name = [pm.get("name") for pm in self.param_meta]

        fi = tensor(fi).to(tkwargs.get("device"))
        if fi.ndim == 1:
            fi = fi.tolist()
            return [dict((k,v) for k,v in zip(pm_name, fi))]
  
        fi_per_trial = []
        for _fi in fi:
            _fi = _fi.tolist()
            fi_dict = dict((k,v) for k,v in zip(pm_name, _fi))
            fi_per_trial.append(fi_dict)
        return fi_per_trial

    def run_solver(self, params):
        '''
        Expects: {material_param_name : value} (one dict, no list!)
        Returns: Releases, bom, materials. orders (all needed for sim)
        '''
        try:
            releases = MRPSolver(self.bom, self.materials, self.orders, self.stock, params,horizon=200).run()
            return releases
        except Exception as e:
            logger.error(f"Error at MRP Run: {traceback.format_exc()}")
            return None




