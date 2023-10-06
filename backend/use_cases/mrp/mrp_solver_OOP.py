import copy
import sys
import traceback
import logging
import itertools

#DEVELOPMENT
import pandas as pd
from pandas import DataFrame
sys.path.insert(0,"/code/SimBO/utils")
from gsheet_utils import read_gsheet, formatDF
from dotenv import load_dotenv

load_dotenv()
import os
from icecream import ic
logger = logging.getLogger("mrpoop")
#########################
def get_releases_from_results(mrp_results,materials, parameters):
    write_data = []
    for r in mrp_results:
        id = r["id"]
        lead_time = [m["lead_time"] for m in materials if m["id"] == id][0]
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
def filter_relevant_materials(bom_id, bom, data, filter_by_id=False):
    if isinstance(data, DataFrame):
        data = data.to_dict('records')
    ids = []
    for b in bom:
        for k,v in b.items():
            if k == 'child_id' or k == 'parent_id':
                if int(v) not in ids:
                    ids.append(int(v))
    if filter_by_id == True:
        data = [d for d in data if int(d["id"]) in ids and d.get("bom_id") == bom_id]
    else:
        data = [d for d in data if int(d["id"]) in ids]

    return data
def format_params_for_mrp(params):
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
def init_sheets(bom_id):
    params =  {"1001_safety_stock": 0,
        "1001_safety_time": 0,
        "1002_safety_stock": 0,
        "1002_safety_time": 0,
        "1003_safety_stock": 0,
        "1003_safety_time": 0,
        "1004_safety_stock": 0,
        "1004_safety_time": 0,
        "1010_safety_stock": 0,
        "1010_safety_time": 0,
        "1011_safety_stock": 0,
        "1011_safety_time": 0,
        "1020_safety_stock": 0,
        "1020_safety_time": 0,
        "1_safety_stock": 0,
        "1_safety_time": 100}
    sheet_id = os.getenv("SHEET_ID")
    bom = [b for b in formatDF(read_gsheet(sheet_id, "bom")) if b["bom_id"] == int(bom_id)]

    materials = filter_relevant_materials(bom_id, bom, formatDF(read_gsheet(sheet_id, "materials")))
    orders = filter_relevant_materials(bom_id, bom, formatDF(read_gsheet(sheet_id, "demand")), filter_by_id=True)
    inventory = filter_relevant_materials(bom_id, bom, formatDF(read_gsheet(sheet_id, "inventory")))
    return bom, materials, orders, inventory, params

def start_mrp(bom_id, run=False):
    bom, materials, orders, inventory, params = init_sheets(bom_id)
    mrp = MRPSolver(bom, materials, orders, inventory, format_params_for_mrp(params))
    if run:
        mrp.run()
    # releases = mrp.run()
    # ic(releases)




class Material():
    instances = []
    def __init__(self,material_dict:dict, bom, initial_stock, safety_stock_param, safety_time_param, horizon) -> None:
        for k, v in material_dict.items():
            setattr(self, k, v)
            self.bom = bom
            self.stock = [0] * (horizon+1)
            # greq = grossrequirement
            self.greq = [0] * (horizon+1)
            self.nreq = None
            self.released_orders = [0] * (horizon+1)
            self.stock[0] = initial_stock
            # self.stock = [initial_stock]
            self.safety_stock = safety_stock_param
            self.safety_time = safety_time_param
            self.first_period_bom = False
            Material.instances.append(self)
    @classmethod
    def get_material_by_id(cls, material_id):
        return [m for m in cls.instances if m.id == material_id][0]
    def add_safety_stock():
        # the safety stock will be added to the first order, when all requirements are calculated
        pass
    def release_order(self):
        releases = list()
 
        for i,r in  enumerate(self.greq,1):
            if r > 0:
                releases.append({
                "period" : i, 
                "material" : self.id, 
                "quantity" : float("{:.2f}".format(r)),
                "lead_time": self.lead_time,
                "period_due" :i+int(self.safety_time) + int(self.lead_time)
                })
        return releases

class MRPSolver():
    def __init__(self, bom, materials, orders, inventory, parameters, horizon=100) -> None:
        self.horizon = horizon
        self.bom = bom # 
        self.parameters = parameters
        self.orders = orders
        self.parameters = parameters
        self.inventory = inventory
        self.materials = [self.init_material(m) for m in materials]

    def init_material(self, m):
        mid = m.get("id")
        bom = [b for b in self.bom if b.get("parent_id") == mid]
        ss = [p.get("value") for p in self.parameters if p.get("id") == mid and p.get("name") == "safety_stock"][0]
        st = [p.get("value") for p in self.parameters if p.get("id") == mid and p.get("name") == "safety_time"][0]
        _is = [i.get("stock") for i in self.inventory if i.get("id") == mid]
        initial_stock = _is[0] if len(_is) > 0 else 0
        return Material(m, bom, initial_stock, ss, st, self.horizon)
    
    # equal to gross requirements
    def explode_bom(self, material:Material, quantity:float, period:int):
        if period < 0:
            logger.warning(f"Period Warning: Period {period} of Material {material.id} is < 0, so place it at Period 0")
            period = 0
        lead_time = material.lead_time + material.safety_time # NOTE: not sure if should add safety time here or in release orders. Check later
        
        # we calc gross and net req at the same time because we only handle initial stock and no lot sizes. This will change later
        material.greq[period] += quantity if material.first_period_bom == True else (quantity + material.safety_stock) 
        material.first_period_bom = True
        for b in material.bom:

            child = Material.get_material_by_id(b.get("child_id"))
            child_quantity = quantity * b.get("quantity")
            self.explode_bom(child, child_quantity, (period-lead_time))
    

    def run(self):
        '''
        This MRP Run is the simpliest version. Normally you got 4 steps:
        - 1 Explode BOM to get needed quantity of all materials and their children. Initial quantity based on orders (in our case interal and external orders are handled equal)
        - 2: Pplace it pending on lead time of parent (grossrequirement). 
        - 3: Calculate Net netrequirements (greq - (stock_at_period - safety_stock))
        - 4: place orders based on nreq and order lead time (in our case 0)

        In our case we got it that simple, that we can do everything in one step, just called explode bom
        '''
        # STEP 1: Explode BOM to get needed quantity for each material
        for o in self.orders:
            material = Material.get_material_by_id(o.get("id"))
            self.explode_bom(material, o.get("quantity"), o.get("period"))
        

        releases_listed_by_material = [m.release_order() for m in self.materials]
       
        releases = list(itertools.chain.from_iterable(releases_listed_by_material))
        releases = sorted(releases,key=lambda x: x["period"])

        return releases
        

        

start_mrp(1, run=True)