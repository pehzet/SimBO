import copy
from platform import release
import sys
import traceback
import logging
import itertools
logger = logging.getLogger("mrpsolver")
from icecream import ic



class Material():

    def __init__(self,material_dict:dict, bom, initial_stock, safety_stock_param, safety_time_param, horizon) -> None:
        for k, v in material_dict.items():
            setattr(self, k, v)
            self.bom = bom
            self.greq = [0] * (horizon+1)
            self.released_orders = [0] * (horizon+1)
            self.initial_stock = initial_stock
            self.safety_stock = safety_stock_param
            self.safety_time = safety_time_param
            self.first_period_bom = False
            

    def release_order(self):
        releases = list()
        for i,r in  enumerate(self.greq):
            if r > 0:
                releases.append({
                "period" : i , 
                "material" : self.id, 
                "quantity" : float("{:.2f}".format(r)),
                "lead_time": self.lead_time,
                "period_due" : i + (int(self.safety_time) + int(self.lead_time))
                })
        return releases

class MRPSolver():
    def __init__(self, bom, materials, orders, stock, parameters, horizon=200) -> None:
        self.horizon = horizon
        self.bom = bom 
        self.parameters = parameters
        self.orders = orders
        self.parameters = parameters
        self.stock = stock
        self.materials = [self.init_material(m) for m in materials]

    def init_material(self, m):
        mid = m.get("id")
        bom = [b for b in self.bom if b.get("parent_id") == mid]
        ss = [p.get("value") for p in self.parameters if p.get("id") == mid and p.get("name") == "safety_stock"][0]
        st = [p.get("value") for p in self.parameters if p.get("id") == mid and p.get("name") == "safety_time"][0]
        _is = [i.get("stock") for i in self.stock if i.get("id") == mid]
        initial_stock = _is[0] if len(_is) > 0 else 0
        return Material(m, bom, initial_stock, ss, st, self.horizon)

    def get_material_by_id(self, material_id):
        return [m for m in self.materials if m.id == material_id][0]
    def explode_bom(self, material:Material, quantity:float, period:int):
        if period < 0:
            logger.warning(f"Period Warning: Period {period} of Material {material.id} is less than 0, so place it at Period 0")
            period = 0
        else:
            period = period - (material.lead_time + material.safety_time)

        quantity = quantity if material.first_period_bom == True else (quantity + material.safety_stock - material.initial_stock)
    
        material.greq[period] += quantity
   
        material.first_period_bom = True
        for b in material.bom:
            child = self.get_material_by_id(b.get("child_id"))
            child_quantity = quantity * b.get("quantity")

            self.explode_bom(child, child_quantity, period)
    

    def run(self):
        '''
        This MRP Run is the simpliest version. Normally you got 4 steps:
        - 1: Explode BOM to get needed quantity of all materials and their children. Initial quantity based on orders (in our case interal and external orders are handled equal)
        - 2: Place it pending on lead time of parent (grossrequirement). 
        - 3: Calculate Net netrequirements (greq - (stock_at_period - safety_stock))
        - 4: place orders based on nreq and order lead time (in our case 0)

        In our case we got it that simple, that we can do everything in one step, just called explode bom
        '''
        # STEP 1: Explode BOM to get needed quantity for each material
        for o in self.orders:
            material = self.get_material_by_id(o.get("id"))
            self.explode_bom(material, o.get("quantity"), o.get("period"))

        releases_listed_by_material = [m.release_order() for m in self.materials]
        releases = list(itertools.chain.from_iterable(releases_listed_by_material))
        # ic(sorted(releases,key=lambda x: x["period"]))

        return sorted(releases,key=lambda x: x["period"])
        

        

