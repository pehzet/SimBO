import random
import math
import copy
import logging
import sys
from icecream import ic
import csv
import numpy as np
import datetime
logger = logging.getLogger("mrpsim")

# class g:
#     pass

# def init_mrp_sim(bom, materials, orders, sim_time=200):
#     g.bom = bom 
#     g.materials = materials
#     g.orders = orders
#     g.sim_time = sim_time

 

class Material():

    def __init__(self, material_dict) -> None:
        for k, v in material_dict.items():
            setattr(self, k, v)
        self.quantity_in_stock = 0
        self.penalty_costs_list = np.array([])
        self.storage_costs_list = np.array([])
        

    def calc_storage_costs(self):
        sc = round(self.quantity_in_stock * self.storage_cost_rate,2)
        self.storage_costs_list = np.append(self.storage_costs_list, sc)

        return sc
    def calc_penalty_costs(self, quantity):
        pc = round(self.penalty_cost_rate * quantity)
        self.penalty_costs_list = np.append(self.penalty_costs_list, pc)
        return pc
    # def calc_penalty_costs_exponential(self, order, period):
    #     delay = order.due_date - period
    #     pc = round(self.penalty_cost_rate**delay * order.quantity)

    def reduce_quantity_in_stock(self, quantity, is_parent_quant=False, quantity_per_unit = 0):
        if is_parent_quant:
            self.quantity_in_stock = max((math.floor(self.quantity_in_stock - (quantity*quantity_per_unit))),0)
        else:
            if quantity > self.quantity_in_stock:
                logger.warning("Reduction Quantity > Quantity in Stock. Going to set it to 0")
                self.quantity_in_stock = 0
            else:
                self.quantity_in_stock -= quantity
        return self.quantity_in_stock
class Order():

    def __init__(self, order_dict) -> None:
        for k, v in order_dict.items():
            if k.startswith("Unnamed"):
                continue
            setattr(self, k, v)
        self.backorder = False
        




class Release():
    def __init__(self, release_dict) -> None:
        for k, v in release_dict.items():
            setattr(self, k, v)
        self.arrival = -1
        self.backorder = False
        self.material_id = self.material
        
     
    

class Child():
    def __init__(self, child_id, quantity, quantity_per_unit) -> None:
        self.id = child_id
        self.quantity = quantity
        self.quantity_per_unit = quantity_per_unit
class MRPSimulation():
    def __init__(self, releases, materials, bom, orders, stochastic_method="discrete") -> None:
        self.releases = [Release(r) for r in copy.deepcopy(releases)]
        self.costs = np.array([])
        self.sl = np.array([])

        self.materials = [Material(m) for m in copy.deepcopy(materials)]
        self.materials_dict = {m.id: m for m in self.materials}
        self.bom = bom 
        self.bom_cache = {}
        self.orders = [Order(o) for o in copy.deepcopy(orders)]
        self.stochastic_method = stochastic_method
        self.storage_costs = 0
        self.penalty_costs = 0
        # self.check()

    def get_material_by_id(self, material_id):
        return self.materials_dict[str(material_id)]
    def get_material_in_stock_by_id(self, material_id):
        material = self.get_material_by_id(material_id)
        if material and material.quantity_in_stock > 0:
            return material
        return None
    def check_if_materials_unique(self):
        material_ids = [m.id for m in self.materials]
        if len(material_ids) > len(set(material_ids)):
            logger.error("Materials not unique! Can not run valide Simulation. Going to exit. Check Spreadsheet for errors")
            return False
        return True
    def calc_storage_costs_all(self):
        sc = [m.calc_storage_costs() for m in self.materials_dict.values() ]
        # sc = [m.quantity_in_stock * m.storage_cost_rate for m in self.materials if m.quantity_in_stock > 0]
        self.storage_costs += sum(sc)
        return sc

    def get_orders_in_period(self, period):
        return [o for o in self.orders if ((o.period == period or o.backorder == True) and o.quantity > 0)]

    def get_arrivals(self, period):
        # return [r for r in self.releases if r.arrival == period or (r.backorder == True and r.quantity > 0)]
        return [r for r in self.releases if r.arrival == period and r.quantity > 0]
    def get_releases(self, period):
        return [r for r in self.releases if (r.period == period) or (r.backorder == True and r.quantity > 0)]

    def check(self):
        """
        Check if MRP Simulation Data is valid
        """
        checks = list()
        checks.append(self.check_if_materials_unique())
        if False in checks:
            sys.exit()
        return True
    def get_bom_childs_with_quantity(self, parent_id, parent_quantity):
        cache_key = (parent_id, parent_quantity)
        if cache_key in self.bom_cache:
            return self.bom_cache[cache_key]
        children = [b for b in self.bom if str(b["parent_id"]) == str(parent_id)]
        if len(children) == 0:
            return None
        bom_children_with_quant = []
        for c in children:
            child_quant = c["quantity"] * parent_quantity
            bom_children_with_quant.append(
                Child(c.get("child_id"),child_quant, c.get("quantity")))
        self.bom_cache[cache_key] = bom_children_with_quant
        return bom_children_with_quant
    
    def sample_lead_time_delay(self):
        if self.stochastic_method == "discrete":
            values = [0,1,2,3]
            weights = [120,5,3,2]

            return random.choices(values, weights,k=1)[0]
        elif self.stochastic_method in ["deterministic", "None", None]:
            return 0
        logging.warn("No method for sample lead time delay selected. Return 0")
        return 0
   
    def sample_quantity_reduction(self, quantity):
        if self.stochastic_method == "discrete":
            values = [0,math.ceil(quantity*0.01),math.ceil(quantity*0.02),math.ceil(quantity*0.05)]
            weights = [60,5,3,2]
            assert len(values) == len(weights)
            value = random.choices(values, weights,k=1)
            return value[0]
        if self.stochastic_method in ["deterministic", "None", None]:
            return 0
        logging.warn("No method for sample quantity reduction selected. Return 0")
        return 0

    def run_simulation(self, sim_time=200):
        # range starts with 1 and ends with sim_time + 1 (from start) + 1(to calc cost of previous period in case of backorder) 
        for period in range(1, sim_time + 20):
            # STEP 1: record storage sosts of last period
   
            self.costs = np.append(self.costs, self.calc_storage_costs_all())

            # STEP 2: Receive Releases from previous periods (called arrivals)  
            arrivals_in_period = self.get_arrivals(period)
            for a in arrivals_in_period:
                m = self.get_material_by_id(a.material_id)
                _quantity = a.quantity - self.sample_quantity_reduction(a.quantity)

                m.quantity_in_stock += _quantity

                a.quantity -= _quantity
                if a.quantity > 0:
                    a.arrival += 1
                else:
                    a.fulfillment_date = period
                    


            # STEP 3: Check if MRP-Release is possible and follow different release-cases 
            releases_in_period = self.get_releases(period)
            # STEP 3.1: Check possible quantity of release depending on bom-children
            for r in releases_in_period:

                children = self.get_bom_childs_with_quantity(r.material_id, r.quantity)
                if children is not None:
                    quantities_possible = []
                    for child in children:
                        child_in_stock = self.get_material_in_stock_by_id(child.id)
                        # Case 1: No child quantity in stock i 0 -> no production possible
                        if child_in_stock == None:
                            quantity_possible = 0
                        # Case 2: child quantity < required quantity -> calc possible quantity based on child quantity
                        elif child_in_stock.quantity_in_stock < child.quantity:
                            quantity_possible = math.ceil((child_in_stock.quantity_in_stock/child.quantity_per_unit))
                        # Case 3: enough in stock -> full quantity can be release
                        else:
                            quantity_possible = r.quantity
                        quantities_possible.append(quantity_possible)

                    # STEP 3.2 : Create seperate Backorder if quantity in stock of one child is not sufficient
                    if any(qp < r.quantity for qp in quantities_possible):

                        _release = Release(r.__dict__)
        
                        _release.backorder = True
                        _release.quantity -= min(quantities_possible)
                        _release.arrival = period + _release.lead_time + self.sample_lead_time_delay()
            
                        self.releases.append(_release)
                        r.quantity = min(quantities_possible)

                
                    # STEP 3.3: reduce quantity in stock of each child depending on release quantity
                    for child in children:
                        child_in_stock = self.get_material_in_stock_by_id(child.id)
                        if child_in_stock is not None:
                            child_in_stock.reduce_quantity_in_stock(r.quantity, is_parent_quant=True, quantity_per_unit = child.quantity_per_unit)
                # STEP 3.4: set arrival time of release and mark as released 
                r.arrival = r.period + r.lead_time + self.sample_lead_time_delay()

            
            # STEP 4: Fulfill orders in Period
            orders_in_period = self.get_orders_in_period(period)  
      
            for o in orders_in_period:
                if not hasattr(o, "first_apperance"):
                    o.first_apperance = period
                # STEP 4.1 Append Penalty costs of last period if is_backorder
                m = self.get_material_by_id(o.id)
         
                if o.backorder:
                    self.costs = np.append(self.costs, m.calc_penalty_costs(o.quantity))
               
                # Set fullfilment quantity case specific
                _fulfill_quant = 0
                # Case Fulfillment = 0
                if m.quantity_in_stock <= 0:
                    if not o.backorder:
                            self.sl = np.append(self.sl, 0)
                            o.backorder = True
                    self.costs = np.append(self.costs, m.calc_penalty_costs(o.quantity))
    
                # Case partly Fulfillment
                elif o.quantity > m.quantity_in_stock:
                    _fulfill_quant = m.quantity_in_stock
                    if not o.backorder:
                        self.sl = np.append(self.sl, 0)
                        o.backorder = True
                    self.costs = np.append(self.costs, m.calc_penalty_costs(o.quantity - _fulfill_quant))
          
                    o.quantity -= _fulfill_quant
                    m.quantity_in_stock -= _fulfill_quant
                # Case complete Fullfilment
                elif o.quantity <= m.quantity_in_stock:
                    if not o.backorder:
                        self.sl= np.append(self.sl, 1)
                    _fulfill_quant = o.quantity
                    o.quantity = 0
                    o.fulfillment_date = period
                    m.quantity_in_stock -= _fulfill_quant
                else:
                    logging.error("Error at Order Fulfillment. Let order pass.")


        assert len(self.sl) > 0
        return {"costs" : int(np.sum(self.costs)), "service_level" : np.average(self.sl)}

    def write_costs_csv(self, all_costs, storage_costs, penalty_costs, sl):
        with open("costs.csv", "a",encoding='utf-8',newline='') as f:
            w = csv.writer(f, delimiter=";")
            w.writerow([datetime.datetime.now().isoformat(), int(all_costs), int(storage_costs), int(penalty_costs), float(sl)] )

    # FUNCTION FOR DEBUG
    def save_object_instances_csv(self):
        # orders
        file_präfix = "data/" + str(datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        with open(file_präfix +"_orders.csv", 'w',encoding='utf-8',newline='') as f:
            w = csv.DictWriter(f, self.orders[0].__dict__.keys())
            w.writeheader()
            for o in self.orders:
                if hasattr(o, "fulfillment_date"):
                    o.delay = max((o.fulfillment_date  - o.period),0)
            orders = sorted(self.orders, key=lambda m: m.period)
            for o in orders:
                w.writerow(o.__dict__)
        with open(file_präfix +"_releases.csv", 'w',encoding='utf-8',newline='') as f:
            header = list(self.releases[0].__dict__.keys()) + ["delay"]
            w = csv.DictWriter(f, header)
            w.writeheader()
            releases = sorted(self.releases, key=lambda m: m.period)
            for r in releases:
                r.delay = max(r.arrival - r.period_due,0)
                w.writerow(r.__dict__)

            num_delays = [r.delay for r in releases if r.delay > 0]

        with open(file_präfix +"_materials.csv", 'w',encoding='utf-8',newline='') as f:
            header = list(self.materials[0].__dict__.keys()) + ["storage_costs","penalty_costs"]
            w = csv.DictWriter(f, header)
            w.writeheader()
            
            for m in self.materials:
                m.storage_costs = sum(m.storage_costs_list)
                m.penalty_costs = sum(m.penalty_costs_list)
                del m.penalty_costs_list
                del m.storage_costs_list
                
                w.writerow(m.__dict__)



