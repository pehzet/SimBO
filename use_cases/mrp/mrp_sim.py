import random
import math
import copy
import logging
import sys
logger = logging.getLogger("sim")

class g:
    pass

def init_mrp_sim(bom, materials, orders, sim_time=100):
    g.bom = bom 
    g.materials = materials
    g.orders = orders
    g.sim_time = sim_time

 

class Material():
    instances = []
    def __init__(self, material_dict) -> None:
        for k, v in material_dict.items():
            setattr(self, k, v)
        self.quantity_in_stock = 0
        Material.instances.append(self)

    def calc_storage_costs(self):
        return round(self.quantity_in_stock * self.storage_cost_rate,2)
    def calc_penalty_costs(self, quantity):
        return round(self.penalty_cost_rate * quantity)
    @classmethod
    def get_material_by_id(cls, material_id):
        return [m for m in cls.instances if m.id == material_id][0]
    @classmethod
    def get_material_in_stock_by_id(cls, material_id):
        m_in_stock_list = [m for m in cls.instances if m.id == material_id and m.quantity_in_stock >0]
        if len(m_in_stock_list) == 0:
            return None
        return m_in_stock_list[0]
    @classmethod
    def check_if_materials_unique(cls):
        material_ids = [m.id for m in cls.instances]
        if len(material_ids) > len(set(material_ids)):
            logger.error("Materials not unique! Can not run valide Simulation. Going to exit. Check Spreadsheet for errors")
            return False
        return True
    @classmethod
    def calc_storage_costs_all(cls):
        return [m.quantity_in_stock * m.storage_cost_rate for m in cls.instances if m.quantity_in_stock > 0]
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
    instances = []
    def __init__(self, order_dict) -> None:
        for k, v in order_dict.items():
            setattr(self, k, v)
        self.backorder = False
        Order.instances.append(self)

    @classmethod
    def get_orders_in_period(cls, period):
        return [o for o in cls.instances if (o.period == period or o.backorder == True) and o.quantity >= 0]
class Release():
    instances = []
    def __init__(self, release_dict) -> None:
        for k, v in release_dict.items():
            setattr(self, k, v)
        self.arrival = -1
        self.backorder = False
        self.material_id = self.material
        self.is_released = False
        Release.instances.append(self)
    
    @classmethod
    def get_arrivals(cls, period):
        return [r for r in cls.instances if r.arrival == period or (r.backorder == True and r.quantity >= 0)]
    @classmethod
    def get_releases(cls, period):
        return [r for r in cls.instances if (r.period == period or r.backorder == True) and r.is_released == False]
class Child():
    def __init__(self, child_id, quantity, quantity_per_unit) -> None:
        self.id = child_id
        self.quantity = quantity
        self.quantity_per_unit = quantity_per_unit
class MRPSimulation():
    def __init__(self, releases, stochastic_method="discrete") -> None:
        # Reset class instances
        Material.instances = []
        Release.instances = []
        Order.instances = []
        self.releases = [Release(r) for r in copy.deepcopy(releases)]
        self.costs = []
        self.stock = []
        self.sl = []
        self.fulfilled_orders = []
        self.materials = [Material(m) for m in copy.deepcopy(g.materials)]
        self.bom = copy.deepcopy(g.bom) # no deepcopy required?
        self.orders = [Order(o) for o in copy.deepcopy(g.orders)]
        self.stochastic_method = stochastic_method
        
        self.check()

    def check(self):
        """
        Check if MRP Simulation Data is valid
        """
        checks = list()
        checks.append(Material.check_if_materials_unique())
        if False in checks:
            sys.exit()
        return True
    def get_bom_childs_with_quantity(self, parent_id, parent_quantity):

        children = [b for b in self.bom if str(b["parent_id"]) == str(parent_id)]
        if len(children) == 0:
            return None
        bom_children_with_quant = []
        for c in children:
            child_quant = c["quantity"] * parent_quantity
            bom_children_with_quant.append(
                Child(c.get("child_id"),child_quant, c.get("quantity")))
        return bom_children_with_quant
    
    def sample_lead_time_delay(self):

        if self.stochastic_method == "discrete":
            values = [0,1,2,3,4,5,6,7]
            weights = [10,5,3,2,1,0.5,0.1,0.05]
            assert len(values) == len(weights)
            value = random.choices(values, weights,k=1)
            return value[0]
        if self.stochastic_method == "deterministic" or self.stochastic_method == None or self.stochastic_method == "None":
            return 0
        logging.warn("No method for sample lead time delay selected. Return 0")
        return 0
   
    def sample_quantity_reduction(self, quantity):

        if self.stochastic_method == "discrete":
            values = [0,math.ceil(quantity*0.01),math.ceil(quantity*0.05),math.ceil(quantity*0.1),math.ceil(quantity*0.25),math.ceil(quantity*0.5),math.ceil(quantity)]
            weights = [10,5,3,2,1,1,0.5]
            assert len(values) == len(weights)
            value = random.choices(values, weights,k=1)
            return value[0]
        if self.stochastic_method == "deterministic" or self.stochastic_method == None or self.stochastic_method == "None":
            return 0
        logging.warn("No method for sample quantity reduction selected. Return 0")
        return 0

    def run_simulation(self):
        # range starts with 1 and ends with sim_time + 1 (from start) + 1(to calc cost of previous period in case of backorder) 
        for period in range(1, g.sim_time + 2):
            
            # STEP 1: record storage sosts of last period
            self.costs.extend(Material.calc_storage_costs_all())
            
            # STEP 2: Receive Releases from previous periods (called arrivals)  
            arrivals_in_period = Release.get_arrivals(period)
            for a in arrivals_in_period:
                m = Material.get_material_by_id(a.material_id)
                _quantity = a.quantity - self.sample_quantity_reduction(a.quantity)
                if m.quantity_in_stock > 0:
                    m.quantity_in_stock += _quantity
                else:
                    m.quantity_in_stock = _quantity
                a.quantity -= _quantity
          
            # STEP 3: Check if MRP-Release is possible and follow different release-cases 
            releases_in_period = Release.get_releases(period)
            # STEP 3.1: Check possible quantity of release depending on bom-children
            for r in releases_in_period:
                children = self.get_bom_childs_with_quantity(r.material_id, r.quantity)
                if children is not None:
                    quantities_possible = []
                    for child in children:
                        child_in_stock = Material.get_material_in_stock_by_id(child.id)
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
                        r.quantity = min(quantities_possible)
                    
                    # STEP 3.3: reduce quantity in stock of each child depending on release quantity
                    for child in children:
                        child_in_stock = Material.get_material_in_stock_by_id(child.id)
                        if child_in_stock is not None:
                            child_in_stock.reduce_quantity_in_stock(r.quantity, is_parent_quant=True, quantity_per_unit = child.quantity_per_unit)
                # STEP 3.4: set arrival time of release and mark as released 
                r.arrival = r.period + r.lead_time + self.sample_lead_time_delay()
                r.is_released = True
            
            # STEP 4: Fulfill orders in Period
            orders_in_period = Order.get_orders_in_period(period)         
            for o in orders_in_period:
                # STEP 4.1 Append Penalty costs of last period if is_backorder
                m = Material.get_material_by_id(o.id)
                if o.backorder:
                    self.costs.append(m.calc_penalty_costs(o.quantity))
                
                
                # Set fullfilment quantity case specific
                _fulfill_quant = 0
                # Case Fulfillment = 0
                if m.quantity_in_stock <= 0:
                    if not o.backorder:
                            self.sl.append(0)
                            o.backorder = True
                    self.costs.append(m.calc_penalty_costs(o.quantity))
                # Case partly Fulfillment
                elif o.quantity > m.quantity_in_stock:
                    _fulfill_quant = m.quantity_in_stock
                    if not o.backorder:
                        o.backorder = True
                        self.sl.append(0)
                    self.costs.append(m.calc_penalty_costs((o.quantity - _fulfill_quant)))
                    o.quantity -= _fulfill_quant
                    m.quantity_in_stock -= _fulfill_quant
                # Case complete Fullfilment
                elif o.quantity <= m.quantity_in_stock:
                    if not o.backorder:
                        self.sl.append(1)
                    _fulfill_quant = o.quantity
                    o.quantity = 0
                
                else:
                    logging.error("Error at Order Fulfillment. Let order pass.")
                
        assert len(self.costs) > 0 and len(self.sl) > 0
        return {"costs" : int(sum(self.costs)), "service_level" : float(int((sum(self.sl)/len(self.sl))*100)/100)}
