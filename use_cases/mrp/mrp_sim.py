import random
import math
import copy
from icecream import ic
import sys
class g:
    pass



def init_mrp_sim(bom, materials, orders, sim_time=100):
    g.bom = bom 
    g.materials = materials
    g.orders = orders

    g.sim_time = sim_time
    for o in g.orders:
        o["backorder"] = False
 
def get_bom_childs_with_quantity(parent_id, parent_quantity, bom):
    try: 
        parent_id = str(parent_id)
    except:
        if parent_id.startswith("M") or parent_id.startswith("O"):
            parent_id = parent_id.split("_")[1]
        else:
            print("ERROR: SOMETHING WRONG WITH PARENT ID")

    children = [b for b in bom if str(b["parent_id"]) == str(parent_id)]
    bom_children_with_quant = []
    for child in children:
        child_quant = child["quantity"] * parent_quantity
        bom_children_with_quant.append({
            "child_id" : child["child_id"],
            "quantity" : child_quant,
            "quantity_per_unit" : child["quantity"]
        })
    return bom_children_with_quant


def sample_lead_time_delay(method="discrete"):
    if method == "discrete":
        values = [0,1,2,3]
        weights = [10,5,3,2]
        value = random.choices(values, weights,k=1)
        return value[0]
    if method == "deterministic" or method == None:
        return 0
    print("No method for sample lead time delay selected. Return 0")
    return 0

def sample_quantity_reduction(quantity, method="discrete"):
    if method == "discrete":
        values = [0,math.ceil(quantity*0.01),math.ceil(quantity*0.05),math.ceil(quantity*0.1)]
        weights = [10,5,3,2]
        value = random.choices(values, weights,k=1)
        return value[0]
    if method == "deterministic" or method == None:
        return 0
    print("No method for sample quantity reduction selected. Return 0")
    return 0

all_order_ids = []
class mrp_simulation:
    def __init__(self):

        self.costs = []
        self.stock = []
        self.sl = []
        self.fulfilled_orders = []
        self.materials = copy.deepcopy(g.materials)
        self.bom = copy.deepcopy(g.bom)
        self.orders = copy.deepcopy(g.orders)
  
    def run_simulation(self, releases):
      
        for r in releases:
            r["arrival"] = -1
            r["backorder"] = False

        for _ in range(g.sim_time):
            
            period = _ + 1
            for s in self.stock:
                self.costs.append(s.get("quantity") * s.get("storage_cost_rate"))
            arrivals_in_period = [r for r in releases if r.get("arrival") == period or (r.get("backorder") == True and r.get("quantity")>0)]
            for arrival in arrivals_in_period:
                material_id = str(arrival["material"])
                mat_in_stock = [s for s in self.stock if s["material"] == material_id]
                _quantity = arrival.get("quantity") - sample_quantity_reduction(arrival.get("quantity"), method="discrete")
                if len(mat_in_stock) == 0:
                    unit_cost, storage_cost_rate, penalty_cost_rate = [(m.get("unit_costs"), m.get("storage_cost_rate"), m.get("penalty_cost_rate")) for m in self.materials if str(m.get("id")) == material_id][0]
                    self.stock.append({
                        "material" : material_id,
                        "quantity" : _quantity,
                        "unit_cost" : unit_cost,
                        "storage_cost_rate" : storage_cost_rate,
                        "penalty_cost_rate" : penalty_cost_rate,

                    })
                else:
                    mat_in_stock[0]["quantity"] += _quantity

                arrival["quantity"] -= _quantity

            releases_in_period = [r for r in releases if r.get("period") == period or (r.get("backorder") == True and r.get("quantity")>0)]
            for release in releases_in_period:
                if not "backorder" in release.keys():
                    release["backorder"] = False
                material_id = str(release["material"])
                _quant = release.get("quantity")
                children = get_bom_childs_with_quantity(material_id, _quant, self.bom)
                quantities_possible = []
                for child in children:
                    child_in_stock_list = [s for s in self.stock if str(s["material"]) == str(child.get("child_id"))]
                    if len(child_in_stock_list) == 0:
                        quantity_possible = 0
                    else:
                        child_in_stock = child_in_stock_list[0]
                        if child_in_stock["quantity"] < child.get("quantity"):
                            quantity_possible = math.ceil((child_in_stock["quantity"]/child.get("quantity_per_unit")))
                        else:
                            quantity_possible = _quant
                    quantities_possible.append(quantity_possible)

                if len(quantities_possible) > 0 and any(qp < _quant for qp in quantities_possible):
                
                    _release = copy.deepcopy(release)
                    _release["backorder"] = True
                    _release["quantity"] = _quant - min(quantities_possible)
                    _release["arrival"] = period + _release.get("lead_time") + sample_lead_time_delay(method="discrete")
                    _quant = min(quantities_possible)
                    releases.append(_release)
            
                release["arrival"] = release["period"] + release["lead_time"] + sample_lead_time_delay(method="discrete")

                release["quantity"] =  _quant
                for child in children:
                    child_in_stock_list = [s for s in self.stock if s["material"] == str(child.get("child_id"))]
                    if len(child_in_stock_list) > 0:
                        child_in_stock = child_in_stock_list[0]
                        child_in_stock["quantity"] = max((math.floor(child_in_stock["quantity"] - (_quant*child.get("quantity_per_unit")))),0)
                    

            orders_in_period = [o for o in self.orders if (o["period"] == period or o["backorder"] == True) and o["order_id"] not in self.fulfilled_orders]
            
            for o in orders_in_period:

                m_id = str(o.get("id"))
                                
                if o["backorder"] == True:
                    penalty_cost_rate = [(m.get("penalty_cost_rate")) for m in self.materials if str(m.get("id")) == m_id][0]
                    self.costs.append(o.get("quantity")*penalty_cost_rate)
                _fulfill_quant = 0
                product_in_stock_list = [s for s in self.stock if s["material"] == m_id]
                product_in_stock = product_in_stock_list[0] if len(product_in_stock_list) > 0 else None
                
                # Case Fulfillment = 0
                if product_in_stock == None or product_in_stock.get("quantity") == 0:
                    if o["backorder"] == False:
                            self.sl.append(0)
                            o["backorder"] = True
                    penalty_cost_rate = [(m.get("penalty_cost_rate")) for m in self.materials if str(m.get("id")) == m_id][0]
                    self.costs.append(o.get("quantity")*penalty_cost_rate)
                # Case partly Fulfillment
                elif o.get("quantity") > product_in_stock.get("quantity"):
                    _fulfill_quant = product_in_stock.get("quantity")
                    penalty_cost_rate = product_in_stock.get("penalty_cost_rate")
                    if o["backorder"] == False:
                        o["backorder"] = True
                        self.sl.append(0)
                    self.costs.append((o.get("quantity") - _fulfill_quant)*penalty_cost_rate)
                    o["quantity"] -= _fulfill_quant
                    product_in_stock["quantity"] -= _fulfill_quant
                # Case complete Fullfilment
                elif o.get("quantity") <= product_in_stock.get("quantity"):
                    if o["backorder"] == False:
                        self.sl.append(1)
                    _fulfill_quant = o.get("quantity")
                    self.fulfilled_orders.append(o.get("order_id"))
                else:
                    print("Mistake at Order Fulfillment. Let order pass")
                    

        assert len(self.costs) > 0 and len(self.sl) > 0
        print(f"Finished with costs: {int(sum(self.costs))} and service level : {float(int((sum(self.sl)/len(self.sl))*100)/100)} ")
        #return({"costs" : int(sum(self.costs)), "service_level" : float(int((sum(self.sl)/len(self.sl))*100)/100)})


        return {"costs" : int(sum(self.costs))}

