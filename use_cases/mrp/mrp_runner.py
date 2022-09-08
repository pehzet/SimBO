
# from utils import *
import os
import sys
# from main import run_solver
from icecream import ic
import pandas as pd
from pandas import DataFrame
from utils.gsheet_utils import read_gsheet, formatDF
from use_cases.mrp.mrp_solver import run_mrp
import os
import traceback
from dotenv import load_dotenv
load_dotenv()
class g:
    pass
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

def filter_relevant_materials(bom, bom_id, data):
    if isinstance(data, DataFrame):
        data = data.to_dict('records')
    ids = []
    for b in bom:
        for k,v in b.items():
            if k == 'child_id' or k == 'parent_id':
                if int(v) not in ids:
                    ids.append(int(v))

  
    
    data = [d for d in data if int(d["id"]) in ids and d.get("bom_id", bom_id) == bom_id]

    return data

def get_releases_from_results(results, materials, parameters):
    
    write_data = []
    for r in results:
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

def init_sheets(bom_id=1):
    sheet_id = os.getenv("SHEET_ID")
    file_pre = ""
    bom = [b for b in formatDF(read_gsheet(sheet_id, "bom")) if b["bom_id"] == int(bom_id)]

    materials = filter_relevant_materials(bom, bom_id, formatDF(read_gsheet(sheet_id, "materials")))
    orders = filter_relevant_materials(bom, bom_id,formatDF(read_gsheet(sheet_id, "demand")))
    inventory = filter_relevant_materials(bom, bom_id,formatDF(read_gsheet(sheet_id, "inventory")))
    print("Sheets initialized")
    return bom, materials, orders, inventory


def get_param_meta_from_materials(materials, format_type="ax"):
    """
    format_types = ["ax", "lb_ub_name", "lb_ub_tuple", ]
    """
    param_meta = []
    if format_type =="ax":

        for m in materials:
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
        for m in materials:
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
            

def get_param_meta(bom_id=1):
    sheet_id = sheet_id = os.getenv("SHEET_ID")
    bom = formatDF(read_gsheet(sheet_id, "bom"))

    ids = []
    for b in bom:
        for k,v in b.items():
            if k == 'child_id' or k == 'parent_id':
                if v not in ids and b["bom_id"] == bom_id:
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

def init_mrp_runner(bom_id, bom, materials, orders, inventory):
    g.bom_id = bom_id
    g.bom = bom
    g.materials = materials
    g.orders = orders
    g.inventory = inventory

def run_solver(params):
    '''
    Expects: {material_param_name : value} (one dict, no list!)
    Returns: Releases, bom, materials. orders (all needed for sim)
    '''
    # print(f"params: {params}")
    #params = format_params_for_mrp(params)
    try:
        mrp_results = run_mrp(g.bom, g.materials, g.orders, g.inventory, params,horizon=120)
        releases = get_releases_from_results(mrp_results,g.materials, params)
        return releases
    except Exception as e:
        print("Error at MRP Run")
        print(traceback.format_exc())
     
        return None




