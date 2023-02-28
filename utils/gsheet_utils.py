import pandas as pd

import urllib
import sys
import json
import os
from dotenv import load_dotenv
from colorama import init, Fore
init(autoreset=True)

#https://docs.google.com/spreadsheets/d/1UoBr31gjg601LVGIOlSKAXTGeYvkDiTinPSu8hWIp3o/edit?usp=sharing

def read_gsheet(sheet_id, sheet_name):
    try:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
        file =pd.read_csv(url)
    except urllib.error.HTTPError as err:
        print(Fore.RED + str(err))
        print(Fore.RED + "Invalid sheet_id or sheet_name. Make sure that Sheet is published")
        sys.exit()

    return file
def get_experiment_runner_type(experiment_id):
    fpath = "configs/config" + str() +".json"
    with open(fpath, 'r') as file:
        config = json.load(file)
    
    return config.get("experiment_runner_type")

def formatDF(file):
    f = file.to_dict('records')
    for line in f:
        for k, v in line.items():
            if k == "id" or k == "parent_id" or k == "child_id" or k=='id':
                line[k] = str(v)
    return f

def get_configs_from_gsheet(from_main=False):
    load_dotenv()
    if not os.path.exists("configs"):
        os.makedirs("configs",exist_ok=True)
    sheet_id = os.getenv("SHEET_ID")
    configs = read_gsheet(sheet_id, "experiments")
    configs = configs.to_dict('records')
    for c in configs:
        obj = dict()
        experiment_id = c.get("experiment_id")
        obj["experiment_id"] = experiment_id
        obj["experiment_runner_type"] = c.get("experiment_runner_type")
        obj["algorithm_config"] = {}
        obj["use_case_config"] = {}

        for k,v in c.items():
            if k == "experiment_id":
                continue
            config_type, key = k.split("_",1)
            if config_type == "a":
                obj["algorithm_config"][key] = v
            elif config_type == "u":
                obj["use_case_config"][key] = v
            else:
                print(f"Key {k} not assignable")
        if from_main:
            fpath = os.path.join("configs",("config" + str(experiment_id) + ".json")).replace("\\","/")
        else:
            fpath = os.path.join(os.pardir, "configs",("config" + str(experiment_id) + ".json")).replace("\\","/")
        with open(fpath, "w+") as fo:
            json.dump(obj, fo)
        print(f"Saved Config{experiment_id} to path: {fpath}")

if __name__ == '__main__':
    get_configs_from_gsheet()

# read_sheet_local("bom", "C:\code\mrpSimulation\data\master", format="csv")