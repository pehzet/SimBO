import os
from supabase import create_client, Client
from icecream import ic
from datetime import datetime
import json
from dotenv import load_dotenv
from pathlib import Path
import pathlib
load_dotenv()


def _init_supabase():
    URL: str = os.getenv("SUPABASE_URL")
    KEY: str = os.getenv("SUPABASE_KEY")
    client = create_client(URL, KEY)
    return client

def sb_tables_insert_data(data:dict, table:str = "experiments"):

    supabase = _init_supabase()
    res = supabase.table(table).insert(data).execute()
    print("Uploaded Data to Supabase Tables")

def sb_update_arms_completed(experiment_name: str, num_arms_completed: int):
    update_object = {
        "arms_completed" : num_arms_completed
    }
    supabase = _init_supabase()
    try:
        res = supabase.table("experiments").update(update_object).eq("name", experiment_name).execute()
        print("Updated Arms completed in Supabase")
    except BaseException as e:
        print(f"Error < {e} < at updating Supabase Table Experiments with new Number of Arms, Experiment: {experiment_name}")

def sb_update_experiment_status(experiment_name: str, status: str = "closed"):
    if not status in ["running", "pending", "error", "closed"]:
        print(f"ValidationError: Status > {status} < not in ['running', 'pending', 'error', 'closed']")
        return
    update_object = {
        "status" : status
    }
    supabase = _init_supabase()
    try:
        res = supabase.table("experiments").update(update_object).eq("name", experiment_name).execute()
        print("Updated Arms completed in Supabase")
    except BaseException as e:
        print(f"Error < {e} < at updating Supabase Table Experiments with new Number of Arms, Experiment: {experiment_name}")


def sb_upload_experiment_jsons(experiment_name: str):
    supabase = _init_supabase()
    filenames = [experiment_name + ".json", experiment_name + "_responses.json"]
    for f in filenames:
        path = Path(f"C:/code/black-box-opt-server-api/data/{f}")
        try:
            res = supabase.storage().from_("experiment-jsons").upload(f, path)
        except BaseException as e:
            print(f"Error > {e} <at uploading JSON files")
    print("Uploaded Experiment JSONs to Supabase Storage")


# ename = "2022_08_03_124511_Model_Experiment1"
