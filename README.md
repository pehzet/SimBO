# SimBO
Simulation Based Optimization in Python using Model-based Optimization Algorithms
The Bayesian Optimization Algorithms run with botorch. Normally the newest version of botorch works, except of MorBO, here we need botorch 0.7.0.
## Setup and run experiments

**Step 1**: Install requirement.txt (ATTENTION! No Versions provided - will install the newest):

```
pip -r requirements.txt
```

**Step 2:** Create config.py file in the root directory and set the following variables:

SHEET_ID = "SET YOUR SHEET ID HERE"
FIREBASE_CONFIG = "SET YOUR FIREBASE CONFIG PATH HERE"
BUCKET = "SET YOUR BUCKET NAME HERE"
BIGQUERY_DATASET = "SET YOUR BIGQUERY DATASET NAME HERE"
GCLOUD_PROJECT = "SET YOUR GCLOUD PROJECT NAME HERE"
GCLOUD_SERVICE_ACCOUNT = "SET YOUR GCLOUD SERVICE ACCOUNT PATH HERE"
DB_NAME = "YOU DB NAME"

**Step 3:** initialize database with:

```
python backend/databases/init_sql.py
```

**Step 4:** Start experiments with:
```
python main.py 
```
