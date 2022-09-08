# SimBO
Simulation Based Optimization in Python using Model-based Optimization Algorithms

## Setup and run experiments

**Step 1**: Install requirement.txt (ATTENTION! No Versions provided - will install the newest):

```
pip -r requirements.txt
```

**Step 2:** Make .env file with google spreadsheet id

Alternative, type on terminal:

```
SET SHEET_ID=<google_spreadsheet_id
```

**Step 3:** Get the configs by running `gheet_utils.py`

**Step 4:** Start experiment with given ID:

```
main.py [experiment_id]
```
