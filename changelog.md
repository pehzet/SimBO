# Changelog

## 07.10.2022
- rounding bug fixed in mrp
- reduced stochastic propability in mrp_sim (got to eval if promising) 

## 05.10.2022
- wording change: batch_* -> trial_*
- small change cma-es. Skipps initial now and generates {trial_size} arms directly
- turbo state updates correctly now
- reduced tolerance of turbo to force exploration
- increased penalty cost rate from 0.25 to 0.33 (reason: at 50D best value found at sl of 0.78)

## 04.10.2022
- set surrogate model für turbo and gpei as experiment parameter (gsheet)
- identify best trial fixed
- Bug at CMA-ES fixed. Now finds minimum instead of maximum

## 30.09.2022
- Brute Force with logs
- log every new best point

## 28.09.2022
- MRP: reduced propability of delay and reduction (stochastic)
- MRP: bug fix at releases. Now it works as it should :)
- MRP: added function to get data of every class instance (csv) 

## 23.09.2022
- wording: configs -> experiment
- fixed solver and simulation
- configs folder will be created if not exists
- further minor changes

## 21.09.2022
- wording changes at mrp_runner (inventory -> stock, demand -> orders)
- Implemented new BOMs etc. in Gsheet

## 16.09.2022
- Sobol Runner 
- Brute Force Runner
- MRP Solver OOP

## 15.09.2022
- MRP Simulation OOP

## 14.09.2022 (Phil)
- acquisition value implementiert
- feature importance hinzugefügt

## 13.09.2022 (Phil)
- Algorithmus Runner Klasse erstellt, aktuelle Runner erben davon
- MRP stochastic_method via Sheets konfigurierbar
- MRP method discrete -> "Tail" hinzugefügt
- changelog file hinzugefügt (offensichtlich)
- laden der Sheets aus der main heraus, dazu als sysarg "load" eintragen
  - Habe Funktion geschrieben zum checken der sysargs
  - Genereller Aufbau der sysargs: main.py [experiment_id] [opt:replication] [opt:"load"]
  - aber auch möglich main.py [opt:"load"] [experiment_id] [opt:replication] 
- wenn num_init/n_init -1 dann num_init = 2*dim (definiert im Algorithmus Runner)
- cmaes implementiert (WIP und hartes theoretisches Defizit meinerseits, muss lernen was die Parametrisierungen bedeuten und diese ggf. implementieren...)
- Update Gsheet Felder u_stochastic_method, a_sigma0 (Hyperparameter cma-es)
- algo runner Funktion "get_technical_specs" für experiment json Informationen. sm and acqf im SubRunner definieren.
- *_stepwise files gelöscht
- requirements.txt upgedated (cma)

# general note
- Changes einfach chronologisch (neueste oben) eintragen mit Datum und Name (siehe oben)