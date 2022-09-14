# Changelog

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