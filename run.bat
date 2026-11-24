@echo off  
set DR_WORKSTATION=1  
set PYTHONPATH=C:\Users\sapna\DR_DigitalTwin  
set OLIVES_LABEL_CSV_OVERRIDE=C:\Users\sapna\DR_DigitalTwin\data\labels\Biomarker_Clinical_Data_Images.csv  
set OLIVES_PRIME_ROOT_OVERRIDE=C:\Users\sapna\DR_DigitalTwin\data\OLIVES\Prime_FULL\Prime_FULL  
set OLIVES_TREX_ROOT_OVERRIDE=C:\Users\sapna\DR_DigitalTwin\data\OLIVES\TREX_DME\TREX DME  
cd C:\Users\sapna\DR_DigitalTwin  
.venv\Scripts\python.exe run_pipeline.py > pipeline_log.txt 2>&1  
