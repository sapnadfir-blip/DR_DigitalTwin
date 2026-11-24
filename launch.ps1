$env:DR_WORKSTATION = "1"
$env:PYTHONPATH = "C:\Users\sapna\DR_DigitalTwin"
$env:OLIVES_LABEL_CSV_OVERRIDE = "C:\Users\sapna\DR_DigitalTwin\data\labels\Biomarker_Clinical_Data_Images.csv"
$env:OLIVES_PRIME_ROOT_OVERRIDE = "C:\Users\sapna\DR_DigitalTwin\data\OLIVES\Prime_FULL\Prime_FULL"
$env:OLIVES_TREX_ROOT_OVERRIDE = "C:\Users\sapna\DR_DigitalTwin\data\OLIVES\TREX_DME\TREX DME"
Set-Location "C:\Users\sapna\DR_DigitalTwin"
& ".venv\Scripts\python.exe" run_pipeline.py *> pipeline_log.txt
