import os, sys

os.environ["DR_WORKSTATION"] = "1"
os.environ["PYTHONPATH"] = r"C:\Users\sapna\DR_DigitalTwin"
os.environ["OLIVES_LABEL_CSV_OVERRIDE"] = r"C:\Users\sapna\DR_DigitalTwin\data\labels\Biomarker_Clinical_Data_Images.csv"
os.environ["OLIVES_PRIME_ROOT_OVERRIDE"] = r"C:\Users\sapna\DR_DigitalTwin\data\OLIVES\Prime_FULL\Prime_FULL"
os.environ["OLIVES_TREX_ROOT_OVERRIDE"] = r"C:\Users\sapna\DR_DigitalTwin\data\OLIVES\TREX_DME\TREX DME"
os.chdir(r"C:\Users\sapna\DR_DigitalTwin")
sys.path.insert(0, r"C:\Users\sapna\DR_DigitalTwin")

class FlushFile:
    def __init__(self, f):
        self.f = f
    def write(self, x):
        self.f.write(x)
        self.f.flush()
    def flush(self):
        self.f.flush()

log = open("pipeline_log.txt", "a")
sys.stdout = FlushFile(log)
sys.stderr = FlushFile(log)

from run_pipeline import run_full_pipeline
run_full_pipeline()
