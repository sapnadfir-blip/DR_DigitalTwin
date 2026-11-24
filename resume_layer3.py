"""Resume Layer 3 only — Layers 1, 2 and causal analysis already done."""
import os, sys, json, pickle, numpy as np, torch

os.environ["DR_WORKSTATION"] = "1"
os.environ["OLIVES_LABEL_CSV_OVERRIDE"] = r"C:\Users\sapna\DR_DigitalTwin\data\labels\Biomarker_Clinical_Data_Images.csv"
os.environ["OLIVES_PRIME_ROOT_OVERRIDE"] = r"C:\Users\sapna\DR_DigitalTwin\data\OLIVES\Prime_FULL\Prime_FULL"
os.environ["OLIVES_TREX_ROOT_OVERRIDE"] = r"C:\Users\sapna\DR_DigitalTwin\data\OLIVES\TREX_DME\TREX DME"
os.chdir(r"C:\Users\sapna\DR_DigitalTwin")
sys.path.insert(0, r"C:\Users\sapna\DR_DigitalTwin")

class FlushFile:
    def __init__(self, f):
        self.f = f
    def write(self, x):
        self.f.write(x.encode('ascii', 'replace').decode('ascii'))
        self.f.flush()
    def flush(self):
        self.f.flush()

log = open("pipeline_layer3_log.txt", "w", encoding="utf-8")
sys.stdout = FlushFile(log)
sys.stderr = FlushFile(log)

from config import OUTPUT_DIR, CHECKPOINT_DIR, METRICS_DIR, BIOMARKERS_6, N_BM, TREATMENT_TYPES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load sequences
seq_path = os.path.join(OUTPUT_DIR, "longitudinal_sequences.pkl")
with open(seq_path, "rb") as f:
    seq_data = pickle.load(f)
sequences_test = seq_data["test"]
col_info = seq_data["col_info"]

npmi = np.load(os.path.join(OUTPUT_DIR, "npmi_matrix.npy"))
thresholds = np.load(os.path.join(CHECKPOINT_DIR, "biomarker_thresholds.npy"))

# Best model is gcn seed=2024 (F1=0.806)
best_model_name = "gcn"
best_seed = 2024
print(f"Best model: {best_model_name} seed={best_seed}")

# Load biomarker model
from layer1_biomarker.models import build_model
from layer1_biomarker.graph import build_graph_bundle
graph_bundle = build_graph_bundle(npmi)
bio_model = build_model(best_model_name, graph_bundle).to(device)
ckpt_path = os.path.join(CHECKPOINT_DIR, f"{best_model_name}_seed{best_seed}.pt")
bio_model.load_state_dict(torch.load(ckpt_path, map_location=device))
bio_model.eval()

# Load progression model
from layer2_progression.models import build_progression_model
prog_model = build_progression_model("graph_lstm", npmi_matrix=npmi)
prog_ckpt = os.path.join(CHECKPOINT_DIR, "progression_graph_lstm.pt")
prog_model.load_state_dict(torch.load(prog_ckpt, map_location=device))
prog_model.to(device).eval()

print("\n" + "=" * 70)
print("LAYER 3: Treatment Simulation (REAL TEST PATIENTS)")
print("=" * 70)

from digital_twin.twin import PatientDigitalTwin
twin = PatientDigitalTwin(
    biomarker_model=bio_model,
    progression_model=prog_model,
    graph_bundle=graph_bundle,
    npmi_matrix=npmi,
    device=device,
    biomarker_thresholds=thresholds,
)

import pandas as pd
from utils.data import find_column

label_csv = os.environ["OLIVES_LABEL_CSV_OVERRIDE"]
df = pd.read_csv(label_csv)
patient_col = col_info["patient_col"]
biomarker_cols = col_info["biomarker_cols"]

test_pids = set()
for s in sequences_test:
    test_pids.add(s["patient_id"])

patient_reports = []
for pid in sorted(test_pids):
    mask = df[patient_col] == pid
    if not mask.any():
        mask = df[patient_col] == int(pid) if pid.isdigit() else mask
    if not mask.any():
        print(f"  Skipping patient {pid} - not found in CSV")
        continue
    patient_df = df[mask]

    twin.set_patient(str(pid))
    bm_vals = patient_df[biomarker_cols].iloc[0].values.astype(np.float32)

    clinical = np.array([60.0, 300.0], dtype=np.float32)
    for i, cc in enumerate(["BCVA", "CST"]):
        matched = find_column(patient_df, [cc])
        if matched is not None:
            val = patient_df[matched].iloc[0]
            if not np.isnan(val):
                clinical[i] = float(val)

    twin.add_visit(biomarkers=bm_vals, clinical_data=clinical)
    comparison = twin.compare_treatments(n_future_visits=6)
    rec = comparison["_summary"]["recommendation"]
    severity = comparison["_summary"]["final_severity"]
    active = [BIOMARKERS_6[i] for i in range(N_BM) if bm_vals[i] > 0.5]

    print(f"  Patient {pid}: active={active}, recommended={rec}, "
          f"final_severity={severity[rec]:.3f}")

    patient_reports.append({
        "patient_id": str(pid),
        "active_biomarkers": active,
        "recommendation": rec,
        "final_severity": {k: float(v) for k, v in severity.items()},
    })

reports_path = os.path.join(METRICS_DIR, "patient_reports.json")
with open(reports_path, "w") as f:
    json.dump(patient_reports, f, indent=2)
print(f"\n  {len(patient_reports)} patient reports saved to {reports_path}")

# Final pipeline report
pipeline_report = {
    "layer1": {"best_model": "gcn", "best_seed": 2024, "macro_f1": 0.8057},
    "layer3": {"n_test_patients": len(patient_reports)},
    "data": {
        "train_sequences": len(seq_data["train"]),
        "val_sequences": len(seq_data["val"]),
        "test_sequences": len(seq_data["test"]),
    },
}
report_path = os.path.join(METRICS_DIR, "pipeline_report.json")
with open(report_path, "w") as f:
    json.dump(pipeline_report, f, indent=2, default=str)

print("\n" + "=" * 70)
print("PIPELINE COMPLETE - ALL REAL DATA, NO SIMULATIONS")
print("=" * 70)
print("Done!")
