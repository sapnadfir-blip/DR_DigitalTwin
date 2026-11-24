"""Resume from causal analysis onwards — Layers 1 & 2 already done."""
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

log = open("pipeline_resume_log.txt", "w", encoding="utf-8")
sys.stdout = FlushFile(log)
sys.stderr = FlushFile(log)

from config import OUTPUT_DIR, CHECKPOINT_DIR, METRICS_DIR, BIOMARKERS_6, N_BM, TREATMENT_TYPES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Resuming from causal analysis...")

# Load saved sequences
seq_path = os.path.join(OUTPUT_DIR, "longitudinal_sequences.pkl")
with open(seq_path, "rb") as f:
    seq_data = pickle.load(f)

all_sequences = seq_data["all"]
sequences_test = seq_data["test"]
print(f"Loaded {len(all_sequences)} sequences")

# Load NPMI
npmi = np.load(os.path.join(OUTPUT_DIR, "npmi_matrix.npy"))

# ── Causal Analysis ──────────────────────────────────────
print("\n" + "=" * 70)
print("TEMPORAL CAUSAL BIOMARKER GRAPH (REAL DATA)")
print("=" * 70)

from layer2_progression.causal_graph import build_full_causal_analysis
causal_results = build_full_causal_analysis(all_sequences, BIOMARKERS_6, N_BM)

print(f"  Analyzed {causal_results['n_sequences']} patient-eye sequences")
print(f"  Total transitions: {causal_results['n_transitions']}")
print(f"  Significant causal edges: {len(causal_results['significant_edges'])}")

if causal_results["significant_edges"]:
    print("\n  Top causal edges:")
    for edge in causal_results["significant_edges"][:5]:
        print(f"    {edge['source']} -> {edge['target']}: "
              f"effect={edge['causal_effect']:+.3f} (p={edge['p_value']:.4f}, {edge['direction']})")

if causal_results["causal_chains"]:
    print("\n  Top causal chains:")
    for chain in causal_results["causal_chains"][:5]:
        print(f"    {' -> '.join(chain['chain'])}: strength={chain['total_strength']:.3f}")

causal_save = {
    k: v.tolist() if isinstance(v, np.ndarray) else v
    for k, v in causal_results.items()
}
causal_path = os.path.join(METRICS_DIR, "causal_analysis.json")
with open(causal_path, "w") as f:
    json.dump(causal_save, f, indent=2, default=str)
print(f"\n  Causal analysis saved to {causal_path}")

# ── Layer 3: Treatment Simulation ────────────────────────
print("\n" + "=" * 70)
print("LAYER 3: Treatment Simulation (REAL TEST PATIENTS)")
print("=" * 70)

# Load best model info from training history
hist = json.load(open(os.path.join(METRICS_DIR, "training_history.json")))
best_entry = max(hist, key=lambda r: r.get("macro_f1", 0))
best_model_name = best_entry["model"]
best_seed = best_entry["seed"]
print(f"  Best model: {best_model_name} seed={best_seed} F1={best_entry['macro_f1']:.4f}")

# Load best checkpoint
ckpt_path = os.path.join(CHECKPOINT_DIR, f"{best_model_name}_seed{best_seed}.pt")
print(f"  Loading checkpoint: {ckpt_path}")

# Load progression model
from layer2_progression.models import GraphLSTMProgression
prog_model = GraphLSTMProgression(
    n_biomarkers=N_BM, hidden_dim=512, n_layers=6,
    npmi_matrix=npmi,
)
prog_ckpt = os.path.join(CHECKPOINT_DIR, "progression_graph_lstm.pt")
prog_model.load_state_dict(torch.load(prog_ckpt, map_location=device))
prog_model.to(device).eval()

# Load biomarker model
from layer1_biomarker.models import build_model
from layer1_biomarker.train import load_graph_bundle
graph_bundle = load_graph_bundle(npmi)
bio_model = build_model(best_model_name, N_BM, graph_bundle, device)
bio_model.load_state_dict(torch.load(ckpt_path, map_location=device))
bio_model.to(device).eval()

# Load thresholds
thresh_path = os.path.join(METRICS_DIR, f"thresholds_{best_model_name}_seed{best_seed}.npy")
if os.path.exists(thresh_path):
    thresholds = np.load(thresh_path)
else:
    thresholds = np.full(N_BM, 0.5)
print(f"  Thresholds: {thresholds}")

from digital_twin.twin import PatientDigitalTwin
twin = PatientDigitalTwin(
    biomarker_model=bio_model,
    progression_model=prog_model,
    graph_bundle=graph_bundle,
    npmi_matrix=npmi,
    device=device,
    biomarker_thresholds=thresholds,
)

# Load test data
col_info = seq_data["col_info"]
from utils.data import find_column
import pandas as pd

label_csv = os.environ["OLIVES_LABEL_CSV_OVERRIDE"]
df = pd.read_csv(label_csv)
patient_col = col_info["patient_col"]
biomarker_cols = col_info["biomarker_cols"]

# Get test patient IDs from sequences
test_pids = set()
for s in sequences_test:
    test_pids.add(s["patient_id"])

patient_reports = []
for pid in sorted(test_pids):
    patient_df = df[df[patient_col] == pid] if pid in df[patient_col].values else None
    if patient_df is None or len(patient_df) == 0:
        continue

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

print("\n" + "=" * 70)
print("PIPELINE COMPLETE - ALL REAL DATA, NO SIMULATIONS")
print("=" * 70)
print("Done!")
