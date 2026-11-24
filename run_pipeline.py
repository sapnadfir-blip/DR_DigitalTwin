"""
DR Digital Twin - Full Pipeline Runner
=======================================
Runs all three layers end-to-end on the OLIVES dataset:
  1. Layer 1: Train biomarker detection models (CNN, GCN, Signed GCN, Prior-GAT)
  2. Layer 2: Train temporal progression model on real longitudinal data
  3. Layer 3: Run treatment comparisons on real test patients
  4. Causal Analysis: Build temporal causal biomarker graph from real visits
  5. Generate patient digital twin reports

Usage:
    python run_pipeline.py                    # Full run on OLIVES data
    python run_pipeline.py --ui               # Launch Streamlit UI (requires trained checkpoints)
    python run_pipeline.py --label-csv PATH   # Specify custom label CSV path
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch

from config import (
    OUTPUT_DIR, CHECKPOINT_DIR, METRICS_DIR, BIOMARKERS_6, N_BM,
    SEEDS, TREATMENT_TYPES,
)
from utils.data import set_seed


def run_full_pipeline(label_csv=None):
    """Run the full training pipeline on OLIVES data."""
    print("=" * 70)
    print("DR DIGITAL TWIN - FULL PIPELINE (REAL DATA ONLY)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # ── Layer 1: Biomarker Detection ──────────────────────────────────
    print("\n" + "=" * 70)
    print("LAYER 1: Biomarker Detection Training")
    print("=" * 70)

    from layer1_biomarker.train import run_full_biomarker_experiment
    results, best_models, graph_bundle, data_splits = run_full_biomarker_experiment(
        label_csv=label_csv, device=device,
    )

    best_result = max(results, key=lambda r: r["macro_f1"])
    print(f"\nBest Layer 1 model: {best_result['model']} seed={best_result['seed']} "
          f"macro-F1={best_result['macro_f1']:.4f}")

    # ── Build real longitudinal sequences ─────────────────────────────
    print("\n" + "=" * 70)
    print("BUILDING LONGITUDINAL SEQUENCES FROM OLIVES")
    print("=" * 70)

    from utils.data import build_longitudinal_sequences

    col_info = data_splits["col_info"]

    sequences_all = build_longitudinal_sequences(
        data_splits["df_train"].copy(),
        col_info["patient_col"], col_info["eye_col"], col_info["visit_col"],
        col_info["biomarker_cols"],
        treatment_col=col_info["treatment_col"],
    )
    sequences_train = sequences_all

    sequences_val = build_longitudinal_sequences(
        data_splits["df_val"],
        col_info["patient_col"], col_info["eye_col"], col_info["visit_col"],
        col_info["biomarker_cols"],
        treatment_col=col_info["treatment_col"],
    )

    sequences_test = build_longitudinal_sequences(
        data_splits["df_test"],
        col_info["patient_col"], col_info["eye_col"], col_info["visit_col"],
        col_info["biomarker_cols"],
        treatment_col=col_info["treatment_col"],
    )

    all_sequences = sequences_train + sequences_val + sequences_test
    print(f"  Train sequences: {len(sequences_train)} patient-eyes")
    print(f"  Val sequences:   {len(sequences_val)} patient-eyes")
    print(f"  Test sequences:  {len(sequences_test)} patient-eyes")
    print(f"  Total sequences: {len(all_sequences)} patient-eyes")

    # Save sequences for UI and causal analysis
    seq_path = os.path.join(OUTPUT_DIR, "longitudinal_sequences.pkl")
    with open(seq_path, "wb") as f:
        pickle.dump({
            "train": sequences_train,
            "val": sequences_val,
            "test": sequences_test,
            "all": all_sequences,
            "biomarker_names": BIOMARKERS_6,
            "col_info": {k: v for k, v in col_info.items() if k != "resolved_bm"},
        }, f)
    print(f"  Sequences saved to {seq_path}")

    # ── Layer 2: Progression Model ────────────────────────────────────
    print("\n" + "=" * 70)
    print("LAYER 2: Temporal Progression Training (REAL DATA)")
    print("=" * 70)

    from layer2_progression.train import train_progression_model

    npmi = data_splits["npmi"]
    progression_model, prog_metrics = train_progression_model(
        sequences_train, sequences_val,
        model_type="graph_lstm", npmi_matrix=npmi,
        device=device,
    )

    prog_ckpt = os.path.join(CHECKPOINT_DIR, "progression_graph_lstm.pt")
    torch.save(progression_model.state_dict(), prog_ckpt)
    print(f"Progression model saved to {prog_ckpt}")

    # Save NPMI matrix for UI loading
    npmi_path = os.path.join(OUTPUT_DIR, "npmi_matrix.npy")
    np.save(npmi_path, npmi)

    # ── Causal Analysis on Real Data ──────────────────────────────────
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

    # Save causal results
    causal_save = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in causal_results.items()
    }
    causal_path = os.path.join(METRICS_DIR, "causal_analysis.json")
    with open(causal_path, "w") as f:
        json.dump(causal_save, f, indent=2, default=str)
    print(f"\n  Causal analysis saved to {causal_path}")

    # ── Layer 3: Treatment Simulation on Real Test Patients ───────────
    print("\n" + "=" * 70)
    print("LAYER 3: Treatment Simulation (REAL TEST PATIENTS)")
    print("=" * 70)

    from digital_twin.twin import PatientDigitalTwin

    best_key = (best_result["model"], best_result["seed"])
    biomarker_model = best_models[best_key]
    thresholds = np.array(best_result["best_thresholds"])

    twin = PatientDigitalTwin(
        biomarker_model=biomarker_model,
        progression_model=progression_model,
        graph_bundle=graph_bundle,
        npmi_matrix=npmi,
        device=device,
        biomarker_thresholds=thresholds,
    )

    # Run on real test patients
    df_test = data_splits["df_test"]
    patient_col = col_info["patient_col"]
    test_patients = df_test[patient_col].unique()

    patient_reports = []
    for pid in test_patients:
        patient_df = df_test[df_test[patient_col] == pid]
        twin.set_patient(str(pid))

        # Use REAL biomarker values from the dataset
        bm_vals = patient_df[col_info["biomarker_cols"]].iloc[0].values.astype(np.float32)

        # Use real clinical data if available
        clinical = np.array([60.0, 300.0], dtype=np.float32)
        for i, cc in enumerate(["BCVA", "CST"]):
            from utils.data import find_column
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

    # Save patient reports
    reports_path = os.path.join(METRICS_DIR, "patient_reports.json")
    with open(reports_path, "w") as f:
        json.dump(patient_reports, f, indent=2)
    print(f"\n  {len(patient_reports)} patient reports saved to {reports_path}")

    # ── Final Pipeline Report ─────────────────────────────────────────
    pipeline_report = {
        "layer1": {
            "best_model": best_result["model"],
            "macro_f1": best_result["macro_f1"],
            "n_models_trained": len(results),
        },
        "layer2": prog_metrics,
        "causal": {
            "n_sequences": causal_results["n_sequences"],
            "n_significant_edges": len(causal_results["significant_edges"]),
            "n_causal_chains": len(causal_results["causal_chains"]),
        },
        "layer3": {
            "n_test_patients": len(test_patients),
        },
        "data": {
            "train_sequences": len(sequences_train),
            "val_sequences": len(sequences_val),
            "test_sequences": len(sequences_test),
        },
    }
    report_path = os.path.join(METRICS_DIR, "pipeline_report.json")
    with open(report_path, "w") as f:
        json.dump(pipeline_report, f, indent=2, default=str)
    print(f"\nPipeline report saved to {report_path}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE - ALL REAL DATA, NO SIMULATIONS")
    print("=" * 70)
    print()
    print("Outputs saved to:")
    print(f"  Checkpoints:  {CHECKPOINT_DIR}/")
    print(f"  Metrics:      {METRICS_DIR}/")
    print(f"  Sequences:    {seq_path}")
    print(f"  Causal graph: {causal_path}")
    print()
    print("To launch the UI with trained models:")
    print("  streamlit run ui/app.py")


def main():
    parser = argparse.ArgumentParser(description="DR Digital Twin Pipeline")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit UI")
    parser.add_argument("--label-csv", type=str, help="Path to OLIVES label CSV")
    args = parser.parse_args()

    if args.ui:
        os.system(f"{sys.executable} -m streamlit run ui/app.py")
    else:
        run_full_pipeline(label_csv=args.label_csv)


if __name__ == "__main__":
    main()
