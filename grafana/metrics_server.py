"""Metrics API server for the Grafana dashboard.

Serves training metrics, patient data, and causal analysis results
as JSON endpoints that Grafana can query via the JSON API datasource.

Usage:
    python grafana/metrics_server.py          # starts on port 3001
    python grafana/metrics_server.py --port 8080
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, METRICS_DIR, CHECKPOINT_DIR, BIOMARKERS_6, N_BM


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_sequences():
    seq_path = os.path.join(OUTPUT_DIR, "longitudinal_sequences.pkl")
    if not os.path.exists(seq_path):
        return None
    with open(seq_path, "rb") as f:
        return pickle.load(f)


def get_pipeline_report():
    return load_json(os.path.join(METRICS_DIR, "pipeline_report.json"))


def get_causal_analysis():
    return load_json(os.path.join(METRICS_DIR, "causal_analysis.json"))


def get_patient_reports():
    return load_json(os.path.join(METRICS_DIR, "patient_reports.json"))


def get_training_history():
    """Load per-epoch training metrics if saved."""
    path = os.path.join(METRICS_DIR, "training_history.json")
    return load_json(path)


def build_overview_metrics():
    """Build summary metrics for the overview dashboard."""
    report = get_pipeline_report()
    if report is None:
        return {"status": "no_data", "message": "Run the pipeline first"}

    return {
        "best_model": report.get("layer1", {}).get("best_model", "N/A"),
        "macro_f1": report.get("layer1", {}).get("macro_f1", 0),
        "n_models_trained": report.get("layer1", {}).get("n_models_trained", 0),
        "progression_mae": report.get("layer2", {}).get("mae", 0),
        "progression_rmse": report.get("layer2", {}).get("rmse", 0),
        "n_causal_edges": report.get("causal", {}).get("n_significant_edges", 0),
        "n_causal_chains": report.get("causal", {}).get("n_causal_chains", 0),
        "n_test_patients": report.get("layer3", {}).get("n_test_patients", 0),
        "train_sequences": report.get("data", {}).get("train_sequences", 0),
        "val_sequences": report.get("data", {}).get("val_sequences", 0),
        "test_sequences": report.get("data", {}).get("test_sequences", 0),
        "total_sequences": (
            report.get("data", {}).get("train_sequences", 0) +
            report.get("data", {}).get("val_sequences", 0) +
            report.get("data", {}).get("test_sequences", 0)
        ),
    }


def build_biomarker_prevalence():
    """Compute biomarker prevalence from real sequences."""
    seq_data = load_sequences()
    if seq_data is None:
        return []

    all_seqs = seq_data["all"]
    total_visits = 0
    active_counts = np.zeros(N_BM)

    for seq in all_seqs:
        bm = seq["biomarker_seq"]
        total_visits += bm.shape[0]
        active_counts += (bm > 0.5).sum(axis=0)

    rows = []
    for i, name in enumerate(BIOMARKERS_6):
        rows.append({
            "biomarker": name,
            "prevalence": round(float(active_counts[i] / max(total_visits, 1)) * 100, 1),
            "active_visits": int(active_counts[i]),
            "total_visits": total_visits,
        })
    return rows


def build_patient_severity_distribution():
    """Compute per-patient severity for distribution chart."""
    seq_data = load_sequences()
    if seq_data is None:
        return []

    rows = []
    for seq in seq_data["all"]:
        bm = seq["biomarker_seq"]
        last_visit = bm[-1]
        severity = float(np.mean(last_visit))
        n_active = int((last_visit > 0.5).sum())
        rows.append({
            "patient_id": str(seq["patient_id"]),
            "eye_id": str(seq.get("eye_id", "N/A")),
            "severity": round(severity, 3),
            "active_biomarkers": n_active,
            "n_visits": bm.shape[0],
        })
    return rows


def build_causal_edges_table():
    """Get significant causal edges for the table panel."""
    causal = get_causal_analysis()
    if causal is None:
        return []
    return causal.get("significant_edges", [])


def build_causal_chains_table():
    """Get causal chains for the table panel."""
    causal = get_causal_analysis()
    if causal is None:
        return []
    return causal.get("causal_chains", [])[:15]


def build_causal_effect_matrix():
    """Get the causal effect matrix as rows for heatmap."""
    causal = get_causal_analysis()
    if causal is None:
        return []

    matrix = causal.get("causal_effect_matrix", [])
    rows = []
    for i, src in enumerate(BIOMARKERS_6):
        for j, tgt in enumerate(BIOMARKERS_6):
            if i < len(matrix) and j < len(matrix[i]):
                rows.append({
                    "source": src,
                    "target": tgt,
                    "effect": round(matrix[i][j], 4),
                })
    return rows


def build_treatment_recommendations():
    """Patient treatment recommendations from real data."""
    patients = get_patient_reports()
    if patients is None:
        return []

    summary = {}
    for p in patients:
        rec = p.get("recommendation", "unknown")
        summary[rec] = summary.get(rec, 0) + 1

    return [{"treatment": k, "count": v} for k, v in sorted(summary.items(), key=lambda x: -x[1])]


def build_treatment_effectiveness():
    """Per-treatment severity changes across patients."""
    patients = get_patient_reports()
    if patients is None:
        return []

    rows = []
    for p in patients:
        severity = p.get("final_severity", {})
        for treat, sev in severity.items():
            rows.append({
                "patient_id": p["patient_id"],
                "treatment": treat,
                "final_severity": round(sev, 3),
            })
    return rows


def build_visit_count_distribution():
    """Distribution of visit counts per patient."""
    seq_data = load_sequences()
    if seq_data is None:
        return []

    from collections import Counter
    counts = Counter(seq["biomarker_seq"].shape[0] for seq in seq_data["all"])
    return [{"visits": k, "patients": v} for k, v in sorted(counts.items())]


ENDPOINTS = {
    "/": lambda: {"status": "ok", "endpoints": list(ENDPOINTS.keys())},
    "/overview": build_overview_metrics,
    "/biomarker-prevalence": build_biomarker_prevalence,
    "/patient-severity": build_patient_severity_distribution,
    "/causal-edges": build_causal_edges_table,
    "/causal-chains": build_causal_chains_table,
    "/causal-matrix": build_causal_effect_matrix,
    "/treatment-recommendations": build_treatment_recommendations,
    "/treatment-effectiveness": build_treatment_effectiveness,
    "/visit-distribution": build_visit_count_distribution,
    "/pipeline-report": get_pipeline_report,
    "/patient-reports": get_patient_reports,
    "/training-history": get_training_history,
}


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        handler = ENDPOINTS.get(path)
        if handler is None:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "not found"}).encode())
            return

        try:
            data = handler()
            body = json.dumps(data, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress request logs


def main():
    parser = argparse.ArgumentParser(description="DR Digital Twin Metrics Server")
    parser.add_argument("--port", type=int, default=3001)
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), MetricsHandler)
    print(f"DR Digital Twin Metrics Server running on http://localhost:{args.port}")
    print(f"Endpoints: {list(ENDPOINTS.keys())}")
    print(f"Data dir: {OUTPUT_DIR}")
    print()
    print("Add this as a JSON API datasource in Grafana:")
    print(f"  URL: http://localhost:{args.port}")
    print()
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
