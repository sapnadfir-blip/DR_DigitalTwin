"""JSON + Prometheus metrics API for Grafana dashboard."""
import json, os, http.server, socketserver, pathlib

BASE = pathlib.Path(__file__).resolve().parent
METRICS_DIR = BASE / "outputs" / "metrics"

layer1 = json.loads((METRICS_DIR / "layer1_results.json").read_text())
causal = json.loads((METRICS_DIR / "causal_analysis.json").read_text())
patients = json.loads((METRICS_DIR / "patient_reports.json").read_text())
history = json.loads((METRICS_DIR / "training_history.json").read_text())

best = max(layer1, key=lambda r: r["macro_f1"])


def _build_prometheus():
    lines = [
        "# HELP dr_macro_f1 Best macro F1 score",
        "# TYPE dr_macro_f1 gauge",
        f'dr_macro_f1 {best["macro_f1"]}',
        "# TYPE dr_micro_f1 gauge",
        f'dr_micro_f1 {best["micro_f1"]}',
        "# TYPE dr_auroc gauge",
        f'dr_auroc {best["mean_auroc"]}',
        "# TYPE dr_causal_edges gauge",
        f'dr_causal_edges {len(causal["significant_edges"])}',
        "# TYPE dr_test_patients gauge",
        f'dr_test_patients {len(patients)}',
        "# TYPE dr_models_trained gauge",
        f'dr_models_trained {len(layer1)}',
        "# TYPE dr_total_epochs gauge",
        f'dr_total_epochs {len(history)}',
    ]
    for m in layer1:
        lbl = f'model="{m["model"]}",seed="{m["seed"]}"'
        lines.append(f"dr_model_macro_f1{{{lbl}}} {m['macro_f1']}")
        lines.append(f"dr_model_auroc{{{lbl}}} {m['mean_auroc']}")
    for e in causal["significant_edges"]:
        lbl = f'source="{e["source"]}",target="{e["target"]}"'
        lines.append(f"dr_causal_effect{{{lbl}}} {e['causal_effect']}")
    return "\n".join(lines) + "\n"


PROM_TEXT = _build_prometheus()


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/metrics", "/api/prometheus"):
            self._text(PROM_TEXT)
        elif self.path == "/api/health":
            self._json({"status": "ok"})
        elif self.path == "/api/metrics":
            self._json([{
                "best_model": best["model"],
                "best_seed": best["seed"],
                "macro_f1": best["macro_f1"],
                "micro_f1": best["micro_f1"],
                "mean_auroc": best["mean_auroc"],
                "causal_edges": len(causal["significant_edges"]),
                "causal_chains": len(causal["causal_chains"]),
                "test_patients": len(patients),
                "models_trained": len(layer1),
                "total_epochs": len(history),
            }])
        elif self.path == "/api/models":
            self._json(layer1)
        elif self.path == "/api/training":
            self._json(history)
        elif self.path == "/api/causal":
            self._json(causal)
        elif self.path == "/api/patients":
            self._json(patients)
        else:
            self._json({"endpoints": ["/metrics", "/api/health", "/api/metrics",
                                      "/api/models", "/api/training", "/api/causal",
                                      "/api/patients"]})

    def _json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _text(self, text):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(text.encode())

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    with socketserver.TCPServer(("", 9100), Handler) as httpd:
        print("Metrics server on port 9100 (JSON + Prometheus)")
        httpd.serve_forever()
