"""Simple JSON metrics API for Grafana dashboard."""
import json, os, http.server, socketserver

os.chdir(r"C:\Users\sapna\DR_DigitalTwin")
METRICS_DIR = "./outputs/metrics"

layer1 = json.load(open(os.path.join(METRICS_DIR, "layer1_results.json")))
causal = json.load(open(os.path.join(METRICS_DIR, "causal_analysis.json")))
patients = json.load(open(os.path.join(METRICS_DIR, "patient_reports.json")))
history = json.load(open(os.path.join(METRICS_DIR, "training_history.json")))

best = max(layer1, key=lambda r: r["macro_f1"])

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/health":
            self._json({"status": "ok"})
        elif self.path == "/api/metrics":
            self._json({
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
            })
        elif self.path == "/api/models":
            self._json(layer1)
        elif self.path == "/api/training":
            self._json(history)
        elif self.path == "/api/causal":
            self._json(causal)
        elif self.path == "/api/patients":
            self._json(patients)
        else:
            self._json({"endpoints": ["/api/health", "/api/metrics", "/api/models", "/api/training", "/api/causal", "/api/patients"]})

    def _json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass

with socketserver.TCPServer(("", 9100), Handler) as httpd:
    print("Metrics server on port 9100")
    httpd.serve_forever()
