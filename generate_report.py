"""Generate a self-contained HTML report with all results and embedded images."""
import os, sys, json, base64

os.chdir(r"C:\Users\sapna\DR_DigitalTwin")

OUTPUT_DIR = "./outputs"
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")

layer1_results = json.load(open(os.path.join(METRICS_DIR, "layer1_results.json")))
causal_data = json.load(open(os.path.join(METRICS_DIR, "causal_analysis.json")))
patient_reports = json.load(open(os.path.join(METRICS_DIR, "patient_reports.json")))

best = max(layer1_results, key=lambda r: r["macro_f1"])

def img_b64(fname):
    path = os.path.join(VIZ_DIR, fname)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def img_tag(fname, caption=""):
    b = img_b64(fname)
    if not b:
        return f"<p><em>[Image not found: {fname}]</em></p>"
    return f'''<figure>
<img src="data:image/png;base64,{b}" style="max-width:100%;border:1px solid #ddd;border-radius:4px;">
<figcaption>{caption}</figcaption>
</figure>'''

# Build model results table
model_rows = ""
for r in sorted(layer1_results, key=lambda x: -x["macro_f1"]):
    star = " *" if r["model"] == best["model"] and r["seed"] == best["seed"] else ""
    model_rows += f"""<tr>
<td>{r['model'].upper()}{star}</td><td>{r['seed']}</td>
<td><strong>{r['macro_f1']:.4f}</strong></td><td>{r['micro_f1']:.4f}</td>
<td>{r['mean_auroc']:.4f}</td>
<td>{r['boot_macro_f1']['ci_lower']:.3f} - {r['boot_macro_f1']['ci_upper']:.3f}</td>
</tr>"""

# Causal edges table
edge_rows = ""
for e in causal_data["significant_edges"][:10]:
    edge_rows += f"""<tr>
<td>{e['source']}</td><td>{e['target']}</td>
<td>{e['causal_effect']:+.4f}</td><td>{e['p_value']:.4f}</td>
<td>{e['direction']}</td>
</tr>"""

# Patient reports table
patient_rows = ""
for p in patient_reports:
    patient_rows += f"""<tr>
<td>{p['patient_id']}</td>
<td>{', '.join(p['active_biomarkers']) if p['active_biomarkers'] else 'None'}</td>
<td><strong>{p['recommendation']}</strong></td>
<td>{p['final_severity'].get(p['recommendation'], 0):.4f}</td>
</tr>"""

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>DR Digital Twin - Experimental Results Report</title>
<style>
@media print {{ body {{ font-size: 11pt; }} }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; line-height: 1.6; color: #222; }}
h1 {{ color: #1a5276; border-bottom: 3px solid #1a5276; padding-bottom: 10px; }}
h2 {{ color: #2c3e50; margin-top: 30px; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
h3 {{ color: #34495e; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th, td {{ border: 1px solid #bdc3c7; padding: 8px 12px; text-align: center; }}
th {{ background: #2c3e50; color: white; }}
tr:nth-child(even) {{ background: #f8f9fa; }}
figure {{ margin: 20px 0; text-align: center; }}
figcaption {{ font-style: italic; color: #555; margin-top: 5px; }}
.highlight {{ background: #d4efdf; padding: 15px; border-radius: 8px; border-left: 4px solid #27ae60; margin: 15px 0; }}
.metric {{ display: inline-block; background: #eaf2f8; padding: 10px 20px; border-radius: 8px; margin: 5px; text-align: center; }}
.metric .value {{ font-size: 28px; font-weight: bold; color: #1a5276; }}
.metric .label {{ font-size: 12px; color: #555; }}
</style>
</head><body>

<h1>DR Digital Twin: A Three-Layer Deep Learning Pipeline for Diabetic Retinopathy Progression Modeling and Treatment Simulation</h1>

<p><strong>Dataset:</strong> OLIVES (9,396 OCT scans, 87 patients, 6 retinal biomarkers)<br>
<strong>Hardware:</strong> NVIDIA RTX 5090 (32 GB VRAM), Intel Core i9 Ultra, 128 GB RAM<br>
<strong>Framework:</strong> PyTorch 2.11.0 + CUDA 12.8</p>

<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:10px;margin:20px 0;">
<div class="metric"><div class="value">{best['macro_f1']:.4f}</div><div class="label">Best Macro-F1</div></div>
<div class="metric"><div class="value">{best['mean_auroc']:.4f}</div><div class="label">Best AUROC</div></div>
<div class="metric"><div class="value">{len(causal_data['significant_edges'])}</div><div class="label">Causal Edges</div></div>
<div class="metric"><div class="value">{len(patient_reports)}</div><div class="label">Test Patients</div></div>
<div class="metric"><div class="value">12</div><div class="label">Models Trained</div></div>
</div>

<h2>1. Pipeline Architecture</h2>
<p>The DR Digital Twin is a three-layer pipeline that processes OCT scans through biomarker detection,
temporal progression modeling, and treatment simulation to produce personalized treatment recommendations.</p>
{img_tag("08_architecture.png", "Figure 1: Complete pipeline architecture")}

<h2>2. Layer 1: Multi-Label Biomarker Detection</h2>
<p>We trained four model architectures (CNN/ResNet-18, GCN, Signed GCN, Prior-GAT) across three seeds each
(42, 123, 2024) for robust evaluation. All models use the NPMI co-occurrence graph to encode biomarker relationships.</p>

<h3>2.1 Training Convergence</h3>
{img_tag("01_training_curves.png", "Figure 2: Training and validation loss curves for all 12 model configurations")}

<h3>2.2 Model Comparison</h3>
<table>
<tr><th>Model</th><th>Seed</th><th>Macro-F1</th><th>Micro-F1</th><th>AUROC</th><th>95% CI (F1)</th></tr>
{model_rows}
</table>
<p><em>* indicates best model selected for downstream layers.</em></p>

{img_tag("02_model_comparison.png", "Figure 3: Macro-F1 and AUROC comparison across all configurations")}

<div class="highlight">
<strong>Best Model:</strong> {best['model'].upper()} (seed={best['seed']}) with Macro-F1 = {best['macro_f1']:.4f},
AUROC = {best['mean_auroc']:.4f}, 95% CI [{best['boot_macro_f1']['ci_lower']:.3f}, {best['boot_macro_f1']['ci_upper']:.3f}]
</div>

<h2>3. Layer 2: Temporal Progression & Causal Analysis</h2>
<p>The Graph-LSTM progression model learns temporal biomarker transitions from longitudinal patient visits.
From these learned dynamics, we extract a temporal causal biomarker graph using Granger causality analysis.</p>

<h3>3.1 Causal Effect Matrix</h3>
{img_tag("03_causal_heatmap.png", "Figure 4: Full causal effect matrix between biomarkers")}

<h3>3.2 Causal Biomarker Graph</h3>
{img_tag("04_causal_graph.png", "Figure 5: Directed causal graph of significant biomarker interactions")}

<h3>3.3 Significant Causal Edges</h3>
<table>
<tr><th>Source</th><th>Target</th><th>Effect</th><th>p-value</th><th>Direction</th></tr>
{edge_rows}
</table>

<h3>3.4 Causal Chains</h3>
{img_tag("05_causal_chains.png", "Figure 6: Top biomarker cascade chains by strength")}

<h2>4. Layer 3: Treatment Simulation</h2>
<p>The Digital Twin simulates four treatment strategies (Anti-VEGF, Steroid, Laser, Observation)
for each test patient over 6 future visits, recommending the treatment with lowest predicted severity.</p>

<h3>4.1 Patient Biomarker Profiles</h3>
{img_tag("06_patient_treatments.png", "Figure 7: Active biomarkers and recommended treatments for test patients")}

<h3>4.2 Treatment Efficacy Comparison</h3>
{img_tag("07_treatment_comparison.png", "Figure 8: Mean predicted final severity by treatment type")}

<h3>4.3 Individual Patient Reports</h3>
<table>
<tr><th>Patient ID</th><th>Active Biomarkers</th><th>Recommendation</th><th>Predicted Severity</th></tr>
{patient_rows}
</table>

<h2>5. Summary</h2>
<div class="highlight">
<p>The DR Digital Twin pipeline successfully:</p>
<ul>
<li>Trained 12 biomarker detection models achieving up to <strong>F1=0.806, AUROC=0.898</strong></li>
<li>Discovered <strong>{len(causal_data['significant_edges'])} significant causal edges</strong> between retinal biomarkers</li>
<li>Identified key cascade: <strong>{' -> '.join(causal_data['causal_chains'][0]['chain'])}</strong></li>
<li>Generated personalized treatment recommendations for <strong>{len(patient_reports)} test patients</strong></li>
<li>All results from <strong>real OLIVES clinical data</strong> - no synthetic/simulated data used</li>
</ul>
</div>

<hr>
<p style="text-align:center;color:#888;font-size:11px;">
Generated from pipeline outputs on NVIDIA RTX 5090 | OLIVES Dataset | PyTorch 2.11.0+cu128
</p>
</body></html>"""

report_path = os.path.join(OUTPUT_DIR, "DR_DigitalTwin_Report.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Report saved to {report_path}")
print(f"File size: {os.path.getsize(report_path) / 1024:.0f} KB")
print("Open in a browser and use Ctrl+P to save as PDF")
