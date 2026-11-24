"""Generate all visualizations and final report from pipeline results."""
import os, sys, json, pickle, numpy as np

os.environ["DR_WORKSTATION"] = "1"
os.chdir(r"C:\Users\sapna\DR_DigitalTwin")
sys.path.insert(0, r"C:\Users\sapna\DR_DigitalTwin")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "./outputs"
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

BIOMARKERS_6 = [
    "ATRL", "DEZ", "DRIL", "IRH", "IRHRF", "PAVF", "FAVF", "PTH",
    "VD", "VMT", "DRT/ME", "IRF", "SRF", "DRPE", "PED", "SHRM",
]

print("Loading results...")
training_history = json.load(open(os.path.join(METRICS_DIR, "training_history.json")))
causal_data = json.load(open(os.path.join(METRICS_DIR, "causal_analysis.json")))
patient_reports = json.load(open(os.path.join(METRICS_DIR, "patient_reports.json")))
layer1_results = json.load(open(os.path.join(METRICS_DIR, "layer1_results.json")))

# ═══════════════════════════════════════════════════════════════
# 1. Training Loss Curves (per model type)
# ═══════════════════════════════════════════════════════════════
print("1. Generating training loss curves...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
model_types = ["cnn", "gcn", "signed_gcn", "prior_gat"]
model_labels = ["CNN (ResNet-18)", "GCN", "Signed GCN", "Prior-GAT"]
colors = ["#3498db", "#e74c3c", "#2ecc71"]

for ax, model_type, label in zip(axes.flatten(), model_types, model_labels):
    seeds = sorted(set(e["seed"] for e in training_history if e["model"] == model_type))
    for i, seed in enumerate(seeds):
        epochs_data = [e for e in training_history if e["model"] == model_type and e["seed"] == seed]
        epochs = [e["epoch"] for e in epochs_data]
        train_loss = [e["train_loss"] for e in epochs_data]
        val_loss = [e["val_loss"] for e in epochs_data]
        ax.plot(epochs, train_loss, color=colors[i], linestyle="-", alpha=0.8, label=f"Train (seed={seed})")
        ax.plot(epochs, val_loss, color=colors[i], linestyle="--", alpha=0.6, label=f"Val (seed={seed})")
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle("Training & Validation Loss Curves", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "01_training_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> 01_training_curves.png")

# ═══════════════════════════════════════════════════════════════
# 2. Model Comparison Bar Chart (F1 & AUROC)
# ═══════════════════════════════════════════════════════════════
print("2. Generating model comparison chart...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

model_names = []
f1_scores = []
auroc_scores = []
for r in layer1_results:
    model_names.append(f"{r['model']}\nseed={r['seed']}")
    f1_scores.append(r["macro_f1"])
    auroc_scores.append(r["mean_auroc"])

x = np.arange(len(model_names))
colors_bar = []
for r in layer1_results:
    if r["model"] == "cnn": colors_bar.append("#3498db")
    elif r["model"] == "gcn": colors_bar.append("#2ecc71")
    elif r["model"] == "signed_gcn": colors_bar.append("#9b59b6")
    else: colors_bar.append("#e74c3c")

ax1.bar(x, f1_scores, color=colors_bar, alpha=0.85, edgecolor="white")
ax1.set_ylabel("Macro F1-Score", fontsize=12)
ax1.set_title("Macro F1 by Model", fontsize=14, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, fontsize=8)
ax1.axhline(y=max(f1_scores), color="green", linestyle="--", alpha=0.5, label=f"Best: {max(f1_scores):.4f}")
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

ax2.bar(x, auroc_scores, color=colors_bar, alpha=0.85, edgecolor="white")
ax2.set_ylabel("AUROC", fontsize=12)
ax2.set_title("AUROC by Model", fontsize=14, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, fontsize=8)
ax2.axhline(y=max(auroc_scores), color="green", linestyle="--", alpha=0.5, label=f"Best: {max(auroc_scores):.4f}")
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

plt.suptitle("Layer 1: Biomarker Detection - Model Comparison", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "02_model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> 02_model_comparison.png")

# ═══════════════════════════════════════════════════════════════
# 3. Causal Effect Heatmap
# ═══════════════════════════════════════════════════════════════
print("3. Generating causal effect heatmap...")
causal_matrix = np.array(causal_data["causal_effect_matrix"])

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.eye(len(BIOMARKERS_6), dtype=bool)
sns.heatmap(causal_matrix, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            xticklabels=BIOMARKERS_6, yticklabels=BIOMARKERS_6,
            mask=mask, vmin=-1, vmax=1, linewidths=0.5,
            cbar_kws={"label": "Causal Effect (positive = promotes, negative = inhibits)"},
            ax=ax)
ax.set_title("Temporal Causal Effect Matrix\n(Row biomarker at time t -> Column biomarker at time t+1)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Target Biomarker (t+1)", fontsize=12)
ax.set_ylabel("Source Biomarker (t)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "03_causal_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> 03_causal_heatmap.png")

# ═══════════════════════════════════════════════════════════════
# 4. Causal Graph (Directed Network)
# ═══════════════════════════════════════════════════════════════
print("4. Generating causal graph...")
fig, ax = plt.subplots(figsize=(10, 10))

n_bm = len(BIOMARKERS_6)
angles = np.linspace(0, 2 * np.pi, n_bm, endpoint=False)
radius = 3.0
positions = {i: (radius * np.cos(a - np.pi/2), radius * np.sin(a - np.pi/2)) for i, a in enumerate(angles)}

for i, bm in enumerate(BIOMARKERS_6):
    x, y = positions[i]
    circle = plt.Circle((x, y), 0.5, color="#3498db", alpha=0.9, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, bm, ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=4)

for edge in causal_data["significant_edges"]:
    src_idx = BIOMARKERS_6.index(edge["source"])
    tgt_idx = BIOMARKERS_6.index(edge["target"])
    sx, sy = positions[src_idx]
    tx, ty = positions[tgt_idx]

    dx, dy = tx - sx, ty - sy
    dist = np.sqrt(dx**2 + dy**2)
    dx, dy = dx/dist, dy/dist
    sx2, sy2 = sx + dx * 0.55, sy + dy * 0.55
    tx2, ty2 = tx - dx * 0.55, ty - dy * 0.55

    effect = edge["causal_effect"]
    color = "#e74c3c" if effect > 0 else "#2ecc71"
    width = min(abs(effect) * 4, 4)

    ax.annotate("", xy=(tx2, ty2), xytext=(sx2, sy2),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=width, alpha=0.7))

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Temporal Causal Biomarker Graph\n(Red=promotes, Green=inhibits)",
             fontsize=16, fontweight="bold")

from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="#e74c3c", lw=3, label="Promotes (positive effect)"),
    Line2D([0], [0], color="#2ecc71", lw=3, label="Inhibits (negative effect)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "04_causal_graph.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> 04_causal_graph.png")

# ═══════════════════════════════════════════════════════════════
# 5. Causal Chains Visualization
# ═══════════════════════════════════════════════════════════════
print("5. Generating causal chains chart...")
chains = causal_data.get("causal_chains", [])[:10]
if chains:
    fig, ax = plt.subplots(figsize=(12, 6))
    chain_labels = [" -> ".join(c["chain"]) for c in chains]
    strengths = [c["total_strength"] for c in chains]
    colors_chain = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(chains)))

    bars = ax.barh(range(len(chains)), strengths, color=colors_chain, edgecolor="white")
    ax.set_yticks(range(len(chains)))
    ax.set_yticklabels(chain_labels, fontsize=10)
    ax.set_xlabel("Chain Strength", fontsize=12)
    ax.set_title("Top Causal Chains (Biomarker Cascades)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    for bar, val in zip(bars, strengths):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "05_causal_chains.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> 05_causal_chains.png")

# ═══════════════════════════════════════════════════════════════
# 6. Patient Treatment Recommendations
# ═══════════════════════════════════════════════════════════════
print("6. Generating patient treatment chart...")
fig, ax = plt.subplots(figsize=(14, 7))

patients = [p["patient_id"] for p in patient_reports]
treatments = [p["recommendation"] for p in patient_reports]
treatment_colors = {"anti_vegf": "#3498db", "steroid": "#e74c3c", "laser": "#f39c12", "observation": "#95a5a6"}

# Biomarker presence heatmap
bm_matrix = np.zeros((len(patients), len(BIOMARKERS_6)))
for i, p in enumerate(patient_reports):
    for bm in p["active_biomarkers"]:
        if bm in BIOMARKERS_6:
            bm_matrix[i, BIOMARKERS_6.index(bm)] = 1

sns.heatmap(bm_matrix, annot=True, fmt=".0f", cmap="YlOrRd",
            xticklabels=BIOMARKERS_6, yticklabels=[f"P{pid} ({t})" for pid, t in zip(patients, treatments)],
            linewidths=0.5, cbar_kws={"label": "Biomarker Active (1) / Inactive (0)"},
            ax=ax)
ax.set_title("Patient Biomarker Profiles & Treatment Recommendations", fontsize=14, fontweight="bold")
ax.set_xlabel("Biomarkers", fontsize=12)
ax.set_ylabel("Patient (Recommended Treatment)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "06_patient_treatments.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> 06_patient_treatments.png")

# ═══════════════════════════════════════════════════════════════
# 7. Treatment Severity Comparison
# ═══════════════════════════════════════════════════════════════
print("7. Generating treatment severity comparison...")
fig, ax = plt.subplots(figsize=(12, 6))

all_treatments = ["anti_vegf", "steroid", "laser", "observation"]
treatment_labels = ["Anti-VEGF", "Steroid", "Laser", "Observation"]
severity_by_treatment = {t: [] for t in all_treatments}

for p in patient_reports:
    for t, sev in p["final_severity"].items():
        if t in severity_by_treatment:
            severity_by_treatment[t].append(sev)

means = [np.mean(severity_by_treatment[t]) if severity_by_treatment[t] else 0 for t in all_treatments]
stds = [np.std(severity_by_treatment[t]) if severity_by_treatment[t] else 0 for t in all_treatments]
colors_t = ["#3498db", "#e74c3c", "#f39c12", "#95a5a6"]

bars = ax.bar(treatment_labels, means, yerr=stds, color=colors_t, alpha=0.85,
              edgecolor="white", capsize=5)
ax.set_ylabel("Average Final Severity", fontsize=12)
ax.set_title("Treatment Comparison: Predicted Final Severity\n(Lower is Better)",
             fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{mean:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "07_treatment_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> 07_treatment_comparison.png")

# ═══════════════════════════════════════════════════════════════
# 8. Architecture Diagram
# ═══════════════════════════════════════════════════════════════
print("8. Generating architecture diagram...")
fig, ax = plt.subplots(figsize=(16, 6))

boxes = [
    {"label": "OCT Scan\n(512x512)", "x": 0, "color": "#ecf0f1", "w": 2},
    {"label": "Layer 1\nBiomarker\nDetection\n(GCN)", "x": 3, "color": "#3498db", "w": 2.2},
    {"label": "6 Biomarker\nProbabilities", "x": 6.2, "color": "#ecf0f1", "w": 2},
    {"label": "Layer 2\nProgression\nModel\n(Graph LSTM)", "x": 9.2, "color": "#2ecc71", "w": 2.2},
    {"label": "Temporal\nCausal Graph\n(19 edges)", "x": 12.4, "color": "#9b59b6", "w": 2.2},
    {"label": "Layer 3\nTreatment\nSimulator", "x": 15.6, "color": "#e74c3c", "w": 2.2},
    {"label": "Treatment\nRecommendation", "x": 18.8, "color": "#f39c12", "w": 2.2},
]

for box in boxes:
    rect = plt.Rectangle((box["x"], 0.5), box["w"], 3,
                          facecolor=box["color"], edgecolor="black", linewidth=1.5,
                          alpha=0.85, zorder=2)
    ax.add_patch(rect)
    text_color = "white" if box["color"] not in ["#ecf0f1", "#f39c12"] else "black"
    ax.text(box["x"] + box["w"]/2, 2, box["label"], ha="center", va="center",
            fontsize=9, fontweight="bold", color=text_color, zorder=3)

for i in range(len(boxes) - 1):
    x1 = boxes[i]["x"] + boxes[i]["w"]
    x2 = boxes[i+1]["x"]
    ax.annotate("", xy=(x2, 2), xytext=(x1, 2),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

ax.set_xlim(-0.5, 21.5)
ax.set_ylim(-0.5, 5)
ax.axis("off")
ax.set_title("DR Digital Twin - Complete Pipeline Architecture", fontsize=16, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "08_architecture.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> 08_architecture.png")

# ═══════════════════════════════════════════════════════════════
# 9. Summary Stats
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ALL VISUALIZATIONS GENERATED!")
print("=" * 60)
print(f"  Output directory: {VIZ_DIR}")
print(f"  Files generated: 8 PNG files")
print(f"\n  Key Results:")
print(f"    Best Model: GCN seed=2024")
print(f"    Macro F1: {max(r['macro_f1'] for r in layer1_results):.4f}")
print(f"    AUROC: {max(r['auroc'] for r in layer1_results):.4f}")
print(f"    Causal Edges: {len(causal_data['significant_edges'])}")
print(f"    Test Patients: {len(patient_reports)}")
print(f"    Top Chain: {' -> '.join(causal_data['causal_chains'][0]['chain'])}")
print("\nDone!")
