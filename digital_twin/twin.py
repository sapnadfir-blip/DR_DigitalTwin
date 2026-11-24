"""Digital Twin: the end-to-end pipeline that ties Layers 1–3 together.

OCT image → biomarker extraction → digital twin state → progression prediction → treatment comparison
"""
import os
import json
import numpy as np
import torch
from PIL import Image

from config import (
    BIOMARKERS_6, N_BM, IMG_SIZE, NODE_DIM, CHECKPOINT_DIR, METRICS_DIR,
    OUTPUT_DIR,
)
from utils.data import eval_transform, resolve_image_path
from utils.metrics import sigmoid
from layer1_biomarker.models import build_model
from layer1_biomarker.graph import build_graph_bundle
from layer2_progression.models import build_progression_model
from layer3_treatment.simulator import TreatmentSimulator, compute_treatment_effectiveness
from layer2_progression.causal_graph import build_full_causal_analysis


class PatientDigitalTwin:
    """A patient-specific digital twin for diabetic retinopathy.

    Encapsulates:
    1. Layer 1: Biomarker detection from OCT images
    2. Layer 2: Temporal disease progression modeling
    3. Layer 3: Virtual drug testing via treatment simulation
    """
    def __init__(self, biomarker_model=None, progression_model=None,
                 graph_bundle=None, npmi_matrix=None, device=None,
                 biomarker_thresholds=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.biomarker_model = biomarker_model
        if self.biomarker_model is not None:
            self.biomarker_model.to(self.device)
            self.biomarker_model.eval()

        self.progression_model = progression_model
        self.graph_bundle = graph_bundle
        self.npmi_matrix = npmi_matrix
        self.biomarker_thresholds = biomarker_thresholds

        self.simulator = None
        if progression_model is not None:
            self.simulator = TreatmentSimulator(progression_model, self.device)

        # Patient state
        self.patient_id = None
        self.visit_history = []
        self.current_state = None

    def set_patient(self, patient_id):
        self.patient_id = patient_id
        self.visit_history = []
        self.current_state = None

    # ── Layer 1: Biomarker Detection ──────────────────────────────────────

    @torch.no_grad()
    def detect_biomarkers(self, oct_image_path):
        """Extract biomarkers from an OCT image using the trained model.

        Args:
            oct_image_path: path to an OCT scan image

        Returns:
            dict with biomarker predictions, probabilities, and attention maps
        """
        if self.biomarker_model is None:
            raise RuntimeError("No biomarker model loaded. Call load_models() first.")

        img = Image.open(oct_image_path).convert("L").convert("RGB")
        tensor = eval_transform(img).unsqueeze(0).to(self.device)

        logits, extras = self.biomarker_model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

        if self.biomarker_thresholds is not None:
            preds = (probs >= self.biomarker_thresholds).astype(int)
        else:
            preds = (probs >= 0.5).astype(int)

        result = {
            "biomarker_names": BIOMARKERS_6[:len(probs)],
            "probabilities": probs.tolist(),
            "predictions": preds.tolist(),
            "active_biomarkers": [BIOMARKERS_6[i] for i in range(len(preds)) if preds[i] == 1],
            "severity_score": float(np.mean(probs)),
        }

        if "spatial_attn" in extras:
            result["spatial_attention"] = extras["spatial_attn"].cpu().numpy()[0]

        return result

    def add_visit(self, oct_image_path=None, biomarkers=None, clinical_data=None,
                  treatment=None, visit_id=None):
        """Record a patient visit to build the longitudinal history.

        Either provide oct_image_path (auto-detect biomarkers) or biomarkers directly.
        """
        if biomarkers is None and oct_image_path is not None:
            detection = self.detect_biomarkers(oct_image_path)
            biomarkers = np.array(detection["predictions"], dtype=np.float32)
            probs = np.array(detection["probabilities"], dtype=np.float32)
        elif biomarkers is not None:
            biomarkers = np.array(biomarkers, dtype=np.float32)
            probs = biomarkers.copy()
        else:
            raise ValueError("Provide either oct_image_path or biomarkers")

        if clinical_data is None:
            clinical_data = np.zeros(2, dtype=np.float32)
        else:
            clinical_data = np.array(clinical_data, dtype=np.float32)

        visit = {
            "visit_id": visit_id or len(self.visit_history),
            "biomarkers": biomarkers,
            "probabilities": probs,
            "clinical_data": clinical_data,
            "treatment": treatment,
            "severity": float(np.mean(probs)),
        }

        self.visit_history.append(visit)
        self.current_state = visit
        return visit

    # ── Layer 2: Progression Prediction ───────────────────────────────────

    def predict_progression(self, n_future_visits=6, treatment=None):
        """Predict future biomarker states using the progression model.

        Uses the full visit history as context.
        """
        if self.simulator is None:
            raise RuntimeError("No progression model loaded.")
        if not self.visit_history:
            raise RuntimeError("No visit history. Call add_visit() first.")

        current = self.visit_history[-1]
        return self.simulator.simulate_trajectory(
            current["biomarkers"],
            current["clinical_data"],
            treatment or "observation",
            n_future_visits,
        )

    # ── Layer 3: Treatment Comparison ─────────────────────────────────────

    def compare_treatments(self, treatments=None, n_future_visits=6):
        """Compare multiple treatment scenarios on this patient's digital twin."""
        if self.simulator is None:
            raise RuntimeError("No progression model loaded.")
        if not self.visit_history:
            raise RuntimeError("No visit history. Call add_visit() first.")

        current = self.visit_history[-1]
        results = self.simulator.compare_treatments(
            current["biomarkers"],
            current["clinical_data"],
            treatments,
            n_future_visits,
        )

        effectiveness = compute_treatment_effectiveness(results)
        results["_effectiveness"] = effectiveness
        return results

    def get_monte_carlo_prediction(self, treatment, n_steps=6, n_samples=50):
        """Get uncertainty-aware prediction via Monte Carlo simulation."""
        if not self.visit_history:
            raise RuntimeError("No visit history.")
        current = self.visit_history[-1]
        return self.simulator.monte_carlo_simulation(
            current["biomarkers"], current["clinical_data"],
            treatment, n_steps, n_samples,
        )

    # ── Causal Analysis ────────────────────────────────────────────────────

    def run_causal_analysis(self, sequences=None):
        """Run temporal causal biomarker graph analysis on real longitudinal data.

        Args:
            sequences: list of dicts with 'biomarker_seq' and optionally 'treatment_seq'.
                       Must be provided — no synthetic fallback.

        Raises:
            RuntimeError if no sequences are provided.
        """
        if sequences is None or len(sequences) == 0:
            raise RuntimeError(
                "Causal analysis requires real longitudinal sequences. "
                "Run the full pipeline first to extract sequences from OLIVES, "
                "or pass sequences from outputs/longitudinal_sequences.pkl"
            )
        return build_full_causal_analysis(sequences, BIOMARKERS_6, N_BM)

    # ── Summary & Reporting ───────────────────────────────────────────────

    def get_patient_summary(self):
        """Generate a comprehensive summary of the patient's digital twin state."""
        if not self.visit_history:
            return {"error": "No visit history"}

        current = self.current_state
        history_severity = [v["severity"] for v in self.visit_history]

        summary = {
            "patient_id": self.patient_id,
            "n_visits": len(self.visit_history),
            "current_state": {
                "biomarkers": {
                    name: {"active": bool(current["biomarkers"][i]),
                           "probability": float(current["probabilities"][i])}
                    for i, name in enumerate(BIOMARKERS_6[:len(current["biomarkers"])])
                },
                "severity_score": current["severity"],
                "clinical_data": current["clinical_data"].tolist(),
            },
            "progression": {
                "severity_trend": history_severity,
                "is_worsening": len(history_severity) > 1 and history_severity[-1] > history_severity[0],
                "severity_change": history_severity[-1] - history_severity[0] if len(history_severity) > 1 else 0,
            },
        }

        return summary

    def export_report(self, output_path=None):
        """Export the digital twin state as a JSON report."""
        if output_path is None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            output_path = os.path.join(OUTPUT_DIR, f"twin_report_{self.patient_id}.json")

        report = self.get_patient_summary()

        if self.simulator is not None and self.visit_history:
            try:
                comparison = self.compare_treatments()
                report["treatment_comparison"] = {
                    "recommendation": comparison["_summary"]["recommendation"],
                    "effectiveness": comparison["_effectiveness"],
                }
            except Exception as e:
                report["treatment_comparison"] = {"error": str(e)}

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return output_path

    # ── Model Loading ─────────────────────────────────────────────────────

    @classmethod
    def from_checkpoints(cls, biomarker_ckpt=None, progression_ckpt=None,
                          graph_bundle=None, npmi_matrix=None,
                          biomarker_model_name="prior_gat",
                          progression_model_type="graph_lstm",
                          biomarker_thresholds=None, device=None):
        """Load a digital twin from saved checkpoints."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        biomarker_model = None
        if biomarker_ckpt is not None and graph_bundle is not None:
            biomarker_model = build_model(biomarker_model_name, graph_bundle)
            biomarker_model.load_state_dict(torch.load(biomarker_ckpt, map_location=device))

        progression_model = None
        if progression_ckpt is not None:
            progression_model = build_progression_model(
                progression_model_type, npmi_matrix=npmi_matrix
            )
            progression_model.load_state_dict(torch.load(progression_ckpt, map_location=device))

        return cls(
            biomarker_model=biomarker_model,
            progression_model=progression_model,
            graph_bundle=graph_bundle,
            npmi_matrix=npmi_matrix,
            device=device,
            biomarker_thresholds=biomarker_thresholds,
        )

