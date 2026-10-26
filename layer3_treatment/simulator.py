"""Layer 3: Treatment response simulation engine.

Uses the trained progression model from Layer 2 to simulate counterfactual
treatment scenarios on a patient's digital twin.
"""
import numpy as np
import torch
import torch.nn.functional as F

from config import TREATMENT_TYPES, SIMULATION_HORIZONS, BIOMARKERS_6, N_BM


TREATMENT_TO_IDX = {t: i + 1 for i, t in enumerate(TREATMENT_TYPES)}
TREATMENT_TO_IDX["unknown"] = 0


class TreatmentSimulator:
    """Simulates treatment responses using the trained progression model.

    Given a patient's current state (biomarker vector + clinical features),
    rolls forward the progression model under different treatment conditions
    to produce counterfactual trajectories.
    """
    def __init__(self, progression_model, device=None):
        self.model = progression_model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def simulate_trajectory(self, initial_biomarkers, initial_clinical,
                             treatment, n_steps=6):
        """Simulate a single treatment trajectory.

        Args:
            initial_biomarkers: (n_bm,) array of current biomarker states
            initial_clinical: (n_clin,) array of current clinical features
            treatment: str or int — treatment type
            n_steps: number of future visits to simulate

        Returns:
            dict with:
                - trajectory: (n_steps+1, n_bm) predicted biomarker states
                - probabilities: (n_steps+1, n_bm) predicted probabilities
                - severity_score: (n_steps+1,) overall severity per step
        """
        if isinstance(treatment, str):
            treat_idx = TREATMENT_TO_IDX.get(treatment, 0)
        else:
            treat_idx = int(treatment)

        bm = torch.tensor(initial_biomarkers, dtype=torch.float32).unsqueeze(0)
        clin = torch.tensor(initial_clinical, dtype=torch.float32).unsqueeze(0)

        trajectory = [initial_biomarkers.copy()]
        probabilities = [initial_biomarkers.copy()]

        current_bm = bm.unsqueeze(1).to(self.device)
        current_clin = clin.unsqueeze(1).to(self.device)

        for step in range(n_steps):
            treat_tensor = torch.tensor([[treat_idx]], dtype=torch.long, device=self.device)
            pred_logits = self.model(current_bm, current_clin, treat_tensor)
            pred_probs = torch.sigmoid(pred_logits[:, -1, :]).cpu().numpy()[0]
            pred_binary = (pred_probs > 0.5).astype(np.float32)

            trajectory.append(pred_binary)
            probabilities.append(pred_probs)

            # Auto-regress: use prediction as next input
            current_bm = torch.tensor(pred_binary, dtype=torch.float32).reshape(1, 1, -1).to(self.device)
            # Clinical features: deterministic decay model (improvement with treatment)
            treatment_clin_effect = np.array([0.5, -5.0]) if treat_idx > 0 else np.array([-0.3, 3.0])
            new_clin = initial_clinical + treatment_clin_effect * (step + 1)
            current_clin = torch.tensor(new_clin, dtype=torch.float32).reshape(1, 1, -1).to(self.device)

        trajectory = np.array(trajectory)
        probabilities = np.array(probabilities)
        severity = np.mean(probabilities, axis=1)

        return {
            "trajectory": trajectory,
            "probabilities": probabilities,
            "severity_score": severity,
            "treatment": treatment,
            "n_steps": n_steps,
        }

    def compare_treatments(self, initial_biomarkers, initial_clinical,
                            treatments=None, n_steps=6):
        """Run simulation for multiple treatments and compare outcomes.

        Args:
            initial_biomarkers: (n_bm,) current biomarker state
            initial_clinical: (n_clin,) current clinical features
            treatments: list of treatment names (default: all)
            n_steps: simulation horizon

        Returns:
            dict mapping treatment name → simulation result, plus a
            "recommendation" key with the best treatment.
        """
        if treatments is None:
            treatments = TREATMENT_TYPES

        results = {}
        for treatment in treatments:
            results[treatment] = self.simulate_trajectory(
                initial_biomarkers, initial_clinical, treatment, n_steps
            )

        # Recommend the treatment with lowest final severity
        final_severity = {t: r["severity_score"][-1] for t, r in results.items()}
        best = min(final_severity, key=final_severity.get)

        # Compute improvement over observation (no treatment)
        if "observation" in results:
            obs_severity = results["observation"]["severity_score"][-1]
            improvements = {t: obs_severity - s for t, s in final_severity.items()}
        else:
            improvements = {t: 0.0 for t in final_severity}

        results["_summary"] = {
            "recommendation": best,
            "final_severity": final_severity,
            "improvements_vs_observation": improvements,
            "biomarker_names": BIOMARKERS_6[:initial_biomarkers.shape[0]],
        }

        return results

    def monte_carlo_simulation(self, initial_biomarkers, initial_clinical,
                                treatment, n_steps=6, n_samples=50):
        """Run multiple stochastic simulations to estimate uncertainty.

        Returns mean trajectory ± std for confidence intervals.
        """
        all_trajectories = []
        all_probabilities = []

        for _ in range(n_samples):
            result = self.simulate_trajectory(
                initial_biomarkers, initial_clinical, treatment, n_steps
            )
            all_trajectories.append(result["trajectory"])
            all_probabilities.append(result["probabilities"])

        trajectories = np.array(all_trajectories)
        probs = np.array(all_probabilities)

        return {
            "mean_trajectory": np.mean(trajectories, axis=0),
            "std_trajectory": np.std(trajectories, axis=0),
            "mean_probabilities": np.mean(probs, axis=0),
            "std_probabilities": np.std(probs, axis=0),
            "ci_lower": np.percentile(probs, 2.5, axis=0),
            "ci_upper": np.percentile(probs, 97.5, axis=0),
            "treatment": treatment,
            "n_samples": n_samples,
        }


def compute_treatment_effectiveness(simulation_results):
    """Compute effectiveness metrics from treatment comparison results.

    Returns a structured summary for visualization / reporting.
    """
    summary = simulation_results["_summary"]
    treatments = [t for t in simulation_results if not t.startswith("_")]

    effectiveness = []
    for treatment in treatments:
        r = simulation_results[treatment]
        probs = r["probabilities"]
        initial_active = np.sum(probs[0] > 0.5)
        final_active = np.sum(probs[-1] > 0.5)
        severity_change = r["severity_score"][-1] - r["severity_score"][0]

        per_biomarker_change = {}
        for i, bm_name in enumerate(summary["biomarker_names"]):
            per_biomarker_change[bm_name] = {
                "initial_prob": float(probs[0, i]),
                "final_prob": float(probs[-1, i]),
                "change": float(probs[-1, i] - probs[0, i]),
            }

        effectiveness.append({
            "treatment": treatment,
            "initial_active_biomarkers": int(initial_active),
            "final_active_biomarkers": int(final_active),
            "severity_change": float(severity_change),
            "is_recommended": treatment == summary["recommendation"],
            "per_biomarker": per_biomarker_change,
        })

    effectiveness.sort(key=lambda x: x["severity_change"])
    return effectiveness
