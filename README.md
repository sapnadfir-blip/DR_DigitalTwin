# Digital Twin for Diabetic Retinopathy
## Graph-Aware Biomarker Detection, Progression Modeling & Virtual Drug Testing

A three-layer AI framework that creates patient-specific digital twins for diabetic retinopathy, enabling personalized disease progression modeling and virtual drug testing on **all 16 OLIVES biomarkers**.

> **Trained on 9,396 OCT scans from 96 patient-eyes across two clinical trials (PRIME & TREX DME)**

## Results

### Layer 1: Biomarker Detection (12 models — 4 architectures × 3 seeds)

| Model | Seed 42 | Seed 123 | Seed 2024 | Mean |
|-------|---------|----------|-----------|------|
| **CNN Baseline** | F1=0.423 / AUC=0.893 | F1=0.417 / AUC=0.856 | F1=0.411 / AUC=0.882 | F1=0.417 / AUC=0.877 |
| **GCN** | F1=0.422 / AUC=0.872 | F1=0.423 / AUC=0.870 | F1=0.416 / AUC=0.854 | F1=0.420 / AUC=0.865 |
| **Signed GCN** | F1=0.376 / AUC=0.865 | **F1=0.429 / AUC=0.893** | F1=0.421 / AUC=0.866 | F1=0.409 / AUC=0.875 |
| **Prior-GAT** | F1=0.372 / AUC=0.805 | F1=0.342 / AUC=0.780 | F1=0.353 / AUC=0.828 | F1=0.356 / AUC=0.804 |

**Best model: Signed GCN (seed=123) — macro-F1 = 0.4292, AUROC = 0.8932, 95% CI [0.374, 0.442]**

### Layer 2: Temporal Progression

| Metric | Value |
|--------|-------|
| Model | Graph-LSTM (NPMI-constrained attention) |
| MAE | 0.2839 |
| RMSE | 0.3853 |
| Train sequences | 68 patient-eyes |
| Val sequences | 15 patient-eyes |
| Test sequences | 13 patient-eyes |

### Layer 3: Treatment Simulation

| Patient | Active Biomarkers | Recommended Treatment | Severity Score |
|---------|-------------------|----------------------|----------------|
| 204 | IRHRF, DRT/ME, IRF | Observation | 0.207 |
| 213 | PAVF, VD | Observation | 0.207 |
| 251 | IRHRF, FAVF, VD, DRT/ME | Observation | 0.207 |
| 222 | PAVF, VD | Observation | 0.206 |
| 230 | FAVF, PTH, VD | Observation | 0.206 |
| 237 | PAVF, VD | Observation | 0.207 |
| 60 | FAVF, VD | Observation | 0.206 |
| 72 | FAVF | Observation | 0.206 |
| 73 | FAVF, VD | Observation | 0.206 |
| 75 | IRHRF, FAVF, VD | Observation | 0.206 |

### Causal Analysis

**58 significant causal edges** discovered from 1,190,400 temporal transitions across 96 patient-eye sequences.

**Top causal edges:**
| Source → Target | Effect | Direction |
|----------------|--------|-----------|
| PAVF → FAVF | -0.768 | Inhibits |
| DRT/ME → IRF | +0.731 | Promotes |
| FAVF → PAVF | -0.669 | Inhibits |
| IRF → DRT/ME | +0.648 | Promotes |
| SRF → DRT/ME | +0.636 | Promotes |

**Top causal chains:**
| Chain | Strength |
|-------|----------|
| SRF → DRT/ME → IRF | 1.366 |
| IRH → DRT/ME → IRF | 1.237 |
| SRF → IRF → DRT/ME | 1.187 |
| DRPE → IRF → DRT/ME | 1.113 |
| DRPE → DRT/ME → IRF | 1.111 |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DIGITAL TWIN PIPELINE                     │
├──────────────┬──────────────────┬───────────────────────────┤
│   LAYER 1    │     LAYER 2      │         LAYER 3           │
│  Biomarker   │   Progression    │    Treatment Simulation   │
│  Detection   │    Modeling      │    & Virtual Drug Testing  │
├──────────────┼──────────────────┼───────────────────────────┤
│ OCT Image    │ Visit History    │ Treatment Scenarios        │
│     ↓        │     ↓            │     ↓                     │
│ ResNet-18    │ Graph-LSTM       │ Counterfactual Rollout     │
│     ↓        │ (NPMI-constrained│     ↓                     │
│ Prior-GAT    │  attention)      │ Monte Carlo Simulation     │
│     ↓        │     ↓            │     ↓                     │
│ 16 Biomarker │ Next-Visit       │ Treatment Recommendation   │
│ Predictions  │ Prediction       │ + Confidence Intervals     │
└──────────────┴──────────────────┴───────────────────────────┘
        ↑               ↑                    ↑
        └───── NPMI Co-occurrence Graph ──────┘
              (shared structural prior)
```

## Key Innovation

Unlike treating biomarker detection and disease modeling separately, this framework:
1. Uses **GNN-based co-occurrence modeling** (NPMI graph) for biomarker detection
2. **Reuses the same co-occurrence graph** to constrain temporal progression predictions
3. Enables **counterfactual treatment simulation** through auto-regressive rollout
4. Discovers **causal chains** (e.g., SRF → DRT/ME → IRF) from longitudinal data

## Quick Start

```bash
# Demo mode (no dataset needed)
python run_pipeline.py --demo

# Interactive UI
python run_pipeline.py --ui
# or
streamlit run ui/app.py

# Full training on OLIVES dataset
python run_pipeline.py

# Workstation mode (RTX 5090 / high-end GPU)
DR_WORKSTATION=1 python run_pipeline.py
```

## Training Configuration

| Setting | Default | Workstation Mode |
|---------|---------|-----------------|
| Image size | 224 | 384 |
| Batch size | 16 | 64 |
| Node dim | 128 | 512 |
| GNN layers | 2 | 4 |
| Learning rate | 1e-4 | 3e-4 |
| Early stop patience | 6 | 10 |
| Progression hidden | 64 | 512 |
| Progression layers | 2 | 6 |

## Project Structure

```
DR_DigitalTwin/
├── config.py                          # Central configuration
├── run_pipeline.py                    # Main entry point
├── requirements.txt                   # Dependencies
├── DR_DigitalTwin_Full_Pipeline.ipynb # Kaggle notebook (self-contained)
│
├── layer1_biomarker/                  # Layer 1: Biomarker Detection
│   ├── models.py                      # CNN, GCN, Signed GCN, Prior-GAT
│   ├── graph.py                       # NPMI graph construction
│   └── train.py                       # Training loop
│
├── layer2_progression/                # Layer 2: Temporal Progression
│   ├── models.py                      # LSTM, Graph-LSTM, Transformer
│   ├── causal_graph.py                # Temporal causal analysis
│   └── train.py                       # Training + synthetic data generation
│
├── layer3_treatment/                  # Layer 3: Treatment Simulation
│   └── simulator.py                   # Counterfactual + Monte Carlo simulation
│
├── digital_twin/                      # End-to-End Pipeline
│   └── twin.py                        # PatientDigitalTwin class
│
├── ui/                                # Interactive Prototype
│   └── app.py                         # Streamlit dashboard
│
└── utils/                             # Shared utilities
    ├── data.py                        # Data loading, splitting, transforms
    └── metrics.py                     # Evaluation metrics, bootstrap
```

## Dataset

**OLIVES** (Ophthalmic Labels for Investigating Visual Eye Semantics)
- 9,396 labeled OCT slices from 87 patients (96 patient-eyes)
- Two clinical trials: PRIME and TREX DME
- Patient-level train/val/test split (68/15/13 patient-eyes)
- **All 16 biomarkers** from the OLIVES dataset:

| # | Abbreviation | Full Name | Prevalence |
|---|-------------|-----------|-----------|
| 1 | ATRL | Atrophy / thinning of retinal layers | 1.8% |
| 2 | DEZ | Disruption of EZ | 6.4% |
| 3 | DRIL | Disorganization of Retinal Inner Layers | 0.3% |
| 4 | IRH | Intraretinal hemorrhages | 4.0% |
| 5 | IRHRF | Intraretinal Hyperreflective Foci | 67.4% |
| 6 | PAVF | Partially attached vitreous face | 31.7% |
| 7 | FAVF | Fully attached vitreous face | 55.5% |
| 8 | PTH | Preretinal tissue/hemorrhage | 8.6% |
| 9 | VD | Vitreous debris | 30.2% |
| 10 | VMT | Vitreomacular traction | 0.1% |
| 11 | DRT/ME | Diffuse Retinal Thickening / Macular Edema | 31.9% |
| 12 | IRF | Intraretinal Fluid | 43.5% |
| 13 | SRF | Subretinal Fluid | 2.5% |
| 14 | DRPE | Disruption of RPE | 0.1% |
| 15 | PED | Pigment Epithelial Detachment (serous) | 0.1% |
| 16 | SHRM | Subretinal Hyperreflective Material | 0.8% |

> **Note:** 5 biomarkers have extreme class imbalance (VMT, DRPE, PED at 0.1%; DRIL at 0.3%; SHRM at 0.8%). The pipeline handles this automatically via `pos_weight` in `BCEWithLogitsLoss`.

## Models

### Layer 1: Biomarker Detection (4 variants)
| Model | Description | Graph Type |
|-------|-------------|------------|
| CNN Baseline | ResNet-18 + GAP + MLP | None |
| GCN | + positive co-occurrence edges | Positive NPMI |
| Signed GCN | + separate pos/neg message passing | Signed NPMI |
| Prior-GAT | + NPMI-biased attention | Full signed + learned |

### Layer 2: Temporal Progression
| Model | Description |
|-------|-------------|
| Graph-LSTM | LSTM + NPMI-constrained graph attention |

### Layer 3: Treatment Simulation
- Auto-regressive rollout of progression model
- Counterfactual comparison across treatment types
- Monte Carlo simulation for uncertainty quantification

## Treatments Modeled
- **Anti-VEGF**: Primary DR treatment (e.g., ranibizumab, aflibercept)
- **Steroid**: Intravitreal corticosteroids
- **Laser**: Photocoagulation
- **Observation**: No active treatment (baseline)

## Hardware

Trained on:
- **GPU**: NVIDIA GeForce RTX 5090 (32 GB VRAM)
- **Mode**: Workstation (IMG_SIZE=384, BATCH_SIZE=64, NODE_DIM=512)
- **Training time**: ~2 hours (12 Layer 1 models + Layer 2 + Layer 3)

## Citation

If you use this work, please cite the OLIVES dataset:

```bibtex
@inproceedings{prabhushankar2022olives,
  title={OLIVES Dataset: Ophthalmic Labels for Investigating Visual Eye Semantics},
  author={Prabhushankar, Mohit and Kokilepersaud, Kiran and Logan, Yash and Corona, Stephanie Trejo and AlRegib, Ghassan and Wykoff, Charles},
  booktitle={NeurIPS},
  year={2022}
}
```
