# Digital Twin for Diabetic Retinopathy
## Graph-Aware Biomarker Detection, Progression Modeling & Virtual Drug Testing

A three-layer AI framework that creates patient-specific digital twins for diabetic retinopathy, enabling personalized disease progression modeling and virtual drug testing.

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

## Quick Start

```bash
# Demo mode (no dataset needed)
python run_pipeline.py --demo

# Interactive UI
python run_pipeline.py --ui
# or
streamlit run ui/app.py

# Full training on OLIVES dataset (Kaggle)
python run_pipeline.py
```

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

- Longitudinal visits per patient

## Models

### Layer 1: Biomarker Detection (4 variants)
| Model | Description | Graph Type |
|-------|-------------|------------|
| CNN Baseline | ResNet-18 + GAP + MLP | None |
| GCN | + positive co-occurrence edges | Positive NPMI |
| Signed GCN | + separate pos/neg message passing | Signed NPMI |
| Prior-GAT | + NPMI-biased attention | Full signed + learned |

### Layer 2: Temporal Progression (3 variants)
| Model | Description |
|-------|-------------|
| LSTM | Standard sequence model |
| Graph-LSTM | LSTM + NPMI-constrained graph attention |
| Transformer | Causal transformer encoder |

### Layer 3: Treatment Simulation
- Auto-regressive rollout of progression model
- Counterfactual comparison across treatment types
- Monte Carlo simulation for uncertainty quantification

## Treatments Modeled
- **Anti-VEGF**: Primary DR treatment (e.g., ranibizumab, aflibercept)
- **Steroid**: Intravitreal corticosteroids
- **Laser**: Photocoagulation
- **Observation**: No active treatment (baseline)
