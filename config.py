"""Central configuration for the DR Digital Twin project."""
import os
import numpy as np

# ── Dataset paths (Kaggle defaults, env-overridable) ──────────────────────
KAGGLE_LABELS_DIR = "/kaggle/input/datasets/saiamarnathg/datalabels"
KAGGLE_LABEL_CSV = os.path.join(KAGGLE_LABELS_DIR, "Biomarker_Clinical_Data_Images.csv")
KAGGLE_OLIVES_ROOT = "/kaggle/input/datasets/abhinavk1609/olives/OLIVES"
PRIME_ROOT_DEFAULT = os.path.join(KAGGLE_OLIVES_ROOT, "Prime_FULL", "Prime_FULL")
TREX_ROOT_DEFAULT = os.path.join(KAGGLE_OLIVES_ROOT, "TREX_DME", "TREX DME")

LABEL_CSV = os.environ.get("OLIVES_LABEL_CSV_OVERRIDE", KAGGLE_LABEL_CSV)
PRIME_ROOT = os.environ.get("OLIVES_PRIME_ROOT_OVERRIDE", PRIME_ROOT_DEFAULT)
TREX_ROOT = os.environ.get("OLIVES_TREX_ROOT_OVERRIDE", TREX_ROOT_DEFAULT)

# ── Biomarker definitions (all 16 OLIVES biomarkers) ─────────────────────
BIOMARKERS_16 = [
    "ATRL", "DEZ", "DRIL", "IRH", "IRHRF", "PAVF", "FAVF", "PTH",
    "VD", "VMT", "DRT/ME", "IRF", "SRF", "DRPE", "PED", "SHRM",
]
BIOMARKERS_6 = BIOMARKERS_16  # backward compat alias
N_BM = len(BIOMARKERS_16)

BIOMARKER_COLUMN_CANDIDATES = {
    "ATRL":   ["Atrophy / thinning of retinal layers", "Atrophy", "ATRL",
               "Atrophy/thinning", "Thinning of retinal layers"],
    "DEZ":    ["Disruption of EZ", "DEZ", "EZ disruption", "Ellipsoid Zone Disruption"],
    "DRIL":   ["DRIL", "Disorganization of Retinal Inner Layers"],
    "IRH":    ["IR hemorrhages", "IRH", "IR Hemorrhages", "Intraretinal hemorrhages",
               "Intraretinal Hemorrhages"],
    "IRHRF":  ["IR HRF", "IRHRF", "IR-HRF", "Intraretinal Hyperreflective Foci", "HRF"],
    "PAVF":   ["Partially attached vitreous face", "PAVF"],
    "FAVF":   ["Fully attached vitreous face", "FAVF"],
    "PTH":    ["Preretinal tissue/hemorrhage", "PTH", "Preretinal tissue",
               "Preretinal hemorrhage"],
    "VD":     ["Vitreous debris", "VD", "Vitreous Debris"],
    "VMT":    ["VMT", "Vitreomacular traction", "Vitreomacular Traction"],
    "DRT/ME": ["DRT/ME", "DRT ME", "DRT", "Diffuse Retinal Thickening",
               "Macular Edema"],
    "IRF":    ["Fluid (IRF)", "IRF", "Intraretinal Fluid"],
    "SRF":    ["Fluid (SRF)", "SRF", "Subretinal Fluid"],
    "DRPE":   ["Disruption of RPE", "DRPE", "RPE Disruption",
               "Retinal Pigment Epithelium Disruption"],
    "PED":    ["PED (serous)", "PED", "Pigment Epithelial Detachment",
               "Serous PED"],
    "SHRM":   ["SHRM", "Subretinal Hyperreflective Material"],
}
PATH_CANDIDATES = ["Path", "Image Path", "File Path"]
PATIENT_CANDIDATES = ["Patient_ID", "PatientID", "Patient ID"]
EYE_CANDIDATES = ["Eye_ID", "EyeID", "Eye ID"]

# Clinical columns used by the progression model
CLINICAL_COLS = ["BCVA", "CST"]
VISIT_COL_CANDIDATES = ["Week", "Visit", "Timepoint"]
TREATMENT_COL_CANDIDATES = ["Treatment", "Drug", "Injection"]

# ── Model hyperparameters ─────────────────────────────────────────────────
IMG_SIZE = 224
NODE_DIM = 128
N_GNN_LAYERS = 2
GNN_DROPOUT = 0.3
GRAPH_TOP_K = 5

# ── Training ──────────────────────────────────────────────────────────────
LR = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 6
GRAD_CLIP_NORM = 2.0
BATCH_SIZE = 16
NUM_WORKERS = 2
SEEDS = [42, 123, 2024]
SPLIT_SEED = 2025
THRESHOLD_GRID = np.arange(0.05, 1.0, 0.05)

# ── Progression model ────────────────────────────────────────────────────
PROGRESSION_HIDDEN = 64
PROGRESSION_LAYERS = 2
PROGRESSION_LR = 5e-4
PROGRESSION_EPOCHS = 100

# ── Treatment simulation ─────────────────────────────────────────────────
TREATMENT_TYPES = ["anti_vegf", "steroid", "laser", "observation"]
SIMULATION_HORIZONS = [1, 3, 6, 12]  # months

# ── Workstation mode (RTX 5090 32GB / i9 Ultra / 128GB RAM) ─────────────
WORKSTATION_MODE = os.environ.get("DR_WORKSTATION", "0") == "1"
if WORKSTATION_MODE:
    IMG_SIZE = 384
    BATCH_SIZE = 64
    NUM_WORKERS = 0
    NODE_DIM = 512
    N_GNN_LAYERS = 4
    MAX_EPOCHS = 40
    SEEDS = [42, 123, 2024]
    PROGRESSION_HIDDEN = 512
    PROGRESSION_LAYERS = 6
    PROGRESSION_EPOCHS = 150
    LR = 3e-4
    GRAD_CLIP_NORM = 1.0
    EARLY_STOP_PATIENCE = 10

# ── Output ────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.environ.get("DR_OUTPUT_DIR", "./outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
