"""Data loading, column resolution, image path resolution, and patient-level splitting."""
import os
import re
import math
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from config import (
    LABEL_CSV, PRIME_ROOT, TREX_ROOT, BIOMARKERS_6, N_BM,
    BIOMARKER_COLUMN_CANDIDATES, PATH_CANDIDATES, PATIENT_CANDIDATES,
    EYE_CANDIDATES, VISIT_COL_CANDIDATES, TREATMENT_COL_CANDIDATES,
    CLINICAL_COLS, IMG_SIZE,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(4),
    transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.99, 1.01)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_col(c):
    return re.sub(r'[^a-z0-9]', '', str(c).lower())


def find_column(df, candidates):
    norm_map = {normalize_col(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_col(cand)
        if key in norm_map:
            return norm_map[key]
    for cand in candidates:
        key = normalize_col(cand)
        for k, orig in norm_map.items():
            if key in k or k in key:
                return orig
    return None


def _normalize_prefix(s):
    return re.sub(r'[^A-Za-z]', '', s).upper()


def resolve_image_path(raw_path, prime_root=None, trex_root=None):
    if prime_root is None:
        prime_root = PRIME_ROOT
    if trex_root is None:
        trex_root = TREX_ROOT

    p = str(raw_path).strip().replace("\\", "/")
    if p.startswith("/"):
        p = p[1:]
    parts = [part for part in p.split("/") if part != ""]
    if not parts:
        return None

    prefix_norm = _normalize_prefix(parts[0])
    rest = parts[1:]

    candidate_roots = []
    if prefix_norm.startswith("TREX"):
        candidate_roots.append(trex_root)
    elif prefix_norm.startswith("PRIME"):
        candidate_roots.append(prime_root)
    for r in (trex_root, prime_root):
        if r not in candidate_roots:
            candidate_roots.append(r)

    for root in candidate_roots:
        candidate = os.path.join(root, *rest) if rest else root
        if os.path.exists(candidate):
            return candidate

    best_root = candidate_roots[0]
    return os.path.join(best_root, *rest) if rest else best_root


def load_olives_dataframe(label_csv=None):
    """Load the OLIVES CSV and resolve all column names."""
    if label_csv is None:
        label_csv = LABEL_CSV
    df_raw = pd.read_csv(label_csv)

    path_col = find_column(df_raw, PATH_CANDIDATES)
    patient_col = find_column(df_raw, PATIENT_CANDIDATES)
    eye_col = find_column(df_raw, EYE_CANDIDATES)
    visit_col = find_column(df_raw, VISIT_COL_CANDIDATES)
    treatment_col = find_column(df_raw, TREATMENT_COL_CANDIDATES)

    resolved_bm = {}
    for bm, cands in BIOMARKER_COLUMN_CANDIDATES.items():
        col = find_column(df_raw, cands)
        if col is not None:
            resolved_bm[bm] = col

    biomarker_cols = [resolved_bm[b] for b in BIOMARKERS_6 if b in resolved_bm]

    # Drop rows with missing biomarker labels
    df = df_raw.dropna(subset=biomarker_cols).reset_index(drop=True)
    for c in biomarker_cols:
        df[c] = df[c].astype(int)

    # Resolve image paths
    df["resolved_path"] = df[path_col].apply(resolve_image_path)
    exists_mask = df["resolved_path"].apply(lambda p: p is not None and os.path.exists(str(p)))
    df = df[exists_mask].reset_index(drop=True)

    col_info = {
        "path_col": path_col,
        "patient_col": patient_col,
        "eye_col": eye_col,
        "visit_col": visit_col,
        "treatment_col": treatment_col,
        "biomarker_cols": biomarker_cols,
        "resolved_bm": resolved_bm,
    }

    return df, col_info


def make_patient_level_split(df, patient_col, biomarker_cols,
                              train_frac=0.7, val_frac=0.15, test_frac=0.15,
                              n_trials=200, seed=0):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.RandomState(seed)
    patients = df[patient_col].unique().tolist()
    overall_prev = df[biomarker_cols].mean().values
    counts = df.groupby(patient_col).size()
    n_total = counts.sum()
    target_train = train_frac * n_total
    target_val = val_frac * n_total

    best_split, best_score = None, np.inf
    for _ in range(n_trials):
        perm = patients.copy()
        rng.shuffle(perm)
        train_ids, val_ids, test_ids = [], [], []
        train_n = val_n = 0
        for pid in perm:
            c = counts[pid]
            if train_n < target_train:
                train_ids.append(pid); train_n += c
            elif val_n < target_val:
                val_ids.append(pid); val_n += c
            else:
                test_ids.append(pid)

        masks = [df[patient_col].isin(ids) for ids in [train_ids, val_ids, test_ids]]
        if any(m.sum() == 0 for m in masks):
            continue

        prevs = [df.loc[m, biomarker_cols].mean().values for m in masks]
        score = sum(np.sum((p - overall_prev) ** 2) for p in prevs)
        if score < best_score:
            best_score = score
            best_split = (train_ids, val_ids, test_ids)

    if best_split is None:
        patients_sorted = sorted(patients)
        n = len(patients_sorted)
        n_train = max(1, int(round(train_frac * n)))
        n_val = max(1, int(round(val_frac * n)))
        train_ids = patients_sorted[:n_train]
        val_ids = patients_sorted[n_train:n_train + n_val]
        test_ids = patients_sorted[n_train + n_val:]
        if not test_ids:
            test_ids = [val_ids.pop()]
        best_split = (train_ids, val_ids, test_ids)

    train_ids, val_ids, test_ids = best_split
    df_train = df[df[patient_col].isin(train_ids)].reset_index(drop=True)
    df_val = df[df[patient_col].isin(val_ids)].reset_index(drop=True)
    df_test = df[df[patient_col].isin(test_ids)].reset_index(drop=True)
    return df_train, df_val, df_test, best_split


class OlivesBiomarkerDataset(Dataset):
    def __init__(self, df, path_col, biomarker_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.path_col = path_col
        self.biomarker_cols = biomarker_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = resolve_image_path(row[self.path_col])
        try:
            image = Image.open(img_path).convert("L").convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        if self.transform is not None:
            image = self.transform(image)
        labels = row[self.biomarker_cols].values.astype(np.float32)
        return image, torch.from_numpy(labels)


def build_longitudinal_sequences(df, patient_col, eye_col, visit_col,
                                  biomarker_cols, clinical_cols=None,
                                  treatment_col=None):
    """Build per-patient-eye visit sequences for the progression model.

    Returns a list of dicts, each with:
      - patient_id, eye_id
      - visits: sorted list of visit timepoints
      - biomarker_seq: (T, n_bm) array of biomarker states per visit
      - clinical_seq: (T, n_clinical) array (NaN-filled if cols missing)
      - treatment_seq: (T,) array of treatment labels (None if col missing)
    """
    if clinical_cols is None:
        clinical_cols = CLINICAL_COLS

    sequences = []
    group_cols = [patient_col]
    if eye_col is not None:
        group_cols.append(eye_col)

    for group_key, group_df in df.groupby(group_cols):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        if visit_col is not None and visit_col in group_df.columns:
            group_df = group_df.sort_values(visit_col)
            visits = group_df[visit_col].values
        else:
            visits = np.arange(len(group_df))

        if len(group_df) < 2:
            continue

        bm_seq = group_df[biomarker_cols].values.astype(np.float32)

        clin_seq = np.full((len(group_df), len(clinical_cols)), np.nan, dtype=np.float32)
        for i, cc in enumerate(clinical_cols):
            matched = find_column(group_df, [cc])
            if matched is not None:
                clin_seq[:, i] = group_df[matched].values.astype(np.float32)

        treat_seq = None
        if treatment_col is not None and treatment_col in group_df.columns:
            treat_seq = group_df[treatment_col].values

        seq = {
            "patient_id": group_key[0],
            "eye_id": group_key[1] if len(group_key) > 1 else None,
            "visits": visits,
            "biomarker_seq": bm_seq,
            "clinical_seq": clin_seq,
            "treatment_seq": treat_seq,
        }
        sequences.append(seq)

    return sequences
