"""Training loop for Layer 2 progression models."""
import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import (
    PROGRESSION_LR, PROGRESSION_EPOCHS, EARLY_STOP_PATIENCE,
    N_BM, OUTPUT_DIR, METRICS_DIR, CHECKPOINT_DIR,
)
from utils.data import set_seed, build_longitudinal_sequences
from utils.metrics import compute_progression_metrics
from layer2_progression.models import build_progression_model


class ProgressionDataset(Dataset):
    """Dataset of patient visit sequences for the progression model.

    Each sample is a (input_seq, target_seq) pair where:
    - input at time t = (biomarkers_t, clinical_t, treatment_t)
    - target at time t = biomarkers_{t+1}
    """
    def __init__(self, sequences, n_clinical=2):
        self.sequences = []
        for seq in sequences:
            bm = seq["biomarker_seq"]  # (T, n_bm)
            clin = seq["clinical_seq"]  # (T, n_clin)
            T = bm.shape[0]
            if T < 2:
                continue

            # Fill NaN clinical values with 0
            clin = np.nan_to_num(clin, nan=0.0)

            # Treatment: encode as integers (0=unknown if missing)
            if seq["treatment_seq"] is not None:
                treat = self._encode_treatment(seq["treatment_seq"])
            else:
                treat = np.zeros(T, dtype=np.int64)

            # Input: visits 0..T-2, Target: visits 1..T-1
            self.sequences.append({
                "biomarker_input": torch.tensor(bm[:-1], dtype=torch.float32),
                "clinical_input": torch.tensor(clin[:-1], dtype=torch.float32),
                "treatment_input": torch.tensor(treat[:-1], dtype=torch.long),
                "biomarker_target": torch.tensor(bm[1:], dtype=torch.float32),
                "length": T - 1,
            })

    def _encode_treatment(self, treatments):
        treatment_map = {}
        encoded = []
        for t in treatments:
            t_str = str(t).lower().strip()
            if t_str in ("nan", "none", ""):
                encoded.append(0)
            else:
                if t_str not in treatment_map:
                    treatment_map[t_str] = len(treatment_map) + 1
                encoded.append(treatment_map[t_str])
        return np.array(encoded, dtype=np.int64)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_progression(batch):
    """Pad sequences to the same length within a batch."""
    max_len = max(item["length"] for item in batch)
    n_bm = batch[0]["biomarker_input"].shape[-1]
    n_clin = batch[0]["clinical_input"].shape[-1]
    B = len(batch)

    bm_input = torch.zeros(B, max_len, n_bm)
    clin_input = torch.zeros(B, max_len, n_clin)
    treat_input = torch.zeros(B, max_len, dtype=torch.long)
    bm_target = torch.zeros(B, max_len, n_bm)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        L = item["length"]
        bm_input[i, :L] = item["biomarker_input"]
        clin_input[i, :L] = item["clinical_input"]
        treat_input[i, :L] = item["treatment_input"]
        bm_target[i, :L] = item["biomarker_target"]
        mask[i, :L] = True

    return bm_input, clin_input, treat_input, bm_target, mask


def train_progression_model(sequences_train, sequences_val, model_type="graph_lstm",
                             npmi_matrix=None, device=None, seed=42,
                             epochs=PROGRESSION_EPOCHS, lr=PROGRESSION_LR,
                             batch_size=32, verbose=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    train_ds = ProgressionDataset(sequences_train)
    val_ds = ProgressionDataset(sequences_val)

    if len(train_ds) == 0:
        raise RuntimeError(
            "No valid training sequences found (need >=2 visits per patient-eye). "
            "Ensure the OLIVES dataset has longitudinal visit data with the correct "
            "patient/eye/visit columns. Cannot train on synthetic data."
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_progression)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_progression)

    model = build_progression_model(model_type, npmi_matrix=npmi_matrix).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss, n_samples = 0.0, 0
        for bm_in, clin_in, treat_in, bm_target, mask in train_loader:
            bm_in, clin_in, treat_in = bm_in.to(device), clin_in.to(device), treat_in.to(device)
            bm_target, mask = bm_target.to(device), mask.to(device)

            optimizer.zero_grad()
            pred = model(bm_in, clin_in, treat_in)
            loss = criterion(pred[mask], bm_target[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            train_loss += loss.item() * mask.sum().item()
            n_samples += mask.sum().item()

        train_loss /= max(n_samples, 1)

        # Validation
        model.eval()
        val_loss, val_n = 0.0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for bm_in, clin_in, treat_in, bm_target, mask in val_loader:
                bm_in, clin_in, treat_in = bm_in.to(device), clin_in.to(device), treat_in.to(device)
                bm_target, mask = bm_target.to(device), mask.to(device)
                pred = model(bm_in, clin_in, treat_in)
                loss = criterion(pred[mask], bm_target[mask])
                val_loss += loss.item() * mask.sum().item()
                val_n += mask.sum().item()
                all_preds.append(torch.sigmoid(pred[mask]).cpu().numpy())
                all_targets.append(bm_target[mask].cpu().numpy())

        val_loss /= max(val_n, 1)
        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)

    # Final evaluation
    if all_preds and all_targets:
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        metrics = compute_progression_metrics(targets, preds)
        if verbose:
            print(f"  Progression MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f}")
    else:
        metrics = {"mae": float("nan"), "rmse": float("nan")}

    return model, metrics


