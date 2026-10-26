"""Training loop for Layer 1 biomarker detection models."""
import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    LR, WEIGHT_DECAY, MAX_EPOCHS, EARLY_STOP_PATIENCE, GRAD_CLIP_NORM,
    BATCH_SIZE, NUM_WORKERS, NODE_DIM, N_GNN_LAYERS, GNN_DROPOUT,
    SEEDS, SPLIT_SEED, CHECKPOINT_DIR, METRICS_DIR, N_BM,
)
from utils.data import (
    set_seed, OlivesBiomarkerDataset, train_transform, eval_transform,
    load_olives_dataframe, make_patient_level_split,
)
from utils.metrics import (
    compute_biomarker_metrics, optimize_thresholds, patient_level_bootstrap,
)
from layer1_biomarker.models import build_model
from layer1_biomarker.graph import compute_npmi_matrix, build_graph_bundle


def compute_pos_weight(df_train, biomarker_cols, device):
    pos = df_train[biomarker_cols].sum().values
    neg = len(df_train) - pos
    pos = np.clip(pos, 1, None)
    pw = np.clip(neg / pos, 0.05, 50.0)
    return torch.tensor(pw, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, use_amp=True):
    model.eval()
    total_loss, n = 0.0, 0
    all_logits, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            logits, _ = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    avg_loss = total_loss / max(n, 1)
    all_logits = np.concatenate(all_logits) if all_logits else np.zeros((0, N_BM))
    all_labels = np.concatenate(all_labels) if all_labels else np.zeros((0, N_BM))
    return avg_loss, all_logits, all_labels


def train_one_model(model_name, graph_bundle, df_train, df_val, df_test,
                     seed, path_col, biomarker_cols, device,
                     max_epochs=MAX_EPOCHS, patience=EARLY_STOP_PATIENCE,
                     lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE,
                     verbose=True):
    set_seed(seed)
    device = torch.device(device) if isinstance(device, str) else device

    train_ds = OlivesBiomarkerDataset(df_train, path_col, biomarker_cols, train_transform)
    val_ds = OlivesBiomarkerDataset(df_val, path_col, biomarker_cols, eval_transform)
    test_ds = OlivesBiomarkerDataset(df_test, path_col, biomarker_cols, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model(model_name, graph_bundle).to(device)
    pos_weight = compute_pos_weight(df_train, biomarker_cols, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        running_loss, n_seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _ = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
            n_seen += images.size(0)

        train_loss = running_loss / max(n_seen, 1)
        val_loss, _, _ = evaluate_loss(model, val_loader, nn.BCEWithLogitsLoss(), device, use_amp)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch + 1,
            "model": model_name,
            "seed": seed,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
        })

        if verbose:
            print(f"  Epoch {epoch+1:3d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)

    # Save training history for Grafana
    os.makedirs(METRICS_DIR, exist_ok=True)
    hist_path = os.path.join(METRICS_DIR, "training_history.json")
    existing = []
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            existing = json.load(f)
    existing.extend(history)
    with open(hist_path, "w") as f:
        json.dump(existing, f, indent=2)

    return model, (train_loader, val_loader, test_loader)


def run_full_biomarker_experiment(label_csv=None, device=None, seeds=None,
                                   model_names=None, save=True):
    """Run the complete Layer 1 experiment: 4 models × 3 seeds."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seeds is None:
        seeds = SEEDS
    if model_names is None:
        model_names = ["cnn", "gcn", "signed_gcn", "prior_gat"]

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    df, col_info = load_olives_dataframe(label_csv)
    path_col = col_info["path_col"]
    patient_col = col_info["patient_col"]
    biomarker_cols = col_info["biomarker_cols"]

    df_train, df_val, df_test, _ = make_patient_level_split(
        df, patient_col, biomarker_cols, seed=SPLIT_SEED
    )

    npmi, _ = compute_npmi_matrix(df_train, biomarker_cols)
    graph_bundle = build_graph_bundle(npmi)

    all_results = []
    best_models = {}

    for model_name in model_names:
        for seed in seeds:
            print(f"\n{'='*60}\nTRAINING {model_name} seed={seed}\n{'='*60}")

            model, (_, val_loader, test_loader) = train_one_model(
                model_name, graph_bundle, df_train, df_val, df_test,
                seed, path_col, biomarker_cols, device,
            )

            eval_criterion = nn.BCEWithLogitsLoss()
            _, val_logits, val_labels = evaluate_loss(model, val_loader, eval_criterion, device)
            _, test_logits, test_labels = evaluate_loss(model, test_loader, eval_criterion, device)

            best_thresholds = optimize_thresholds(val_labels, val_logits)
            opt_metrics = compute_biomarker_metrics(test_labels, test_logits,
                                                     per_biomarker_thresholds=best_thresholds)
            boot = patient_level_bootstrap(df_test, patient_col, test_labels, test_logits,
                                            best_thresholds, n_boot=300, seed=seed)

            result = {
                "model": model_name, "seed": seed,
                "macro_f1": opt_metrics["macro_f1"],
                "micro_f1": opt_metrics["micro_f1"],
                "mean_auroc": opt_metrics["mean_auroc"],
                "boot_macro_f1": boot["macro_f1"],
                "best_thresholds": best_thresholds.tolist(),
            }
            all_results.append(result)

            if save:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_seed{seed}.pt")
                torch.save(model.state_dict(), ckpt_path)

            key = (model_name, seed)
            best_models[key] = model

            print(f"  macro-F1={opt_metrics['macro_f1']:.4f}  "
                  f"AUROC={opt_metrics['mean_auroc']:.4f}  "
                  f"boot CI=[{boot['macro_f1']['ci_lower']:.3f}, {boot['macro_f1']['ci_upper']:.3f}]")

    if save:
        with open(os.path.join(METRICS_DIR, "layer1_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        # Save best model checkpoint + thresholds for UI loading
        best_result = max(all_results, key=lambda r: r["macro_f1"])
        best_key = (best_result["model"], best_result["seed"])
        best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_biomarker_model.pt")
        torch.save(best_models[best_key].state_dict(), best_ckpt_path)
        np.save(os.path.join(CHECKPOINT_DIR, "biomarker_thresholds.npy"),
                np.array(best_result["best_thresholds"]))

    return all_results, best_models, graph_bundle, {
        "df_train": df_train, "df_val": df_val, "df_test": df_test,
        "col_info": col_info, "npmi": npmi,
    }
