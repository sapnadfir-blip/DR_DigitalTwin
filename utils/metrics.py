"""Evaluation metrics, threshold optimization, and patient-level bootstrap."""
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error,
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def compute_biomarker_metrics(y_true, logits, threshold=0.5, per_biomarker_thresholds=None):
    probs = sigmoid(logits)
    if per_biomarker_thresholds is not None:
        thr = np.asarray(per_biomarker_thresholds)
        preds = (probs >= thr[None, :]).astype(int)
    else:
        preds = (probs >= threshold).astype(int)

    n_labels = y_true.shape[1]
    per_bm = {"f1": [], "precision": [], "recall": [], "auroc": [], "auprc": []}
    for i in range(n_labels):
        yt, pr, pb = y_true[:, i], preds[:, i], probs[:, i]
        per_bm["f1"].append(f1_score(yt, pr, zero_division=0))
        per_bm["precision"].append(precision_score(yt, pr, zero_division=0))
        per_bm["recall"].append(recall_score(yt, pr, zero_division=0))
        if len(np.unique(yt)) > 1:
            per_bm["auroc"].append(roc_auc_score(yt, pb))
            per_bm["auprc"].append(average_precision_score(yt, pb))
        else:
            per_bm["auroc"].append(float("nan"))
            per_bm["auprc"].append(float("nan"))

    return {
        "macro_f1": float(np.mean(per_bm["f1"])),
        "micro_f1": f1_score(y_true.flatten(), preds.flatten(), zero_division=0),
        "per_biomarker": per_bm,
        "mean_auroc": float(np.nanmean(per_bm["auroc"])),
        "mean_auprc": float(np.nanmean(per_bm["auprc"])),
    }


def optimize_thresholds(y_val, val_logits, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)
    val_probs = sigmoid(val_logits)
    n_labels = y_val.shape[1]
    best_thresholds = np.zeros(n_labels)
    for i in range(n_labels):
        yt = y_val[:, i]
        best_f1, best_t = -1.0, 0.5
        for t in thresholds:
            f1 = f1_score(yt, (val_probs[:, i] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[i] = best_t
    return best_thresholds


def patient_level_bootstrap(df_test, patient_col, y_test, test_logits,
                             per_biomarker_thresholds, n_boot=300, seed=0):
    rng = np.random.RandomState(seed)
    patients = df_test[patient_col].values
    unique_patients = np.unique(patients)

    macro_samples, micro_samples = [], []
    for _ in range(n_boot):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        idx = np.concatenate([np.where(patients == p)[0] for p in sampled])
        m = compute_biomarker_metrics(y_test[idx], test_logits[idx],
                                       per_biomarker_thresholds=per_biomarker_thresholds)
        macro_samples.append(m["macro_f1"])
        micro_samples.append(m["micro_f1"])

    def summarize(samples):
        arr = np.array(samples)
        return {"mean": float(np.mean(arr)),
                "ci_lower": float(np.percentile(arr, 2.5)),
                "ci_upper": float(np.percentile(arr, 97.5))}

    return {"macro_f1": summarize(macro_samples), "micro_f1": summarize(micro_samples)}


def compute_progression_metrics(y_true, y_pred):
    """Metrics for the temporal progression model (continuous biomarker predictions)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Per-biomarker MAE
    per_bm_mae = np.mean(np.abs(y_true - y_pred), axis=0)
    return {"mae": mae, "rmse": rmse, "per_biomarker_mae": per_bm_mae.tolist()}
