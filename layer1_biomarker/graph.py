"""NPMI co-occurrence graph construction and sparsification."""
import math
import numpy as np


def compute_npmi_matrix(df_train, biomarker_cols, eps=1e-12):
    X = df_train[biomarker_cols].values.astype(np.float64)
    n, k = X.shape
    p_single = X.mean(axis=0)
    npmi = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                npmi[i, j] = 1.0
                continue
            p_ij = np.mean((X[:, i] == 1) & (X[:, j] == 1))
            p_i, p_j = p_single[i], p_single[j]
            if p_i < eps or p_j < eps:
                npmi[i, j] = 0.0
                continue
            if p_ij < eps:
                npmi[i, j] = -1.0
                continue
            pmi = math.log(p_ij / (p_i * p_j))
            denom = -math.log(p_ij)
            npmi[i, j] = float(np.clip(pmi / denom, -1.0, 1.0))
    return npmi, p_single


def sparsify_signed_graph(npmi, top_k=3):
    k = npmi.shape[0]
    mask = np.zeros((k, k), dtype=bool)
    for i in range(k):
        row = npmi[i].copy()
        row[i] = -np.inf
        idx_sorted = np.argsort(-np.abs(row))
        mask[i, idx_sorted[:top_k]] = True
    sym_mask = mask | mask.T
    np.fill_diagonal(sym_mask, False)

    signed_graph = np.where(sym_mask, npmi, 0.0)
    positive_graph = np.where(signed_graph > 0, signed_graph, 0.0)
    negative_graph = np.where(signed_graph < 0, signed_graph, 0.0)
    return signed_graph, positive_graph, negative_graph, sym_mask


def normalize_adjacency(A, add_self_loop=True):
    k = A.shape[0]
    A_hat = A + np.eye(k) if add_self_loop else A.copy()
    deg = np.sum(np.abs(A_hat), axis=1)
    deg[deg == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return (D_inv_sqrt @ A_hat @ D_inv_sqrt).astype(np.float32)


def build_graph_bundle(npmi, top_k=3):
    """Build all adjacency matrices needed by the four model variants."""
    n_bm = npmi.shape[0]
    signed_graph, positive_graph, negative_graph, sym_mask = sparsify_signed_graph(npmi, top_k)

    pos_adj_norm = normalize_adjacency(positive_graph, add_self_loop=True)
    pos_adj_norm_nosl = normalize_adjacency(positive_graph, add_self_loop=False)
    neg_adj_norm_nosl = normalize_adjacency(negative_graph, add_self_loop=False)
    gat_edge_mask = (sym_mask | np.eye(n_bm, dtype=bool))

    return {
        "npmi": npmi,
        "signed_graph": signed_graph.astype(np.float32),
        "positive_graph": positive_graph,
        "negative_graph": negative_graph,
        "pos_adj_norm": pos_adj_norm,
        "pos_adj_norm_nosl": pos_adj_norm_nosl,
        "neg_adj_norm_nosl": neg_adj_norm_nosl,
        "gat_edge_mask": gat_edge_mask,
        "sym_mask": sym_mask,
    }
