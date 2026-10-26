"""Layer 2: Temporal progression models for biomarker trajectory prediction.

Given a patient's biomarker state at visit t (+ clinical features),
predict the biomarker state at visit t+1. The co-occurrence graph from
Layer 1 provides structural constraints.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import BIOMARKERS_6, N_BM, PROGRESSION_HIDDEN, PROGRESSION_LAYERS


class ProgressionLSTM(nn.Module):
    """LSTM-based progression model.

    Input per timestep: biomarker vector (n_bm) + clinical features (n_clin) + treatment embedding
    Output: predicted biomarker probabilities at next visit
    """
    def __init__(self, n_biomarkers=N_BM, n_clinical=2, n_treatments=4,
                 hidden_dim=PROGRESSION_HIDDEN, n_layers=PROGRESSION_LAYERS, dropout=0.2):
        super().__init__()
        self.n_biomarkers = n_biomarkers
        self.treatment_emb = nn.Embedding(n_treatments + 1, 16)  # +1 for unknown
        input_dim = n_biomarkers + n_clinical + 16  # bm + clinical + treatment

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_biomarkers),
        )

    def forward(self, biomarker_seq, clinical_seq, treatment_seq):
        """
        Args:
            biomarker_seq: (B, T, n_bm) — biomarker states per visit
            clinical_seq: (B, T, n_clin) — clinical features per visit
            treatment_seq: (B, T) — treatment indices per visit (int)
        Returns:
            pred_seq: (B, T, n_bm) — predicted biomarker logits for the NEXT visit
        """
        treat_emb = self.treatment_emb(treatment_seq)
        x = torch.cat([biomarker_seq, clinical_seq, treat_emb], dim=-1)
        h, _ = self.lstm(x)
        return self.head(h)


class GraphConstrainedProgressionLSTM(nn.Module):
    """Progression LSTM with co-occurrence graph constraint.

    After LSTM prediction, a GNN message-passing step refines the predictions
    using the biomarker co-occurrence structure from Layer 1.
    """
    def __init__(self, n_biomarkers=N_BM, n_clinical=2, n_treatments=4,
                 hidden_dim=PROGRESSION_HIDDEN, n_layers=PROGRESSION_LAYERS,
                 dropout=0.2, npmi_matrix=None):
        super().__init__()
        self.n_biomarkers = n_biomarkers
        self.treatment_emb = nn.Embedding(n_treatments + 1, 16)
        input_dim = n_biomarkers + n_clinical + 16

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.node_proj = nn.Linear(hidden_dim, hidden_dim)

        # Graph refinement: use NPMI as attention bias
        if npmi_matrix is not None:
            self.register_buffer("npmi", torch.as_tensor(npmi_matrix, dtype=torch.float32))
        else:
            self.register_buffer("npmi", torch.zeros(n_biomarkers, n_biomarkers))

        self.graph_q = nn.Linear(hidden_dim, hidden_dim)
        self.graph_k = nn.Linear(hidden_dim, hidden_dim)
        self.graph_v = nn.Linear(hidden_dim, hidden_dim)
        self.graph_alpha = nn.Parameter(torch.tensor(0.5))
        self.graph_norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, biomarker_seq, clinical_seq, treatment_seq):
        B, T, _ = biomarker_seq.shape
        treat_emb = self.treatment_emb(treatment_seq)
        x = torch.cat([biomarker_seq, clinical_seq, treat_emb], dim=-1)
        h_seq, _ = self.lstm(x)  # (B, T, hidden)

        # Expand hidden state into per-biomarker nodes
        # (B, T, hidden) → (B*T, n_bm, hidden)
        h_flat = h_seq.reshape(B * T, -1)
        nodes = self.node_proj(h_flat).unsqueeze(1).expand(-1, self.n_biomarkers, -1)

        # Graph-constrained attention refinement
        Q = self.graph_q(nodes)
        K = self.graph_k(nodes)
        V = self.graph_v(nodes)
        D = Q.shape[-1]

        scores = torch.einsum("bid,bjd->bij", Q, K) / math.sqrt(D)
        scores = scores + self.graph_alpha * self.npmi.unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        context = torch.einsum("bij,bjd->bid", attn, V)
        refined = F.relu(self.graph_norm(nodes + context))

        # Per-biomarker prediction
        logits = self.head(refined).squeeze(-1)  # (B*T, n_bm)
        return logits.reshape(B, T, self.n_biomarkers)


class ProgressionTransformer(nn.Module):
    """Transformer-based progression model for longer sequences."""
    def __init__(self, n_biomarkers=N_BM, n_clinical=2, n_treatments=4,
                 hidden_dim=PROGRESSION_HIDDEN, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_biomarkers = n_biomarkers
        self.treatment_emb = nn.Embedding(n_treatments + 1, 16)
        input_dim = n_biomarkers + n_clinical + 16

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 50, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_biomarkers),
        )

    def forward(self, biomarker_seq, clinical_seq, treatment_seq):
        B, T, _ = biomarker_seq.shape
        treat_emb = self.treatment_emb(treatment_seq)
        x = torch.cat([biomarker_seq, clinical_seq, treat_emb], dim=-1)
        x = self.input_proj(x) + self.pos_encoding[:, :T, :]

        # Causal mask: each position can only attend to itself and earlier positions
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.transformer(x, mask=causal_mask)
        return self.head(h)


def build_progression_model(model_type="graph_lstm", npmi_matrix=None, **kwargs):
    if model_type == "lstm":
        return ProgressionLSTM(**kwargs)
    elif model_type == "graph_lstm":
        return GraphConstrainedProgressionLSTM(npmi_matrix=npmi_matrix, **kwargs)
    elif model_type == "transformer":
        return ProgressionTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown progression model type: {model_type}")
