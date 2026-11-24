"""Layer 1: Biomarker detection models — CNN baseline, GCN, Signed GCN, Prior-GAT."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import BIOMARKERS_6, NODE_DIM, N_GNN_LAYERS, GNN_DROPOUT


# ── Backbone ──────────────────────────────────────────────────────────────

def build_spatial_resnet18():
    try:
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        net = models.resnet18(weights=None)
    body = nn.Sequential(
        net.conv1, net.bn1, net.relu, net.maxpool,
        net.layer1, net.layer2, net.layer3, net.layer4,
    )
    return body, 512


class SpatialBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.body, self.out_dim = build_spatial_resnet18()

    def forward(self, x):
        return self.body(x)


class BiomarkerSpatialEncoder(nn.Module):
    def __init__(self, in_channels=512, node_dim=128, n_biomarkers=None):
        super().__init__()
        if n_biomarkers is None:
            n_biomarkers = len(BIOMARKERS_6)
        self.n = n_biomarkers
        self.query = nn.Parameter(torch.randn(n_biomarkers, in_channels) * 0.02)
        self.value_proj = nn.Linear(in_channels, node_dim)
        self.node_embedding = nn.Embedding(n_biomarkers, node_dim)

    def forward(self, fmap):
        B, C, H, W = fmap.shape
        tokens = fmap.flatten(2).transpose(1, 2)
        scores = torch.einsum("bsc,nc->bns", tokens, self.query) / math.sqrt(C)
        attn = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bns,bsc->bnc", attn, tokens)
        ids = torch.arange(self.n, device=fmap.device)
        node_features = self.value_proj(pooled) + self.node_embedding(ids).unsqueeze(0)
        return node_features, attn


# ── GNN Layers ────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    def __init__(self, dim, adj_norm):
        super().__init__()
        self.register_buffer("adj_norm", torch.as_tensor(adj_norm, dtype=torch.float32))
        self.lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h):
        agg = torch.einsum("ij,bjd->bid", self.adj_norm, h)
        return F.relu(self.norm(self.lin(agg)))


class SignedGCNLayer(nn.Module):
    def __init__(self, dim, pos_adj_norm, neg_adj_norm):
        super().__init__()
        self.register_buffer("pos_adj", torch.as_tensor(pos_adj_norm, dtype=torch.float32))
        self.register_buffer("neg_adj", torch.as_tensor(neg_adj_norm, dtype=torch.float32))
        self.self_lin = nn.Linear(dim, dim)
        self.pos_lin = nn.Linear(dim, dim)
        self.neg_lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h):
        self_term = self.self_lin(h)
        pos_msg = self.pos_lin(torch.einsum("ij,bjd->bid", self.pos_adj, h))
        neg_msg = self.neg_lin(torch.einsum("ij,bjd->bid", self.neg_adj, h))
        return F.relu(self.norm(self_term + pos_msg + neg_msg))


class PriorGATLayer(nn.Module):
    def __init__(self, dim, npmi_prior, edge_mask):
        super().__init__()
        self.register_buffer("npmi_prior", torch.as_tensor(npmi_prior, dtype=torch.float32))
        self.register_buffer("edge_mask", torch.as_tensor(edge_mask, dtype=torch.bool))
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.self_proj = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, h, return_attn=False):
        B, N, D = h.shape
        Q, K, V = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        raw_scores = torch.einsum("bid,bjd->bij", Q, K) / math.sqrt(D)
        scores = raw_scores + self.alpha * self.npmi_prior.unsqueeze(0)
        mask = self.edge_mask.unsqueeze(0).expand(B, -1, -1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        context = torch.einsum("bij,bjd->bid", attn, V)
        out = F.relu(self.norm(self.self_proj(h) + context))
        return (out, attn) if return_attn else out


# ── Full Models ───────────────────────────────────────────────────────────

class CNNBaseline(nn.Module):
    def __init__(self, n_biomarkers=None, hidden_dim=256, dropout=0.3):
        super().__init__()
        if n_biomarkers is None:
            n_biomarkers = len(BIOMARKERS_6)
        self.backbone = SpatialBackbone()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.backbone.out_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_biomarkers),
        )

    def forward(self, x):
        fmap = self.backbone(x)
        pooled = self.pool(fmap).flatten(1)
        return self.mlp(pooled), {}


class CNNGCNModel(nn.Module):
    def __init__(self, pos_adj_norm, n_biomarkers=None, node_dim=128, n_layers=2, dropout=0.3):
        super().__init__()
        if n_biomarkers is None:
            n_biomarkers = len(BIOMARKERS_6)
        self.backbone = SpatialBackbone()
        self.encoder = BiomarkerSpatialEncoder(self.backbone.out_dim, node_dim, n_biomarkers)
        self.gcn_layers = nn.ModuleList([GCNLayer(node_dim, pos_adj_norm) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(node_dim, 1)

    def forward(self, x):
        fmap = self.backbone(x)
        nodes, spatial_attn = self.encoder(fmap)
        h = nodes
        for layer in self.gcn_layers:
            h = self.dropout(layer(h))
        logits = self.classifier(h).squeeze(-1)
        return logits, {"spatial_attn": spatial_attn}


class CNNSignedGCNModel(nn.Module):
    def __init__(self, pos_adj_norm, neg_adj_norm, n_biomarkers=None, node_dim=128, n_layers=2, dropout=0.3):
        super().__init__()
        if n_biomarkers is None:
            n_biomarkers = len(BIOMARKERS_6)
        self.backbone = SpatialBackbone()
        self.encoder = BiomarkerSpatialEncoder(self.backbone.out_dim, node_dim, n_biomarkers)
        self.gcn_layers = nn.ModuleList(
            [SignedGCNLayer(node_dim, pos_adj_norm, neg_adj_norm) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(node_dim, 1)

    def forward(self, x):
        fmap = self.backbone(x)
        nodes, spatial_attn = self.encoder(fmap)
        h = nodes
        for layer in self.gcn_layers:
            h = self.dropout(layer(h))
        logits = self.classifier(h).squeeze(-1)
        return logits, {"spatial_attn": spatial_attn}


class CNNPriorGATModel(nn.Module):
    def __init__(self, npmi_prior, edge_mask, n_biomarkers=None, node_dim=128, n_layers=2, dropout=0.3):
        super().__init__()
        if n_biomarkers is None:
            n_biomarkers = len(BIOMARKERS_6)
        self.backbone = SpatialBackbone()
        self.encoder = BiomarkerSpatialEncoder(self.backbone.out_dim, node_dim, n_biomarkers)
        self.gat_layers = nn.ModuleList(
            [PriorGATLayer(node_dim, npmi_prior, edge_mask) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(node_dim, 1)

    def forward(self, x, return_attn=False):
        fmap = self.backbone(x)
        nodes, spatial_attn = self.encoder(fmap)
        h = nodes
        graph_attn = None
        for i, layer in enumerate(self.gat_layers):
            if return_attn and i == len(self.gat_layers) - 1:
                h, graph_attn = layer(h, return_attn=True)
            else:
                h = layer(h)
            h = self.dropout(h)
        logits = self.classifier(h).squeeze(-1)
        extras = {"spatial_attn": spatial_attn}
        if graph_attn is not None:
            extras["graph_attn"] = graph_attn
        return logits, extras


def build_model(model_name, graph_bundle, node_dim=NODE_DIM, n_layers=N_GNN_LAYERS,
                dropout=GNN_DROPOUT, n_biomarkers=None):
    if model_name == "cnn":
        return CNNBaseline(n_biomarkers=n_biomarkers, dropout=dropout)
    elif model_name == "gcn":
        return CNNGCNModel(graph_bundle["pos_adj_norm"], n_biomarkers=n_biomarkers,
                           node_dim=node_dim, n_layers=n_layers, dropout=dropout)
    elif model_name == "signed_gcn":
        return CNNSignedGCNModel(graph_bundle["pos_adj_norm_nosl"], graph_bundle["neg_adj_norm_nosl"],
                                  n_biomarkers=n_biomarkers, node_dim=node_dim,
                                  n_layers=n_layers, dropout=dropout)
    elif model_name == "prior_gat":
        return CNNPriorGATModel(graph_bundle["signed_graph"], graph_bundle["gat_edge_mask"],
                                 n_biomarkers=n_biomarkers, node_dim=node_dim,
                                 n_layers=n_layers, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")
