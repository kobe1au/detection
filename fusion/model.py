from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from fusion.constants import NUM_EDGE_TYPES, NUM_NODE_TYPES, NUM_SOURCE_TYPES, NODE_TYPES, SOURCE_TYPES
from fusion.semantic_categories import SEMANTIC_CATEGORY_DIM


def _batch_size(data: Batch) -> int:
    if hasattr(data, "num_graphs"):
        return int(data.num_graphs)
    if hasattr(data, "batch") and data.batch.numel() > 0:
        return int(data.batch.max().item()) + 1
    return 1


def _graph_scalar(data: Batch, name: str, default: float = 0.0) -> torch.Tensor:
    bsz = _batch_size(data)
    value = getattr(data, name, None)
    if isinstance(value, torch.Tensor) and value.numel() > 0:
        out = value.float().view(-1)
        if out.numel() >= bsz:
            return out[:bsz]
    return torch.full((bsz,), float(default), dtype=torch.float32, device=data.x.device)


def _masked_mean(h: torch.Tensor, mask: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    if h.numel() == 0:
        return h.new_zeros((num_graphs, h.size(-1)))
    mask = mask.view(-1).bool()
    if not bool(mask.any()):
        return h.new_zeros((num_graphs, h.size(-1)))
    out = h.new_zeros((num_graphs, h.size(-1)))
    count = h.new_zeros((num_graphs, 1))
    out.index_add_(0, batch[mask], h[mask])
    count.index_add_(0, batch[mask], torch.ones((int(mask.sum()), 1), dtype=h.dtype, device=h.device))
    return out / count.clamp_min(1.0)


def _masked_semantic_mean(data: Batch, mask: torch.Tensor, num_graphs: int) -> torch.Tensor:
    sem = getattr(data, "node_semantic", None)
    if not isinstance(sem, torch.Tensor) or sem.numel() == 0:
        return data.x.new_zeros((num_graphs, SEMANTIC_CATEGORY_DIM))
    return _masked_mean(sem.float(), mask, data.batch, num_graphs)


class RelationalGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.rel_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(NUM_EDGE_TYPES)])
        self.edge_source_emb = nn.Embedding(NUM_SOURCE_TYPES, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_source: torch.Tensor,
        edge_quality: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.norm(h + self.update(torch.cat([h, torch.zeros_like(h)], dim=-1)))

        src_all, dst_all = edge_index.long()
        if edge_quality is None or edge_quality.numel() != edge_type.numel():
            edge_quality = torch.ones_like(edge_type, dtype=h.dtype, device=h.device)
        else:
            edge_quality = edge_quality.to(device=h.device, dtype=h.dtype).view(-1)

        agg = torch.zeros_like(h)
        deg = torch.zeros((h.size(0), 1), dtype=h.dtype, device=h.device)
        for rel_id, proj in enumerate(self.rel_proj):
            mask = edge_type == rel_id
            if not bool(mask.any()):
                continue
            src = src_all[mask]
            dst = dst_all[mask]
            src_type = edge_source[mask].long().clamp(0, NUM_SOURCE_TYPES - 1)
            weight = edge_quality[mask].clamp_min(0.0).unsqueeze(-1)
            msg = proj(h[src] + self.edge_source_emb(src_type)) * weight
            agg.index_add_(0, dst, msg)
            deg.index_add_(0, dst, weight)
        agg = agg / deg.clamp_min(1.0)
        return self.norm(h + self.update(torch.cat([h, agg], dim=-1)))


class LatentReliabilityFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_latents: int = 16,
        dropout: float = 0.1,
        reliability_bias_weight: float = 1.0,
        conflict_bias_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim) / math.sqrt(hidden_dim))
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.source_score_bias = nn.Embedding(NUM_SOURCE_TYPES, 1)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reliability_bias_weight = float(reliability_bias_weight)
        self.conflict_bias_weight = float(conflict_bias_weight)

    def forward(
        self,
        tokens: torch.Tensor,
        token_reliability: torch.Tensor,
        conflict: torch.Tensor,
        token_source: torch.Tensor,
        manifest_token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, dim = tokens.shape
        query = self.q_proj(self.latents).unsqueeze(0).expand(bsz, -1, -1)
        key = self.k_proj(tokens)
        value = self.v_proj(tokens)
        scores = torch.einsum("bld,bsd->bls", query, key) / math.sqrt(dim)

        rel = token_reliability.clamp_min(1e-4).to(dtype=scores.dtype)
        scores = scores + self.reliability_bias_weight * torch.log(rel).unsqueeze(1)
        source_bias = self.source_score_bias(token_source.long().clamp(0, NUM_SOURCE_TYPES - 1)).view(1, 1, -1)
        scores = scores + source_bias
        manifest_mask = manifest_token_mask.to(device=scores.device, dtype=torch.bool).view(1, 1, -1)
        scores = torch.where(
            manifest_mask,
            scores - self.conflict_bias_weight * conflict.view(-1, 1, 1),
            scores,
        )

        attn = torch.softmax(scores, dim=-1)
        fused_latents = torch.einsum("bls,bsd->bld", attn, value)
        fused = self.out(fused_latents.mean(dim=1))
        token_mass = attn.mean(dim=1)
        return fused, token_mass


class AEGModel(nn.Module):
    def __init__(
        self,
        *,
        node_input_dim: int = 128,
        hidden_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.15,
        num_classes: int = 2,
        num_latents: int = 16,
        reliability_bias_weight: float = 1.0,
        conflict_bias_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_proj = nn.Linear(int(node_input_dim), hidden_dim)
        self.node_type_emb = nn.Embedding(NUM_NODE_TYPES, hidden_dim)
        self.source_emb = nn.Embedding(NUM_SOURCE_TYPES, hidden_dim)
        self.quality_proj = nn.Linear(1, hidden_dim)
        self.semantic_proj = nn.Linear(SEMANTIC_CATEGORY_DIM, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([RelationalGraphLayer(hidden_dim, dropout) for _ in range(int(layers))])
        self.fusion = LatentReliabilityFusion(
            hidden_dim,
            num_latents=num_latents,
            dropout=dropout,
            reliability_bias_weight=reliability_bias_weight,
            conflict_bias_weight=conflict_bias_weight,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _initial_node_state(self, data: Batch) -> torch.Tensor:
        x = data.x.float()
        if x.size(1) != self.input_proj.in_features:
            if x.size(1) < self.input_proj.in_features:
                pad = x.new_zeros((x.size(0), self.input_proj.in_features - x.size(1)))
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[:, : self.input_proj.in_features]
        node_type = data.node_type.long().clamp(0, NUM_NODE_TYPES - 1)
        node_source = data.node_source.long().clamp(0, NUM_SOURCE_TYPES - 1)
        quality = data.node_quality.float().view(-1, 1).clamp(0.0, 1.0)
        semantic = data.node_semantic.float()
        h = (
            self.input_proj(x)
            + self.node_type_emb(node_type)
            + self.source_emb(node_source)
            + self.quality_proj(quality)
            + self.semantic_proj(semantic)
        )
        return self.dropout(h)

    def forward(self, data: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not hasattr(data, "batch"):
            data.batch = data.x.new_zeros((data.x.size(0),), dtype=torch.long)
        num_graphs = _batch_size(data)
        h = self._initial_node_state(data)
        edge_type = data.edge_type.long().to(data.x.device)
        edge_quality = data.edge_quality.float().to(data.x.device)
        edge_source = data.edge_source.long().to(data.x.device)
        for layer in self.layers:
            h = layer(h, data.edge_index.to(data.x.device), edge_type, edge_source, edge_quality)

        source_code = data.node_source == SOURCE_TYPES["code"]
        source_manifest = data.node_source == SOURCE_TYPES["manifest"]
        method_nodes = data.node_type == NODE_TYPES["METHOD"]
        api_family_nodes = data.node_type == NODE_TYPES["API_FAMILY"]
        permission_nodes = data.node_type == NODE_TYPES["PERMISSION"]
        component_nodes = data.node_type == NODE_TYPES["COMPONENT"]
        risk_nodes = data.node_type == NODE_TYPES["RISK_SEMANTIC"]
        string_hint_nodes = data.node_type == NODE_TYPES["STRING_HINT"]

        method_emb = _masked_mean(h, method_nodes, data.batch, num_graphs)
        api_family_emb = _masked_mean(h, api_family_nodes, data.batch, num_graphs)
        permission_emb = _masked_mean(h, permission_nodes, data.batch, num_graphs)
        component_emb = _masked_mean(h, component_nodes, data.batch, num_graphs)
        risk_emb = _masked_mean(h, risk_nodes, data.batch, num_graphs)
        string_hint_emb = _masked_mean(h, string_hint_nodes, data.batch, num_graphs)
        global_emb = global_mean_pool(h, data.batch, size=num_graphs)
        code_emb = 0.5 * (method_emb + api_family_emb)
        manifest_emb = 0.5 * (permission_emb + component_emb)

        code_sem = _masked_semantic_mean(data, source_code, num_graphs)
        manifest_sem = _masked_semantic_mean(data, source_manifest, num_graphs)
        sim = F.cosine_similarity(code_sem, manifest_sem, dim=-1).clamp(-1.0, 1.0)
        conflict = ((1.0 - sim) * 0.5).clamp(0.0, 1.0)

        q_api = _graph_scalar(data, "q_api", 0.0).to(data.x.device)
        q_graph = _graph_scalar(data, "q_graph", 0.0).to(data.x.device)
        q_manifest = _graph_scalar(data, "q_manifest", 0.0).to(data.x.device)
        q_align = _graph_scalar(data, "q_align", 0.0).to(data.x.device)
        code_rel = (q_api.clamp_min(0.0) * q_graph.clamp_min(0.0)).sqrt()
        risk_rel = torch.stack([code_rel, q_manifest, q_align.clamp_min(0.0)], dim=-1).amax(dim=-1)
        global_rel = torch.stack([code_rel, q_manifest], dim=-1).amax(dim=-1)
        token_rel = torch.stack(
            [q_graph, q_api, q_manifest, q_manifest, risk_rel, q_api, global_rel],
            dim=-1,
        ).clamp(0.0, 1.0)
        tokens = torch.stack(
            [method_emb, api_family_emb, permission_emb, component_emb, risk_emb, string_hint_emb, global_emb],
            dim=1,
        )
        token_source = torch.tensor(
            [
                SOURCE_TYPES["code"],
                SOURCE_TYPES["code"],
                SOURCE_TYPES["manifest"],
                SOURCE_TYPES["manifest"],
                SOURCE_TYPES["derived"],
                SOURCE_TYPES["derived"],
                SOURCE_TYPES["derived"],
            ],
            dtype=torch.long,
            device=data.x.device,
        )
        manifest_token_mask = torch.tensor([False, False, True, True, False, False, False], device=data.x.device)
        fused, attention_mass = self.fusion(tokens, token_rel, conflict, token_source, manifest_token_mask)
        logits = self.classifier(fused)
        extra = {
            "fused_emb": fused,
            "method_emb": method_emb,
            "api_family_emb": api_family_emb,
            "permission_emb": permission_emb,
            "component_emb": component_emb,
            "code_emb": code_emb,
            "manifest_emb": manifest_emb,
            "risk_emb": risk_emb,
            "string_hint_emb": string_hint_emb,
            "global_emb": global_emb,
            "attention_mass": attention_mass.detach(),
            "code_manifest_conflict": conflict.detach(),
            "code_manifest_similarity": sim.detach(),
            "q_api": q_api.detach(),
            "q_graph": q_graph.detach(),
            "q_manifest": q_manifest.detach(),
            "q_align": q_align.detach(),
            "code_reliability": code_rel.detach(),
            "manifest_reliability": q_manifest.detach(),
            "view_type_id": _graph_scalar(data, "view_type_id", 0.0).detach(),
            "cf_weight": _graph_scalar(data, "cf_weight", 0.0).detach(),
        }
        return logits, extra


def build_model(cfg: dict[str, Any], node_input_dim: int) -> AEGModel:
    model_cfg = cfg.get("model", {}) or {}
    return AEGModel(
        node_input_dim=int(model_cfg.get("node_input_dim", node_input_dim)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        layers=int(model_cfg.get("layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        num_classes=int(model_cfg.get("num_classes", 2)),
        num_latents=int(model_cfg.get("num_latents", 16)),
        reliability_bias_weight=float(model_cfg.get("reliability_bias_weight", 1.0)),
        conflict_bias_weight=float(model_cfg.get("conflict_bias_weight", 0.5)),
    )
