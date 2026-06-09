from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from fusion.constants import EDGE_TYPES, NUM_EDGE_TYPES, NUM_NODE_TYPES, NUM_SOURCE_TYPES, NODE_TYPES, SOURCE_TYPES
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


def _resolve_type_ids(values: Any, table: dict[str, int], *, kind: str) -> tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, int)):
        values = [values]
    out: list[int] = []
    for value in values:
        if isinstance(value, str):
            key = value.strip()
            if key not in table:
                raise ValueError(f"Unknown {kind} type name: {value!r}")
            out.append(int(table[key]))
        else:
            out.append(int(value))
    return tuple(sorted(set(out)))


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
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        *,
        use_relation_types: bool = True,
        use_edge_source: bool = True,
    ):
        super().__init__()
        self.rel_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(NUM_EDGE_TYPES)])
        self.shared_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_source_emb = nn.Embedding(NUM_SOURCE_TYPES, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.use_relation_types = bool(use_relation_types)
        self.use_edge_source = bool(use_edge_source)

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
        relation_iter = enumerate(self.rel_proj) if self.use_relation_types else [(None, self.shared_proj)]
        for rel_id, proj in relation_iter:
            mask = torch.ones_like(edge_type, dtype=torch.bool) if rel_id is None else edge_type == rel_id
            if not bool(mask.any()):
                continue
            src = src_all[mask]
            dst = dst_all[mask]
            src_type = edge_source[mask].long().clamp(0, NUM_SOURCE_TYPES - 1)
            weight = edge_quality[mask].clamp_min(0.0).unsqueeze(-1)
            msg_input = h[src]
            if self.use_edge_source:
                msg_input = msg_input + self.edge_source_emb(src_type)
            msg = proj(msg_input) * weight
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
        source_bias_weight: float = 1.0,
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
        self.source_bias_weight = float(source_bias_weight)

    def forward(
        self,
        tokens: torch.Tensor,
        token_reliability: torch.Tensor,
        conflict: torch.Tensor,
        token_source: torch.Tensor,
        token_conflict_sensitivity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, dim = tokens.shape
        query = self.q_proj(self.latents).unsqueeze(0).expand(bsz, -1, -1)
        key = self.k_proj(tokens)
        value = self.v_proj(tokens)
        scores = torch.einsum("bld,bsd->bls", query, key) / math.sqrt(dim)

        rel = token_reliability.clamp_min(1e-4).to(dtype=scores.dtype)
        scores = scores + self.reliability_bias_weight * torch.log(rel).unsqueeze(1)
        source_bias = self.source_score_bias(token_source.long().clamp(0, NUM_SOURCE_TYPES - 1)).view(1, 1, -1)
        scores = scores + self.source_bias_weight * source_bias
        conflict_sensitivity = token_conflict_sensitivity.to(device=scores.device, dtype=scores.dtype).view(1, 1, -1)
        scores = scores - self.conflict_bias_weight * conflict.view(-1, 1, 1) * conflict_sensitivity

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
        source_bias_weight: float = 1.0,
        use_node_source: bool = True,
        use_node_types: bool = True,
        use_edge_source: bool = True,
        use_node_quality: bool = True,
        use_edge_quality: bool = True,
        use_relation_types: bool = True,
        fusion_mode: str = "latent",
        masked_node_types: list[str | int] | tuple[str | int, ...] | None = None,
        masked_edge_types: list[str | int] | tuple[str | int, ...] | None = None,
        allow_node_dim_adapt: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_node_source = bool(use_node_source)
        self.use_node_types = bool(use_node_types)
        self.use_node_quality = bool(use_node_quality)
        self.use_edge_quality = bool(use_edge_quality)
        self.allow_node_dim_adapt = bool(allow_node_dim_adapt)
        self.fusion_mode = str(fusion_mode or "latent").lower()
        if self.fusion_mode not in {"latent", "mean_pool"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode!r}")
        masked_node_ids = torch.tensor(
            _resolve_type_ids(masked_node_types, NODE_TYPES, kind="node"),
            dtype=torch.long,
        )
        masked_edge_ids = torch.tensor(
            _resolve_type_ids(masked_edge_types, EDGE_TYPES, kind="edge"),
            dtype=torch.long,
        )
        self.register_buffer("masked_node_type_ids", masked_node_ids, persistent=False)
        self.register_buffer("masked_edge_type_ids", masked_edge_ids, persistent=False)
        self.input_proj = nn.Linear(int(node_input_dim), hidden_dim)
        self.node_type_emb = nn.Embedding(NUM_NODE_TYPES, hidden_dim)
        self.source_emb = nn.Embedding(NUM_SOURCE_TYPES, hidden_dim)
        self.quality_proj = nn.Linear(1, hidden_dim)
        self.semantic_proj = nn.Linear(SEMANTIC_CATEGORY_DIM, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                RelationalGraphLayer(
                    hidden_dim,
                    dropout,
                    use_relation_types=use_relation_types,
                    use_edge_source=use_edge_source,
                )
                for _ in range(int(layers))
            ]
        )
        self.fusion = LatentReliabilityFusion(
            hidden_dim,
            num_latents=num_latents,
            dropout=dropout,
            reliability_bias_weight=reliability_bias_weight,
            conflict_bias_weight=conflict_bias_weight,
            source_bias_weight=source_bias_weight,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _effective_node_quality(self, data: Batch) -> torch.Tensor:
        quality = data.node_quality.float().view(-1, 1).clamp(0.0, 1.0)
        if self.masked_node_type_ids.numel() > 0:
            masked_ids = self.masked_node_type_ids.to(device=data.x.device)
            mask = torch.isin(data.node_type.long().to(data.x.device), masked_ids).view(-1, 1)
            quality = torch.where(mask, torch.zeros_like(quality), quality)
        return quality

    def _effective_edge_quality(self, data: Batch) -> torch.Tensor:
        edge_quality = data.edge_quality.float().to(data.x.device)
        if self.masked_edge_type_ids.numel() > 0 and edge_quality.numel() > 0:
            masked_ids = self.masked_edge_type_ids.to(device=data.x.device)
            mask = torch.isin(data.edge_type.long().to(data.x.device), masked_ids)
            edge_quality = torch.where(mask, torch.zeros_like(edge_quality), edge_quality)
        return edge_quality

    def _initial_node_state(self, data: Batch) -> torch.Tensor:
        x = data.x.float()
        if x.size(1) != self.input_proj.in_features:
            if not self.allow_node_dim_adapt:
                raise ValueError(
                    "AEG node feature dimension mismatch: "
                    f"data.x.size(1)={x.size(1)} model_input_dim={self.input_proj.in_features}. "
                    "Regenerate matching PT files or set model.allow_node_dim_adapt=true for explicit compatibility mode."
                )
            if x.size(1) < self.input_proj.in_features:
                pad = x.new_zeros((x.size(0), self.input_proj.in_features - x.size(1)))
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[:, : self.input_proj.in_features]
        node_type = data.node_type.long().clamp(0, NUM_NODE_TYPES - 1)
        node_source = data.node_source.long().clamp(0, NUM_SOURCE_TYPES - 1)
        quality = self._effective_node_quality(data)
        semantic = data.node_semantic.float()
        h = self.input_proj(x) + self.semantic_proj(semantic)
        if self.use_node_types:
            h = h + self.node_type_emb(node_type)
        if self.use_node_source:
            h = h + self.source_emb(node_source)
        if self.use_node_quality:
            h = h + self.quality_proj(quality)
        alive = (quality > 0).to(dtype=h.dtype)
        return self.dropout(h) * alive

    def forward(self, data: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not hasattr(data, "batch"):
            data.batch = data.x.new_zeros((data.x.size(0),), dtype=torch.long)
        num_graphs = _batch_size(data)
        h = self._initial_node_state(data)
        edge_type = data.edge_type.long().to(data.x.device)
        edge_quality = self._effective_edge_quality(data)
        edge_source = data.edge_source.long().to(data.x.device)
        node_quality = self._effective_node_quality(data).to(data.x.device)
        node_alive_mask = node_quality.view(-1) > 0
        node_weight = node_quality if self.use_node_quality else node_alive_mask.to(dtype=node_quality.dtype).view(-1, 1)
        edge_index = data.edge_index.to(data.x.device)
        if edge_index.numel() > 0 and edge_quality.numel() == edge_index.size(1):
            src, dst = edge_index.long()
            if not self.use_edge_quality:
                edge_quality = (edge_quality > 0).to(dtype=edge_quality.dtype)
            edge_quality = edge_quality * node_weight.view(-1)[src] * node_weight.view(-1)[dst]
        for layer in self.layers:
            h = layer(h, edge_index, edge_type, edge_source, edge_quality) * node_weight

        source_code = (data.node_source == SOURCE_TYPES["code"]) & node_alive_mask
        source_manifest = (data.node_source == SOURCE_TYPES["manifest"]) & node_alive_mask
        method_nodes = (data.node_type == NODE_TYPES["METHOD"]) & node_alive_mask
        api_family_nodes = (data.node_type == NODE_TYPES["API_FAMILY"]) & node_alive_mask
        permission_nodes = (data.node_type == NODE_TYPES["PERMISSION"]) & node_alive_mask
        component_nodes = (data.node_type == NODE_TYPES["COMPONENT"]) & node_alive_mask
        risk_nodes = (data.node_type == NODE_TYPES["RISK_SEMANTIC"]) & node_alive_mask
        string_hint_nodes = (data.node_type == NODE_TYPES["STRING_HINT"]) & node_alive_mask

        method_emb = _masked_mean(h, method_nodes, data.batch, num_graphs)
        api_family_emb = _masked_mean(h, api_family_nodes, data.batch, num_graphs)
        permission_emb = _masked_mean(h, permission_nodes, data.batch, num_graphs)
        component_emb = _masked_mean(h, component_nodes, data.batch, num_graphs)
        risk_emb = _masked_mean(h, risk_nodes, data.batch, num_graphs)
        string_hint_emb = _masked_mean(h, string_hint_nodes, data.batch, num_graphs)
        global_emb = _masked_mean(h, node_alive_mask, data.batch, num_graphs)
        code_emb = 0.5 * (method_emb + api_family_emb)
        manifest_emb = 0.5 * (permission_emb + component_emb)

        code_sem = _masked_semantic_mean(data, source_code, num_graphs)
        manifest_sem = _masked_semantic_mean(data, source_manifest, num_graphs)
        sim = F.cosine_similarity(code_sem, manifest_sem, dim=-1).clamp(0.0, 1.0)
        both_semantic_available = (
            (code_sem.norm(dim=-1) > 1e-8)
            & (manifest_sem.norm(dim=-1) > 1e-8)
        ).to(dtype=sim.dtype)
        # Absence is handled by reliability; disagreement is only meaningful
        # when both sources expose observable semantic evidence.
        conflict = ((1.0 - sim) * both_semantic_available).clamp(0.0, 1.0)

        q_api = _graph_scalar(data, "q_api", 0.0).to(data.x.device)
        q_graph = _graph_scalar(data, "q_graph", 0.0).to(data.x.device)
        q_manifest = _graph_scalar(data, "q_manifest", 0.0).to(data.x.device)
        q_align = _graph_scalar(data, "q_align", 0.0).to(data.x.device)
        pert_api = _graph_scalar(data, "pert_api", 0.0).to(data.x.device).clamp(0.0, 1.0)
        pert_graph = _graph_scalar(data, "pert_graph", 0.0).to(data.x.device).clamp(0.0, 1.0)
        pert_manifest = _graph_scalar(data, "pert_manifest", 0.0).to(data.x.device).clamp(0.0, 1.0)
        r_api = q_api.clamp(0.0, 1.0) * (1.0 - pert_api)
        r_graph = q_graph.clamp(0.0, 1.0) * (1.0 - pert_graph)
        r_manifest = q_manifest.clamp(0.0, 1.0) * (1.0 - pert_manifest)
        # q_align is a soft correspondence-quality cue, not proof that API and
        # graph semantics must agree. It therefore modulates rather than gates
        # the code-side reliability.
        code_rel = (
            (r_api * r_graph).sqrt()
            * (0.5 + 0.5 * q_align.clamp(0.0, 1.0))
        ).clamp(0.0, 1.0)
        risk_rel = _masked_mean(node_quality.float().view(-1, 1), risk_nodes, data.batch, num_graphs).view(-1)
        global_rel = torch.stack([code_rel, r_manifest], dim=-1).amax(dim=-1)
        token_rel = torch.stack(
            [r_graph, r_api, r_manifest, r_manifest, risk_rel, r_api, global_rel],
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
        # Direct Manifest tokens are fully conflict-sensitive. Mixed derived
        # tokens are only softly suppressed because they may also contain
        # valid code-side evidence.
        token_conflict_sensitivity = torch.tensor(
            [0.0, 0.0, 1.0, 1.0, 0.5, 0.0, 0.25],
            device=data.x.device,
        )
        if self.fusion_mode == "mean_pool":
            fused = tokens.mean(dim=1)
            attention_mass = tokens.new_full((num_graphs, tokens.size(1)), 1.0 / tokens.size(1))
        else:
            fused, attention_mass = self.fusion(
                tokens,
                token_rel,
                conflict,
                token_source,
                token_conflict_sensitivity,
            )
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
            "pert_api": pert_api.detach(),
            "pert_graph": pert_graph.detach(),
            "pert_manifest": pert_manifest.detach(),
            "r_api": r_api.detach(),
            "r_graph": r_graph.detach(),
            "r_manifest": r_manifest.detach(),
            "code_reliability": code_rel.detach(),
            "manifest_reliability": r_manifest.detach(),
            "view_type_id": _graph_scalar(data, "view_type_id", 0.0).detach(),
            "cf_weight": _graph_scalar(data, "cf_weight", 0.0).detach(),
        }
        return logits, extra


def build_model(cfg: dict[str, Any], node_input_dim: int) -> AEGModel:
    model_cfg = cfg.get("model", {}) or {}
    return AEGModel(
        # The PT schema is the source of truth for node feature width. Keeping
        # YAML-only dimensions authoritative would break extraction ablations
        # such as behavior hints, where node_x can legitimately be wider.
        node_input_dim=int(node_input_dim),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        layers=int(model_cfg.get("layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        num_classes=int(model_cfg.get("num_classes", 2)),
        num_latents=int(model_cfg.get("num_latents", 16)),
        reliability_bias_weight=float(model_cfg.get("reliability_bias_weight", 1.0)),
        conflict_bias_weight=float(model_cfg.get("conflict_bias_weight", 0.5)),
        source_bias_weight=float(model_cfg.get("source_bias_weight", 1.0)),
        use_node_source=bool(model_cfg.get("use_node_source", True)),
        use_node_types=bool(model_cfg.get("use_node_types", True)),
        use_edge_source=bool(model_cfg.get("use_edge_source", True)),
        use_node_quality=bool(model_cfg.get("use_node_quality", True)),
        use_edge_quality=bool(model_cfg.get("use_edge_quality", True)),
        use_relation_types=bool(model_cfg.get("use_relation_types", True)),
        fusion_mode=str(model_cfg.get("fusion_mode", "latent")),
        masked_node_types=model_cfg.get("masked_node_types"),
        masked_edge_types=model_cfg.get("masked_edge_types"),
        allow_node_dim_adapt=bool(model_cfg.get("allow_node_dim_adapt", False)),
    )
