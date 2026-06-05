from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.constants import ArchitectureConstants
from fusion.graph_encoders import GraphEncoderGAT, GraphEncoderGCN, GraphEncoderGPS
from fusion.semantic_categories import SEMANTIC_CATEGORY_DIM, validate_api_type_mapping


TRI_MODAL_FUSION_MODES = {
    "api",
    "api_only",
    "graph",
    "graph_only",
    "manifest",
    "manifest_only",
    "api_graph",
    "api_graph_concat",
    "api_graph_manifest_concat",
    "tri_modal_concat",
    "tri_modal_fixed_gate",
    "tri_modal_reliability_gate",
    "tri_modal_confidence_gate",
    "tri_modal_ours",
}


def build_main_head(in_dim: int, num_classes: int) -> nn.Sequential:
    hidden = ArchitectureConstants.HEAD_HIDDEN_DIMS[1]
    drop = ArchitectureConstants.HEAD_DROPOUT_RATES[1]
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(drop),
        nn.Linear(hidden, num_classes),
    )


class ApiSequenceEncoder(nn.Module):
    """API event encoder for the robust tri-modal pipeline."""

    def __init__(
        self,
        num_hash_buckets: int,
        type_vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
        encoder_type: str = "transformer",
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.num_hash_buckets = int(num_hash_buckets)
        self.type_vocab_size = int(type_vocab_size)
        self.emb_dim = int(emb_dim)
        self.encoder_type = str(encoder_type).lower()
        self.max_seq_len = int(max_seq_len)
        self.api_embedding = nn.Embedding(self.num_hash_buckets + 2, emb_dim, padding_idx=0)
        self.type_embedding = nn.Embedding(self.type_vocab_size, emb_dim)
        self.sensitive_embedding = nn.Embedding(2, emb_dim)
        self.input_norm = nn.LayerNorm(emb_dim)
        self.input_dropout = nn.Dropout(dropout)

        if self.encoder_type == "bigru":
            if emb_dim % 2 != 0:
                raise ValueError("emb_dim must be even for bigru API encoder")
            self.sequence_encoder = nn.GRU(
                input_size=emb_dim,
                hidden_size=emb_dim // 2,
                num_layers=max(1, int(num_layers)),
                batch_first=True,
                bidirectional=True,
                dropout=dropout if int(num_layers) > 1 else 0.0,
            )
            self.pos_embedding = None
        elif self.encoder_type == "transformer":
            if emb_dim % int(num_heads) != 0:
                raise ValueError("api emb_dim must be divisible by api heads")
            self.pos_embedding = nn.Embedding(self.max_seq_len, emb_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=int(num_heads),
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence_encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))
        else:
            raise ValueError(f"Unsupported API encoder type: {encoder_type}")

        self.out_proj = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        self.pool_score = nn.Sequential(
            nn.Linear(emb_dim, max(hidden_dim // 2, 32)),
            nn.Tanh(),
            nn.Linear(max(hidden_dim // 2, 32), 1),
        )

    def _empty_output(self, num_graphs: int, device, dtype):
        return (
            torch.zeros((0, self.emb_dim), device=device, dtype=dtype),
            torch.zeros((num_graphs, self.emb_dim), device=device, dtype=dtype),
            torch.empty((0,), device=device, dtype=torch.long),
        )

    @staticmethod
    def _padded_batch(event_emb: torch.Tensor, api_batch: torch.Tensor, num_graphs: int):
        device = event_emb.device
        lengths = torch.bincount(api_batch, minlength=num_graphs).to(device=device)
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        if max_len <= 0:
            return None, None, lengths, None
        padded = event_emb.new_zeros((num_graphs, max_len, event_emb.size(-1)))
        key_padding_mask = torch.ones((num_graphs, max_len), device=device, dtype=torch.bool)
        offsets = torch.zeros((num_graphs + 1,), device=device, dtype=torch.long)
        offsets[1:] = lengths.cumsum(dim=0)
        restore_pos = torch.arange(event_emb.size(0), device=device) - offsets[api_batch]
        padded[api_batch, restore_pos] = event_emb
        key_padding_mask[api_batch, restore_pos] = False
        empty_rows = lengths == 0
        if empty_rows.any():
            key_padding_mask[empty_rows, 0] = False
        return padded, key_padding_mask, lengths, restore_pos

    def forward(self, graph_data, num_graphs: int, device, dtype):
        api_ids = getattr(graph_data, "api_ids", None)
        api_batch = getattr(graph_data, "api_batch", None)
        if api_ids is None or api_batch is None or api_ids.numel() == 0:
            return self._empty_output(num_graphs, device, dtype)

        api_ids = api_ids.to(device=device, dtype=torch.long).clamp(0, self.num_hash_buckets + 1)
        api_batch = api_batch.to(device=device, dtype=torch.long).clamp(0, max(num_graphs - 1, 0))
        if self.max_seq_len > 0:
            raw_lengths = torch.bincount(api_batch, minlength=num_graphs).to(device=device)
            offsets = torch.zeros((num_graphs + 1,), device=device, dtype=torch.long)
            offsets[1:] = raw_lengths.cumsum(dim=0)
            local_pos = torch.arange(api_batch.numel(), device=device) - offsets[api_batch]
            keep = local_pos < self.max_seq_len
            api_ids = api_ids[keep]
            api_batch = api_batch[keep]
            if api_ids.numel() == 0:
                return self._empty_output(num_graphs, device, dtype)
        else:
            keep = slice(None)

        raw_type_ids = getattr(graph_data, "api_type_ids", None)
        api_type_ids = (
            torch.zeros_like(api_ids)
            if raw_type_ids is None
            else raw_type_ids.to(device=device, dtype=torch.long)[keep].clamp(0, self.type_vocab_size - 1)
        )
        raw_sensitive = getattr(graph_data, "api_sensitive_mask", None)
        api_sensitive = (
            torch.zeros_like(api_ids)
            if raw_sensitive is None
            else raw_sensitive.to(device=device)[keep].float().gt(0.5).long().clamp(0, 1)
        )
        event_emb = self.api_embedding(api_ids) + self.type_embedding(api_type_ids) + self.sensitive_embedding(api_sensitive)
        event_emb = self.input_dropout(self.input_norm(event_emb))
        padded, key_padding_mask, lengths, restore_pos = self._padded_batch(event_emb, api_batch, num_graphs)
        if padded is None:
            return self._empty_output(num_graphs, device, dtype)

        if self.encoder_type == "transformer":
            pos = torch.arange(padded.size(1), device=device).clamp(max=self.max_seq_len - 1)
            encoded = self.sequence_encoder(padded + self.pos_embedding(pos).unsqueeze(0), src_key_padding_mask=key_padding_mask)
        else:
            encoded, _ = self.sequence_encoder(padded)
        token_emb = encoded[api_batch, restore_pos]
        token_emb = self.out_proj(token_emb)
        scores = self.pool_score(token_emb).view(-1)
        scores = scores.masked_fill(~torch.isfinite(scores), 0.0)
        pooled = token_emb.new_zeros((num_graphs, self.emb_dim))
        for i in range(num_graphs):
            idx = api_batch == i
            if idx.any():
                weights = torch.softmax(scores[idx], dim=0).view(-1, 1)
                pooled[i] = (token_emb[idx] * weights).sum(dim=0)
        return token_emb, pooled.to(dtype=dtype), api_batch


class ManifestEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.in_dim = int(in_dim)
        self.emb_dim = int(emb_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class FourBranchEvidenceGate(nn.Module):
    """Evidence-only four-way gate for API, Graph, Manifest, and Joint branches."""

    def __init__(self, evidence_dim: int = 17, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or ArchitectureConstants.GATE_HIDDEN_DIM
        self.evidence_dim = int(evidence_dim)
        self.net = nn.Sequential(
            nn.Linear(self.evidence_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.constant_(self.net[-1].bias, ArchitectureConstants.GATE_INIT_BIAS)
        with torch.no_grad():
            self.net[-1].bias[3] = 0.5

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.size(-1) != self.evidence_dim:
            raise ValueError(f"FourBranchEvidenceGate expected {self.evidence_dim} features, got {evidence.size(-1)}")
        return torch.softmax(self.net(evidence), dim=-1)


class TriModalRobustModel(nn.Module):
    def __init__(
        self,
        in_feat_dim: int = 515,
        num_classes: int = 2,
        fusion_mode: str = "tri_modal_ours",
        api_num_hash_buckets: int = 8192,
        api_type_vocab_size: int = 16,
        api_emb_dim: int = 128,
        api_hidden_dim: int = 256,
        api_dropout: float = 0.15,
        api_encoder_type: str = "transformer",
        api_layers: int = 2,
        api_heads: int = 4,
        api_max_seq_len: int = 1024,
        graph_emb_dim: int = 128,
        graph_hidden: int = 128,
        graph_heads: int = 4,
        graph_layers: int = 2,
        graph_encoder_type: str = "gatv2",
        max_nodes_gnn: int = 12288,
        use_graph_behavior_hint: bool = True,
        manifest_in_dim: int = 256,
        manifest_emb_dim: int = 128,
        manifest_hidden_dim: int = 256,
        manifest_dropout: float = 0.1,
        joint_emb_dim: int = 128,
        gate_hidden_dim: int = 128,
        gate_detach: bool = True,
        use_consistency_evidence: bool = True,
        use_conflict_evidence: bool = True,
        use_perturbation_evidence: bool = False,
        apply_alive_mask_to_learned_gate: bool = True,
    ):
        super().__init__()
        # Defensive check: ensure DEFAULT_API_TYPE_ID_TO_CATEGORY stays
        # consistent with the extractor taxonomy and the 12-D shared space.
        validate_api_type_mapping()
        fusion_mode = str(fusion_mode or "tri_modal_ours")
        if fusion_mode not in TRI_MODAL_FUSION_MODES:
            raise ValueError(f"Unsupported tri-modal fusion_mode: {fusion_mode}")

        self.fusion_mode = fusion_mode
        self.num_classes = int(num_classes)
        self.api_emb_dim = int(api_emb_dim)
        self.graph_emb_dim = int(graph_emb_dim)
        self.manifest_in_dim = int(manifest_in_dim)
        self.manifest_emb_dim = int(manifest_emb_dim)
        self.joint_emb_dim = int(joint_emb_dim)
        self.gate_detach = bool(gate_detach)
        self.use_consistency_evidence = bool(use_consistency_evidence)
        self.use_conflict_evidence = bool(use_conflict_evidence)
        self.use_perturbation_evidence = bool(use_perturbation_evidence)
        self.apply_alive_mask_to_learned_gate = bool(apply_alive_mask_to_learned_gate)

        self.api_encoder = ApiSequenceEncoder(
            num_hash_buckets=api_num_hash_buckets,
            type_vocab_size=api_type_vocab_size,
            emb_dim=api_emb_dim,
            hidden_dim=api_hidden_dim,
            dropout=api_dropout,
            encoder_type=api_encoder_type,
            num_layers=api_layers,
            num_heads=api_heads,
            max_seq_len=api_max_seq_len,
        )

        graph_encoder_type = str(graph_encoder_type or "gatv2").lower()
        if graph_encoder_type in {"gat", "gatv2"}:
            self.graph_encoder = GraphEncoderGAT(
                in_dim=in_feat_dim,
                out_dim=graph_emb_dim,
                hidden=graph_hidden,
                heads=graph_heads,
                num_layers=graph_layers,
                max_nodes=max_nodes_gnn,
                use_behavior_hint=use_graph_behavior_hint,
            )
        elif graph_encoder_type == "gcn":
            self.graph_encoder = GraphEncoderGCN(
                in_dim=in_feat_dim,
                out_dim=graph_emb_dim,
                hidden=graph_hidden,
                max_nodes=max_nodes_gnn,
                use_behavior_hint=use_graph_behavior_hint,
            )
        elif graph_encoder_type == "gps":
            self.graph_encoder = GraphEncoderGPS(
                in_dim=in_feat_dim,
                out_dim=graph_emb_dim,
                hidden=graph_hidden,
                heads=graph_heads,
                num_layers=graph_layers,
                max_nodes=max_nodes_gnn,
                use_behavior_hint=use_graph_behavior_hint,
            )
        else:
            raise ValueError(f"Unsupported graph_encoder_type: {graph_encoder_type}")

        self.manifest_encoder = ManifestEncoder(
            in_dim=manifest_in_dim,
            emb_dim=manifest_emb_dim,
            hidden_dim=manifest_hidden_dim,
            dropout=manifest_dropout,
        )

        self.joint_encoder = nn.Sequential(
            nn.Linear(api_emb_dim + graph_emb_dim + manifest_emb_dim, max(joint_emb_dim * 2, 128)),
            nn.GELU(),
            nn.Dropout(ArchitectureConstants.HEAD_DROPOUT),
            nn.Linear(max(joint_emb_dim * 2, 128), joint_emb_dim),
            nn.LayerNorm(joint_emb_dim),
        )

        self.api_head = build_main_head(api_emb_dim, num_classes)
        self.graph_head = build_main_head(graph_emb_dim, num_classes)
        self.manifest_head = build_main_head(manifest_emb_dim, num_classes)
        self.joint_head = build_main_head(joint_emb_dim, num_classes)
        self.api_semantic_head = nn.Linear(api_emb_dim, SEMANTIC_CATEGORY_DIM)
        self.graph_semantic_head = nn.Linear(graph_emb_dim, SEMANTIC_CATEGORY_DIM)
        self.manifest_semantic_head = nn.Linear(manifest_emb_dim, SEMANTIC_CATEGORY_DIM)
        self.api_graph_concat_head = build_main_head(api_emb_dim + graph_emb_dim, num_classes)
        self.tri_concat_head = build_main_head(api_emb_dim + graph_emb_dim + manifest_emb_dim, num_classes)
        self.gate_net = FourBranchEvidenceGate(
            evidence_dim=20 if self.use_perturbation_evidence else 17,
            hidden_dim=gate_hidden_dim,
        )

    @staticmethod
    def _scalar_attr(graph_data, name: str, batch_size: int, device, dtype, default: float) -> torch.Tensor:
        value = getattr(graph_data, name, None)
        if isinstance(value, torch.Tensor):
            out = value.to(device=device, dtype=dtype).view(batch_size, -1)
            if out.size(1) > 1:
                out = out[:, :1]
            return out.clamp(0.0, 1.0)
        return torch.full((batch_size, 1), float(default), device=device, dtype=dtype)

    @staticmethod
    def _matrix_attr(graph_data, name: str, batch_size: int, device, dtype, width: int | None = None) -> torch.Tensor:
        value = getattr(graph_data, name, None)
        if not isinstance(value, torch.Tensor):
            final_width = int(width or 0)
            return torch.zeros((batch_size, final_width), device=device, dtype=dtype)
        out = value.to(device=device, dtype=dtype)
        if out.ndim == 1:
            out = out.view(1, -1).expand(batch_size, -1)
        else:
            out = out.view(batch_size, -1)
        if width is None:
            return out
        if out.size(1) < width:
            out = torch.cat([out, out.new_zeros((batch_size, width - out.size(1)))], dim=-1)
        elif out.size(1) > width:
            out = out[:, :width]
        return out

    @staticmethod
    def _semantic_counts_attr(graph_data, name: str, batch_size: int, device, dtype) -> torch.Tensor:
        value = getattr(graph_data, name, None)
        if not isinstance(value, torch.Tensor):
            return torch.zeros((batch_size, SEMANTIC_CATEGORY_DIM), device=device, dtype=dtype)
        out = value.to(device=device, dtype=dtype)
        if out.ndim == 1:
            out = out.view(1, -1).expand(batch_size, -1)
        else:
            out = out.view(batch_size, -1)
        if out.size(1) != SEMANTIC_CATEGORY_DIM:
            return torch.zeros((batch_size, SEMANTIC_CATEGORY_DIM), device=device, dtype=dtype)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _manifest_input(self, graph_data, batch_size: int, device, dtype) -> torch.Tensor:
        x = getattr(graph_data, "manifest_x", None)
        if not isinstance(x, torch.Tensor):
            return torch.zeros((batch_size, self.manifest_in_dim), device=device, dtype=dtype)
        x = x.to(device=device, dtype=dtype).view(batch_size, -1)
        if x.size(1) < self.manifest_in_dim:
            x = torch.cat([x, x.new_zeros((batch_size, self.manifest_in_dim - x.size(1)))], dim=-1)
        elif x.size(1) > self.manifest_in_dim:
            x = x[:, : self.manifest_in_dim]
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _confidence(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits.detach(), dim=-1).max(dim=-1, keepdim=True).values.clamp(0.0, 1.0)

    @staticmethod
    def _prob_disagreement(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        pa = torch.softmax(a.detach(), dim=-1)
        pb = torch.softmax(b.detach(), dim=-1)
        return (pa - pb).abs().mean(dim=-1, keepdim=True).clamp(0.0, 1.0)

    @staticmethod
    def _modality_alive(emb: torch.Tensor) -> torch.Tensor:
        alive = emb.detach().abs().sum(dim=-1, keepdim=True) > ArchitectureConstants.MODALITY_ALIVE_THRESHOLD
        return alive.to(dtype=emb.dtype)

    @staticmethod
    def _cosine_counts(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.numel() == 0 or b.numel() == 0:
            return a.new_zeros((a.size(0), 1))
        width = max(a.size(1), b.size(1))
        if a.size(1) < width:
            a = torch.cat([a, a.new_zeros((a.size(0), width - a.size(1)))], dim=-1)
        if b.size(1) < width:
            b = torch.cat([b, b.new_zeros((b.size(0), width - b.size(1)))], dim=-1)
        valid = (a.abs().sum(dim=-1, keepdim=True) > 0) & (b.abs().sum(dim=-1, keepdim=True) > 0)
        sim = F.cosine_similarity(a.float(), b.float(), dim=-1).view(-1, 1).clamp(0.0, 1.0)
        return torch.where(valid, sim, torch.zeros_like(sim))

    @staticmethod
    def _directional_semantic_conflicts(
        api_counts: torch.Tensor,
        graph_counts: torch.Tensor,
        manifest_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        api = api_counts.float().clamp_min(0.0)
        graph = graph_counts.float().clamp_min(0.0)
        manifest = manifest_counts.float().clamp_min(0.0)
        code = torch.maximum(api, graph)
        valid = (code.sum(dim=-1, keepdim=True) > 0) & (manifest.sum(dim=-1, keepdim=True) > 0)
        code = code / code.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        manifest = manifest / manifest.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        manifest_to_code = (manifest - code).clamp_min(0.0).sum(dim=-1, keepdim=True)
        code_to_manifest = (code - manifest).clamp_min(0.0).sum(dim=-1, keepdim=True)
        zero = torch.zeros_like(manifest_to_code)
        return torch.where(valid, manifest_to_code, zero), torch.where(valid, code_to_manifest, zero)

    def _encode_api(self, graph_data, batch_size: int, device, dtype):
        _, pooled, _ = self.api_encoder(graph_data, batch_size, device, dtype)
        return pooled

    def _build_evidence(
        self,
        graph_data,
        api_logits: torch.Tensor,
        graph_logits: torch.Tensor,
        manifest_logits: torch.Tensor,
        joint_logits: torch.Tensor,
        api_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        manifest_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = api_logits.size(0)
        device = api_logits.device
        dtype = api_logits.dtype

        q_api = self._scalar_attr(graph_data, "q_api", batch_size, device, dtype, 1.0)
        q_graph = self._scalar_attr(graph_data, "q_graph", batch_size, device, dtype, 1.0)
        q_manifest = self._scalar_attr(graph_data, "q_manifest", batch_size, device, dtype, 0.0)
        q_align = self._scalar_attr(graph_data, "q_align", batch_size, device, dtype, 0.0)
        pert_api = self._scalar_attr(graph_data, "pert_api", batch_size, device, dtype, 0.0)
        pert_graph = self._scalar_attr(graph_data, "pert_graph", batch_size, device, dtype, 0.0)
        pert_manifest = self._scalar_attr(graph_data, "pert_manifest", batch_size, device, dtype, 1.0)

        # Synthetic perturbation strength is oracle metadata and is unavailable
        # for naturally corrupted APKs. The main method therefore derives
        # reliability from observable post-extraction quality only. An explicit
        # oracle ablation can opt into perturbation evidence.
        if self.use_perturbation_evidence:
            r_api = (q_api * (1.0 - pert_api)).clamp(0.0, 1.0)
            r_graph = (q_graph * (1.0 - pert_graph)).clamp(0.0, 1.0)
            r_manifest = (q_manifest * (1.0 - pert_manifest)).clamp(0.0, 1.0)
        else:
            r_api = q_api
            r_graph = q_graph
            r_manifest = q_manifest

        api_conf = self._confidence(api_logits).to(dtype=dtype)
        graph_conf = self._confidence(graph_logits).to(dtype=dtype)
        manifest_conf = self._confidence(manifest_logits).to(dtype=dtype)
        joint_conf = self._confidence(joint_logits).to(dtype=dtype)

        api_counts = self._semantic_counts_attr(graph_data, "api_semantic_category_counts", batch_size, device, dtype)
        if float(api_counts.detach().abs().sum().item()) <= 0.0:
            api_counts = self._semantic_counts_attr(graph_data, "api_category_counts", batch_size, device, dtype)
        graph_counts = self._semantic_counts_attr(graph_data, "graph_semantic_category_counts", batch_size, device, dtype)
        if float(graph_counts.detach().abs().sum().item()) <= 0.0:
            graph_counts = self._semantic_counts_attr(graph_data, "graph_category_counts", batch_size, device, dtype)
        manifest_counts = self._semantic_counts_attr(graph_data, "manifest_category_counts", batch_size, device, dtype)

        api_graph_consistency = self._cosine_counts(api_counts, graph_counts)
        graph_missing_counts = graph_counts.abs().sum(dim=-1, keepdim=True) <= 0
        api_graph_consistency = torch.where(graph_missing_counts, q_align, api_graph_consistency).clamp(0.0, 1.0)
        api_manifest_consistency = self._cosine_counts(api_counts, manifest_counts)
        graph_manifest_consistency = self._cosine_counts(graph_counts, manifest_counts)
        manifest_to_code_conflict, code_to_manifest_conflict = self._directional_semantic_conflicts(
            api_counts,
            graph_counts,
            manifest_counts,
        )
        evidence_api_graph_consistency = api_graph_consistency
        evidence_api_manifest_consistency = api_manifest_consistency
        evidence_graph_manifest_consistency = graph_manifest_consistency
        evidence_manifest_to_code_conflict = manifest_to_code_conflict
        evidence_code_to_manifest_conflict = code_to_manifest_conflict
        if not self.use_consistency_evidence:
            evidence_api_graph_consistency = torch.zeros_like(api_graph_consistency)
            evidence_api_manifest_consistency = torch.zeros_like(api_manifest_consistency)
            evidence_graph_manifest_consistency = torch.zeros_like(graph_manifest_consistency)
        if not self.use_conflict_evidence:
            evidence_manifest_to_code_conflict = torch.zeros_like(manifest_to_code_conflict)
            evidence_code_to_manifest_conflict = torch.zeros_like(code_to_manifest_conflict)

        api_graph_disagreement = self._prob_disagreement(api_logits, graph_logits).to(dtype=dtype)
        api_alive = self._modality_alive(api_emb).to(dtype=dtype) * (q_api > 0.0).to(dtype=dtype)
        graph_alive = self._modality_alive(graph_emb).to(dtype=dtype) * (q_graph > 0.0).to(dtype=dtype)
        manifest_alive = (
            self._modality_alive(manifest_emb).to(dtype=dtype)
            * (q_manifest > 0.0).to(dtype=dtype)
        )

        evidence = torch.cat(
            [
                q_align,
                r_api,
                r_graph,
                r_manifest,
                api_conf,
                graph_conf,
                manifest_conf,
                joint_conf,
                api_graph_disagreement,
                evidence_api_graph_consistency,
                evidence_api_manifest_consistency,
                evidence_graph_manifest_consistency,
                api_alive,
                graph_alive,
                manifest_alive,
                evidence_manifest_to_code_conflict,
                evidence_code_to_manifest_conflict,
            ],
            dim=-1,
        )
        if self.use_perturbation_evidence:
            evidence = torch.cat([evidence, pert_api, pert_graph, pert_manifest], dim=-1)
        diagnostics = {
            "q_api": q_api.detach().view(batch_size),
            "q_graph": q_graph.detach().view(batch_size),
            "q_manifest": q_manifest.detach().view(batch_size),
            "q_align": q_align.detach().view(batch_size),
            "pert_api": pert_api.detach().view(batch_size),
            "pert_graph": pert_graph.detach().view(batch_size),
            "pert_manifest": pert_manifest.detach().view(batch_size),
            "r_api": r_api.detach().view(batch_size),
            "r_graph": r_graph.detach().view(batch_size),
            "r_manifest": r_manifest.detach().view(batch_size),
            "api_confidence": api_conf.detach().view(batch_size),
            "graph_confidence": graph_conf.detach().view(batch_size),
            "manifest_confidence": manifest_conf.detach().view(batch_size),
            "joint_confidence": joint_conf.detach().view(batch_size),
            "api_graph_disagreement": api_graph_disagreement.detach().view(batch_size),
            "api_graph_consistency": api_graph_consistency.detach().view(batch_size),
            "api_manifest_consistency": api_manifest_consistency.detach().view(batch_size),
            "graph_manifest_consistency": graph_manifest_consistency.detach().view(batch_size),
            "manifest_to_code_conflict": manifest_to_code_conflict.detach().view(batch_size),
            "code_to_manifest_conflict": code_to_manifest_conflict.detach().view(batch_size),
            "api_semantic_category_counts": api_counts.detach(),
            "graph_semantic_category_counts": graph_counts.detach(),
            "manifest_category_counts": manifest_counts.detach(),
            "api_category_counts": api_counts.detach(),
            "graph_category_counts": graph_counts.detach(),
            "api_alive": api_alive.detach().view(batch_size),
            "graph_alive": graph_alive.detach().view(batch_size),
            "manifest_alive": manifest_alive.detach().view(batch_size),
            "gate_uses_perturbation_evidence": torch.full(
                (batch_size,),
                float(self.use_perturbation_evidence),
                device=device,
                dtype=dtype,
            ),
        }
        return evidence, diagnostics

    @staticmethod
    def _heuristic_reliability_gate(evidence: torch.Tensor) -> torch.Tensor:
        r_api = evidence[:, 1:2]
        r_graph = evidence[:, 2:3]
        r_manifest = evidence[:, 3:4]
        api_graph = evidence[:, 9:10]
        api_manifest = evidence[:, 10:11]
        graph_manifest = evidence[:, 11:12]
        api_alive = evidence[:, 12:13]
        graph_alive = evidence[:, 13:14]
        manifest_alive = evidence[:, 14:15]
        alive_sum = (api_alive + graph_alive + manifest_alive).clamp_min(1.0)
        reliability_support = (
            r_api * api_alive + r_graph * graph_alive + r_manifest * manifest_alive
        ) / alive_sum
        pair_support = api_alive * graph_alive + api_alive * manifest_alive + graph_alive * manifest_alive
        pair_consistency = (
            api_graph * api_alive * graph_alive
            + api_manifest * api_alive * manifest_alive
            + graph_manifest * graph_alive * manifest_alive
        ) / pair_support.clamp_min(1.0)
        joint_score = (
            reliability_support.square() * (0.5 + 0.5 * pair_consistency)
        ).clamp(0.0, 1.0)
        joint_availability = (alive_sum / 3.0).clamp(0.0, 1.0)
        scores = torch.cat(
            [
                r_api * api_alive,
                r_graph * graph_alive,
                r_manifest * manifest_alive,
                joint_score * joint_availability,
            ],
            dim=-1,
        )
        denom = scores.sum(dim=-1, keepdim=True)
        normalized = scores / denom.clamp_min(1e-8)
        fallback = torch.full_like(scores, 0.25)
        return torch.where(denom > 1e-8, normalized, fallback)

    @staticmethod
    def _confidence_gate(evidence: torch.Tensor) -> torch.Tensor:
        scores = torch.cat(
            [
                evidence[:, 4:5],
                evidence[:, 5:6],
                evidence[:, 6:7],
                evidence[:, 7:8],
            ],
            dim=-1,
        ).clamp(0.0, 1.0)
        denom = scores.sum(dim=-1, keepdim=True)
        normalized = scores / denom.clamp_min(1e-8)
        fallback = torch.full_like(scores, 0.25)
        return torch.where(denom > 1e-8, normalized, fallback)

    @staticmethod
    def _apply_alive_mask(gate_weights: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        api_alive = evidence[:, 12:13].clamp(0.0, 1.0)
        graph_alive = evidence[:, 13:14].clamp(0.0, 1.0)
        manifest_alive = evidence[:, 14:15].clamp(0.0, 1.0)
        joint_alive = torch.maximum(torch.maximum(api_alive, graph_alive), manifest_alive)
        support = torch.cat([api_alive, graph_alive, manifest_alive, joint_alive], dim=-1)
        masked = gate_weights * support
        denom = masked.sum(dim=-1, keepdim=True)
        fallback = torch.full_like(masked, 0.25)
        return torch.where(denom > 1e-8, masked / denom.clamp_min(1e-8), fallback)

    def forward(
        self,
        graph_data,
        return_features: bool = False,
    ):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        batch_size = int(getattr(graph_data, "num_graphs", 1))

        api_emb = self._encode_api(graph_data, batch_size, device, dtype)
        _, graph_emb, _, _ = self.graph_encoder(graph_data)
        manifest_x = self._manifest_input(graph_data, batch_size, device, dtype)
        manifest_emb = self.manifest_encoder(manifest_x)

        joint_input = torch.cat([api_emb, graph_emb, manifest_emb], dim=-1)
        joint_emb = self.joint_encoder(joint_input)

        api_logits = self.api_head(api_emb)
        graph_logits = self.graph_head(graph_emb)
        manifest_logits = self.manifest_head(manifest_emb)
        joint_logits = self.joint_head(joint_emb)

        extra = {
            "api_logits_aux": api_logits,
            "graph_logits_aux": graph_logits,
            "manifest_logits_aux": manifest_logits,
            "joint_logits_aux": joint_logits,
            "api_semantic_logits": self.api_semantic_head(api_emb),
            "graph_semantic_logits": self.graph_semantic_head(graph_emb),
            "manifest_semantic_logits": self.manifest_semantic_head(manifest_emb),
        }

        if self.fusion_mode in {"api", "api_only"}:
            logits = api_logits
            gate_weights = torch.zeros((batch_size, 4), device=device, dtype=dtype)
            gate_weights[:, 0] = 1.0
        elif self.fusion_mode in {"graph", "graph_only"}:
            logits = graph_logits
            gate_weights = torch.zeros((batch_size, 4), device=device, dtype=dtype)
            gate_weights[:, 1] = 1.0
        elif self.fusion_mode in {"manifest", "manifest_only"}:
            logits = manifest_logits
            gate_weights = torch.zeros((batch_size, 4), device=device, dtype=dtype)
            gate_weights[:, 2] = 1.0
        elif self.fusion_mode in {"api_graph", "api_graph_concat"}:
            logits = self.api_graph_concat_head(torch.cat([api_emb, graph_emb], dim=-1))
            gate_weights = torch.zeros((batch_size, 4), device=device, dtype=dtype)
            gate_weights[:, 3] = 1.0
            extra["joint_logits_aux"] = logits
        elif self.fusion_mode in {"api_graph_manifest_concat", "tri_modal_concat"}:
            logits = self.tri_concat_head(joint_input)
            gate_weights = torch.zeros((batch_size, 4), device=device, dtype=dtype)
            gate_weights[:, 3] = 1.0
            extra["joint_logits_aux"] = logits
        else:
            evidence, diagnostics = self._build_evidence(
                graph_data,
                api_logits,
                graph_logits,
                manifest_logits,
                joint_logits,
                api_emb,
                graph_emb,
                manifest_emb,
            )
            extra.update(diagnostics)
            extra["gate_evidence"] = evidence.detach()
            if self.fusion_mode == "tri_modal_fixed_gate":
                gate_weights = torch.full((batch_size, 4), 0.25, device=device, dtype=dtype)
                extra["gate_prior_enabled"] = False
            elif self.fusion_mode == "tri_modal_reliability_gate":
                gate_weights = self._heuristic_reliability_gate(evidence).to(dtype=dtype)
                extra["gate_prior_enabled"] = False
            elif self.fusion_mode == "tri_modal_confidence_gate":
                gate_weights = self._confidence_gate(evidence).to(dtype=dtype)
                extra["gate_prior_enabled"] = False
            else:
                gate_input = evidence.detach() if self.gate_detach else evidence
                gate_weights = self.gate_net(gate_input)
                if self.apply_alive_mask_to_learned_gate:
                    gate_weights = self._apply_alive_mask(gate_weights, evidence).to(dtype=dtype)
                extra["gate_prior_enabled"] = True
            logits = (
                gate_weights[:, 0:1] * api_logits
                + gate_weights[:, 1:2] * graph_logits
                + gate_weights[:, 2:3] * manifest_logits
                + gate_weights[:, 3:4] * joint_logits
            )

        extra["gate_weights_train"] = gate_weights
        extra["gate_weights"] = gate_weights.detach()
        extra.setdefault("gate_prior_enabled", False)

        if "api_confidence" not in extra:
            evidence, diagnostics = self._build_evidence(
                graph_data,
                api_logits,
                graph_logits,
                manifest_logits,
                joint_logits,
                api_emb,
                graph_emb,
                manifest_emb,
            )
            del evidence
            extra.update(diagnostics)

        if return_features:
            extra["api_emb"] = api_emb
            extra["graph_emb"] = graph_emb
            extra["manifest_emb"] = manifest_emb
            extra["joint_emb"] = joint_emb

        return logits, extra
