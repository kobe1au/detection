from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax, to_dense_batch

from fusion.constants import ArchitectureConstants
from fusion.modules import (
    TriBranchGate,
    CrossAttention,
    build_main_head,
)
from fusion.graph_encoders import GraphEncoderGAT, GraphEncoderGCN, GraphEncoderGPS
from fusion.prototypes import TemporalPrototypeMemory


class ApiSequenceEncoder(nn.Module):
    """
    Mainstream API event sequence encoder.

    Supports:
      - encoder_type="bigru": Embedding + BiGRU + attention pooling
      - encoder_type="transformer": Embedding + positional embedding + TransformerEncoder + attention pooling

    Expected fields from graph_data:
      - api_ids: LongTensor [T]
      - api_type_ids: LongTensor/UInt8Tensor [T]
      - api_sensitive_mask: Bool/UInt8Tensor [T]
      - api_batch: LongTensor [T], mapping each API event to APK/sample index

    Output:
      - token_emb: [T, emb_dim]
      - pooled: [B, emb_dim]
      - api_batch: [T]
    """

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
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.encoder_type = str(encoder_type).lower()
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.max_seq_len = int(max_seq_len)

        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        if self.encoder_type not in {"bigru", "transformer"}:
            raise ValueError(
                f"encoder_type must be 'bigru' or 'transformer', got {encoder_type}"
            )

        self.api_embedding = nn.Embedding(
            self.num_hash_buckets + 2,
            emb_dim,
            padding_idx=0,
        )

        # 注意：api_type_ids 的 0 是 "other"，不是 padding，所以这里不要 padding_idx=0
        self.type_embedding = nn.Embedding(
            self.type_vocab_size,
            emb_dim,
        )

        self.sensitive_embedding = nn.Embedding(2, emb_dim)

        self.input_norm = nn.LayerNorm(emb_dim)
        self.input_dropout = nn.Dropout(dropout)

        if self.encoder_type == "bigru":
            if emb_dim % 2 != 0:
                raise ValueError("For BiGRU encoder, emb_dim must be even.")

            self.sequence_encoder = nn.GRU(
                input_size=emb_dim,
                hidden_size=emb_dim // 2,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if self.num_layers > 1 else 0.0,
            )

        else:
            if emb_dim % self.num_heads != 0:
                raise ValueError(
                    f"emb_dim ({emb_dim}) must be divisible by num_heads ({self.num_heads})"
                )

            self.pos_embedding = nn.Embedding(self.max_seq_len, emb_dim)

            layer = nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=self.num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence_encoder = nn.TransformerEncoder(
                layer,
                num_layers=self.num_layers,
            )

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
        token_emb = torch.zeros((0, self.emb_dim), device=device, dtype=dtype)
        pooled = torch.zeros((num_graphs, self.emb_dim), device=device, dtype=dtype)
        token_batch = torch.empty((0,), device=device, dtype=torch.long)
        return token_emb, pooled, token_batch

    def _build_padded_batch(
        self,
        event_emb: torch.Tensor,
        api_batch: torch.Tensor,
        num_graphs: int,
    ):
        """
        Convert flattened events [T, D] into padded sequence [B, L, D].

        Returns:
          padded: [B, L, D]
          key_padding_mask: [B, L], True means padding
          lengths: [B]
          restore_batch: original api_batch [T]
          restore_pos: original event position inside each sample [T]
        """
        device = event_emb.device
        dtype = event_emb.dtype
        T, D = event_emb.shape

        lengths = torch.bincount(api_batch, minlength=num_graphs).to(device=device)
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0

        if max_len <= 0:
            padded = torch.zeros((num_graphs, 0, D), device=device, dtype=dtype)
            key_padding_mask = torch.ones((num_graphs, 0), device=device, dtype=torch.bool)
            restore_pos = torch.empty((T,), device=device, dtype=torch.long)
            return padded, key_padding_mask, lengths, api_batch, restore_pos

        padded = torch.zeros((num_graphs, max_len, D), device=device, dtype=dtype)
        key_padding_mask = torch.ones((num_graphs, max_len), device=device, dtype=torch.bool)

        offsets = torch.zeros((num_graphs + 1,), device=device, dtype=torch.long)
        offsets[1:] = lengths.cumsum(dim=0)
        restore_pos = torch.arange(T, device=device, dtype=torch.long) - offsets[api_batch]

        padded[api_batch, restore_pos] = event_emb
        key_padding_mask[api_batch, restore_pos] = False

        # TransformerEncoder can produce NaN when a row is fully padded.
        # Make one dummy valid position for empty samples; it will not be restored to token_emb.
        empty_rows = lengths == 0
        if empty_rows.any() and max_len > 0:
            key_padding_mask[empty_rows, 0] = False

        return padded, key_padding_mask, lengths, api_batch, restore_pos

    def forward(self, graph_data, num_graphs: int, device, dtype):
        api_ids = getattr(graph_data, "api_ids", None)
        api_batch = getattr(graph_data, "api_batch", None)

        if api_ids is None or api_batch is None or api_ids.numel() == 0:
            return self._empty_output(num_graphs, device, dtype)

        api_ids = api_ids.to(device=device, dtype=torch.long)
        api_batch = api_batch.to(device=device, dtype=torch.long)

        api_ids = api_ids.clamp(0, self.num_hash_buckets + 1)
        api_batch = api_batch.clamp(0, max(num_graphs - 1, 0))

        # Limit API events per sample to avoid very long Transformer/GRU sequences.
        if self.max_seq_len > 0:
            raw_lengths = torch.bincount(api_batch, minlength=num_graphs).to(device=device)
            offsets = torch.zeros((num_graphs + 1,), device=device, dtype=torch.long)
            offsets[1:] = raw_lengths.cumsum(dim=0)
            local_pos = torch.arange(api_batch.numel(), device=device, dtype=torch.long) - offsets[api_batch]
            keep = local_pos < self.max_seq_len

            api_ids = api_ids[keep]
            api_batch = api_batch[keep]

            if api_ids.numel() == 0:
                return self._empty_output(num_graphs, device, dtype)

        raw_type_ids = getattr(graph_data, "api_type_ids", None)
        if raw_type_ids is None:
            api_type_ids = torch.zeros_like(api_ids)
        else:
            api_type_ids = raw_type_ids.to(device=device, dtype=torch.long)
            if self.max_seq_len > 0:
                api_type_ids = api_type_ids[keep]
            api_type_ids = api_type_ids.clamp(0, self.type_vocab_size - 1)

        raw_sensitive = getattr(graph_data, "api_sensitive_mask", None)
        if raw_sensitive is None:
            api_sensitive = torch.zeros_like(api_ids)
        else:
            api_sensitive = raw_sensitive.to(device=device)
            if self.max_seq_len > 0:
                api_sensitive = api_sensitive[keep]
            api_sensitive = api_sensitive.float().gt(0.5).long().clamp(0, 1)

        event_emb = (
            self.api_embedding(api_ids)
            + self.type_embedding(api_type_ids)
            + self.sensitive_embedding(api_sensitive)
        )

        event_emb = self.input_norm(event_emb)
        event_emb = self.input_dropout(event_emb)

        padded, key_padding_mask, lengths, restore_batch, restore_pos = self._build_padded_batch(
            event_emb=event_emb,
            api_batch=api_batch,
            num_graphs=num_graphs,
        )

        if padded.size(1) == 0:
            return self._empty_output(num_graphs, device, dtype)

        if self.encoder_type == "transformer":
            L = padded.size(1)
            pos = torch.arange(L, device=device).clamp(max=self.max_seq_len - 1)
            pos_emb = self.pos_embedding(pos).unsqueeze(0)
            padded = padded + pos_emb

            encoded = self.sequence_encoder(
                padded,
                src_key_padding_mask=key_padding_mask,
            )

        else:
            # BiGRU
            lengths_cpu = lengths.clamp_min(1).detach().cpu()

            packed = nn.utils.rnn.pack_padded_sequence(
                padded,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.sequence_encoder(packed)
            encoded, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=padded.size(1),
            )

            encoded = encoded.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        encoded = self.out_proj(encoded)

        # Restore token-level sequence back to [T, D]
        token_emb = encoded[restore_batch, restore_pos]

        # Attention pooling per sample
        scores = self.pool_score(token_emb.float()).squeeze(-1)

        weights = softmax(scores, api_batch, num_nodes=num_graphs).to(dtype=token_emb.dtype)
        pooled = torch.zeros((num_graphs, self.emb_dim), device=device, dtype=token_emb.dtype)
        pooled.index_add_(0, api_batch, token_emb * weights.unsqueeze(-1))

        return token_emb.to(dtype=dtype), pooled.to(dtype=dtype), api_batch


class MalwareModelWithXAttn(nn.Module):
    """
    Clean API + Call Graph model.

    Supported fusion modes:
      - api: API sequence only
      - graph: call graph only
      - concat: concat(API, graph)
      - late_fusion: weighted logits fusion
      - cross_attention: graph nodes attend to API events
      - ours: API + graph + cross-attention joint branch + tri-branch gate
    """

    def __init__(
        self,
        num_classes: int = 2,
        api_emb_dim: int = 128,
        graph_emb_dim: int = 128,
        align_dim: int = 128,
        max_nodes_gnn: int = 2048,
        max_xattn_nodes: int = 512,
        in_feat_dim: int = 515,
        use_temporal_regularization: bool = True,
        gate_detach: bool = True,
        xattn_heads: int = 4,
        fusion_mode: str = "ours",
        graph_encoder_type: str = "gatv2",
        graph_hidden: int = 128,
        graph_heads: int = 4,
        graph_layers: int = 2,
        graph_use_behavior_hint: bool = True,

        api_encoder_type: str = "transformer",
        api_max_seq_len: int = 1024,
        api_heads: int = 4,
        api_layers: int = 2,

        api_num_hash_buckets: int = ArchitectureConstants.API_NUM_HASH_BUCKETS,
        api_type_vocab_size: int = ArchitectureConstants.API_TYPE_VOCAB_SIZE,
        alignment_penalty_scale: float = 1.0,
        alignment_bonus_scale: float | None = None,
        alignment_context_scale: float = 0.35,
        use_alignment_bias: bool = True,
        use_adaptive_alignment_bias: bool = False,
        use_alignment_drift_guidance: bool = True,
        use_drift_gate: bool = True,
        use_quality_gate_inputs: bool = True,
        gate_mode: str = "learned",
        late_fusion_api_weight: float = 0.5,
        temporal_num_domains: int = 1,
        temporal_prototype_momentum: float = 0.95,
        temporal_prototype_clusters: int = 4,
        temporal_drift_velocity_scale: float = 0.5,
        temporal_drift_min_history: int = 2,
        use_future_temporal_drift: bool = True,
        use_temporal_risk_calibration: bool = True,
    ):
        super().__init__()

        if api_emb_dim != graph_emb_dim:
            raise ValueError(
                f"api_emb_dim ({api_emb_dim}) must equal graph_emb_dim ({graph_emb_dim})"
            )

        valid_fusion_modes = {
            "api",
            "graph",
            "concat",
            "late_fusion",
            "cross_attention",
            "ours",
        }
        if fusion_mode not in valid_fusion_modes:
            raise ValueError(f"fusion_mode='{fusion_mode}' not in {valid_fusion_modes}")

        self.num_classes = int(num_classes)
        self.api_emb_dim = int(api_emb_dim)
        self.graph_emb_dim = int(graph_emb_dim)
        self.align_dim = int(align_dim)
        self.fusion_mode = str(fusion_mode)
        self.use_temporal_regularization = bool(use_temporal_regularization)
        self.gate_detach = bool(gate_detach)
        self.graph_use_behavior_hint = bool(graph_use_behavior_hint)
        self.temporal_num_domains = max(int(temporal_num_domains), 1)
        self.temporal_prototype_momentum = float(temporal_prototype_momentum)
        self.temporal_prototype_clusters = max(int(temporal_prototype_clusters), 1)
        self.temporal_drift_velocity_scale = float(temporal_drift_velocity_scale)
        self.temporal_drift_min_history = max(int(temporal_drift_min_history), 1)
        self.use_future_temporal_drift = bool(use_future_temporal_drift)
        self.use_temporal_risk_calibration = bool(use_temporal_risk_calibration)

        self.api_encoder_type = str(api_encoder_type).lower()
        self.api_max_seq_len = int(api_max_seq_len)
        self.api_heads = int(api_heads)
        self.api_layers = int(api_layers)

        self.max_xattn_nodes = int(max_xattn_nodes)
        if self.max_xattn_nodes <= 0:
            raise ValueError("max_xattn_nodes must be positive")

        self.alignment_penalty_scale = float(alignment_penalty_scale)
        if not (0.0 <= self.alignment_penalty_scale <= 1.0):
            raise ValueError("alignment_penalty_scale must be in [0, 1]")

        self.alignment_bonus_scale = (
            float(alignment_bonus_scale)
            if alignment_bonus_scale is not None
            else self.alignment_penalty_scale
        )
        if not (0.0 <= self.alignment_bonus_scale <= 1.0):
            raise ValueError("alignment_bonus_scale must be in [0, 1]")

        self.alignment_context_scale = float(alignment_context_scale)
        if not (0.0 <= self.alignment_context_scale <= 2.0):
            raise ValueError("alignment_context_scale must be in [0, 2]")

        self.use_alignment_bias = bool(use_alignment_bias)
        self.use_adaptive_alignment_bias = bool(use_adaptive_alignment_bias)
        self.use_alignment_drift_guidance = bool(use_alignment_drift_guidance)
        self.use_drift_gate = bool(use_drift_gate)
        self.use_quality_gate_inputs = bool(use_quality_gate_inputs)
        self.training_stage = "main"
        self.force_fixed_gate = False
        self.force_disable_alignment = False
        self.force_disable_future_temporal = False
        self.gate_mode = str(gate_mode or "learned").lower()
        if self.gate_mode not in {"learned", "fixed"}:
            raise ValueError("gate_mode must be one of: learned, fixed")

        self.late_fusion_api_weight = float(late_fusion_api_weight)
        if not (0.0 <= self.late_fusion_api_weight <= 1.0):
            raise ValueError("late_fusion_api_weight must be in [0, 1]")

        self._need_api_encoder = fusion_mode != "graph"
        self._need_graph_encoder = fusion_mode != "api"
        self._need_cross_attn = fusion_mode in {"cross_attention", "ours"}
        self._need_gates = fusion_mode == "ours" and self.gate_mode == "learned"
        self._need_joint_head = fusion_mode in {"concat", "cross_attention", "ours"}
        self._need_alignment = fusion_mode == "ours"

        if self._need_api_encoder:
            self.api_encoder = ApiSequenceEncoder(
                num_hash_buckets=int(api_num_hash_buckets),
                type_vocab_size=int(api_type_vocab_size),
                emb_dim=api_emb_dim,
                hidden_dim=ArchitectureConstants.API_PROJ_HIDDEN,
                dropout=ArchitectureConstants.API_DROPOUT,
                encoder_type=self.api_encoder_type,
                num_layers=self.api_layers,
                num_heads=self.api_heads,
                max_seq_len=self.api_max_seq_len,
            )
        else:
            self.api_encoder = None

        if self._need_graph_encoder:
            if graph_encoder_type == "gcn":
                self.graph_encoder = GraphEncoderGCN(
                    in_feat_dim,
                    graph_emb_dim,
                    hidden=graph_hidden,
                    max_nodes=max_nodes_gnn,
                    use_behavior_hint=self.graph_use_behavior_hint,
                )
            elif graph_encoder_type == "gps":
                self.graph_encoder = GraphEncoderGPS(
                    in_feat_dim,
                    graph_emb_dim,
                    hidden=graph_hidden,
                    heads=graph_heads,
                    num_layers=graph_layers,
                    max_nodes=max_nodes_gnn,
                    use_behavior_hint=self.graph_use_behavior_hint,
                )
            else:
                self.graph_encoder = GraphEncoderGAT(
                    in_feat_dim,
                    graph_emb_dim,
                    hidden=graph_hidden,
                    heads=graph_heads,
                    num_layers=graph_layers,
                    max_nodes=max_nodes_gnn,
                    use_behavior_hint=self.graph_use_behavior_hint,
                )
        else:
            self.graph_encoder = None

        if self._need_cross_attn:
            self.cross_attn = CrossAttention(
                graph_emb_dim,
                api_emb_dim,
                graph_emb_dim,
                num_heads=xattn_heads,
            )
        else:
            self.cross_attn = None

        if self._need_gates:
            self.gate_net = TriBranchGate(api_emb_dim, graph_emb_dim, q_dim=10)
        else:
            self.gate_net = None

        if fusion_mode == "api":
            temporal_in_dim = api_emb_dim
        elif fusion_mode == "graph":
            temporal_in_dim = graph_emb_dim
        elif fusion_mode == "cross_attention":
            temporal_in_dim = api_emb_dim + graph_emb_dim + api_emb_dim
        else:
            temporal_in_dim = api_emb_dim + graph_emb_dim

        if temporal_in_dim == api_emb_dim:
            self.temporal_feature_proj = nn.Identity()
        else:
            self.temporal_feature_proj = nn.Sequential(
                nn.Linear(temporal_in_dim, api_emb_dim),
                nn.LayerNorm(api_emb_dim),
            )

        self.temporal_pair_proj = nn.Sequential(
            nn.Linear(api_emb_dim + graph_emb_dim, api_emb_dim),
            nn.LayerNorm(api_emb_dim),
        )
        # Keep downstream classifier initialization seed-compatible with the
        # previous 3-input risk head. The actual 6-input head is created after
        # the main heads below.
        _risk_head_rng_compat = nn.Sequential(
            nn.Linear(3, ArchitectureConstants.GATE_HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Linear(ArchitectureConstants.GATE_HIDDEN_DIM // 2, 1),
        )
        del _risk_head_rng_compat
        self.temporal_risk_head = None

        if self._need_alignment:
            self.api_align_proj = nn.Sequential(
                nn.Linear(api_emb_dim, align_dim),
                nn.LayerNorm(align_dim),
            )
            self.graph_align_proj = nn.Sequential(
                nn.Linear(graph_emb_dim, align_dim),
                nn.LayerNorm(align_dim),
            )
            self.alignment_context_delta = nn.Sequential(
                nn.Linear(graph_emb_dim * 3, graph_emb_dim),
                nn.GELU(),
                nn.Dropout(ArchitectureConstants.HEAD_DROPOUT),
                nn.Linear(graph_emb_dim, graph_emb_dim),
            )
            self.alignment_context_norm = nn.LayerNorm(graph_emb_dim)
        else:
            self.api_align_proj = None
            self.graph_align_proj = None
            self.alignment_context_delta = None
            self.alignment_context_norm = None

        self.api_head = None
        self.graph_head = None
        self.joint_head = None

        if self._need_api_encoder:
            self.api_head = build_main_head(api_emb_dim, num_classes)

        if self._need_graph_encoder:
            self.graph_head = build_main_head(graph_emb_dim, num_classes)

        if self._need_joint_head:
            if fusion_mode == "cross_attention":
                joint_in_dim = api_emb_dim + graph_emb_dim + api_emb_dim
            else:
                joint_in_dim = api_emb_dim + graph_emb_dim
            self.joint_head = build_main_head(joint_in_dim, num_classes)

        self.temporal_prototype_memory = TemporalPrototypeMemory(
            num_domains=self.temporal_num_domains,
            num_classes=num_classes,
            feature_dim=api_emb_dim,
            momentum=self.temporal_prototype_momentum,
            num_clusters=self.temporal_prototype_clusters,
        )
        self.temporal_risk_head = nn.Sequential(
            nn.Linear(7, ArchitectureConstants.GATE_HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Linear(ArchitectureConstants.GATE_HIDDEN_DIM // 2, 1),
        )

    def set_training_stage(self, stage: str):
        """Switch lightweight stage-specific behavior without rebuilding modules."""
        stage = str(stage or "main").lower()
        if stage not in {"warmup", "main"}:
            raise ValueError("training stage must be 'warmup' or 'main'")
        self.training_stage = stage
        self.force_fixed_gate = stage == "warmup"
        self.force_disable_alignment = stage == "warmup"
        self.force_disable_future_temporal = stage == "warmup"

    def _encode_api(self, graph_data, batch_size: int, device, dtype, return_token_seqs: bool = True):
        if self.api_encoder is None:
            pooled = torch.zeros((batch_size, self.api_emb_dim), device=device, dtype=dtype)
            if return_token_seqs:
                token_seqs = [
                    torch.zeros((0, self.api_emb_dim), device=device, dtype=dtype)
                    for _ in range(batch_size)
                ]
                return token_seqs, pooled
            return None, pooled

        token_emb, pooled, api_batch = self.api_encoder(graph_data, batch_size, device, dtype)

        if not return_token_seqs:
            return None, pooled

        token_seqs = []
        for i in range(batch_size):
            idx = api_batch == i
            token_seqs.append(token_emb[idx] if idx.any() else token_emb[:0])

        return token_seqs, pooled

    def build_alignment_features(self, api_emb, graph_emb):
        if self.api_align_proj is None or self.graph_align_proj is None:
            return None, None
        if api_emb is None or graph_emb is None:
            return None, None

        api_z = F.normalize(self.api_align_proj(api_emb), dim=-1)
        graph_z = F.normalize(self.graph_align_proj(graph_emb), dim=-1)

        if not torch.isfinite(api_z).all() or not torch.isfinite(graph_z).all():
            return None, None

        return api_z, graph_z

    def _project_temporal_feature(self, feature: torch.Tensor) -> torch.Tensor:
        projected = self.temporal_feature_proj(feature)
        return F.normalize(projected, dim=-1)

    def _project_temporal_pair(self, api_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([api_emb, graph_emb], dim=-1)
        return F.normalize(self.temporal_pair_proj(pair), dim=-1)

    def _estimate_temporal_drift(
        self,
        feature: torch.Tensor | None,
        time_ids: torch.Tensor | None,
        logits: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if (
            feature is None
            or time_ids is None
            or not self.use_temporal_regularization
            or self.temporal_prototype_memory is None
        ):
            return None

        class_probs = None
        if logits is not None:
            class_probs = torch.softmax(logits.detach().float(), dim=-1)

        drift = self.temporal_prototype_memory.estimate_drift(
            feature.detach(),
            time_ids.detach(),
            class_probs=class_probs,
            include_future=(
                self.use_future_temporal_drift
                and not self.force_disable_future_temporal
            ),
            velocity_scale=self.temporal_drift_velocity_scale,
            min_history=self.temporal_drift_min_history,
        )
        return drift.to(device=feature.device, dtype=feature.dtype)

    def _estimate_temporal_risk_components(
        self,
        feature: torch.Tensor | None,
        time_ids: torch.Tensor | None,
        logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if (
            feature is None
            or time_ids is None
            or not self.use_temporal_regularization
            or self.temporal_prototype_memory is None
        ):
            return None

        class_probs = None
        if logits is not None:
            class_probs = torch.softmax(logits.detach().float(), dim=-1)

        components = self.temporal_prototype_memory.estimate_risk_components(
            feature.detach(),
            time_ids.detach(),
            class_probs=class_probs,
            include_future=(
                self.use_future_temporal_drift
                and not self.force_disable_future_temporal
            ),
            velocity_scale=self.temporal_drift_velocity_scale,
            min_history=self.temporal_drift_min_history,
        )
        return {
            k: v.to(device=feature.device, dtype=feature.dtype)
            for k, v in components.items()
        }

    def _calibrate_temporal_risk(
        self,
        temporal_drift: torch.Tensor | None,
        prototype_pred_distance: torch.Tensor | None,
        prototype_margin_risk: torch.Tensor | None,
        prototype_label_mismatch: torch.Tensor | None,
        prototype_reliability_risk: torch.Tensor | None,
        disagreement: torch.Tensor,
        entropy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map raw prototype drift to a calibrated temporal error-risk score."""
        if temporal_drift is None:
            temporal_drift = torch.zeros_like(entropy)
        temporal_drift = temporal_drift.to(device=entropy.device, dtype=entropy.dtype).view_as(entropy)
        if prototype_pred_distance is None:
            prototype_pred_distance = temporal_drift
        prototype_pred_distance = prototype_pred_distance.to(device=entropy.device, dtype=entropy.dtype).view_as(entropy)
        if prototype_margin_risk is None:
            prototype_margin_risk = torch.zeros_like(entropy)
        prototype_margin_risk = prototype_margin_risk.to(device=entropy.device, dtype=entropy.dtype).view_as(entropy)
        if prototype_label_mismatch is None:
            prototype_label_mismatch = torch.zeros_like(entropy)
        prototype_label_mismatch = prototype_label_mismatch.to(device=entropy.device, dtype=entropy.dtype).view_as(entropy)
        if prototype_reliability_risk is None:
            prototype_reliability_risk = torch.zeros_like(entropy)
        prototype_reliability_risk = prototype_reliability_risk.to(device=entropy.device, dtype=entropy.dtype).view_as(entropy)
        risk_inputs = torch.cat(
            [
                temporal_drift.clamp(0.0, 1.0),
                prototype_pred_distance.clamp(0.0, 1.0),
                prototype_margin_risk.clamp(0.0, 1.0),
                prototype_label_mismatch.clamp(0.0, 1.0),
                prototype_reliability_risk.clamp(0.0, 1.0),
                disagreement.detach().clamp(0.0, 1.0),
                entropy.detach().clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        if not self.use_temporal_risk_calibration:
            raw_logit = torch.logit(temporal_drift.clamp(1e-4, 1.0 - 1e-4))
            return raw_logit, temporal_drift.clamp(0.0, 1.0)
        risk_logit = self.temporal_risk_head(risk_inputs.detach())
        return risk_logit, torch.sigmoid(risk_logit).clamp(0.0, 1.0)

    def _explicit_qs_to_tensor(self, explicit_qs, b, device, dtype):
        if explicit_qs is None:
            return torch.ones((b, 5), device=device, dtype=dtype)

        vals = []
        for i in range(5):
            if i < len(explicit_qs) and explicit_qs[i] is not None:
                vals.append(explicit_qs[i].to(device=device, dtype=dtype).view(b, 1))
            else:
                vals.append(torch.ones((b, 1), device=device, dtype=dtype))

        return torch.cat(vals, dim=-1).clamp(0.0, 1.0)

    def _compute_drift_components(self, api_logits, graph_logits, joint_logits, temporal_drift=None):
        p_api = torch.softmax(api_logits.detach(), dim=-1)
        p_graph = torch.softmax(graph_logits.detach(), dim=-1)
        p_joint = torch.softmax(joint_logits.detach(), dim=-1)

        p_mean = (p_api + p_graph + p_joint) / 3.0

        disagreement = (
            (p_api - p_mean).abs().mean(dim=-1, keepdim=True)
            + (p_graph - p_mean).abs().mean(dim=-1, keepdim=True)
            + (p_joint - p_mean).abs().mean(dim=-1, keepdim=True)
        ).clamp(0.0, 1.0)

        entropy = -(
            p_joint.clamp_min(1e-8)
            * p_joint.clamp_min(1e-8).log()
        ).sum(dim=-1, keepdim=True) / math.log(max(p_joint.size(-1), 2))

        W = ArchitectureConstants
        if temporal_drift is None:
            temporal_drift = torch.zeros_like(entropy)
        else:
            temporal_drift = temporal_drift.to(device=entropy.device, dtype=entropy.dtype).view_as(entropy)

        drift_score = (
            W.DRIFT_W_TEMPORAL * temporal_drift
            + W.DRIFT_W_DISAGREE * disagreement
            + W.DRIFT_W_ENTROPY * entropy
        ).clamp(0.0, 1.0)
        return drift_score, temporal_drift.clamp(0.0, 1.0), disagreement, entropy

    def _attach_temporal_risk(
        self,
        extra: dict,
        temporal_feature: torch.Tensor | None,
        time_ids: torch.Tensor | None,
        logits: torch.Tensor | None,
        api_logits: torch.Tensor | None = None,
        graph_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (
            temporal_feature is None
            or time_ids is None
            or logits is None
            or not self.use_temporal_regularization
        ):
            return None, None

        risk_components = self._estimate_temporal_risk_components(
            temporal_feature,
            time_ids,
            logits,
        )
        raw_temporal_drift = (
            None
            if risk_components is None
            else risk_components.get("temporal_drift")
        )
        proto_pred_dist = (
            None
            if risk_components is None
            else risk_components.get("prototype_pred_distance")
        )
        proto_margin_risk = (
            None
            if risk_components is None
            else risk_components.get("prototype_margin_risk")
        )
        proto_label_mismatch = (
            None
            if risk_components is None
            else risk_components.get("prototype_label_mismatch")
        )
        proto_reliability_risk = (
            None
            if risk_components is None
            else risk_components.get("prototype_reliability_risk")
        )

        if (
            api_logits is not None
            and graph_logits is not None
            and api_logits.shape == logits.shape
            and graph_logits.shape == logits.shape
        ):
            _, raw_gate_temporal_drift, disagreement, entropy = self._compute_drift_components(
                api_logits,
                graph_logits,
                logits,
                temporal_drift=raw_temporal_drift,
            )
        else:
            p = torch.softmax(logits.detach(), dim=-1)
            entropy = -(
                p.clamp_min(1e-8) * p.clamp_min(1e-8).log()
            ).sum(dim=-1, keepdim=True) / math.log(max(p.size(-1), 2))
            disagreement = torch.zeros_like(entropy)
            raw_gate_temporal_drift = (
                torch.zeros_like(entropy)
                if raw_temporal_drift is None
                else raw_temporal_drift.to(device=entropy.device, dtype=entropy.dtype).view_as(entropy)
            )

        risk_logit, risk_score = self._calibrate_temporal_risk(
            raw_gate_temporal_drift,
            proto_pred_dist,
            proto_margin_risk,
            proto_label_mismatch,
            proto_reliability_risk,
            disagreement,
            entropy,
        )

        B = logits.size(0)
        if raw_temporal_drift is not None:
            extra["temporal_drift_score"] = raw_temporal_drift.detach().view(B)
            extra["temporal_drift_raw_for_gate"] = raw_gate_temporal_drift.detach().view(B)
        if proto_pred_dist is not None:
            extra["temporal_proto_pred_dist"] = proto_pred_dist.detach().view(B)
        if proto_margin_risk is not None:
            extra["temporal_proto_margin_risk"] = proto_margin_risk.detach().view(B)
        if proto_label_mismatch is not None:
            extra["temporal_proto_label_mismatch"] = proto_label_mismatch.detach().view(B)
        if proto_reliability_risk is not None:
            extra["temporal_proto_reliability_risk"] = proto_reliability_risk.detach().view(B)
        extra["temporal_risk_logit"] = risk_logit.view(B)
        extra["temporal_risk_score"] = risk_score.detach().view(B)
        return raw_temporal_drift, risk_score

    @staticmethod
    def _modality_alive(emb: torch.Tensor) -> torch.Tensor:
        return (
            emb.abs().sum(dim=-1, keepdim=True)
            > ArchitectureConstants.MODALITY_ALIVE_THRESHOLD
        ).float()

    def _build_hard_mask(self, token_seqs: List[torch.Tensor], max_tokens: int, device, dtype) -> torch.Tensor:
        B = len(token_seqs)
        hard_neg = ArchitectureConstants.MASK_HARD_NEG

        mask = torch.zeros((B, 1, 1, max_tokens), device=device, dtype=dtype)
        for i, ts in enumerate(token_seqs):
            n = ts.size(0)
            if n < max_tokens:
                mask[i, :, :, n:] = hard_neg
        return mask

    def _build_alignment_context(
        self,
        masks,
        node_dense: torch.Tensor,
        node_key_mask: torch.Tensor,
        padded_api: torch.Tensor,
        xattn_pooled: torch.Tensor,
        align_scale: torch.Tensor | None,
    ):
        if self.alignment_context_delta is None or masks is None:
            return xattn_pooled, None, None

        B, max_nodes, _ = node_dense.shape
        max_tokens = padded_api.size(1)
        device = node_dense.device
        dtype = node_dense.dtype

        aligned_nodes = torch.zeros_like(xattn_pooled)
        aligned_apis = torch.zeros_like(xattn_pooled)
        coverage = torch.zeros((B, 1), device=device, dtype=dtype)
        density = torch.zeros((B, 1), device=device, dtype=dtype)

        for i, m in enumerate(masks):
            if m is None or m.numel() == 0 or i >= B:
                continue

            n_m = min(int(m.size(0)), max_nodes)
            t_m = min(int(m.size(1)), max_tokens)
            if n_m <= 0 or t_m <= 0:
                continue

            local_weight = m[:n_m, :t_m].to(device=device, dtype=dtype).clamp(0.0, 1.0)
            local_mask = local_weight > 0.0
            if not local_mask.any():
                continue

            node_w = local_weight.max(dim=1).values
            node_w = node_w * node_key_mask[i, :n_m].to(dtype)
            api_w = local_weight.max(dim=0).values

            node_denom = node_w.sum().clamp_min(1.0)
            api_denom = api_w.sum().clamp_min(1.0)

            aligned_nodes[i] = (
                node_dense[i, :n_m] * node_w.unsqueeze(-1)
            ).sum(dim=0) / node_denom
            aligned_apis[i] = (
                padded_api[i, :t_m] * api_w.unsqueeze(-1)
            ).sum(dim=0) / api_denom

            node_cov = node_w.sum() / max(float(n_m), 1.0)
            api_cov = api_w.sum() / max(float(t_m), 1.0)
            coverage[i, 0] = (node_cov * api_cov).sqrt().clamp(0.0, 1.0)
            density[i, 0] = local_weight.mean().clamp(0.0, 1.0)

        if align_scale is not None:
            scale = align_scale.view(B, -1)[:, :1].to(device=device, dtype=dtype)
        else:
            scale = torch.ones((B, 1), device=device, dtype=dtype)

        context = torch.cat([xattn_pooled, aligned_nodes, aligned_apis], dim=-1)
        delta = self.alignment_context_delta(context)
        xattn_pooled = self.alignment_context_norm(
            xattn_pooled
            + self.alignment_context_scale
            * coverage
            * scale.clamp(0.0, 1.0)
            * delta
        )

        return xattn_pooled, coverage.detach(), density.detach()

    @staticmethod
    def _summarize_alignment_masks(masks, batch_size: int, device, dtype):
        coverage = torch.zeros((batch_size,), device=device, dtype=dtype)
        density = torch.zeros((batch_size,), device=device, dtype=dtype)
        if masks is None:
            return coverage, density

        for i, m in enumerate(masks):
            if i >= batch_size or m is None or m.numel() == 0:
                continue
            weight = m.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            if weight.numel() == 0:
                continue
            node_cov = (weight.max(dim=1).values > 0.0).to(dtype).mean()
            api_cov = (weight.max(dim=0).values > 0.0).to(dtype).mean()
            coverage[i] = (node_cov * api_cov).sqrt().clamp(0.0, 1.0)
            density[i] = weight.mean().clamp(0.0, 1.0)
        return coverage, density

    def _select_xattn_nodes(self, node_emb, graph_batch, masks, batch_size: int):
        if (
            node_emb is None
            or graph_batch is None
            or node_emb.numel() == 0
            or self.max_xattn_nodes <= 0
        ):
            return node_emb, graph_batch, masks

        selected_global = []
        selected_masks = [] if masks is not None else masks
        device = node_emb.device

        for i in range(batch_size):
            idx = torch.where(graph_batch == i)[0]
            if idx.numel() == 0:
                if selected_masks is not None:
                    selected_masks.append(None)
                continue

            if idx.numel() <= self.max_xattn_nodes:
                keep_local = torch.arange(idx.numel(), device=device)
            else:
                align_rows = None
                if masks is not None and i < len(masks) and masks[i] is not None and masks[i].numel() > 0:
                    m = masks[i]
                    n = min(int(m.size(0)), int(idx.numel()))
                    if n > 0:
                        align_rows = (m[:n].to(device=device).float() > 0.0).any(dim=1)

                if align_rows is not None and align_rows.any():
                    aligned = torch.where(align_rows)[0]
                    rest = torch.where(~align_rows)[0]
                    order = torch.cat([aligned, rest], dim=0)
                    if order.numel() < idx.numel():
                        tail = torch.arange(order.numel(), idx.numel(), device=device)
                        order = torch.cat([order, tail], dim=0)
                    keep_local = order[: self.max_xattn_nodes]
                else:
                    keep_local = torch.arange(self.max_xattn_nodes, device=device)

            selected_global.append(idx[keep_local])
            if selected_masks is not None:
                if i < len(masks) and masks[i] is not None and masks[i].numel() > 0:
                    local_for_mask = keep_local.to(device=masks[i].device)
                    valid = local_for_mask < masks[i].size(0)
                    selected_mask = masks[i].new_zeros((keep_local.numel(), masks[i].size(1)))
                    if valid.any():
                        selected_mask[valid] = masks[i][local_for_mask[valid]]
                    selected_masks.append(selected_mask)
                else:
                    selected_masks.append(None)

        if not selected_global:
            return node_emb[:0], graph_batch[:0], selected_masks

        selected_global = torch.cat(selected_global, dim=0)
        return node_emb[selected_global], graph_batch[selected_global], selected_masks

    def forward(
        self,
        graph_data=None,
        y=None,
        explicit_qs=None,
        time_ids=None,
        return_features=False,
        masks=None,
    ):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        B = getattr(graph_data, "num_graphs", 1) if graph_data is not None else 1

        need_api_token_seqs = self._need_cross_attn

        if self._need_api_encoder:
            api_token_seqs, api_emb = self._encode_api(
                graph_data,
                B,
                device,
                dtype,
                return_token_seqs=need_api_token_seqs,
            )
        else:
            api_token_seqs = None
            api_emb = torch.zeros((B, self.api_emb_dim), device=device, dtype=dtype)

        if self._need_graph_encoder and graph_data is not None:
            node_emb, graph_emb, graph_batch, keep_local_parts = self.graph_encoder(graph_data)
        else:
            node_emb = None
            graph_emb = torch.zeros((B, self.graph_emb_dim), device=device, dtype=dtype)
            graph_batch = None
            keep_local_parts = []

        if masks is not None and keep_local_parts:
            truncated_masks = []
            for i, m in enumerate(masks):
                if m is not None and i < len(keep_local_parts) and keep_local_parts[i].numel() > 0:
                    local_idx = keep_local_parts[i]
                    valid_idx = local_idx[local_idx < m.size(0)]
                    truncated_masks.append(m[valid_idx] if valid_idx.numel() > 0 else m[:0])
                else:
                    truncated_masks.append(m)
            masks = truncated_masks

        raw_qs = self._explicit_qs_to_tensor(explicit_qs, B, device, dtype)
        gate_qs = raw_qs.clone()

        if not self.use_quality_gate_inputs:
            neutral_qs = torch.ones_like(gate_qs)
            neutral_qs[:, 3:] = 0.0
            gate_qs = neutral_qs

        feat_drop = ArchitectureConstants.FEATURE_DROPOUT
        if self.training and feat_drop > 0.0:
            api_emb = F.dropout(api_emb, p=feat_drop, training=True)
            graph_emb = F.dropout(graph_emb, p=feat_drop, training=True)

        extra = {}

        temporal_api_feature = F.normalize(api_emb, dim=-1)
        temporal_graph_feature = F.normalize(graph_emb, dim=-1)

        def _set_temporal_features(fusion_feature, projected_feature=None):
            if not (self.training and y is not None and self.use_temporal_regularization and time_ids is not None):
                return (
                    projected_feature
                    if projected_feature is not None
                    else self._project_temporal_feature(fusion_feature)
                )
            fusion_feature = (
                projected_feature
                if projected_feature is not None
                else self._project_temporal_feature(fusion_feature)
            )
            extra["temporal_features_api"] = temporal_api_feature
            extra["temporal_features_graph"] = temporal_graph_feature
            extra["temporal_features_fusion"] = fusion_feature
            extra["temporal_features"] = fusion_feature
            extra["temporal_prototype_memory"] = self.temporal_prototype_memory
            return fusion_feature

        if time_ids is not None:
            extra["time_ids"] = time_ids

        alignment_coverage_prior = None
        alignment_density_prior = None
        if masks is not None and self.fusion_mode == "ours":
            alignment_coverage_prior, alignment_density_prior = self._summarize_alignment_masks(
                masks,
                B,
                device,
                dtype,
            )
            extra["alignment_coverage_prior"] = alignment_coverage_prior.detach()
            extra["alignment_density_prior"] = alignment_density_prior.detach()

        if self.fusion_mode == "api":
            risk_feature = _set_temporal_features(api_emb, projected_feature=temporal_api_feature)
            if self.training and y is not None and self.use_temporal_regularization and time_ids is not None:
                extra["temporal_quality"] = raw_qs[:, 0].detach().clamp(0.0, 1.0)
            logits = self.api_head(api_emb)
            self._attach_temporal_risk(extra, risk_feature, time_ids, logits)
            if return_features:
                extra["api_emb"] = api_emb
            return logits, extra

        if self.fusion_mode == "graph":
            risk_feature = _set_temporal_features(graph_emb, projected_feature=temporal_graph_feature)
            if self.training and y is not None and self.use_temporal_regularization and time_ids is not None:
                extra["temporal_quality"] = raw_qs[:, 1].detach().clamp(0.0, 1.0)
            logits = self.graph_head(graph_emb)
            self._attach_temporal_risk(extra, risk_feature, time_ids, logits)
            if return_features:
                extra["graph_emb"] = graph_emb
            return logits, extra

        api_logits = self.api_head(api_emb) if self.api_head is not None else None
        graph_logits = self.graph_head(graph_emb) if self.graph_head is not None else None

        if self.fusion_mode in {"late_fusion", "concat", "cross_attention", "ours"}:
            if api_logits is not None:
                extra["api_logits_aux"] = api_logits
            if graph_logits is not None:
                extra["graph_logits_aux"] = graph_logits

        if self.fusion_mode == "late_fusion":
            late_joint = torch.cat([api_emb, graph_emb], dim=-1)
            risk_feature = _set_temporal_features(late_joint)
            if self.training and y is not None and self.use_temporal_regularization and time_ids is not None:
                extra["temporal_quality"] = (
                    raw_qs[:, 0] * raw_qs[:, 1]
                ).detach().clamp(0.0, 1.0)
            wa = self.late_fusion_api_weight
            logits = wa * api_logits + (1.0 - wa) * graph_logits
            self._attach_temporal_risk(extra, risk_feature, time_ids, logits, api_logits, graph_logits)
            if return_features:
                extra["api_emb"] = api_emb
                extra["graph_emb"] = graph_emb
            return logits, extra

        temporal_pair_feature = None
        alignment_temporal_drift = None
        if self.fusion_mode == "ours" and time_ids is not None:
            temporal_pair_feature = self._project_temporal_pair(api_emb, graph_emb)
            pre_logits = None
            if api_logits is not None and graph_logits is not None:
                pre_logits = 0.5 * (api_logits + graph_logits)
            alignment_temporal_drift = self._estimate_temporal_drift(
                temporal_pair_feature,
                time_ids,
                pre_logits,
            )
            if alignment_temporal_drift is not None:
                extra["alignment_temporal_drift"] = alignment_temporal_drift.detach().view(B)

        if (
            self.training
            and y is not None
            and self._need_alignment
            and not self.force_disable_alignment
        ):
            api_align, graph_align = self.build_alignment_features(api_emb, graph_emb)
            if api_align is not None and graph_align is not None:
                semantic_quality = raw_qs[:, 2].detach().clamp(0.0, 1.0)
                if alignment_coverage_prior is not None and alignment_density_prior is not None:
                    coverage_gate = alignment_coverage_prior.detach().view(-1).clamp(0.0, 1.0)
                    density_gate = alignment_density_prior.detach().view(-1).clamp(0.0, 1.0).sqrt()
                    semantic_quality = semantic_quality * coverage_gate * density_gate
                    extra["semantic_alignment_coverage_gate"] = coverage_gate.detach()
                    extra["semantic_alignment_density_gate"] = density_gate.detach()
                if self.use_alignment_drift_guidance and alignment_temporal_drift is not None:
                    drift_gate = (
                        0.25
                        + 0.75 * (1.0 - alignment_temporal_drift.detach().view(-1))
                    ).clamp(0.25, 1.0)
                    semantic_quality = semantic_quality * drift_gate
                    extra["semantic_alignment_drift_gate"] = drift_gate.detach()
                extra["semantic_alignment_api"] = api_align
                extra["semantic_alignment_graph"] = graph_align
                extra["semantic_alignment_quality"] = semantic_quality

        if self._need_cross_attn and api_token_seqs is not None and node_emb is not None:
            max_api_tokens = max(ts.size(0) for ts in api_token_seqs) if api_token_seqs else 0

            if max_api_tokens > 0 and node_emb.numel() > 0:
                padded_api = torch.zeros(
                    (B, max_api_tokens, self.api_emb_dim),
                    device=device,
                    dtype=dtype,
                )

                for i, ts in enumerate(api_token_seqs):
                    n = ts.size(0)
                    if n > 0:
                        padded_api[i, :n] = ts

                xattn_node_emb, xattn_graph_batch, xattn_masks = self._select_xattn_nodes(
                    node_emb,
                    graph_batch,
                    masks,
                    B,
                )

                node_dense, node_key_mask = to_dense_batch(
                    xattn_node_emb,
                    xattn_graph_batch,
                    max_num_nodes=self.max_xattn_nodes,
                )
                max_nodes_dense = node_dense.size(1)

                attn_bias = self._build_hard_mask(api_token_seqs, max_api_tokens, device, dtype)

                align_bias = None
                align_scale_for_context = None
                alignment_bias_active = (
                    self.fusion_mode == "ours"
                    and self.use_alignment_bias
                    and not self.force_disable_alignment
                )
                if xattn_masks is not None and alignment_bias_active:
                    if self.use_adaptive_alignment_bias:
                        align_scale = raw_qs[:, 2].view(B, 1, 1, 1).clamp(0.0, 1.0)
                    else:
                        align_scale = torch.ones((B, 1, 1, 1), device=device, dtype=dtype)

                    align_scale = align_scale.to(device=device, dtype=dtype)
                    if self.use_alignment_drift_guidance and alignment_temporal_drift is not None:
                        drift_gate = (
                            0.25
                            + 0.75 * (1.0 - alignment_temporal_drift.view(B, 1, 1, 1))
                        ).clamp(0.25, 1.0)
                        align_scale = align_scale * drift_gate.to(device=device, dtype=dtype)
                        extra["alignment_drift_gate"] = drift_gate.view(B).detach()
                    align_scale_for_context = align_scale

                    penalty_scale = align_scale * self.alignment_penalty_scale
                    bonus_scale = align_scale * self.alignment_bonus_scale

                    penalty = ArchitectureConstants.ALIGN_BIAS_PENALTY * penalty_scale
                    bonus = ArchitectureConstants.ALIGN_BIAS_BONUS * bonus_scale

                    align_bias = torch.zeros(
                        (B, 1, max_nodes_dense, max_api_tokens),
                        device=device,
                        dtype=dtype,
                    )

                    for i, m in enumerate(xattn_masks):
                        if m is not None and m.numel() > 0:
                            n_m, t_m = m.shape
                            n_m = min(n_m, max_nodes_dense)
                            t_m = min(t_m, max_api_tokens)
                            if n_m <= 0 or t_m <= 0:
                                continue

                            local_weight = m[:n_m, :t_m].to(device=device, dtype=dtype).clamp(0.0, 1.0)
                            local_mask = local_weight > 0.0
                            row_has_alignment = local_mask.any(dim=1, keepdim=True)
                            local_bias = torch.zeros((n_m, t_m), device=device, dtype=dtype)
                            local_bias = bonus[i, 0, 0, 0] * local_weight
                            local_bias = torch.where(
                                row_has_alignment & (~local_mask),
                                penalty[i, 0, 0, 0],
                                local_bias,
                            )
                            align_bias[i, 0, :n_m, :t_m] = local_bias

                    padding_mask = ~node_key_mask
                    align_bias = torch.where(
                        padding_mask.unsqueeze(1).unsqueeze(-1),
                        torch.zeros_like(align_bias),
                        align_bias,
                    )

                    extra["alignment_penalty_scale"] = penalty_scale.view(B).detach()
                    extra["alignment_bonus_scale"] = bonus_scale.view(B).detach()

                xattn_out = self.cross_attn(
                    node_dense,
                    padded_api,
                    attn_bias=attn_bias,
                    q_key_mask=node_key_mask,
                    align_bias=align_bias,
                )

                valid_counts = node_key_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
                xattn_pooled = (
                    xattn_out * node_key_mask.unsqueeze(-1)
                ).sum(dim=1) / valid_counts.squeeze(-1)

                if self.fusion_mode == "ours" and self.use_alignment_bias and not self.force_disable_alignment:
                    xattn_pooled, alignment_coverage, alignment_density = self._build_alignment_context(
                        xattn_masks,
                        node_dense,
                        node_key_mask,
                        padded_api,
                        xattn_pooled,
                        align_scale_for_context,
                    )
                    if alignment_coverage is not None:
                        extra["alignment_coverage"] = alignment_coverage.view(B).detach()
                        extra["alignment_density"] = alignment_density.view(B).detach()

            else:
                xattn_pooled = torch.zeros_like(api_emb)
        else:
            xattn_pooled = torch.zeros_like(api_emb)

        if self.fusion_mode == "cross_attention":
            joint = torch.cat([api_emb, graph_emb, xattn_pooled], dim=-1)
            risk_feature = _set_temporal_features(joint)
            if self.training and y is not None and self.use_temporal_regularization and time_ids is not None:
                extra["temporal_quality"] = (
                    raw_qs[:, 0] * raw_qs[:, 1]
                ).detach().clamp(0.0, 1.0)
            logits = self.joint_head(joint)
            self._attach_temporal_risk(extra, risk_feature, time_ids, logits, api_logits, graph_logits)
            if return_features:
                extra["api_emb"] = api_emb
                extra["graph_emb"] = graph_emb
                extra["cross_emb"] = xattn_pooled
            return logits, extra

        if self.fusion_mode == "concat":
            joint = torch.cat([api_emb, graph_emb], dim=-1)
            risk_feature = _set_temporal_features(joint)
            if self.training and y is not None and self.use_temporal_regularization and time_ids is not None:
                extra["temporal_quality"] = (
                    raw_qs[:, 0] * raw_qs[:, 1]
                ).detach().clamp(0.0, 1.0)
            logits = self.joint_head(joint)
            self._attach_temporal_risk(extra, risk_feature, time_ids, logits, api_logits, graph_logits)
            if return_features:
                extra["api_emb"] = api_emb
                extra["graph_emb"] = graph_emb
            return logits, extra

        has_xattn = (
            xattn_pooled.abs().sum(dim=-1, keepdim=True) > 0
        ).to(device=device, dtype=dtype)

        fused_api = has_xattn * xattn_pooled + (1.0 - has_xattn) * api_emb

        api_alive = self._modality_alive(api_emb).to(device=device, dtype=dtype)
        graph_alive = self._modality_alive(graph_emb).to(device=device, dtype=dtype)

        joint = torch.cat([fused_api, graph_emb], dim=-1)
        joint_logits = self.joint_head(joint)
        extra["joint_logits_aux"] = joint_logits
        joint_temporal_feature = (
            temporal_pair_feature
            if temporal_pair_feature is not None
            else self._project_temporal_feature(joint)
        )
        risk_components = self._estimate_temporal_risk_components(
            joint_temporal_feature,
            time_ids,
            joint_logits,
        )
        temporal_drift_score = (
            None
            if risk_components is None
            else risk_components.get("temporal_drift")
        )
        proto_pred_dist = (
            None
            if risk_components is None
            else risk_components.get("prototype_pred_distance")
        )
        proto_margin_risk = (
            None
            if risk_components is None
            else risk_components.get("prototype_margin_risk")
        )
        proto_label_mismatch = (
            None
            if risk_components is None
            else risk_components.get("prototype_label_mismatch")
        )
        proto_reliability_risk = (
            None
            if risk_components is None
            else risk_components.get("prototype_reliability_risk")
        )

        _, raw_gate_temporal_drift, gate_disagreement, gate_entropy = self._compute_drift_components(
            api_logits,
            graph_logits,
            joint_logits,
            temporal_drift=temporal_drift_score,
        )
        temporal_risk_logit, temporal_risk_score = self._calibrate_temporal_risk(
            raw_gate_temporal_drift,
            proto_pred_dist,
            proto_margin_risk,
            proto_label_mismatch,
            proto_reliability_risk,
            gate_disagreement,
            gate_entropy,
        )
        # The gate should consume the calibrated risk estimate as a signal, but
        # not reshape the calibrator through the classification objective.
        gate_temporal_risk = temporal_risk_score.detach()
        drift_score, gate_temporal_drift, gate_disagreement, gate_entropy = self._compute_drift_components(
            api_logits,
            graph_logits,
            joint_logits,
            temporal_drift=gate_temporal_risk,
        )
        drift_score = drift_score.to(device=device, dtype=dtype)
        gate_temporal_drift = gate_temporal_drift.to(device=device, dtype=dtype)
        gate_disagreement = gate_disagreement.to(device=device, dtype=dtype)
        gate_entropy = gate_entropy.to(device=device, dtype=dtype)
        if temporal_drift_score is not None:
            extra["temporal_drift_score"] = temporal_drift_score.detach().view(B)
            extra["temporal_drift_raw_for_gate"] = raw_gate_temporal_drift.detach().view(B)
        if proto_pred_dist is not None:
            extra["temporal_proto_pred_dist"] = proto_pred_dist.detach().view(B)
        if proto_margin_risk is not None:
            extra["temporal_proto_margin_risk"] = proto_margin_risk.detach().view(B)
        if proto_label_mismatch is not None:
            extra["temporal_proto_label_mismatch"] = proto_label_mismatch.detach().view(B)
        if proto_reliability_risk is not None:
            extra["temporal_proto_reliability_risk"] = proto_reliability_risk.detach().view(B)
        extra["temporal_risk_logit"] = temporal_risk_logit.view(B)
        extra["temporal_risk_score"] = temporal_risk_score.detach().view(B)

        if not self.use_drift_gate:
            drift_score = torch.zeros_like(drift_score)
            gate_temporal_drift = torch.zeros_like(gate_temporal_drift)
            gate_disagreement = torch.zeros_like(gate_disagreement)
            gate_entropy = torch.zeros_like(gate_entropy)

        gate_inputs = torch.cat(
            [gate_qs, gate_temporal_drift, gate_disagreement, gate_entropy, api_alive, graph_alive],
            dim=-1,
        )

        if self.gate_mode == "fixed" or self.force_fixed_gate:
            gate_weights = torch.full(
                (B, 3),
                1.0 / 3.0,
                device=device,
                dtype=dtype,
            )
        else:
            gate_weights = self.gate_net(
                api_emb.detach() if self.gate_detach else api_emb,
                graph_emb.detach() if self.gate_detach else graph_emb,
                gate_inputs,
            )

        w_api = gate_weights[:, 0:1]
        w_graph = gate_weights[:, 1:2]
        w_joint = gate_weights[:, 2:3]

        logits = w_api * api_logits + w_graph * graph_logits + w_joint * joint_logits

        extra["gate_weights"] = gate_weights.detach()
        extra["drift_score"] = drift_score.detach()
        extra["gate_temporal_drift"] = gate_temporal_drift.detach()
        extra["gate_disagreement"] = gate_disagreement.detach()
        extra["gate_entropy"] = gate_entropy.detach()

        if self.training and y is not None and self.use_temporal_regularization and time_ids is not None:
            _set_temporal_features(joint, projected_feature=joint_temporal_feature)
            extra["temporal_quality"] = (
                raw_qs[:, 0:1] * raw_qs[:, 1:2]
            ).detach().view(-1).clamp(0.0, 1.0)

        if return_features:
            extra["api_emb"] = api_emb
            extra["graph_emb"] = graph_emb
            extra["joint_emb"] = joint

        return logits, extra
