from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.autograd import Function

from fusion.constants import ArchitectureConstants


# ─────────────────────────────────────────────────────────────────────────────
# Gradient reversal
# ─────────────────────────────────────────────────────────────────────────────

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


# ─────────────────────────────────────────────────────────────────────────────
# Gate modules
# ─────────────────────────────────────────────────────────────────────────────

class TriBranchGate(nn.Module):
    """
    Three-branch softmax gate for API + Graph fusion.

    Minimal q_dim=9:
      [q_api, q_graph, q_align, pert_api, pert_graph,
       branch_disagreement, entropy, api_alive, graph_alive]
    Optional gate features can append temporal reliability and explicit
    time-position features.

    Output:
      [w_api, w_graph, w_joint]
    """

    def __init__(
        self,
        api_dim: int,
        graph_dim: int,
        q_dim: int = 9,
        hidden: int | None = None,
    ):
        super().__init__()
        hidden = hidden or ArchitectureConstants.GATE_HIDDEN_DIM

        self.q_dim = int(q_dim)

        self.net = nn.Sequential(
            nn.Linear(api_dim + graph_dim + self.q_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

        nn.init.constant_(self.net[-1].bias, ArchitectureConstants.GATE_INIT_BIAS)

        # Slightly prefer joint branch at initialization.
        with torch.no_grad():
            self.net[-1].bias[2] = 0.8

    def forward(
        self,
        api_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        qs: torch.Tensor,
    ) -> torch.Tensor:
        if qs.size(-1) != self.q_dim:
            raise ValueError(
                f"TriBranchGate expected q_dim={self.q_dim}, got {qs.size(-1)}"
            )
        x = torch.cat([api_emb, graph_emb, qs], dim=-1)
        return torch.softmax(self.net(x), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Attention modules
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """
    Cross-attention from graph nodes to API tokens.

    q  = graph node features
    kv = API token features
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float | None = None,
    ):
        super().__init__()

        dropout = dropout if dropout is not None else ArchitectureConstants.XATTN_DROPOUT

        if out_dim % num_heads != 0:
            raise ValueError(
                f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.num_heads = int(num_heads)
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim_q, out_dim)
        self.k_proj = nn.Linear(dim_kv, out_dim)
        self.v_proj = nn.Linear(dim_kv, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        q_key_mask: Optional[torch.Tensor] = None,
        align_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q:
                [B, Nq, dim_q], graph node features.
            kv:
                [B, Nk, dim_kv], API token features.
            attn_bias:
                [B, 1, 1, Nk], additive bias on API token side.
                Usually used to hard-mask padded API tokens.
            q_key_mask:
                [B, Nq] bool, True = valid graph node, False = padded node.
            align_bias:
                [B, 1, Nq, Nk], additive prior encoding method-API alignment.

        Returns:
            [B, Nq, out_dim]
        """

        B, Nq, _ = q.shape
        Nk = kv.size(1)
        H, d = self.num_heads, self.head_dim

        q_proj = self.q_proj(q)

        qh = q_proj.view(B, Nq, H, d).transpose(1, 2)
        kh = self.k_proj(kv).view(B, Nk, H, d).transpose(1, 2)
        vh = self.v_proj(kv).view(B, Nk, H, d).transpose(1, 2)

        scores = (qh @ kh.transpose(-2, -1)) * self.scale

        if attn_bias is not None:
            scores = scores + attn_bias

        if align_bias is not None:
            scores = scores + align_bias

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_drop(attn_weights)

        out = (attn_weights @ vh).transpose(1, 2).reshape(B, Nq, -1)
        out = self.out_proj(out)

        result = self.norm(q_proj + out)

        if q_key_mask is not None:
            result = result * q_key_mask.unsqueeze(-1).to(result.dtype)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Classification heads
# ─────────────────────────────────────────────────────────────────────────────

def build_classification_head(in_dim: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, ArchitectureConstants.HEAD_HIDDEN_DIMS[0]),
        nn.ReLU(inplace=True),
        nn.Dropout(ArchitectureConstants.HEAD_DROPOUT_RATES[0]),
        nn.Linear(
            ArchitectureConstants.HEAD_HIDDEN_DIMS[0],
            ArchitectureConstants.HEAD_HIDDEN_DIMS[1],
        ),
        nn.ReLU(inplace=True),
        nn.Dropout(ArchitectureConstants.HEAD_DROPOUT_RATES[1]),
        nn.Linear(ArchitectureConstants.HEAD_HIDDEN_DIMS[1], num_classes),
    )


def build_main_head(in_dim: int, num_classes: int) -> nn.Sequential:
    """
    Main classification head.

    Used for:
      - API-only branch
      - Graph-only branch
      - Joint branch
      - late_fusion / concat / cross_attention / ours modes
    """
    hidden = ArchitectureConstants.HEAD_HIDDEN_DIMS[1]
    drop = ArchitectureConstants.HEAD_DROPOUT_RATES[1]

    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(drop),
        nn.Linear(hidden, num_classes),
    )


def build_aux_head(in_dim: int, num_classes: int) -> nn.Linear:
    """
    Lightweight auxiliary classification head.

    Optional. Use this only if you explicitly want simpler branch heads
    for API / Graph auxiliary logits.
    """
    return nn.Linear(in_dim, num_classes)
