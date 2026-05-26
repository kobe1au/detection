from __future__ import annotations

import torch
import torch.nn.functional as F


BRANCH_AUX_KEYS = (
    "api_logits_aux",
    "graph_logits_aux",
    "manifest_logits_aux",
    "joint_logits_aux",
)


def compute_robust_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    extra: dict | None = None,
    loss_cfg: dict | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """First-version robust objective: CE(final) + branch auxiliary CE."""
    extra = extra or {}
    loss_cfg = loss_cfg or {}
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    branch_aux_weight = float(loss_cfg.get("branch_aux_weight", 0.05))

    ce = F.cross_entropy(logits, labels.long(), label_smoothing=label_smoothing)
    branch_loss = logits.new_tensor(0.0)
    branch_count = 0
    for key in BRANCH_AUX_KEYS:
        aux_logits = extra.get(key)
        if isinstance(aux_logits, torch.Tensor) and aux_logits.shape == logits.shape:
            branch_loss = branch_loss + F.cross_entropy(
                aux_logits,
                labels.long(),
                label_smoothing=label_smoothing,
            )
            branch_count += 1
    if branch_count > 0:
        branch_loss = branch_loss / float(branch_count)

    total = ce + branch_aux_weight * branch_loss
    return total, {
        "loss": float(total.detach().item()),
        "ce": float(ce.detach().item()),
        "branch_aux": float(branch_loss.detach().item()),
        "branch_aux_weight": branch_aux_weight,
    }
