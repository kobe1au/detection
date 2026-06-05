from __future__ import annotations

import torch
import torch.nn.functional as F

from fusion.constants import ArchitectureConstants
from fusion.semantic_categories import SEMANTIC_CATEGORY_DIM


def scalar_attr(
    graph_data,
    name: str,
    batch_size: int,
    device,
    dtype,
    default: float,
) -> torch.Tensor:
    value = getattr(graph_data, name, None)

    if isinstance(value, torch.Tensor):
        out = value.to(device=device, dtype=dtype).view(batch_size, -1)
        if out.size(1) > 1:
            out = out[:, :1]
        return torch.nan_to_num(
            out.clamp(0.0, 1.0),
            nan=float(default),
            posinf=1.0,
            neginf=0.0,
        )

    return torch.full(
        (batch_size, 1),
        float(default),
        device=device,
        dtype=dtype,
    )


def semantic_counts_attr(
    graph_data,
    name: str,
    batch_size: int,
    device,
    dtype,
) -> torch.Tensor:
    value = getattr(graph_data, name, None)

    if not isinstance(value, torch.Tensor):
        return torch.zeros(
            (batch_size, SEMANTIC_CATEGORY_DIM),
            device=device,
            dtype=dtype,
        )

    out = value.to(device=device, dtype=dtype)

    if out.ndim == 1:
        out = out.view(1, -1).expand(batch_size, -1)
    else:
        out = out.view(batch_size, -1)

    if out.size(1) != SEMANTIC_CATEGORY_DIM:
        return torch.zeros(
            (batch_size, SEMANTIC_CATEGORY_DIM),
            device=device,
            dtype=dtype,
        )

    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def confidence(logits: torch.Tensor) -> torch.Tensor:
    return (
        torch.softmax(logits.detach(), dim=-1)
        .max(dim=-1, keepdim=True)
        .values
        .clamp(0.0, 1.0)
    )


def prob_disagreement(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    pa = torch.softmax(a.detach(), dim=-1)
    pb = torch.softmax(b.detach(), dim=-1)
    return (pa - pb).abs().mean(dim=-1, keepdim=True).clamp(0.0, 1.0)


def modality_alive(emb: torch.Tensor) -> torch.Tensor:
    alive = (
        emb.detach().abs().sum(dim=-1, keepdim=True)
        > ArchitectureConstants.MODALITY_ALIVE_THRESHOLD
    )
    return alive.to(dtype=emb.dtype)


def cosine_counts(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.size(0), 1))

    width = max(a.size(1), b.size(1))

    if a.size(1) < width:
        a = torch.cat([a, a.new_zeros((a.size(0), width - a.size(1)))], dim=-1)

    if b.size(1) < width:
        b = torch.cat([b, b.new_zeros((b.size(0), width - b.size(1)))], dim=-1)

    valid = (
        (a.abs().sum(dim=-1, keepdim=True) > 0)
        & (b.abs().sum(dim=-1, keepdim=True) > 0)
    )

    sim = F.cosine_similarity(a.float(), b.float(), dim=-1).view(-1, 1)
    sim = sim.clamp(0.0, 1.0)

    return torch.where(valid, sim, torch.zeros_like(sim))


def directional_semantic_conflicts(
    api_counts: torch.Tensor,
    graph_counts: torch.Tensor,
    manifest_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    api = api_counts.float().clamp_min(0.0)
    graph = graph_counts.float().clamp_min(0.0)
    manifest = manifest_counts.float().clamp_min(0.0)

    code = torch.maximum(api, graph)

    valid = (
        (code.sum(dim=-1, keepdim=True) > 0)
        & (manifest.sum(dim=-1, keepdim=True) > 0)
    )

    code = code / code.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    manifest = manifest / manifest.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    manifest_to_code = (manifest - code).clamp_min(0.0).sum(dim=-1, keepdim=True)
    code_to_manifest = (code - manifest).clamp_min(0.0).sum(dim=-1, keepdim=True)

    zero = torch.zeros_like(manifest_to_code)

    return (
        torch.where(valid, manifest_to_code, zero),
        torch.where(valid, code_to_manifest, zero),
    )


def build_evidence(
    graph_data,
    api_logits: torch.Tensor,
    graph_logits: torch.Tensor,
    manifest_logits: torch.Tensor,
    joint_logits: torch.Tensor,
    api_emb: torch.Tensor,
    graph_emb: torch.Tensor,
    manifest_emb: torch.Tensor,
    *,
    use_consistency_evidence: bool,
    use_conflict_evidence: bool,
    use_perturbation_evidence: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = api_logits.size(0)
    device = api_logits.device
    dtype = api_logits.dtype

    q_api = scalar_attr(graph_data, "q_api", batch_size, device, dtype, 1.0)
    q_graph = scalar_attr(graph_data, "q_graph", batch_size, device, dtype, 1.0)
    q_manifest = scalar_attr(graph_data, "q_manifest", batch_size, device, dtype, 0.0)
    q_align = scalar_attr(graph_data, "q_align", batch_size, device, dtype, 0.0)

    pert_api = scalar_attr(graph_data, "pert_api", batch_size, device, dtype, 0.0)
    pert_graph = scalar_attr(graph_data, "pert_graph", batch_size, device, dtype, 0.0)
    pert_manifest = scalar_attr(graph_data, "pert_manifest", batch_size, device, dtype, 1.0)

    # Synthetic perturbation strength is oracle metadata and is unavailable
    # for naturally corrupted APKs. Main method should normally use q only.
    if use_perturbation_evidence:
        r_api = (q_api * (1.0 - pert_api)).clamp(0.0, 1.0)
        r_graph = (q_graph * (1.0 - pert_graph)).clamp(0.0, 1.0)
        r_manifest = (q_manifest * (1.0 - pert_manifest)).clamp(0.0, 1.0)
    else:
        r_api = q_api
        r_graph = q_graph
        r_manifest = q_manifest

    api_conf = confidence(api_logits).to(dtype=dtype)
    graph_conf = confidence(graph_logits).to(dtype=dtype)
    manifest_conf = confidence(manifest_logits).to(dtype=dtype)
    joint_conf = confidence(joint_logits).to(dtype=dtype)

    api_counts = semantic_counts_attr(
        graph_data,
        "api_semantic_category_counts",
        batch_size,
        device,
        dtype,
    )
    if float(api_counts.detach().abs().sum().item()) <= 0.0:
        api_counts = semantic_counts_attr(
            graph_data,
            "api_category_counts",
            batch_size,
            device,
            dtype,
        )

    graph_counts = semantic_counts_attr(
        graph_data,
        "graph_semantic_category_counts",
        batch_size,
        device,
        dtype,
    )
    if float(graph_counts.detach().abs().sum().item()) <= 0.0:
        graph_counts = semantic_counts_attr(
            graph_data,
            "graph_category_counts",
            batch_size,
            device,
            dtype,
        )

    manifest_counts = semantic_counts_attr(
        graph_data,
        "manifest_category_counts",
        batch_size,
        device,
        dtype,
    )

    api_graph_consistency = cosine_counts(api_counts, graph_counts)

    graph_missing_counts = graph_counts.abs().sum(dim=-1, keepdim=True) <= 0
    api_graph_consistency = torch.where(
        graph_missing_counts,
        q_align,
        api_graph_consistency,
    ).clamp(0.0, 1.0)

    api_manifest_consistency = cosine_counts(api_counts, manifest_counts)
    graph_manifest_consistency = cosine_counts(graph_counts, manifest_counts)

    manifest_to_code_conflict, code_to_manifest_conflict = directional_semantic_conflicts(
        api_counts,
        graph_counts,
        manifest_counts,
    )

    evidence_api_graph_consistency = api_graph_consistency
    evidence_api_manifest_consistency = api_manifest_consistency
    evidence_graph_manifest_consistency = graph_manifest_consistency

    evidence_manifest_to_code_conflict = manifest_to_code_conflict
    evidence_code_to_manifest_conflict = code_to_manifest_conflict

    if not use_consistency_evidence:
        evidence_api_graph_consistency = torch.zeros_like(api_graph_consistency)
        evidence_api_manifest_consistency = torch.zeros_like(api_manifest_consistency)
        evidence_graph_manifest_consistency = torch.zeros_like(graph_manifest_consistency)

    if not use_conflict_evidence:
        evidence_manifest_to_code_conflict = torch.zeros_like(manifest_to_code_conflict)
        evidence_code_to_manifest_conflict = torch.zeros_like(code_to_manifest_conflict)

    api_graph_disagreement = prob_disagreement(api_logits, graph_logits).to(dtype=dtype)

    api_alive = modality_alive(api_emb).to(dtype=dtype) * (q_api > 0.0).to(dtype=dtype)
    graph_alive = modality_alive(graph_emb).to(dtype=dtype) * (q_graph > 0.0).to(dtype=dtype)
    manifest_alive = (
        modality_alive(manifest_emb).to(dtype=dtype)
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

    if use_perturbation_evidence:
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
            float(use_perturbation_evidence),
            device=device,
            dtype=dtype,
        ),
    }

    return evidence, diagnostics