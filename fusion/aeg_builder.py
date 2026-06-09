from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

from fusion.constants import (
    AEG_PAYLOAD_CONTRACT_FINGERPRINT,
    AEG_PAYLOAD_CONTRACT_VERSION,
    AEG_SCHEMA_VERSION,
    AEG_SCHEMA_TABLE_FINGERPRINT,
    AEG_SCHEMA_TABLES,
    EDGE_TYPES,
    NODE_TYPES,
    REVERSE_EDGE_TYPES,
    SOURCE_TYPES,
    STRING_HINT_KEYWORDS,
)
from fusion.manifest_features import category_counts_from_strings
from fusion.quality import (
    compute_align_quality,
    compute_api_quality,
    compute_graph_quality,
    compute_manifest_quality,
)
from fusion.semantic_categories import (
    CATEGORY_TO_INDEX,
    DEFAULT_API_TYPE_ID_TO_CATEGORY,
    SEMANTIC_CATEGORIES,
    SEMANTIC_CATEGORY_DIM,
    api_semantic_counts_from_type_ids,
    sanitize_semantic_counts,
)


def _as_tensor(value, dtype=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        out = value.detach().cpu()
        return out.to(dtype=dtype) if dtype is not None else out
    if value is None:
        return torch.empty((0,), dtype=dtype or torch.float32)
    return torch.as_tensor(value, dtype=dtype).detach().cpu()


def _cat_1d(items: list[torch.Tensor], dtype=torch.long) -> torch.Tensor:
    valid = [x.view(-1).to(dtype=dtype) for x in items if isinstance(x, torch.Tensor) and x.numel() > 0]
    return torch.cat(valid, dim=0) if valid else torch.empty((0,), dtype=dtype)


def _cat_2d(items: list[torch.Tensor], columns: int, dtype=torch.float32) -> torch.Tensor:
    valid = [x.to(dtype=dtype) for x in items if isinstance(x, torch.Tensor) and x.ndim == 2 and x.numel() > 0]
    return torch.cat(valid, dim=0) if valid else torch.empty((0, columns), dtype=dtype)


def _flatten_dex_list(dex_list: list[dict[str, Any]]) -> dict[str, Any]:
    call_x_parts: list[torch.Tensor] = []
    sensitive_parts: list[torch.Tensor] = []
    edge_parts: list[torch.Tensor] = []
    method_name_parts: list[str] = []

    api_ids_parts: list[torch.Tensor] = []
    api_type_parts: list[torch.Tensor] = []
    api_sensitive_parts: list[torch.Tensor] = []
    api_method_parts: list[torch.Tensor] = []
    api_in_graph_parts: list[torch.Tensor] = []
    method_api_parts: list[torch.Tensor] = []
    api_token_parts: list[str] = []

    method_offset = 0
    api_offset = 0
    call_feature_dim = 0
    for item in dex_list:
        call_x = _as_tensor(item.get("call_x"), dtype=torch.float32)
        if call_x.ndim != 2:
            call_x = torch.empty((0, 0), dtype=torch.float32)
        call_feature_dim = max(call_feature_dim, int(call_x.size(1)) if call_x.ndim == 2 else 0)
        n_method = int(call_x.size(0))
        call_x_parts.append(call_x)

        sensitive = _as_tensor(item.get("call_sensitive_mask"), dtype=torch.float32).view(-1)
        if sensitive.numel() < n_method:
            sensitive = torch.cat([sensitive, torch.zeros((n_method - sensitive.numel(),), dtype=torch.float32)])
        sensitive_parts.append(sensitive[:n_method])

        edge = _as_tensor(item.get("call_edge_index"), dtype=torch.long)
        if edge.ndim == 2 and edge.size(0) == 2 and edge.numel() > 0:
            edge_parts.append(edge + method_offset)

        api_ids = _as_tensor(item.get("api_ids"), dtype=torch.long).view(-1)
        api_types = _as_tensor(item.get("api_type_ids"), dtype=torch.long).view(-1)
        api_sensitive = _as_tensor(item.get("api_sensitive_mask"), dtype=torch.float32).view(-1)
        api_method = _as_tensor(item.get("api_method_index"), dtype=torch.long).view(-1)
        api_in_graph = _as_tensor(item.get("api_in_graph_mask"), dtype=torch.float32).view(-1)
        n_api = int(api_ids.numel())
        api_ids_parts.append(api_ids)
        api_type_parts.append(api_types[:n_api] if api_types.numel() >= n_api else torch.cat([api_types, torch.zeros(n_api - api_types.numel(), dtype=torch.long)]))
        api_sensitive_parts.append(api_sensitive[:n_api] if api_sensitive.numel() >= n_api else torch.cat([api_sensitive, torch.zeros(n_api - api_sensitive.numel())]))
        if api_method.numel() < n_api:
            api_method = torch.cat([api_method, torch.full((n_api - api_method.numel(),), -1, dtype=torch.long)])
        valid_method = api_method[:n_api].clone()
        valid_method[valid_method >= 0] += method_offset
        api_method_parts.append(valid_method)
        api_in_graph_parts.append(api_in_graph[:n_api] if api_in_graph.numel() >= n_api else torch.cat([api_in_graph, torch.zeros(n_api - api_in_graph.numel())]))

        method_api = _as_tensor(item.get("method_api_edge_index"), dtype=torch.long)
        if method_api.ndim == 2 and method_api.size(0) == 2 and method_api.numel() > 0:
            aligned = method_api.clone()
            aligned[0] += method_offset
            aligned[1] += api_offset
            method_api_parts.append(aligned)

        if isinstance(item.get("call_method_names"), list):
            method_name_parts.extend([str(v) for v in item["call_method_names"][:n_method]])
        else:
            method_name_parts.extend([""] * n_method)
        if isinstance(item.get("api_tokens"), list):
            api_token_parts.extend([str(v) for v in item["api_tokens"][:n_api]])

        method_offset += n_method
        api_offset += n_api

    padded_call_parts = []
    for x in call_x_parts:
        if x.ndim != 2:
            continue
        if x.size(1) < call_feature_dim:
            x = torch.cat([x, torch.zeros((x.size(0), call_feature_dim - x.size(1)))], dim=1)
        padded_call_parts.append(x[:, :call_feature_dim])

    return {
        "call_x": _cat_2d(padded_call_parts, call_feature_dim, dtype=torch.float32),
        "call_edge_index": torch.cat(edge_parts, dim=1).long() if edge_parts else torch.empty((2, 0), dtype=torch.long),
        "call_sensitive_mask": _cat_1d(sensitive_parts, dtype=torch.float32),
        "api_ids": _cat_1d(api_ids_parts, dtype=torch.long),
        "api_type_ids": _cat_1d(api_type_parts, dtype=torch.long),
        "api_sensitive_mask": _cat_1d(api_sensitive_parts, dtype=torch.float32),
        "api_method_index": _cat_1d(api_method_parts, dtype=torch.long),
        "api_in_graph_mask": _cat_1d(api_in_graph_parts, dtype=torch.float32),
        "method_api_edge_index": torch.cat(method_api_parts, dim=1).long() if method_api_parts else torch.empty((2, 0), dtype=torch.long),
        "method_names": method_name_parts,
        "api_tokens": api_token_parts,
    }


def _node_feature(base_dim: int, values: list[float] | torch.Tensor | None = None) -> torch.Tensor:
    out = torch.zeros((base_dim,), dtype=torch.float32)
    if values is None:
        return out
    value = values.float().view(-1) if isinstance(values, torch.Tensor) else torch.tensor(values, dtype=torch.float32).view(-1)
    n = min(base_dim, int(value.numel()))
    if n:
        out[:n] = torch.nan_to_num(value[:n], nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _compress_method_feature(feature: torch.Tensor, out_dim: int) -> torch.Tensor:
    """Compress a full method-CFG vector without arbitrary prefix truncation.

    The extractor emits two opcode histograms followed by three structural
    statistics. When compression is required, pool the histogram portion and
    preserve the structural statistics explicitly.
    """

    feature = torch.nan_to_num(feature.float().view(-1), nan=0.0, posinf=0.0, neginf=0.0)
    out_dim = int(out_dim)
    if feature.numel() <= out_dim:
        return _node_feature(out_dim, feature)
    if out_dim <= 3 or feature.numel() <= 3:
        return _node_feature(out_dim, feature)
    structural_stats = feature[-3:]
    histogram = feature[:-3].view(1, 1, -1)
    pooled = F.adaptive_avg_pool1d(histogram, out_dim - 3).view(-1)
    return torch.cat([pooled, structural_stats], dim=0)


def _normalize_name(value: str) -> str:
    text = str(value or "").lower()
    text = text.replace("/", ".").replace("$", ".")
    text = re.sub(r"[^a-z0-9_.]+", ".", text)
    return text.strip(".")


def _api_type_semantic(type_id: int) -> torch.Tensor:
    out = torch.zeros((SEMANTIC_CATEGORY_DIM,), dtype=torch.float32)
    category = DEFAULT_API_TYPE_ID_TO_CATEGORY.get(int(type_id))
    if category in CATEGORY_TO_INDEX:
        out[CATEGORY_TO_INDEX[category]] = 1.0
    return out


def _component_records(record: dict[str, Any]) -> list[tuple[str, int, list[str]]]:
    typed = {"activity": 0, "service": 1, "receiver": 2, "provider": 3}
    components = record.get("components")
    if isinstance(components, list) and components:
        out: list[tuple[str, int, list[str]]] = []
        for item in components:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            if not name:
                continue
            comp_type = typed.get(str(item.get("type") or "").lower(), 0)
            intents = [str(v).lower() for v in [*(item.get("intent_actions") or []), *(item.get("intent_categories") or [])] if v]
            out.append((name, comp_type, intents))
        return out

    out = []
    for type_id, key in enumerate(("activities", "services", "receivers", "providers")):
        for name in record.get(key) or []:
            out.append((str(name), type_id, []))
    return out


def _string_hints(method_names: list[str], api_tokens: list[str]) -> dict[str, bool]:
    text = " ".join([*method_names, *api_tokens]).lower()
    return {name: any(token in text for token in tokens) for name, tokens in STRING_HINT_KEYWORDS.items()}


class _GraphBuilder:
    def __init__(self, node_feature_dim: int):
        self.node_feature_dim = int(node_feature_dim)
        self.node_x: list[torch.Tensor] = []
        self.node_type: list[int] = []
        self.node_source: list[int] = []
        self.node_quality: list[float] = []
        self.node_semantic: list[torch.Tensor] = []
        self.edges: list[tuple[int, int, int, float, int]] = []

    def add_node(self, node_type: str, source: str, quality: float, semantic=None, feature=None) -> int:
        idx = len(self.node_type)
        self.node_type.append(NODE_TYPES[node_type])
        self.node_source.append(SOURCE_TYPES[source])
        self.node_quality.append(float(max(0.0, min(1.0, quality))))
        self.node_semantic.append(sanitize_semantic_counts(semantic))
        self.node_x.append(_node_feature(self.node_feature_dim, feature))
        return idx

    def add_edge(self, src: int, dst: int, edge_type: str, quality: float, source: str, *, add_reverse: bool = True) -> None:
        if src < 0 or dst < 0:
            return
        edge = (int(src), int(dst), EDGE_TYPES[edge_type], float(max(0.0, min(1.0, quality))), SOURCE_TYPES[source])
        self.edges.append(edge)
        reverse_type = REVERSE_EDGE_TYPES.get(edge_type)
        if add_reverse and reverse_type and src != dst:
            self.edges.append((edge[1], edge[0], EDGE_TYPES[reverse_type], edge[3], edge[4]))

    def tensors(self) -> dict[str, torch.Tensor]:
        if self.edges:
            edge_index = torch.tensor([[s, d] for s, d, _t, _q, _src in self.edges], dtype=torch.long).t().contiguous()
            edge_type = torch.tensor([t for _s, _d, t, _q, _src in self.edges], dtype=torch.long)
            edge_quality = torch.tensor([q for _s, _d, _t, q, _src in self.edges], dtype=torch.float32)
            edge_source = torch.tensor([src for _s, _d, _t, _q, src in self.edges], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)
            edge_quality = torch.empty((0,), dtype=torch.float32)
            edge_source = torch.empty((0,), dtype=torch.long)
        return {
            "node_x": torch.stack(self.node_x, dim=0).float() if self.node_x else torch.empty((0, self.node_feature_dim)),
            "node_type": torch.tensor(self.node_type, dtype=torch.long),
            "node_source": torch.tensor(self.node_source, dtype=torch.long),
            "node_quality": torch.tensor(self.node_quality, dtype=torch.float32),
            "node_semantic": torch.stack(self.node_semantic, dim=0).float() if self.node_semantic else torch.empty((0, SEMANTIC_CATEGORY_DIM)),
            "edge_index": edge_index,
            "edge_type": edge_type,
            "edge_quality": edge_quality,
            "edge_source": edge_source,
        }


def build_aeg_payload(
    *,
    sid: str,
    apk_name: str,
    split: str,
    dex_list: list[dict[str, Any]],
    manifest_payload: dict[str, Any],
    manifest_record: dict[str, Any],
    direct_meta: dict[str, Any],
    node_feature_dim: int = 128,
    retain_intermediate_features: bool = False,
    storage_dtype: str = "float32",
) -> dict[str, Any]:
    flat = _flatten_dex_list(dex_list)
    call_x = flat["call_x"]
    call_edge_index = flat["call_edge_index"]
    api_ids = flat["api_ids"]
    api_type_ids = flat["api_type_ids"]
    api_method_index = flat["api_method_index"]
    api_in_graph_mask = flat["api_in_graph_mask"]
    method_api_edge_index = flat["method_api_edge_index"]
    num_methods = int(call_x.size(0))
    num_api = int(api_ids.numel())

    q_api = compute_api_quality(api_ids, api_type_ids, api_in_graph_mask)
    q_graph = compute_graph_quality(call_edge_index, num_methods, call_x)
    q_align = compute_align_quality(q_api, q_graph, method_api_edge_index, num_methods, num_api)
    q_manifest = compute_manifest_quality(
        manifest_payload.get("manifest_x"),
        manifest_payload.get("manifest_category_counts"),
        manifest_payload.get("manifest_stats"),
        manifest_payload.get("manifest_meta"),
    )
    dex_success_ratio = float(direct_meta.get("dex_success_ratio", 1.0))
    q_api *= max(0.0, min(1.0, dex_success_ratio))
    q_graph *= max(0.0, min(1.0, dex_success_ratio))
    q_align = compute_align_quality(q_api, q_graph, method_api_edge_index, num_methods, num_api)

    builder = _GraphBuilder(node_feature_dim)
    apk_sem = (
        sanitize_semantic_counts(api_semantic_counts_from_type_ids(api_type_ids))
        + sanitize_semantic_counts(manifest_payload.get("manifest_category_counts"))
    )
    # Quality is represented by node_quality and graph-level q_* fields. Keep
    # node_x content-only so quality ablations cannot recover the same signal
    # through a duplicated feature channel.
    apk_node = builder.add_node("APK", "derived", max(q_api, q_graph, q_manifest), apk_sem)

    method_nodes = []
    method_sem = torch.zeros((num_methods, SEMANTIC_CATEGORY_DIM), dtype=torch.float32)
    if method_api_edge_index.numel() > 0 and api_type_ids.numel() > 0:
        for method_idx, api_idx in method_api_edge_index.long().t().tolist():
            if 0 <= method_idx < num_methods and 0 <= api_idx < int(api_type_ids.numel()):
                method_sem[method_idx] += _api_type_semantic(int(api_type_ids[api_idx]))

    for idx in range(num_methods):
        feature = call_x[idx] if idx < call_x.size(0) else None
        if feature is not None:
            feature = _compress_method_feature(feature, node_feature_dim)
        node = builder.add_node("METHOD", "code", q_graph, method_sem[idx], feature)
        method_nodes.append(node)
        builder.add_edge(apk_node, node, "APK_HAS_METHOD", q_graph, "code")

    for src, dst in call_edge_index.long().t().tolist():
        if 0 <= src < len(method_nodes) and 0 <= dst < len(method_nodes):
            builder.add_edge(method_nodes[src], method_nodes[dst], "METHOD_CALLS_METHOD", q_graph, "code")

    api_family_nodes: dict[int, int] = {}
    api_type_counts = defaultdict(int)
    for value in api_type_ids.long().view(-1).tolist():
        if value > 0:
            api_type_counts[int(value)] += 1
    for type_id, count in sorted(api_type_counts.items()):
        sem = _api_type_semantic(type_id)
        feature = [float(type_id) / 32.0, math.log1p(count) / 8.0]
        api_family_nodes[type_id] = builder.add_node("API_FAMILY", "code", q_api, sem, feature)

    if method_api_edge_index.numel() > 0:
        seen_pairs = set()
        for method_idx, api_idx in method_api_edge_index.long().t().tolist():
            if 0 <= method_idx < len(method_nodes) and 0 <= api_idx < int(api_type_ids.numel()):
                type_id = int(api_type_ids[api_idx])
                family_node = api_family_nodes.get(type_id)
                pair = (method_idx, family_node)
                if family_node is not None and pair not in seen_pairs:
                    seen_pairs.add(pair)
                    builder.add_edge(method_nodes[method_idx], family_node, "METHOD_INVOKES_API_FAMILY", q_api, "code")

    permission_nodes = []
    perm_ids = _as_tensor(manifest_payload.get("manifest_permission_ids"), dtype=torch.long).view(-1)
    perm_map = _as_tensor(manifest_payload.get("manifest_permission_category_map"), dtype=torch.float32)
    for perm_id in perm_ids.tolist():
        row = int(perm_id) - 1
        sem = perm_map[row] if perm_map.ndim == 2 and 0 <= row < perm_map.size(0) else torch.zeros(SEMANTIC_CATEGORY_DIM)
        node = builder.add_node("PERMISSION", "manifest", q_manifest, sem, [float(perm_id) / 512.0])
        permission_nodes.append((int(perm_id), node, sem))
        builder.add_edge(apk_node, node, "APK_REQUESTS_PERMISSION", q_manifest, "manifest")

    intent_nodes = []
    intent_token_to_node: dict[str, int] = {}
    intent_ids = _as_tensor(manifest_payload.get("manifest_intent_ids"), dtype=torch.long).view(-1)
    intent_tokens = [str(v).lower() for v in (manifest_payload.get("manifest_intent_tokens") or [])]
    intent_map = _as_tensor(manifest_payload.get("manifest_intent_category_map"), dtype=torch.float32)
    for idx, intent_id in enumerate(intent_ids.tolist()):
        row = int(intent_id) - 1
        sem = intent_map[row] if intent_map.ndim == 2 and 0 <= row < intent_map.size(0) else torch.zeros(SEMANTIC_CATEGORY_DIM)
        node = builder.add_node("INTENT", "manifest", q_manifest, sem, [float(intent_id) / 512.0])
        intent_nodes.append(node)
        if idx < len(intent_tokens):
            intent_token_to_node[intent_tokens[idx]] = node

    component_nodes = []
    method_name_norm = [_normalize_name(v) for v in flat["method_names"]]
    for comp_name, comp_type, comp_intents in _component_records(manifest_record):
        sem = category_counts_from_strings([comp_name], list(SEMANTIC_CATEGORIES))
        feature = [float(comp_type) / 4.0]
        comp_node = builder.add_node("COMPONENT", "manifest", q_manifest, sem, feature)
        component_nodes.append(comp_node)
        builder.add_edge(apk_node, comp_node, "APK_HAS_COMPONENT", q_manifest, "manifest")
        for token in comp_intents:
            intent_node = intent_token_to_node.get(str(token).lower())
            if intent_node is not None:
                builder.add_edge(comp_node, intent_node, "COMPONENT_DECLARES_INTENT", q_manifest, "manifest")
        comp_norm = _normalize_name(comp_name)
        if comp_norm:
            for method_idx, method_norm in enumerate(method_name_norm):
                if comp_norm and comp_norm in method_norm:
                    builder.add_edge(comp_node, method_nodes[method_idx], "COMPONENT_MATCHES_METHOD", q_align, "alignment")

    manifest_counts = sanitize_semantic_counts(manifest_payload.get("manifest_category_counts"))
    code_risk_active = method_sem.sum(dim=0) > 0
    manifest_risk_active = manifest_counts > 0
    risk_nodes = []
    for idx, category in enumerate(SEMANTIC_CATEGORIES):
        sem = torch.zeros((SEMANTIC_CATEGORY_DIM,), dtype=torch.float32)
        sem[idx] = 1.0
        risk_quality = max(
            q_api if bool(code_risk_active[idx]) else 0.0,
            q_manifest if bool(manifest_risk_active[idx]) else 0.0,
        )
        risk_node = builder.add_node("RISK_SEMANTIC", "derived", risk_quality, sem, [idx / max(1, SEMANTIC_CATEGORY_DIM - 1)])
        risk_nodes.append(risk_node)

    for method_idx, sem in enumerate(method_sem):
        for cat_idx in torch.where(sem > 0)[0].tolist():
            builder.add_edge(method_nodes[method_idx], risk_nodes[cat_idx], "METHOD_HAS_RISK", q_api, "derived")
    for cat_idx in torch.where(manifest_counts > 0)[0].tolist():
        builder.add_edge(apk_node, risk_nodes[cat_idx], "MANIFEST_HAS_RISK", q_manifest, "derived")

    for perm_id, perm_node, perm_sem in permission_nodes:
        for type_id, family_node in api_family_nodes.items():
            if float((perm_sem * _api_type_semantic(type_id)).sum().item()) > 0:
                builder.add_edge(perm_node, family_node, "PERMISSION_RELATED_TO_API_FAMILY", min(q_manifest, q_api), "derived")

    hints = _string_hints(flat["method_names"], flat["api_tokens"])
    string_hint_nodes = []
    for hint_idx, (hint_name, present) in enumerate(sorted(hints.items())):
        if not present:
            continue
        sem = category_counts_from_strings([hint_name], list(SEMANTIC_CATEGORIES))
        node = builder.add_node("STRING_HINT", "derived", q_api, sem, [hint_idx / max(1, len(hints) - 1), 1.0])
        string_hint_nodes.append(node)
        for method_idx, method_name in enumerate(flat["method_names"]):
            if any(token in method_name.lower() for token in STRING_HINT_KEYWORDS.get(hint_name, ())):
                builder.add_edge(method_nodes[method_idx], node, "METHOD_HAS_STRING_HINT", q_api, "derived")

    graph = builder.tensors()

    payload = {
        "schema_version": AEG_SCHEMA_VERSION,
        "aeg_schema_fingerprint": AEG_SCHEMA_TABLE_FINGERPRINT,
        "aeg_payload_contract_version": AEG_PAYLOAD_CONTRACT_VERSION,
        "aeg_payload_contract_fingerprint": AEG_PAYLOAD_CONTRACT_FINGERPRINT,
        "sid": sid,
        "sha256": sid,
        "apk_name": apk_name,
        "split": split,
        "package_name": str(manifest_record.get("package_name", "")),
        "year": int(manifest_record.get("year", 0) or 0),
        "sample_meta": dict(manifest_record.get("sample_meta") or {}),
        "storage_dtype": str(storage_dtype),
        "api_semantic_category_counts": api_semantic_counts_from_type_ids(api_type_ids).float(),
        "graph_semantic_category_counts": method_sem.sum(dim=0).float(),
        "manifest_category_counts": _as_tensor(
            manifest_payload.get("manifest_category_counts"), dtype=torch.float32
        ).view(-1),
        "manifest_component_category_counts": _as_tensor(
            manifest_payload.get("manifest_component_category_counts"), dtype=torch.float32
        ).view(-1),
        "manifest_stats": _as_tensor(manifest_payload.get("manifest_stats"), dtype=torch.float32).view(-1),
        "q_api": torch.tensor([q_api], dtype=torch.float32),
        "q_graph": torch.tensor([q_graph], dtype=torch.float32),
        "q_manifest": torch.tensor([q_manifest], dtype=torch.float32),
        "q_align": torch.tensor([q_align], dtype=torch.float32),
        "pert_api": torch.tensor([0.0], dtype=torch.float32),
        "pert_graph": torch.tensor([0.0], dtype=torch.float32),
        "pert_manifest": torch.tensor([0.0], dtype=torch.float32),
        **graph,
        "manifest_parse_ok": not bool((manifest_payload.get("manifest_meta") or {}).get("parse_error")),
        "manifest_parse_error": str((manifest_payload.get("manifest_meta") or {}).get("parse_error") or ""),
        "dex_success_ratio": float(dex_success_ratio),
        "multi_dex_total": int(direct_meta.get("num_dex_total", len(dex_list))),
        "multi_dex_success": int(direct_meta.get("num_dex_success", len(dex_list))),
        "method_budget_per_dex": int(direct_meta.get("method_budget_per_dex", 0) or 0),
        "api_event_budget_per_dex": int(direct_meta.get("api_event_budget_per_dex", 0) or 0),
        "graph_behavior_hints": bool(direct_meta.get("use_graph_behavior_hints", False)),
        "graph_behavior_hint_start": int(direct_meta.get("graph_behavior_hint_start", 0) or 0),
        "graph_behavior_hint_dim": int(direct_meta.get("graph_behavior_hint_dim", 0) or 0),
        "has_reflection": bool(hints.get("reflection", False)),
        "has_dynamic_loading": bool(hints.get("dynamic_loading", False)),
        "has_native": bool(hints.get("native", False)),
        "has_string_encryption_hint": bool(hints.get("crypto", False)),
        "aeg_meta": {
            **AEG_SCHEMA_TABLES,
            "schema_fingerprint": AEG_SCHEMA_TABLE_FINGERPRINT,
            "num_nodes": int(graph["node_x"].size(0)),
            "num_edges": int(graph["edge_index"].size(1)),
            "node_feature_dim": int(node_feature_dim),
        },
    }
    if retain_intermediate_features:
        payload.update(
            {
                "apk_node": int(apk_node),
                "method_nodes": torch.tensor(method_nodes, dtype=torch.long),
                "api_family_nodes": torch.tensor(list(api_family_nodes.values()), dtype=torch.long),
                "permission_nodes": torch.tensor([node for _pid, node, _sem in permission_nodes], dtype=torch.long),
                "intent_nodes": torch.tensor(intent_nodes, dtype=torch.long),
                "component_nodes": torch.tensor(component_nodes, dtype=torch.long),
                "risk_nodes": torch.tensor(risk_nodes, dtype=torch.long),
                "string_hint_nodes": torch.tensor(string_hint_nodes, dtype=torch.long),
                "method_api_edges": method_api_edge_index.long(),
                "method_call_edges": call_edge_index.long(),
            }
        )
        payload.update(flat)
        payload.update(manifest_payload)
    if storage_dtype == "float16":
        for key in ("node_x", "node_quality", "node_semantic", "edge_quality"):
            payload[key] = payload[key].half()
        payload["edge_index"] = payload["edge_index"].int()
        for key in ("node_type", "node_source", "edge_type", "edge_source"):
            payload[key] = payload[key].to(dtype=torch.uint8)
    elif storage_dtype != "float32":
        raise ValueError(f"Unsupported AEG storage dtype: {storage_dtype}")
    return payload
