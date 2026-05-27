#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.robust.semantic_categories import (
    SEMANTIC_CATEGORY_DIM,
    api_semantic_counts_from_type_ids,
    graph_semantic_counts_from_method_api_edges,
    sanitize_semantic_counts,
)


def _normalize_loaded_pt(raw) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(raw, list):
        dex_list = [d for d in raw if isinstance(d, dict)]
        return dex_list, dex_list
    if isinstance(raw, dict):
        if isinstance(raw.get("dex_list"), list):
            dex_list = [d for d in raw["dex_list"] if isinstance(d, dict)]
            return dex_list, [raw] + dex_list
        if isinstance(raw.get("dexes"), list):
            dex_list = [d for d in raw["dexes"] if isinstance(d, dict)]
            return dex_list, [raw] + dex_list
        return [raw], [raw]
    return [], []


def _first_present(sources: list[dict[str, Any]], key: str):
    for src in sources:
        if key in src and src[key] is not None:
            return src[key]
    return None


def _long_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().long().view(-1)
    if value is None:
        return torch.empty((0,), dtype=torch.long)
    return torch.as_tensor(value, dtype=torch.long).view(-1)


def _edge_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor) and value.ndim == 2 and value.size(0) == 2:
        return value.detach().long()
    return torch.empty((2, 0), dtype=torch.long)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if a.abs().sum().item() <= 0.0 or b.abs().sum().item() <= 0.0:
        return None
    return float(F.cosine_similarity(a.float().view(1, -1), b.float().view(1, -1), dim=-1).item())


def _sample_counts(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    dex_list, sources = _normalize_loaded_pt(raw)

    api_counts = torch.zeros((SEMANTIC_CATEGORY_DIM,), dtype=torch.float32)
    graph_counts = torch.zeros((SEMANTIC_CATEGORY_DIM,), dtype=torch.float32)
    for dex in dex_list:
        explicit_api = sanitize_semantic_counts(dex.get("api_semantic_category_counts"), require_exact=True)
        if explicit_api.abs().sum().item() > 0.0:
            api_counts += explicit_api
        else:
            api_counts += api_semantic_counts_from_type_ids(_long_tensor(dex.get("api_type_ids")))

        explicit_graph = sanitize_semantic_counts(dex.get("graph_semantic_category_counts"), require_exact=True)
        if explicit_graph.abs().sum().item() > 0.0:
            graph_counts += explicit_graph
        else:
            graph_counts += graph_semantic_counts_from_method_api_edges(
                _long_tensor(dex.get("api_type_ids")),
                _edge_tensor(dex.get("method_api_edge_index")),
            )

    manifest_counts = sanitize_semantic_counts(_first_present(sources, "manifest_category_counts"))
    return api_counts, graph_counts, manifest_counts


def summarize_root(root: Path, split: str) -> dict[str, Any]:
    split_dir = root / split
    files = sorted(split_dir.rglob("*.pt")) if split_dir.exists() else []
    api_nonzero = 0
    graph_nonzero = 0
    manifest_nonzero = 0
    api_manifest = []
    graph_manifest = []
    graph_api = []
    failed = []

    for path in files:
        try:
            api_counts, graph_counts, manifest_counts = _sample_counts(path)
        except Exception as exc:
            failed.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"})
            continue

        api_alive = api_counts.abs().sum().item() > 0.0
        graph_alive = graph_counts.abs().sum().item() > 0.0
        manifest_alive = manifest_counts.abs().sum().item() > 0.0
        api_nonzero += int(api_alive)
        graph_nonzero += int(graph_alive)
        manifest_nonzero += int(manifest_alive)
        for bucket, value in (
            (api_manifest, _cosine(api_counts, manifest_counts)),
            (graph_manifest, _cosine(graph_counts, manifest_counts)),
            (graph_api, _cosine(graph_counts, api_counts)),
        ):
            if value is not None:
                bucket.append(value)

    ok = max(len(files) - len(failed), 0)

    def ratio(n: int) -> float:
        return float(n / ok) if ok > 0 else 0.0

    def mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "split": split,
        "num_files": len(files),
        "num_ok": ok,
        "num_failed": len(failed),
        "api_semantic_nonzero_ratio": ratio(api_nonzero),
        "graph_semantic_nonzero_ratio": ratio(graph_nonzero),
        "manifest_semantic_nonzero_ratio": ratio(manifest_nonzero),
        "api_manifest_consistency_mean": mean(api_manifest),
        "graph_manifest_consistency_mean": mean(graph_manifest),
        "api_graph_consistency_mean": mean(graph_api),
        "failed": failed[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize semantic category coverage in tri-modal .pt files.")
    parser.add_argument("--pt-root", required=True, help="Root containing train/val/test .pt split directories.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--out-json", default="", help="Optional JSON summary path.")
    args = parser.parse_args()

    root = Path(args.pt_root)
    summary = {split: summarize_root(root, split) for split in args.splits}
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

