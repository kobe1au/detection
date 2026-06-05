from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from fusion.constants import (
    AEG_SCHEMA_VERSION,
    AEG_SCHEMA_TABLE_FINGERPRINT,
    AEG_SCHEMA_TABLES,
    EDGE_TYPES,
    NODE_TYPES,
    SOURCE_TYPES,
    VIEW_TYPES,
)
from fusion.perturbations import apply_aeg_view, clear_aggregate_apk_semantic, refresh_risk_node_quality
from fusion.semantic_categories import SEMANTIC_CATEGORY_DIM


class AEGDatasetConfigError(RuntimeError):
    pass


SHUFFLED_MANIFEST_STALE_EDGE_TYPES = {
    EDGE_TYPES["APK_REQUESTS_PERMISSION"],
    EDGE_TYPES["PERMISSION_REQUESTED_BY_APK"],
    EDGE_TYPES["APK_HAS_COMPONENT"],
    EDGE_TYPES["COMPONENT_IN_APK"],
    EDGE_TYPES["COMPONENT_DECLARES_INTENT"],
    EDGE_TYPES["INTENT_DECLARED_BY_COMPONENT"],
    EDGE_TYPES["PERMISSION_RELATED_TO_API_FAMILY"],
    EDGE_TYPES["API_FAMILY_RELATED_TO_PERMISSION"],
    EDGE_TYPES["COMPONENT_MATCHES_METHOD"],
    EDGE_TYPES["METHOD_MATCHES_COMPONENT"],
    EDGE_TYPES["MANIFEST_HAS_RISK"],
    EDGE_TYPES["RISK_DECLARED_BY_MANIFEST"],
}


def _norm_id(value: Any) -> str:
    return str(value or "").strip().lower()


def _read_labels(path: str | Path | None) -> dict[str, int]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        raise AEGDatasetConfigError(f"Label CSV not found: {path}")

    labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise AEGDatasetConfigError(f"Empty label CSV: {path}")
        field_map = {name.lower(): name for name in reader.fieldnames}
        id_field = None
        for candidate in ("id", "sha256", "sid", "sample_id", "apk_sha256"):
            if candidate in field_map:
                id_field = field_map[candidate]
                break
        label_field = None
        for candidate in ("label", "y", "class"):
            if candidate in field_map:
                label_field = field_map[candidate]
                break
        if id_field is None or label_field is None:
            raise AEGDatasetConfigError(
                f"{path} must contain an id/sha256/sid column and a label column"
            )

        for row in reader:
            sid = _norm_id(row.get(id_field))
            if not sid:
                continue
            raw_label = str(row.get(label_field, "")).strip()
            if raw_label.lower() in {"", "nan", "none", "null"}:
                continue
            try:
                label = int(float(raw_label))
            except ValueError as exc:
                raise AEGDatasetConfigError(f"Invalid label {raw_label!r} for {sid}") from exc
            if label not in {0, 1}:
                raise AEGDatasetConfigError(f"Only binary labels are supported; got {label} for {sid}")
            if sid in labels:
                raise AEGDatasetConfigError(f"Duplicate label id in {path}: {sid}")
            labels[sid] = label
    return labels


def _tensor(payload: dict[str, Any], key: str, dtype: torch.dtype, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    value = payload.get(key)
    if isinstance(value, torch.Tensor):
        out = value.detach().cpu().to(dtype=dtype)
    elif value is None:
        out = torch.empty(shape or (0,), dtype=dtype)
    else:
        out = torch.as_tensor(value, dtype=dtype)
    if shape is not None and out.numel() == 0:
        out = torch.empty(shape, dtype=dtype)
    return torch.nan_to_num(out.float(), nan=0.0, posinf=0.0, neginf=0.0).to(dtype=dtype) if dtype.is_floating_point else out


def _scalar_tensor(payload: dict[str, Any], key: str, default: float = 0.0) -> torch.Tensor:
    value = payload.get(key)
    if isinstance(value, torch.Tensor) and value.numel() > 0:
        return value.detach().cpu().float().view(-1)[:1]
    if isinstance(value, (float, int)):
        return torch.tensor([float(value)], dtype=torch.float32)
    return torch.tensor([float(default)], dtype=torch.float32)


def _fit_1d(value: torch.Tensor, length: int, *, dtype: torch.dtype, fill: float = 0.0) -> torch.Tensor:
    out = value.to(dtype=dtype).view(-1)
    if out.numel() < length:
        pad = torch.full((length - out.numel(),), fill, dtype=dtype)
        out = torch.cat([out, pad], dim=0)
    return out[:length]


def _fit_2d(value: torch.Tensor, rows: int, cols: int, *, dtype: torch.dtype) -> torch.Tensor:
    out = value.to(dtype=dtype)
    if out.ndim != 2:
        out = torch.empty((0, cols), dtype=dtype)
    if out.size(1) < cols:
        out = torch.cat([out, torch.zeros((out.size(0), cols - out.size(1)), dtype=dtype)], dim=1)
    elif out.size(1) > cols:
        out = out[:, :cols]
    if out.size(0) < rows:
        out = torch.cat([out, torch.zeros((rows - out.size(0), cols), dtype=dtype)], dim=0)
    return out[:rows]


def payload_to_data(payload: dict[str, Any], *, label: int | None = None) -> Data:
    schema = int(payload.get("schema_version", 0) or 0)
    if schema != AEG_SCHEMA_VERSION:
        raise AEGDatasetConfigError(
            f"Expected AEG schema_version={AEG_SCHEMA_VERSION}, got {schema}. "
            "Regenerate PT files with scripts/build_aeg_pts_direct.py."
        )
    if payload.get("aeg_schema_fingerprint") != AEG_SCHEMA_TABLE_FINGERPRINT:
        meta = payload.get("aeg_meta") or {}
        if meta.get("schema_fingerprint") != AEG_SCHEMA_TABLE_FINGERPRINT:
            raise AEGDatasetConfigError(
                f"AEG schema table fingerprint mismatch for {payload.get('sid', '<unknown>')}. "
                "Regenerate PT files because node/edge/source/view type tables changed."
            )
    meta = payload.get("aeg_meta") or {}
    for key, expected in AEG_SCHEMA_TABLES.items():
        if key in meta and dict(meta[key]) != expected:
            raise AEGDatasetConfigError(
                f"AEG {key} table mismatch for {payload.get('sid', '<unknown>')}. "
                "Regenerate PT files with the current code."
            )

    node_x = _tensor(payload, "node_x", torch.float32)
    if node_x.ndim != 2 or node_x.size(0) == 0:
        raise AEGDatasetConfigError(f"AEG sample {payload.get('sid', '<unknown>')} has no nodes")
    num_nodes = int(node_x.size(0))
    edge_index = _tensor(payload, "edge_index", torch.long, (2, 0))
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise AEGDatasetConfigError(f"AEG sample {payload.get('sid', '<unknown>')} has invalid edge_index")
    num_edges = int(edge_index.size(1))

    edge_type = _fit_1d(_tensor(payload, "edge_type", torch.long, (0,)), num_edges, dtype=torch.long, fill=0)
    edge_quality = _fit_1d(_tensor(payload, "edge_quality", torch.float32, (0,)), num_edges, dtype=torch.float32, fill=1.0)
    edge_source = _fit_1d(_tensor(payload, "edge_source", torch.long, (0,)), num_edges, dtype=torch.long, fill=0)
    node_type = _fit_1d(_tensor(payload, "node_type", torch.long, (num_nodes,)), num_nodes, dtype=torch.long, fill=0)
    node_source = _fit_1d(_tensor(payload, "node_source", torch.long, (num_nodes,)), num_nodes, dtype=torch.long, fill=0)
    node_quality = _fit_1d(_tensor(payload, "node_quality", torch.float32, (num_nodes,)), num_nodes, dtype=torch.float32, fill=0.0)
    node_semantic = _fit_2d(_tensor(payload, "node_semantic", torch.float32), num_nodes, SEMANTIC_CATEGORY_DIM, dtype=torch.float32)

    y_value = label if label is not None else int(payload.get("label", 0) or 0)
    data = Data(
        x=node_x.float(),
        edge_index=edge_index.long(),
        edge_type=edge_type,
        edge_quality=edge_quality,
        edge_source=edge_source,
        node_type=node_type,
        node_source=node_source,
        node_quality=node_quality,
        node_semantic=node_semantic,
        q_api=_scalar_tensor(payload, "q_api"),
        q_graph=_scalar_tensor(payload, "q_graph"),
        q_manifest=_scalar_tensor(payload, "q_manifest"),
        q_align=_scalar_tensor(payload, "q_align"),
        y=torch.tensor([int(y_value)], dtype=torch.long),
    )
    data.sid = str(payload.get("sid") or payload.get("sha256") or "").lower()
    data.apk_name = str(payload.get("apk_name") or "")
    data.package_name = str(payload.get("package_name") or "")
    data.split = str(payload.get("split") or "")
    data.view_type_id = torch.tensor([0], dtype=torch.long)
    data.cf_weight = torch.tensor([0.0], dtype=torch.float32)
    data.manifest_parse_ok = torch.tensor([1.0 if payload.get("manifest_parse_ok", True) else 0.0], dtype=torch.float32)
    data.dex_success_ratio = torch.tensor([float(payload.get("dex_success_ratio", 1.0) or 0.0)], dtype=torch.float32)
    return data


class AEGDataset(Dataset):
    def __init__(
        self,
        pt_dir: str | Path,
        label_csv: str | Path | None = None,
        *,
        split: str = "",
        train_aug: bool = False,
        aug_views: list[str] | tuple[str, ...] | None = None,
        aug_strengths: list[float] | tuple[float, ...] | None = None,
        aug_prob: float = 0.5,
        seed: int = 42,
        strict_integrity: bool = True,
    ) -> None:
        self.pt_dir = Path(pt_dir)
        if not self.pt_dir.exists():
            raise AEGDatasetConfigError(f"PT directory not found: {self.pt_dir}")
        self.split = split
        self.labels = _read_labels(label_csv)
        self.train_aug = bool(train_aug)
        self.aug_views = list(aug_views or ("api_degraded", "graph_degraded", "api_graph_degraded", "manifest_degraded", "all_degraded"))
        self.aug_strengths = [float(v) for v in (aug_strengths or (0.1, 0.3, 0.5))]
        self.aug_prob = float(aug_prob)
        _ = seed

        pt_files = sorted(self.pt_dir.glob("*.pt"))
        pt_ids = {path.stem.lower() for path in pt_files}
        if self.labels and strict_integrity:
            csv_ids = set(self.labels)
            csv_only = sorted(csv_ids - pt_ids)
            pt_only = sorted(pt_ids - csv_ids)
            if csv_only or pt_only:
                raise AEGDatasetConfigError(
                    f"CSV/PT mismatch for {self.pt_dir}: "
                    f"csv_only={len(csv_only)} examples={csv_only[:5]}, "
                    f"pt_only={len(pt_only)} examples={pt_only[:5]}"
                )
        samples: list[tuple[Path, int | None]] = []
        for path in pt_files:
            sid = path.stem.lower()
            label = self.labels.get(sid)
            if self.labels and label is None:
                continue
            samples.append((path, label))
        if not samples:
            raise AEGDatasetConfigError(f"No AEG PT samples found in {self.pt_dir}")
        self.samples = samples
        self.manifest_donor_indices = _build_manifest_donor_indices(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Data]:
        path, label = self.samples[idx]
        payload = torch.load(path, map_location="cpu")
        clean = payload_to_data(payload, label=label)
        out = {"clean": clean}
        selected_view = "clean"
        if self.train_aug and float(torch.rand(()).item()) < self.aug_prob and self.aug_views:
            selected_view = self.aug_views[int(torch.randint(len(self.aug_views), (1,)).item())]
            strength = self.aug_strengths[int(torch.randint(len(self.aug_strengths), (1,)).item())]
            out["aug"] = apply_aeg_view(clean, view=selected_view, strength=strength)
        else:
            out["aug"] = apply_aeg_view(clean, view="clean", strength=0.0)
        if selected_view == "manifest_shuffled":
            donor_idx = self.manifest_donor_indices[idx] if idx < len(self.manifest_donor_indices) else None
            if donor_idx is not None:
                donor_path, donor_label = self.samples[donor_idx]
                donor_payload = torch.load(donor_path, map_location="cpu")
                out["manifest_donor"] = payload_to_data(donor_payload, label=donor_label)
        return out


def aeg_collate_fn(items: list[dict[str, Data]]) -> dict[str, Any]:
    clean_items = [item["clean"] for item in items]
    aug_items = [item.get("aug", item["clean"]) for item in items]
    donor_items = [item.get("manifest_donor") for item in items]
    _apply_manifest_shuffle(clean_items, aug_items, donor_items)
    return {
        "clean": Batch.from_data_list(clean_items),
        "aug": Batch.from_data_list(aug_items),
        "sid": [getattr(item, "sid", "") for item in clean_items],
        "manifest_donor_sid": [getattr(item.get("manifest_donor"), "sid", "") if item.get("manifest_donor") is not None else "" for item in items],
    }


def _copy_manifest_content(target: Data, donor: Data) -> None:
    target_mask = target.node_source == SOURCE_TYPES["manifest"]
    donor_mask = donor.node_source == SOURCE_TYPES["manifest"]
    if not bool(target_mask.any()):
        return
    if not bool(donor_mask.any()):
        _zero_manifest_nodes(target)
        return
    for node_type in (NODE_TYPES["PERMISSION"], NODE_TYPES["INTENT"], NODE_TYPES["COMPONENT"]):
        target_idx = torch.where(target_mask & (target.node_type == node_type))[0]
        if target_idx.numel() == 0:
            continue
        donor_idx = torch.where(donor_mask & (donor.node_type == node_type))[0]
        if donor_idx.numel() == 0:
            target.x[target_idx] = 0.0
            target.node_quality[target_idx] = 0.0
            target.node_semantic[target_idx] = 0.0
            continue
        repeat = donor_idx.repeat((int(target_idx.numel()) + int(donor_idx.numel()) - 1) // int(donor_idx.numel()))[: target_idx.numel()]
        target.x[target_idx] = donor.x[repeat]
        target.node_quality[target_idx] = donor.node_quality[repeat]
        target.node_semantic[target_idx] = donor.node_semantic[repeat]
    target.q_manifest = donor.q_manifest.clone()
    _zero_shuffled_manifest_edges(target)
    clear_aggregate_apk_semantic(target)
    refresh_risk_node_quality(target)


def _zero_manifest_nodes(data: Data) -> None:
    mask = data.node_source == SOURCE_TYPES["manifest"]
    if bool(mask.any()):
        data.x[mask] = 0.0
        data.node_quality[mask] = 0.0
        data.node_semantic[mask] = 0.0
    data.q_manifest = torch.tensor([0.0], dtype=torch.float32)
    _zero_shuffled_manifest_edges(data)
    clear_aggregate_apk_semantic(data)
    refresh_risk_node_quality(data)


def _zero_shuffled_manifest_edges(data: Data) -> None:
    if not hasattr(data, "edge_quality") or not hasattr(data, "edge_type") or data.edge_type.numel() == 0:
        return
    edge_type_tensor = torch.tensor(sorted(SHUFFLED_MANIFEST_STALE_EDGE_TYPES), dtype=torch.long, device=data.edge_type.device)
    mask = torch.isin(data.edge_type, edge_type_tensor)
    if bool(mask.any()):
        data.edge_quality[mask] = 0.0


def _apply_manifest_shuffle(clean_items: list[Data], aug_items: list[Data], donor_items: list[Data | None] | None = None) -> None:
    for idx, aug in enumerate(aug_items):
        if int(aug.view_type_id.view(-1)[0].item()) != VIEW_TYPES["manifest_shuffled"]:
            continue
        donor = donor_items[idx] if donor_items is not None and idx < len(donor_items) else None
        if donor is None:
            # A one-sample split cannot supply a donor; use zeroed Manifest
            # evidence instead of silently keeping the original.
            _zero_manifest_nodes(aug)
        else:
            _copy_manifest_content(aug, donor)


def _build_manifest_donor_indices(samples: list[tuple[Path, int | None]]) -> list[int | None]:
    n = len(samples)
    if n <= 1:
        return [None] * n
    donors: list[int | None] = []
    labels = [label for _path, label in samples]
    for idx, label in enumerate(labels):
        donor_idx: int | None = None
        if label is not None:
            for offset in range(1, n):
                cand = (idx + offset) % n
                cand_label = labels[cand]
                if cand_label is not None and cand_label != label:
                    donor_idx = cand
                    break
        if donor_idx is None:
            donor_idx = (idx + 1) % n
        donors.append(donor_idx)
    return donors


def split_label_stats(dataset: AEGDataset) -> dict[str, float]:
    labels = []
    for _, label in dataset.samples:
        if label is not None:
            labels.append(int(label))
    if not labels:
        return {"num_samples": float(len(dataset)), "positive_ratio": math.nan}
    positives = sum(labels)
    return {
        "num_samples": float(len(labels)),
        "positive_ratio": float(positives / max(1, len(labels))),
    }
