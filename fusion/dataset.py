from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from fusion.constants import AEG_SCHEMA_VERSION
from fusion.perturbations import apply_aeg_view
from fusion.semantic_categories import SEMANTIC_CATEGORY_DIM


class AEGDatasetConfigError(RuntimeError):
    pass


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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Data]:
        path, label = self.samples[idx]
        payload = torch.load(path, map_location="cpu")
        clean = payload_to_data(payload, label=label)
        out = {"clean": clean}
        if self.train_aug and float(torch.rand(()).item()) < self.aug_prob and self.aug_views:
            view = self.aug_views[int(torch.randint(len(self.aug_views), (1,)).item())]
            strength = self.aug_strengths[int(torch.randint(len(self.aug_strengths), (1,)).item())]
            out["aug"] = apply_aeg_view(clean, view=view, strength=strength)
        else:
            out["aug"] = apply_aeg_view(clean, view="clean", strength=0.0)
        return out


def aeg_collate_fn(items: list[dict[str, Data]]) -> dict[str, Any]:
    clean_items = [item["clean"] for item in items]
    aug_items = [item.get("aug", item["clean"]) for item in items]
    return {
        "clean": Batch.from_data_list(clean_items),
        "aug": Batch.from_data_list(aug_items),
        "sid": [getattr(item, "sid", "") for item in clean_items],
    }


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
