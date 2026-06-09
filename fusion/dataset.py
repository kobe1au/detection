from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from fusion.constants import (
    EDGE_TYPES,
    NODE_TYPES,
    SOURCE_TYPES,
    VIEW_TYPES,
)
from fusion.io_utils import load_aeg_payload
from fusion.payload_contract import AEGPayloadContractError, validate_aeg_payload
from fusion.perturbations import (
    apply_aeg_view,
    clear_aggregate_apk_semantic,
    refresh_apk_node_quality,
    refresh_risk_node_quality,
)
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
        for candidate in ("sha256", "apk_sha256", "id", "sid", "sample_id"):
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


def payload_to_data(
    payload: dict[str, Any],
    *,
    label: int | None = None,
    validate_payload: bool = True,
) -> Data:
    if validate_payload:
        try:
            validate_aeg_payload(payload)
        except AEGPayloadContractError as exc:
            raise AEGDatasetConfigError(
                f"Invalid AEG PT payload for {payload.get('sid', '<unknown>')}: {exc}"
            ) from exc

    node_x = _tensor(payload, "node_x", torch.float32)
    edge_index = _tensor(payload, "edge_index", torch.long, (2, 0))
    edge_type = _tensor(payload, "edge_type", torch.long).view(-1)
    edge_quality = _tensor(payload, "edge_quality", torch.float32).view(-1)
    edge_source = _tensor(payload, "edge_source", torch.long).view(-1)
    node_type = _tensor(payload, "node_type", torch.long).view(-1)
    node_source = _tensor(payload, "node_source", torch.long).view(-1)
    node_quality = _tensor(payload, "node_quality", torch.float32).view(-1)
    node_semantic = _tensor(payload, "node_semantic", torch.float32)

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
        pert_api=_scalar_tensor(payload, "pert_api"),
        pert_graph=_scalar_tensor(payload, "pert_graph"),
        pert_manifest=_scalar_tensor(payload, "pert_manifest"),
        y=torch.tensor([int(y_value)], dtype=torch.long),
    )
    data.sid = str(payload.get("sid") or payload.get("sha256") or "").lower()
    data.apk_name = str(payload.get("apk_name") or "")
    data.package_name = str(payload.get("package_name") or "")
    data.split = str(payload.get("split") or "")
    data.view_type_id = torch.tensor([0], dtype=torch.long)
    data.requested_view_type_id = torch.tensor([0], dtype=torch.long)
    data.effective_view_type_id = torch.tensor([0], dtype=torch.long)
    data.manifest_shuffle_fallback = torch.tensor([0], dtype=torch.long)
    data.cf_weight = torch.tensor([0.0], dtype=torch.float32)
    data.manifest_parse_ok = torch.tensor([1.0 if payload.get("manifest_parse_ok", True) else 0.0], dtype=torch.float32)
    data.dex_success_ratio = torch.tensor([float(payload.get("dex_success_ratio", 1.0) or 0.0)], dtype=torch.float32)
    data.year = torch.tensor([int(payload.get("year", 0) or 0)], dtype=torch.long)
    data.multi_dex_total = torch.tensor([int(payload.get("multi_dex_total", 0) or 0)], dtype=torch.long)
    data.multi_dex_success = torch.tensor([int(payload.get("multi_dex_success", 0) or 0)], dtype=torch.long)
    data.has_reflection = torch.tensor([1.0 if payload.get("has_reflection", False) else 0.0], dtype=torch.float32)
    data.has_dynamic_loading = torch.tensor([1.0 if payload.get("has_dynamic_loading", False) else 0.0], dtype=torch.float32)
    data.has_native = torch.tensor([1.0 if payload.get("has_native", False) else 0.0], dtype=torch.float32)
    data.has_string_encryption_hint = torch.tensor(
        [1.0 if payload.get("has_string_encryption_hint", False) else 0.0],
        dtype=torch.float32,
    )
    data.graph_behavior_hints = torch.tensor([1.0 if payload.get("graph_behavior_hints", False) else 0.0], dtype=torch.float32)
    data.graph_behavior_hint_start = torch.tensor([int(payload.get("graph_behavior_hint_start", 0) or 0)], dtype=torch.long)
    data.graph_behavior_hint_dim = torch.tensor([int(payload.get("graph_behavior_hint_dim", 0) or 0)], dtype=torch.long)
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
        validate_payload_on_load: bool = True,
        manifest_donor_mode: str = "cyclic",
        deterministic_aug: bool = False,
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
        self.validate_payload_on_load = bool(validate_payload_on_load)
        self.manifest_donor_mode = str(manifest_donor_mode or "cyclic").lower()
        self.seed = int(seed)
        self.deterministic_aug = bool(deterministic_aug)

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
        self.manifest_donor_indices = _build_manifest_donor_indices(
            samples,
            mode=self.manifest_donor_mode,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Data]:
        path, label = self.samples[idx]
        # Use centralized safe loading; validation handled by load_aeg_payload
        payload = load_aeg_payload(path, validate=self.validate_payload_on_load)
        # Skip redundant validation in payload_to_data since load_aeg_payload already validated
        clean = payload_to_data(payload, label=label, validate_payload=False)
        out = {"clean": clean}
        selected_view = "clean"
        if self.train_aug and float(torch.rand(()).item()) < self.aug_prob and self.aug_views:
            selected_view = self.aug_views[int(torch.randint(len(self.aug_views), (1,)).item())]
            strength = self.aug_strengths[int(torch.randint(len(self.aug_strengths), (1,)).item())]
            if self.deterministic_aug:
                digest = hashlib.blake2b(
                    f"{self.seed}:{clean.sid}:{selected_view}:{strength:.8f}".encode("utf-8"),
                    digest_size=8,
                ).digest()
                deterministic_seed = int.from_bytes(digest, byteorder="little", signed=False)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(deterministic_seed)
                    out["aug"] = apply_aeg_view(clean, view=selected_view, strength=strength)
            else:
                out["aug"] = apply_aeg_view(clean, view=selected_view, strength=strength)
        else:
            out["aug"] = apply_aeg_view(clean, view="clean", strength=0.0)
        if selected_view in {"manifest_shuffled", "manifest_shuffled_blind"}:
            donor_idx = self.manifest_donor_indices[idx] if idx < len(self.manifest_donor_indices) else None
            if donor_idx is not None:
                donor_path, donor_label = self.samples[donor_idx]
                donor_payload = load_aeg_payload(donor_path, validate=self.validate_payload_on_load)
                out["manifest_donor"] = payload_to_data(
                    donor_payload,
                    label=donor_label,
                    validate_payload=False,
                )
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
        "requested_view_type_id": [int(getattr(item.get("aug", item["clean"]), "requested_view_type_id", getattr(item.get("aug", item["clean"]), "view_type_id")).view(-1)[0].item()) for item in items],
        "effective_view_type_id": [int(getattr(item.get("aug", item["clean"]), "effective_view_type_id", getattr(item.get("aug", item["clean"]), "view_type_id")).view(-1)[0].item()) for item in items],
        "manifest_shuffle_fallback": [int(getattr(item.get("aug", item["clean"]), "manifest_shuffle_fallback", torch.tensor([0])).view(-1)[0].item()) for item in items],
    }


def _copy_manifest_content(target: Data, donor: Data, *, blind: bool = False) -> None:
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
        target.node_semantic[target_idx] = donor.node_semantic[repeat]
        if not blind:
            target.node_quality[target_idx] = donor.node_quality[repeat]
    if not blind:
        target.q_manifest = donor.q_manifest.clone()
        target.pert_manifest = torch.tensor([1.0], dtype=torch.float32, device=target.x.device)
        _zero_shuffled_manifest_edges(target)
    clear_aggregate_apk_semantic(target)
    refresh_apk_node_quality(target)
    refresh_risk_node_quality(target)


def _zero_manifest_nodes(data: Data) -> None:
    mask = data.node_source == SOURCE_TYPES["manifest"]
    if bool(mask.any()):
        data.x[mask] = 0.0
        data.node_quality[mask] = 0.0
        data.node_semantic[mask] = 0.0
    data.pert_manifest = torch.tensor([1.0], dtype=torch.float32, device=data.x.device)
    _zero_shuffled_manifest_edges(data)
    clear_aggregate_apk_semantic(data)
    refresh_apk_node_quality(data)
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
        view_id = int(aug.view_type_id.view(-1)[0].item())
        if view_id not in {VIEW_TYPES["manifest_shuffled"], VIEW_TYPES["manifest_shuffled_blind"]}:
            continue
        donor = donor_items[idx] if donor_items is not None and idx < len(donor_items) else None
        if donor is None:
            # A one-sample split cannot supply a donor; use zeroed Manifest
            # evidence instead of silently keeping the original.
            _zero_manifest_nodes(aug)
            aug.effective_view_type_id = torch.tensor([VIEW_TYPES["manifest_missing"]], dtype=torch.long, device=aug.x.device)
            aug.manifest_shuffle_fallback = torch.tensor([1], dtype=torch.long, device=aug.x.device)
        else:
            _copy_manifest_content(aug, donor, blind=view_id == VIEW_TYPES["manifest_shuffled_blind"])
            aug.effective_view_type_id = aug.view_type_id.clone().to(device=aug.x.device)
            aug.manifest_shuffle_fallback = torch.tensor([0], dtype=torch.long, device=aug.x.device)


def _build_manifest_donor_indices(
    samples: list[tuple[Path, int | None]],
    *,
    mode: str = "cyclic",
) -> list[int | None]:
    n = len(samples)
    if n <= 1:
        return [None] * n
    mode = str(mode or "cyclic").lower()
    if mode not in {"cyclic", "opposite_label"}:
        raise AEGDatasetConfigError(
            "manifest_donor_mode must be 'cyclic' or 'opposite_label'; "
            f"got {mode!r}"
        )
    donors: list[int | None] = []
    labels = [label for _path, label in samples]
    for idx, label in enumerate(labels):
        donor_idx: int | None = None
        if mode == "opposite_label" and label is not None:
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
