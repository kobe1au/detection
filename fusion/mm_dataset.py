#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from fusion.constants import AugmentationConstants

logger = logging.getLogger(__name__)


class MultiModalMalwareDataset(Dataset):
    """
    APK-level Dataset for the API + call-graph representation produced by
    extract/extract_graph_api.py.

    Expected per-dex fields:
      - call_x: [N, D]
      - call_edge_index: [2, E]
      - call_sensitive_mask: [N]
      - api_ids: [T]
      - api_type_ids: [T]
      - api_sensitive_mask: [T]
      - api_method_index: [T]
      - api_in_graph_mask: [T]
      - method_api_edge_index: [2, K]
    """

    def __init__(
        self,
        pt_dir: str,
        csv_path: str,
        is_train: bool = True,
        robust_aug: bool = False,
        max_api_events_per_sample: int | None = None,
        eval_perturb_type: str | None = None,
        eval_perturb_strength: float = 0.0,
        fusion_mode: str = "ours",
        need_alignment_mask: bool | None = None,
        drop_graph_behavior_hints: bool = False,
        domain_years: list[int] | tuple[int, ...] | None = None,
        **_unused,
    ):
        super().__init__()
        self.pt_dir = Path(pt_dir)
        self.is_train = bool(is_train)
        self.robust_aug = bool(robust_aug)
        self.drop_graph_behavior_hints = bool(drop_graph_behavior_hints)
        self.max_api_events_per_sample = (
            int(max_api_events_per_sample)
            if max_api_events_per_sample is not None
            else None
        )
        self.eval_perturb_type = eval_perturb_type
        self.eval_perturb_strength = float(eval_perturb_strength)
        self.fusion_mode = str(fusion_mode or "ours")
        self.need_alignment_mask = (
            self.fusion_mode == "ours"
            if need_alignment_mask is None
            else bool(need_alignment_mask)
        )

        if self.fusion_mode not in {"api", "graph", "concat", "late_fusion", "cross_attention", "ours"}:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")
        if self.eval_perturb_type not in AugmentationConstants.EVAL_PERTURB_TYPES:
            raise ValueError(f"Unsupported eval_perturb_type: {self.eval_perturb_type}")

        self.aug_strength_min = 0.1
        self.aug_strength_max = 0.4

        df = pd.read_csv(csv_path)
        id_col = next((c for c in ["id", "ID", "Id", "sha256"] if c in df.columns), None)
        if id_col is None:
            raise ValueError("CSV 文件中必须包含 'id' 或 'sha256' 列")
        if "label" not in df.columns:
            raise ValueError("CSV 文件中必须包含 'label' 列")

        year_col = next((c for c in ["year", "Year", "年份", "vt_year", "dex_year"] if c in df.columns), None)
        if year_col is None:
            raise ValueError("Temporal setting requires a year column in CSV")
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce").fillna(0).astype(int)

        sid_series = df[id_col].astype(str).str.lower()
        self.labels = dict(zip(sid_series, df["label"].astype(int)))
        self.sid_to_year: Dict[str, int] = dict(zip(sid_series, df[year_col].astype(int)))
        self.unique_years = sorted(df[year_col].astype(int).unique().tolist())
        if domain_years is None:
            self.domain_years = list(self.unique_years)
        else:
            self.domain_years = sorted({int(y) for y in domain_years})
            missing_years = sorted(set(self.unique_years) - set(self.domain_years))
            if missing_years:
                raise ValueError(
                    "domain_years must include every year in the split CSV; "
                    f"missing={missing_years}"
                )
        self.year_to_domain_id = {year: idx for idx, year in enumerate(self.domain_years)}
        self.domain_id_to_year = {idx: year for year, idx in self.year_to_domain_id.items()}

        self.samples: List[Tuple[Path, int, str, int, int]] = []
        for pt_file in sorted(self.pt_dir.rglob("*.pt")):
            sid = pt_file.stem.lower()
            if sid in self.labels:
                year = int(self.sid_to_year.get(sid, 0))
                domain_id = int(self.year_to_domain_id.get(year, 0))
                self.samples.append((pt_file, self.labels[sid], sid, year, domain_id))

        self.sample_sids = [sid for _, _, sid, _, _ in self.samples]
        self.sample_years = [year for _, _, _, year, _ in self.samples]
        self.sid_to_domain_id = {sid: domain_id for _, _, sid, _, domain_id in self.samples}
        self.year_to_indices = defaultdict(list)
        for idx, (_, _, _, year, _) in enumerate(self.samples):
            self.year_to_indices[int(year)].append(idx)

        self.feature_dim = self._infer_feature_dim(default_dim=515)
        if not self.samples:
            raise RuntimeError(
                f"No matching samples found: pt_dir='{self.pt_dir}', "
                f"csv has {len(self.labels)} ids, but no .pt files matched."
            )

        logger.info(
            f"[{'Train' if is_train else 'Eval'} Dataset] loaded {len(self.samples)} "
            f"samples (fusion_mode={self.fusion_mode}, robust_aug={self.robust_aug})"
        )

    def __len__(self):
        return len(self.samples)

    def _infer_feature_dim(self, default_dim: int = 515) -> int:
        for pt_file, _, _, _, _ in self.samples:
            try:
                dex_list = torch.load(pt_file, map_location="cpu", weights_only=False)
                dex_list = dex_list if isinstance(dex_list, list) else [dex_list]
                for dex in dex_list:
                    x = dex.get("call_x", None) if isinstance(dex, dict) else None
                    if isinstance(x, torch.Tensor) and x.ndim == 2 and x.shape[1] > 0:
                        dim = int(x.shape[1])
                        if self.drop_graph_behavior_hints and dim == 519:
                            return 515
                        return dim
            except Exception as e:
                logger.warning(f"infer feature_dim failed for {pt_file}: {e}")
        return default_dim

    def _get_temporal_aug_strength(self, year: int):
        if len(self.unique_years) <= 1:
            return self.aug_strength_min, self.aug_strength_max
        min_year = min(self.unique_years)
        max_year = max(self.unique_years)
        ratio = (year - min_year) / max(max_year - min_year, 1)
        s_min = self.aug_strength_min + AugmentationConstants.TEMPORAL_AUG_DELTA_MIN * ratio
        s_max = self.aug_strength_max + AugmentationConstants.TEMPORAL_AUG_DELTA_MAX * ratio
        return min(s_min, 0.5), min(s_max, 0.7)

    @staticmethod
    def _sanitize_call_x(x, feature_dim: int, drop_graph_behavior_hints: bool = False):
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            return torch.zeros((0, feature_dim), dtype=torch.float32)
        x = x.float()
        if drop_graph_behavior_hints and x.shape[1] == 519:
            x = x[:, :515]
        if x.shape[1] > feature_dim:
            x = x[:, :feature_dim]
        elif x.shape[1] < feature_dim:
            pad = torch.zeros((x.shape[0], feature_dim - x.shape[1]), dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _sanitize_edge_index(edge_index, num_nodes: int):
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index = edge_index.long()
        if edge_index.numel() == 0 or num_nodes <= 0:
            return torch.empty((2, 0), dtype=torch.long)
        valid = (
            (edge_index[0] >= 0) & (edge_index[0] < num_nodes)
            & (edge_index[1] >= 0) & (edge_index[1] < num_nodes)
        )
        return edge_index[:, valid]

    @staticmethod
    def _sanitize_sensitive_mask(mask, num_nodes: int):
        if not isinstance(mask, torch.Tensor) or mask.ndim != 1:
            return torch.zeros((num_nodes,), dtype=torch.uint8)
        mask = mask.to(torch.uint8)
        if mask.shape[0] < num_nodes:
            pad = torch.zeros((num_nodes - mask.shape[0],), dtype=torch.uint8)
            mask = torch.cat([mask, pad], dim=0)
        elif mask.shape[0] > num_nodes:
            mask = mask[:num_nodes]
        return mask

    @staticmethod
    def _sanitize_api_1d(values, length: int | None = None, dtype=torch.long, fill_value=0):
        if not isinstance(values, torch.Tensor) or values.ndim != 1:
            values = torch.empty((0,), dtype=dtype)
        else:
            values = values.to(dtype=dtype).view(-1)
        if length is None:
            return values
        if values.numel() < length:
            pad = torch.full((length - values.numel(),), fill_value, dtype=dtype, device=values.device)
            values = torch.cat([values, pad], dim=0)
        elif values.numel() > length:
            values = values[:length]
        return values

    @staticmethod
    def _sanitize_method_api_edges(edge_index, num_nodes: int, num_api: int):
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.size(0) != 2:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index = edge_index.long()
        if edge_index.numel() == 0 or num_nodes <= 0 or num_api <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        valid = (
            (edge_index[0] >= 0) & (edge_index[0] < num_nodes)
            & (edge_index[1] >= 0) & (edge_index[1] < num_api)
        )
        return edge_index[:, valid]

    @staticmethod
    def _compute_api_category_counts(api_type_ids: torch.Tensor, vocab_size: int = 16):
        counts = torch.zeros((vocab_size,), dtype=torch.float32)
        if isinstance(api_type_ids, torch.Tensor) and api_type_ids.numel() > 0:
            ids = api_type_ids.long().clamp(0, vocab_size - 1).cpu()
            counts.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
        return counts

    @staticmethod
    def _build_method_api_mask(
        method_api_edge_index: torch.Tensor,
        num_nodes: int,
        num_api: int,
        api_strong_mask: torch.Tensor | None = None,
        api_weak_mask: torch.Tensor | None = None,
    ):
        mask = torch.zeros((num_nodes, max(num_api, 1)), dtype=torch.float32)
        if num_nodes <= 0 or num_api <= 0 or method_api_edge_index.numel() == 0:
            return mask, 0.0
        src, dst = method_api_edge_index[0], method_api_edge_index[1]
        valid = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_api)
        if not valid.any():
            return mask, 0.0

        edge_weight = torch.zeros((method_api_edge_index.size(1),), dtype=torch.float32, device=dst.device)
        safe_dst = dst.clamp(0, num_api - 1)
        if isinstance(api_weak_mask, torch.Tensor) and api_weak_mask.numel() == num_api:
            weak = api_weak_mask.to(device=dst.device, dtype=torch.bool)[safe_dst]
            edge_weight = torch.where(weak, torch.full_like(edge_weight, 0.5), edge_weight)
        if isinstance(api_strong_mask, torch.Tensor) and api_strong_mask.numel() == num_api:
            strong = api_strong_mask.to(device=dst.device, dtype=torch.bool)[safe_dst]
            edge_weight = torch.where(strong, torch.ones_like(edge_weight), edge_weight)

        weak_valid = valid & (edge_weight >= 0.5)
        if weak_valid.any():
            mask[src[weak_valid], dst[weak_valid]] = 0.5
        strong_valid = valid & (edge_weight >= 1.0)
        if strong_valid.any():
            mask[src[strong_valid], dst[strong_valid]] = 1.0
        q_align = float((mask.max(dim=1).values > 0.0).float().mean().item()) if num_nodes > 0 else 0.0
        return mask, q_align

    def _limit_api_events(
        self,
        api_ids: torch.Tensor,
        api_type_ids: torch.Tensor,
        api_sensitive_mask: torch.Tensor,
        api_method_index: torch.Tensor,
        api_in_graph_mask: torch.Tensor,
        method_api_edge_index: torch.Tensor,
    ):
        limit = self.max_api_events_per_sample
        if limit is None or limit <= 0 or api_ids.numel() <= limit:
            return api_ids, api_type_ids, api_sensitive_mask, api_method_index, api_in_graph_mask, method_api_edge_index

        num_api = int(api_ids.numel())
        if self.is_train:
            keep_idx = torch.randperm(num_api)[:limit].sort().values
        else:
            keep_idx = torch.linspace(0, num_api - 1, steps=limit).round().long().unique(sorted=True)
            if keep_idx.numel() > limit:
                keep_idx = keep_idx[:limit]

        old_to_new = torch.full((num_api,), -1, dtype=torch.long, device=api_ids.device)
        old_to_new[keep_idx] = torch.arange(keep_idx.numel(), dtype=torch.long, device=api_ids.device)

        if method_api_edge_index.numel() > 0:
            old_api = method_api_edge_index[1].long()
            valid = (old_api >= 0) & (old_api < num_api) & (old_to_new[old_api] >= 0)
            method_api_edge_index = method_api_edge_index[:, valid].clone()
            method_api_edge_index[1] = old_to_new[method_api_edge_index[1].long()]

        return (
            api_ids[keep_idx],
            api_type_ids[keep_idx],
            api_sensitive_mask[keep_idx],
            api_method_index[keep_idx],
            api_in_graph_mask[keep_idx],
            method_api_edge_index,
        )

    def _compute_graph_intrinsic_quality(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        sensitive_mask: torch.Tensor,
    ) -> float:
        if num_nodes <= 0:
            return 0.0
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.size(0) != 2:
            return 0.0
        if edge_index.numel() == 0:
            return 0.1

        involved_nodes = torch.cat([edge_index[0], edge_index[1]], dim=0).unique()
        connected_ratio = involved_nodes.numel() / max(num_nodes, 1)
        avg_degree = (2.0 * edge_index.size(1)) / max(num_nodes, 1)
        degree_score = min(avg_degree / 4.0, 1.0)

        if isinstance(sensitive_mask, torch.Tensor) and sensitive_mask.numel() > 0 and sensitive_mask.sum() > 0:
            sens_idx = torch.where(sensitive_mask.bool())[0]
            sens_connected_ratio = torch.isin(sens_idx, involved_nodes).float().mean().item()
        else:
            sens_connected_ratio = connected_ratio

        quality = 0.5 * connected_ratio + 0.3 * degree_score + 0.2 * sens_connected_ratio
        return float(max(0.0, min(1.0, quality)))

    def _compute_api_intrinsic_quality(
        self,
        api_ids: torch.Tensor,
        api_type_ids: torch.Tensor,
        api_sensitive_mask: torch.Tensor,
        api_in_graph_mask: torch.Tensor,
    ) -> float:
        if not isinstance(api_ids, torch.Tensor) or api_ids.numel() == 0:
            return 0.0
        n_api = int(api_ids.numel())
        count_score = min(math.log1p(n_api) / math.log1p(128), 1.0)
        if isinstance(api_type_ids, torch.Tensor) and api_type_ids.numel() > 0:
            nonzero_types = api_type_ids[api_type_ids > 0].unique().numel()
            diversity_score = min(float(nonzero_types) / 6.0, 1.0)
        else:
            diversity_score = 0.0
        if isinstance(api_sensitive_mask, torch.Tensor) and api_sensitive_mask.numel() == n_api:
            sensitive_score = min(float(api_sensitive_mask.float().mean().item()) * 4.0, 1.0)
        else:
            sensitive_score = 0.0
        if isinstance(api_in_graph_mask, torch.Tensor) and api_in_graph_mask.numel() == n_api:
            coverage_score = float(api_in_graph_mask.float().mean().item())
        else:
            coverage_score = 0.0
        quality = 0.35 * count_score + 0.25 * diversity_score + 0.20 * sensitive_score + 0.20 * coverage_score
        return float(max(0.0, min(1.0, quality)))

    def _compute_alignment_quality(
        self,
        q_api: float,
        q_graph: float,
        mask: torch.Tensor,
        sensitive_mask: torch.Tensor,
    ) -> float:
        if isinstance(mask, torch.Tensor) and mask.numel() > 0:
            weighted_mask = mask.float().clamp(0.0, 1.0)
            node_cover = float((weighted_mask.max(dim=1).values > 0.0).float().mean().item())
        else:
            node_cover = 0.0

        if (
            isinstance(mask, torch.Tensor)
            and mask.numel() > 0
            and isinstance(sensitive_mask, torch.Tensor)
            and sensitive_mask.numel() == mask.size(0)
            and sensitive_mask.bool().any()
        ):
            sens_weight = mask.float().clamp(0.0, 1.0)[sensitive_mask.bool()]
            sens_cover = float((sens_weight.max(dim=1).values > 0.0).float().mean().item())
        else:
            sens_cover = node_cover

        q_align = (
            AugmentationConstants.ALIGN_QAPI_WEIGHT * q_api
            + AugmentationConstants.ALIGN_QGRAPH_WEIGHT * q_graph
            + AugmentationConstants.ALIGN_NODE_COVER_WEIGHT * node_cover
            + AugmentationConstants.ALIGN_SENSITIVE_COVER_WEIGHT * sens_cover
        )
        return float(max(0.0, min(1.0, q_align)))

    def _get_dummy(self, label: int, sid: str, year: int = 0, time_id: int = 0):
        data = Data(
            x=torch.zeros((1, self.feature_dim), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            y=torch.tensor(label, dtype=torch.long),
        )
        data.attn_mask = torch.zeros((1, 1), dtype=torch.float32)
        data.sensitive_mask = torch.zeros((1,), dtype=torch.uint8)
        data.api_ids = torch.zeros((0,), dtype=torch.long)
        data.api_type_ids = torch.zeros((0,), dtype=torch.long)
        data.api_sensitive_mask = torch.zeros((0,), dtype=torch.float32)
        data.api_method_index = torch.zeros((0,), dtype=torch.long)
        data.api_in_graph_mask = torch.zeros((0,), dtype=torch.float32)
        data.method_api_edge_index = torch.zeros((2, 0), dtype=torch.long)
        data.api_category_counts = torch.zeros((16,), dtype=torch.float32)
        data.node_mask = torch.zeros((1,), dtype=torch.bool)
        data.api_event_mask = torch.zeros((0,), dtype=torch.bool)
        data.sid = sid
        data.year = torch.tensor(int(year), dtype=torch.long)
        data.time_id = torch.tensor(int(time_id), dtype=torch.long)
        data.is_dummy = True
        data.q_api = torch.tensor([0.0], dtype=torch.float32)
        data.q_graph = torch.tensor([0.0], dtype=torch.float32)
        data.q_align = torch.tensor([0.0], dtype=torch.float32)
        data.pert_api = torch.tensor([0.0], dtype=torch.float32)
        data.pert_graph = torch.tensor([0.0], dtype=torch.float32)
        data.align_penalty = torch.tensor([0.0], dtype=torch.float32)
        data.api_aug_strength = torch.tensor([0.0], dtype=torch.float32)
        data.graph_aug_strength = torch.tensor([0.0], dtype=torch.float32)
        data.overall_aug_strength = torch.tensor([0.0], dtype=torch.float32)
        data.api_aug_type = "none"
        data.graph_aug_type = "none"
        return data

    def _load_dex_list(self, pt_path: Path):
        dex_list = torch.load(pt_path, map_location="cpu", weights_only=False)
        return dex_list if isinstance(dex_list, list) else [dex_list]

    def _process_api_only_dex(self, dex: dict):
        api_ids = self._sanitize_api_1d(dex.get("api_ids", None), dtype=torch.long).clamp_min(0)
        num_api = int(api_ids.numel())
        api_type_ids = self._sanitize_api_1d(
            dex.get("api_type_ids", None), length=num_api, dtype=torch.long, fill_value=0
        ).clamp_min(0)
        api_sensitive_mask = self._sanitize_api_1d(
            dex.get("api_sensitive_mask", None), length=num_api, dtype=torch.float32, fill_value=0.0
        ).clamp(0.0, 1.0)
        api_in_graph_mask = self._sanitize_api_1d(
            dex.get("api_in_graph_mask", None), length=num_api, dtype=torch.float32, fill_value=0.0
        ).clamp(0.0, 1.0)
        api_method_index = torch.full((num_api,), -1, dtype=torch.long)
        method_api_edge_index = torch.empty((2, 0), dtype=torch.long)

        api_ids, api_type_ids, api_sensitive_mask, api_method_index, api_in_graph_mask, method_api_edge_index = (
            self._limit_api_events(
                api_ids,
                api_type_ids,
                api_sensitive_mask,
                api_method_index,
                api_in_graph_mask,
                method_api_edge_index,
            )
        )
        num_api = int(api_ids.numel())
        x = torch.zeros((1, self.feature_dim), dtype=torch.float32)
        return {
            "x": x,
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "mask": torch.empty((1, 0), dtype=torch.float32),
            "sensitive_mask": torch.zeros((1,), dtype=torch.uint8),
            "api_ids": api_ids,
            "api_type_ids": api_type_ids,
            "api_sensitive_mask": api_sensitive_mask,
            "api_method_index": api_method_index,
            "api_in_graph_mask": api_in_graph_mask,
            "method_api_edge_index": method_api_edge_index,
            "num_nodes": 1,
            "num_api": num_api,
        }

    def _process_single_dex(self, dex: dict, node_offset: int, api_offset: int):
        if self.fusion_mode == "api":
            return self._process_api_only_dex(dex)

        x = self._sanitize_call_x(
            dex.get("call_x", None),
            self.feature_dim,
            self.drop_graph_behavior_hints,
        )
        n = int(x.size(0))
        if n == 0:
            return None

        sensitive_mask = self._sanitize_sensitive_mask(dex.get("call_sensitive_mask", None), n)
        edge_index = self._sanitize_edge_index(dex.get("call_edge_index", None), n)
        if edge_index.numel() > 0:
            edge_index = edge_index + node_offset

        if self.fusion_mode == "graph":
            return {
                "x": x,
                "edge_index": edge_index,
                "mask": torch.zeros((n, 0), dtype=torch.float32, device=x.device),
                "sensitive_mask": sensitive_mask,
                "api_ids": torch.empty((0,), dtype=torch.long, device=x.device),
                "api_type_ids": torch.empty((0,), dtype=torch.long, device=x.device),
                "api_sensitive_mask": torch.empty((0,), dtype=torch.float32, device=x.device),
                "api_method_index": torch.empty((0,), dtype=torch.long, device=x.device),
                "api_in_graph_mask": torch.empty((0,), dtype=torch.float32, device=x.device),
                "method_api_edge_index": torch.empty((2, 0), dtype=torch.long, device=x.device),
                "num_nodes": n,
                "num_api": 0,
            }

        api_ids = self._sanitize_api_1d(dex.get("api_ids", None), dtype=torch.long).clamp_min(0)
        num_api = int(api_ids.numel())
        api_type_ids = self._sanitize_api_1d(
            dex.get("api_type_ids", None), length=num_api, dtype=torch.long, fill_value=0
        ).clamp_min(0)
        api_sensitive_mask = self._sanitize_api_1d(
            dex.get("api_sensitive_mask", None), length=num_api, dtype=torch.float32, fill_value=0.0
        ).clamp(0.0, 1.0)
        api_method_index = self._sanitize_api_1d(
            dex.get("api_method_index", None), length=num_api, dtype=torch.long, fill_value=-1
        )
        valid_method = (api_method_index >= 0) & (api_method_index < n)
        api_method_index = torch.where(
            valid_method,
            api_method_index + int(node_offset),
            torch.full_like(api_method_index, -1),
        )
        api_in_graph_mask = self._sanitize_api_1d(
            dex.get("api_in_graph_mask", None), length=num_api, dtype=torch.float32, fill_value=0.0
        ).clamp(0.0, 1.0)

        method_api_edge_index = self._sanitize_method_api_edges(dex.get("method_api_edge_index", None), n, num_api)

        api_ids, api_type_ids, api_sensitive_mask, api_method_index, api_in_graph_mask, method_api_edge_index = (
            self._limit_api_events(
                api_ids,
                api_type_ids,
                api_sensitive_mask,
                api_method_index,
                api_in_graph_mask,
                method_api_edge_index,
            )
        )
        num_api = int(api_ids.numel())

        if method_api_edge_index.numel() > 0:
            local_method_api = method_api_edge_index.clone()
            method_api_edge_index = method_api_edge_index.clone()
            method_api_edge_index[0] += int(node_offset)
            method_api_edge_index[1] += int(api_offset)
        else:
            local_method_api = torch.empty((2, 0), dtype=torch.long, device=x.device)

        if self.need_alignment_mask:
            # Sensitive-First Sparse Alignment:
            # only keep method-API edges linked to sensitive API events.
            if n > 0 and num_api > 0 and api_sensitive_mask.numel() == num_api:
                # 只有同时满足 sensitive flag 和非 other API type 的 API 才参与 alignment。
                api_strong = (api_sensitive_mask > 0.5) & (api_type_ids > 0)
                api_weak = ((api_in_graph_mask > 0.5) | (api_type_ids > 0)) & (~api_strong)
                mask, _ = self._build_method_api_mask(local_method_api, n, num_api, api_strong, api_weak)
            else:
                mask = torch.zeros((n, max(num_api, 1)), dtype=torch.float32, device=x.device)
        else:
            mask = torch.empty((n, 0), dtype=torch.float32, device=x.device)

        return {
            "x": x,
            "edge_index": edge_index,
            "mask": mask.to(device=x.device),
            "sensitive_mask": sensitive_mask,
            "api_ids": api_ids.to(device=x.device),
            "api_type_ids": api_type_ids.to(device=x.device),
            "api_sensitive_mask": api_sensitive_mask.to(device=x.device),
            "api_method_index": api_method_index.to(device=x.device),
            "api_in_graph_mask": api_in_graph_mask.to(device=x.device),
            "method_api_edge_index": method_api_edge_index.to(device=x.device),
            "num_nodes": n,
            "num_api": num_api,
        }

    def _aggregate_dex_data(self, dex_list):
        total_x, total_edge_index, total_masks, total_sensitive = [], [], [], []
        total_api_ids, total_api_type_ids, total_api_sensitive = [], [], []
        total_api_method_index, total_api_in_graph, total_method_api_edges = [], [], []
        node_offset, api_offset = 0, 0

        if self.is_train and self.robust_aug and len(dex_list) > 1 and random.random() < 0.15:
            n_keep = max(1, int(len(dex_list) * random.uniform(0.5, 0.9)))
            dex_list = random.sample(dex_list, n_keep)

        for dex in dex_list:
            if not isinstance(dex, dict):
                continue
            dex_data = self._process_single_dex(dex, node_offset, api_offset)
            if dex_data is None:
                continue

            total_x.append(dex_data["x"])
            total_edge_index.append(dex_data["edge_index"])
            total_masks.append(dex_data["mask"])
            total_sensitive.append(dex_data["sensitive_mask"])
            total_api_ids.append(dex_data["api_ids"])
            total_api_type_ids.append(dex_data["api_type_ids"])
            total_api_sensitive.append(dex_data["api_sensitive_mask"])
            total_api_method_index.append(dex_data["api_method_index"])
            total_api_in_graph.append(dex_data["api_in_graph_mask"])
            total_method_api_edges.append(dex_data["method_api_edge_index"])

            node_offset += int(dex_data["num_nodes"])
            api_offset += int(dex_data["num_api"])

        if node_offset == 0:
            return None

        return self._merge_dex_data(
            total_x,
            total_edge_index,
            total_masks,
            total_sensitive,
            total_api_ids,
            total_api_type_ids,
            total_api_sensitive,
            total_api_method_index,
            total_api_in_graph,
            total_method_api_edges,
            node_offset,
            api_offset,
        )

    def _merge_dex_data(
        self,
        total_x,
        total_edge_index,
        total_masks,
        total_sensitive,
        total_api_ids,
        total_api_type_ids,
        total_api_sensitive,
        total_api_method_index,
        total_api_in_graph,
        total_method_api_edges,
        node_offset: int,
        api_offset: int,
    ):
        final_x = torch.cat(total_x, dim=0)
        edge_device = final_x.device
        final_edge_index = (
            torch.cat(total_edge_index, dim=1)
            if total_edge_index and any(e.numel() > 0 for e in total_edge_index)
            else torch.empty((2, 0), dtype=torch.long, device=edge_device)
        )
        final_sensitive_mask = (
            torch.cat(total_sensitive, dim=0)
            if total_sensitive
            else torch.zeros((node_offset,), dtype=torch.uint8, device=edge_device)
        )
        final_api_ids = (
            torch.cat(total_api_ids, dim=0).long()
            if total_api_ids and any(a.numel() > 0 for a in total_api_ids)
            else torch.empty((0,), dtype=torch.long, device=edge_device)
        )
        final_api_type_ids = (
            torch.cat(total_api_type_ids, dim=0).long()
            if total_api_type_ids and any(a.numel() > 0 for a in total_api_type_ids)
            else torch.empty((0,), dtype=torch.long, device=edge_device)
        )
        final_api_sensitive_mask = (
            torch.cat(total_api_sensitive, dim=0).float()
            if total_api_sensitive and any(a.numel() > 0 for a in total_api_sensitive)
            else torch.empty((0,), dtype=torch.float32, device=edge_device)
        )
        final_api_method_index = (
            torch.cat(total_api_method_index, dim=0).long()
            if total_api_method_index and any(a.numel() > 0 for a in total_api_method_index)
            else torch.empty((0,), dtype=torch.long, device=edge_device)
        )
        final_api_in_graph_mask = (
            torch.cat(total_api_in_graph, dim=0).float()
            if total_api_in_graph and any(a.numel() > 0 for a in total_api_in_graph)
            else torch.empty((0,), dtype=torch.float32, device=edge_device)
        )
        final_method_api_edge_index = (
            torch.cat(total_method_api_edges, dim=1).long()
            if total_method_api_edges and any(e.numel() > 0 for e in total_method_api_edges)
            else torch.empty((2, 0), dtype=torch.long, device=edge_device)
        )

        if self.fusion_mode == "ours" and self.need_alignment_mask:
            mask_device = total_masks[0].device if total_masks else edge_device
            final_mask = torch.zeros((node_offset, max(api_offset, 1)), dtype=torch.float32, device=mask_device)
            cur_n, cur_t = 0, 0
            for m, api_part in zip(total_masks, total_api_ids):
                n = int(m.size(0))
                # _build_method_api_mask uses width 1 for zero-API dex files so
                # downstream tensors stay well-formed. That placeholder column
                # must not advance the real API offset during multi-dex merge.
                real_t = int(api_part.numel()) if isinstance(api_part, torch.Tensor) else 0
                if n > 0 and real_t > 0 and cur_n < node_offset and cur_t < final_mask.size(1):
                    copy_n = min(n, node_offset - cur_n, int(m.size(0)))
                    copy_t = min(real_t, final_mask.size(1) - cur_t, int(m.size(1)))
                    if copy_n > 0 and copy_t > 0:
                        final_mask[cur_n:cur_n + copy_n, cur_t:cur_t + copy_t] = m[:copy_n, :copy_t]
                cur_n += n
                cur_t += real_t
        else:
            final_mask = torch.empty((node_offset, 0), dtype=torch.float32, device=edge_device)

        q_api = (
            self._compute_api_intrinsic_quality(
                final_api_ids,
                final_api_type_ids,
                final_api_sensitive_mask,
                final_api_in_graph_mask,
            )
            if self.fusion_mode != "graph"
            else 1.0
        )
        q_graph = (
            self._compute_graph_intrinsic_quality(final_edge_index, node_offset, final_sensitive_mask)
            if self.fusion_mode != "api"
            else 1.0
        )

        if self.fusion_mode == "api":
            q_align = q_api
        elif self.fusion_mode == "graph":
            q_align = q_graph
        else:
            q_align = self._compute_alignment_quality(q_api, q_graph, final_mask, final_sensitive_mask)

        return {
            "x": final_x,
            "edge_index": final_edge_index,
            "mask": final_mask,
            "sensitive_mask": final_sensitive_mask,
            "api_ids": final_api_ids,
            "api_type_ids": final_api_type_ids,
            "api_sensitive_mask": final_api_sensitive_mask,
            "api_method_index": final_api_method_index,
            "api_in_graph_mask": final_api_in_graph_mask,
            "method_api_edge_index": final_method_api_edge_index,
            "api_category_counts": self._compute_api_category_counts(final_api_type_ids),
            "q_api": q_api,
            "q_graph": q_graph,
            "q_align": q_align,
            "align_penalty": 0.0,
            "pert_api": 0.0,
            "pert_graph": 0.0,
            "api_aug_type": "none",
            "graph_aug_type": "none",
        }

    @staticmethod
    def _adjust_q_graph_semantic(
        q_graph_topo: float,
        edge_index: torch.Tensor,
        sensitive_mask: torch.Tensor,
        alpha: float | None = None,
    ) -> float:
        alpha = alpha if alpha is not None else AugmentationConstants.SEMANTIC_QUALITY_ALPHA
        if sensitive_mask is None or sensitive_mask.sum() == 0:
            return q_graph_topo
        sens_idx = torch.where(sensitive_mask.bool())[0]
        if edge_index.ndim != 2 or edge_index.size(1) == 0:
            q_sens_conn = 0.0
        else:
            all_nodes_in_edges = torch.cat([edge_index[0], edge_index[1]]).unique()
            connected_sens = torch.isin(sens_idx, all_nodes_in_edges).sum().item()
            q_sens_conn = connected_sens / max(sens_idx.numel(), 1)
        return (1.0 - alpha) * q_graph_topo + alpha * q_sens_conn

    def _apply_api_event_dropout(self, data: dict, strength: float, sensitive_only: bool = False):
        api_ids = data.get("api_ids")
        if not isinstance(api_ids, torch.Tensor) or api_ids.numel() == 0:
            return data

        strength = max(0.0, min(1.0, float(strength)))
        n_api = int(api_ids.numel())
        device = api_ids.device

        if sensitive_only and isinstance(data.get("api_sensitive_mask"), torch.Tensor):
            candidate = torch.where(data["api_sensitive_mask"].to(device).float() > 0.5)[0]
            if candidate.numel() == 0:
                candidate = torch.arange(n_api, device=device)
        else:
            candidate = torch.arange(n_api, device=device)

        n_drop = max(1, int(candidate.numel() * strength))
        n_drop = min(n_drop, candidate.numel())
        drop_idx = candidate[torch.randperm(candidate.numel(), device=device)[:n_drop]]

        keep_token = torch.ones((n_api,), dtype=torch.bool, device=device)
        keep_token[drop_idx] = False

        data["api_ids"] = api_ids.clone()
        data["api_ids"][drop_idx] = 0

        for key, fill in [
            ("api_type_ids", 0),
            ("api_sensitive_mask", 0),
            ("api_in_graph_mask", 0),
        ]:
            if isinstance(data.get(key), torch.Tensor) and data[key].numel() == n_api:
                data[key] = data[key].clone()
                data[key][drop_idx] = fill

        if isinstance(data.get("api_method_index"), torch.Tensor) and data["api_method_index"].numel() == n_api:
            data["api_method_index"] = data["api_method_index"].clone()
            data["api_method_index"][drop_idx] = -1

        if isinstance(data.get("mask"), torch.Tensor) and data["mask"].ndim == 2 and data["mask"].size(1) == n_api:
            data["mask"] = data["mask"].clone()
            data["mask"][:, drop_idx] = False

        edge = data.get("method_api_edge_index")
        if isinstance(edge, torch.Tensor) and edge.ndim == 2 and edge.size(0) == 2 and edge.numel() > 0:
            valid_edge = keep_token[edge[1].long().clamp(0, n_api - 1)]
            data["method_api_edge_index"] = edge[:, valid_edge]

        data["q_api"] = min(float(data.get("q_api", 1.0)), 1.0 - float(drop_idx.numel()) / max(n_api, 1))
        data["pert_api"] = 1.0
        data["api_aug_type"] = "api_sensitive_event_dropout" if sensitive_only else "api_event_dropout"
        return data

    def _apply_graph_node_feature_mask(self, data: dict, mask_ratio: float = 0.2):
        x = data["x"]
        if x.size(0) == 0:
            data["node_mask"] = torch.zeros((0,), dtype=torch.bool, device=x.device)
            return data
        num_nodes = x.size(0)
        num_mask = max(1, int(num_nodes * mask_ratio))
        idx = torch.randperm(num_nodes, device=x.device)[:num_mask]
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        node_mask[idx] = True
        x_new = x.clone()
        x_new[node_mask] = 0.0
        data["x"] = x_new
        data["node_mask"] = node_mask
        q_graph_feat = 1.0 - float(node_mask.float().mean().item())
        data["q_graph"] = min(float(data.get("q_graph", 1.0)), q_graph_feat)
        data["pert_graph"] = 1.0
        return data

    def apply_graph_sparsify(self, edge_index: torch.Tensor, strength: float):
        if edge_index.ndim != 2 or edge_index.size(1) == 0:
            return edge_index, 0.0
        strength = max(0.0, min(1.0, float(strength)))
        num_edges = edge_index.size(1)
        keep = torch.rand(num_edges, device=edge_index.device) > strength
        if keep.sum() == 0:
            keep[random.randrange(num_edges)] = True
        return edge_index[:, keep], float(keep.float().mean().item())

    def apply_graph_local_break(self, edge_index: torch.Tensor, sensitive_mask: torch.Tensor, strength: float):
        if edge_index.ndim != 2 or edge_index.size(1) == 0:
            return edge_index, 0.0
        strength = max(0.0, min(1.0, float(strength)))
        num_nodes = int(edge_index.max().item()) + 1
        sens_nodes = (
            torch.where(sensitive_mask.bool())[0]
            if isinstance(sensitive_mask, torch.Tensor) and sensitive_mask.numel() > 0 and sensitive_mask.bool().any()
            else torch.empty((0,), dtype=torch.long, device=edge_index.device)
        )
        center = (
            int(sens_nodes[random.randrange(sens_nodes.numel())].item())
            if sens_nodes.numel() > 0 and random.random() < 0.7
            else random.randrange(num_nodes)
        )
        local = (edge_index[0] == center) | (edge_index[1] == center)
        keep = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        drop_local = local & (torch.rand(edge_index.size(1), device=edge_index.device) < strength)
        keep[drop_local] = False
        if keep.sum() == 0:
            keep[random.randrange(edge_index.size(1))] = True
        return edge_index[:, keep], float(keep.float().mean().item())

    def apply_graph_target_redirection(self, edge_index: torch.Tensor, sensitive_mask: torch.Tensor, strength: float):
        if edge_index.ndim != 2 or edge_index.size(1) == 0:
            return edge_index, 0.0
        strength = max(0.0, min(1.0, float(strength)))
        edge_index = edge_index.clone()
        num_edges = edge_index.size(1)
        num_nodes = int(edge_index.max().item()) + 1
        n_rewire = max(1, int(num_edges * strength * AugmentationConstants.GRAPH_REWIRE_RATIO))
        cand = torch.randperm(num_edges, device=edge_index.device)[:n_rewire]
        for eidx in cand.tolist():
            src = int(edge_index[0, eidx].item())
            dst = random.randrange(num_nodes)
            if dst != src:
                edge_index[1, eidx] = dst
        return edge_index, max(0.0, 1.0 - 0.4 * (cand.numel() / max(num_edges, 1)))

    def apply_graph_control_flow_flattening(self, edge_index: torch.Tensor, sensitive_mask: torch.Tensor, strength: float):
        if edge_index.ndim != 2 or edge_index.size(1) == 0:
            return edge_index, 1.0
        num_nodes = int(edge_index.max().item()) + 1
        center = random.randrange(num_nodes)
        src, dst = edge_index[0], edge_index[1]
        affected = ((src == center) | (dst == center))
        keep = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        keep[affected & (torch.rand(edge_index.size(1), device=edge_index.device) < strength)] = False
        if keep.sum() == 0:
            keep[random.randrange(edge_index.size(1))] = True
        return edge_index[:, keep], float(keep.float().mean().item())

    def _apply_graph_dead_code_injection_data(self, data: dict, strength: float):
        x = data["x"]
        if x.ndim != 2 or x.size(0) == 0:
            return data
        strength = max(0.0, min(1.0, float(strength)))
        num_nodes, feat_dim = x.size()
        n_new = min(256, max(1, int(num_nodes * strength * 0.12)))
        new_x = 0.02 * torch.randn((n_new, feat_dim), dtype=x.dtype, device=x.device)
        data["x"] = torch.cat([x, new_x], dim=0)
        if data["edge_index"].ndim == 2:
            new_ids = torch.arange(num_nodes, num_nodes + n_new, device=x.device)
            dst = torch.randint(0, num_nodes, (n_new,), device=x.device)
            new_edges = torch.cat([torch.stack([new_ids, dst]), torch.stack([dst, new_ids])], dim=1)
            data["edge_index"] = torch.cat([data["edge_index"], new_edges], dim=1)
        data["sensitive_mask"] = torch.cat([
            data["sensitive_mask"],
            torch.zeros((n_new,), dtype=data["sensitive_mask"].dtype, device=data["sensitive_mask"].device),
        ])
        if isinstance(data.get("mask"), torch.Tensor) and data["mask"].ndim == 2:
            data["mask"] = torch.cat([
                data["mask"],
                torch.zeros((n_new, data["mask"].size(1)), dtype=data["mask"].dtype, device=data["mask"].device),
            ], dim=0)
        data["q_graph"] = min(float(data.get("q_graph", 1.0)), max(0.45, 1.0 - 0.35 * strength))
        data["pert_graph"] = 1.0
        return data

    def _apply_graph_feature_obfuscation(self, data: dict, strength: float):
        x = data["x"]
        if x.ndim != 2 or x.size(0) == 0:
            return data
        strength = max(0.0, min(1.0, float(strength)))
        num_nodes = x.size(0)
        n_mask = max(1, int(num_nodes * strength * 0.35))
        idx = torch.randperm(num_nodes, device=x.device)[:n_mask]
        x_new = x.clone()
        x_new[idx] = 0.15 * x_new[idx] + 0.03 * torch.randn_like(x_new[idx])
        data["x"] = x_new
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        node_mask[idx] = True
        data["node_mask"] = node_mask
        data["q_graph"] = min(float(data.get("q_graph", 1.0)), max(0.35, 1.0 - 0.6 * (idx.numel() / max(num_nodes, 1))))
        data["pert_graph"] = 1.0
        return data

    def _apply_random_graph_augmentation(self, data: dict, aug_strength_min: float, aug_strength_max: float):
        graph_aug_type = random.choice(AugmentationConstants.GRAPH_AUG_TYPES)
        graph_strength = random.uniform(aug_strength_min, aug_strength_max)
        if graph_aug_type == "graph_sparsify":
            data["edge_index"], data["q_graph"] = self.apply_graph_sparsify(data["edge_index"], graph_strength)
        elif graph_aug_type == "graph_local_break":
            data["edge_index"], data["q_graph"] = self.apply_graph_local_break(
                data["edge_index"], data["sensitive_mask"], graph_strength
            )
        elif graph_aug_type == "graph_target_redirection":
            data["edge_index"], data["q_graph"] = self.apply_graph_target_redirection(
                data["edge_index"], data["sensitive_mask"], graph_strength
            )
        elif graph_aug_type == "graph_control_flow_flattening":
            data["edge_index"], data["q_graph"] = self.apply_graph_control_flow_flattening(
                data["edge_index"], data["sensitive_mask"], graph_strength
            )
        elif graph_aug_type == "graph_dead_code_injection":
            data = self._apply_graph_dead_code_injection_data(data, graph_strength)
        elif graph_aug_type == "graph_feature_obfuscation":
            data = self._apply_graph_feature_obfuscation(data, graph_strength)
        if data["q_graph"] < 0.999:
            data["pert_graph"] = 1.0
        data["graph_aug_type"] = graph_aug_type
        return data

    def _recalculate_alignment_quality(self, data: dict):
        if self.fusion_mode == "api":
            base_align = float(data["q_api"])
        elif self.fusion_mode == "graph":
            base_align = float(data["q_graph"])
        else:
            base_align = self._compute_alignment_quality(
                float(data["q_api"]),
                float(data["q_graph"]),
                data["mask"],
                data["sensitive_mask"],
            )
        align_penalty = float(data.get("align_penalty", 0.0))
        data["q_align"] = max(0.0, base_align * (1.0 - align_penalty))
        return data

    def _apply_training_augmentation(self, data: dict, aug_strength_min: float, aug_strength_max: float):
        if self.fusion_mode != "graph" and random.random() < 0.5:
            api_aug_type = random.choices(
                population=AugmentationConstants.API_AUG_TYPES,
                weights=AugmentationConstants.API_AUG_WEIGHTS,
                k=1,
            )[0]
            api_strength = random.uniform(aug_strength_min, aug_strength_max)
            data = self._apply_api_event_dropout(
                data,
                api_strength,
                sensitive_only=(api_aug_type == "api_sensitive_event_dropout"),
            )

        if self.fusion_mode != "api" and data["edge_index"].numel() > 0:
            data = self._apply_random_graph_augmentation(data, aug_strength_min, aug_strength_max)
            data["q_graph"] = self._adjust_q_graph_semantic(
                data["q_graph"],
                data["edge_index"],
                data["sensitive_mask"],
            )

        if self.fusion_mode in {"graph", "ours"} and random.random() < 0.1:
            data = self._apply_graph_node_feature_mask(data, mask_ratio=0.15)

        data = self._recalculate_alignment_quality(data)
        api_strength = 1.0 - float(data.get("q_api", 1.0))
        graph_strength = 1.0 - float(data.get("q_graph", 1.0))
        data["api_aug_strength"] = api_strength
        data["graph_aug_strength"] = graph_strength
        data["overall_aug_strength"] = 0.5 * api_strength + 0.5 * graph_strength
        return data

    def _apply_eval_perturbation_wrapper(self, data: dict):
        t = self.eval_perturb_type
        s = float(self.eval_perturb_strength)
        if t in {"api_event_dropout", "api_sensitive_event_dropout", "modality_dropout_api"}:
            data = self._apply_api_event_dropout(
                data,
                1.0 if t == "modality_dropout_api" else s,
                sensitive_only=(t == "api_sensitive_event_dropout"),
            )
            data = self._recalculate_alignment_quality(data)
            return data

        if t == "modality_dropout_graph":
            data["edge_index"] = torch.empty((2, 0), dtype=torch.long, device=data["edge_index"].device)
            data["q_graph"] = 0.0
            data["q_align"] = 0.0
            data["pert_graph"] = 1.0
            data["graph_aug_type"] = t
            return data

        if t == "graph_sparsify":
            data["edge_index"], data["q_graph"] = self.apply_graph_sparsify(data["edge_index"], s)
        elif t == "graph_local_break":
            data["edge_index"], data["q_graph"] = self.apply_graph_local_break(
                data["edge_index"], data["sensitive_mask"], s
            )
        elif t == "graph_target_redirection":
            data["edge_index"], data["q_graph"] = self.apply_graph_target_redirection(
                data["edge_index"], data["sensitive_mask"], s
            )
        elif t == "graph_control_flow_flattening":
            data["edge_index"], data["q_graph"] = self.apply_graph_control_flow_flattening(
                data["edge_index"], data["sensitive_mask"], s
            )
        elif t == "graph_dead_code_injection":
            data = self._apply_graph_dead_code_injection_data(data, s)
        elif t == "graph_feature_obfuscation":
            data = self._apply_graph_feature_obfuscation(data, s)

        if t in {
            "graph_sparsify",
            "graph_local_break",
            "graph_target_redirection",
            "graph_control_flow_flattening",
            "graph_dead_code_injection",
            "graph_feature_obfuscation",
        }:
            data["q_graph"] = self._adjust_q_graph_semantic(data["q_graph"], data["edge_index"], data["sensitive_mask"])
            data["pert_graph"] = 1.0
            data["graph_aug_type"] = t

        data = self._recalculate_alignment_quality(data)
        return data

    def _build_data_object(self, data: dict, label: int, sid: str, year: int, time_id: int):
        dev_x = data["x"].device
        data_obj = Data(
            x=data["x"],
            edge_index=data["edge_index"],
            y=torch.tensor(label, dtype=torch.long, device=dev_x),
        )
        data_obj.attn_mask = data["mask"]
        data_obj.sensitive_mask = data["sensitive_mask"]
        data_obj.api_ids = data.get("api_ids", torch.empty((0,), dtype=torch.long, device=dev_x))
        data_obj.api_type_ids = data.get("api_type_ids", torch.empty((0,), dtype=torch.long, device=dev_x))
        data_obj.api_sensitive_mask = data.get("api_sensitive_mask", torch.empty((0,), dtype=torch.float32, device=dev_x))
        data_obj.api_method_index = data.get("api_method_index", torch.empty((0,), dtype=torch.long, device=dev_x))
        data_obj.api_in_graph_mask = data.get("api_in_graph_mask", torch.empty((0,), dtype=torch.float32, device=dev_x))
        data_obj.method_api_edge_index = data.get("method_api_edge_index", torch.empty((2, 0), dtype=torch.long, device=dev_x))
        data_obj.api_category_counts = data.get("api_category_counts", torch.zeros((16,), dtype=torch.float32, device=dev_x)).to(
            device=dev_x,
            dtype=torch.float32,
        )
        data_obj.sid = sid
        data_obj.year = torch.tensor(int(year), dtype=torch.long, device=dev_x)
        data_obj.time_id = torch.tensor(int(time_id), dtype=torch.long, device=dev_x)
        data_obj.is_dummy = False
        data_obj.q_api = torch.tensor([data["q_api"]], dtype=torch.float32, device=dev_x)
        data_obj.q_graph = torch.tensor([data["q_graph"]], dtype=torch.float32, device=dev_x)
        data_obj.q_align = torch.tensor([data["q_align"]], dtype=torch.float32, device=dev_x)
        data_obj.pert_api = torch.tensor([data["pert_api"]], dtype=torch.float32, device=dev_x)
        data_obj.pert_graph = torch.tensor([data["pert_graph"]], dtype=torch.float32, device=dev_x)
        data_obj.align_penalty = torch.tensor([data.get("align_penalty", 0.0)], dtype=torch.float32, device=dev_x)
        data_obj.api_aug_type = data.get("api_aug_type", "none")
        data_obj.graph_aug_type = data.get("graph_aug_type", "none")
        data_obj.api_aug_strength = torch.tensor([data.get("api_aug_strength", 0.0)], dtype=torch.float32, device=dev_x)
        data_obj.graph_aug_strength = torch.tensor([data.get("graph_aug_strength", 0.0)], dtype=torch.float32, device=dev_x)
        data_obj.overall_aug_strength = torch.tensor([data.get("overall_aug_strength", 0.0)], dtype=torch.float32, device=dev_x)
        data_obj.node_mask = data.get("node_mask", torch.zeros((data["x"].size(0),), dtype=torch.bool, device=dev_x))
        data_obj.api_event_mask = data.get(
            "api_event_mask",
            torch.zeros((data_obj.api_ids.numel(),), dtype=torch.bool, device=dev_x),
        )
        return data_obj

    def _get_dummy_with_error(self, label: int, sid: str, year: int, time_id: int, pt_path: Path, error):
        data = self._get_dummy(label, sid, year=year, time_id=time_id)
        data.load_failed = True
        data.fail_reason = (
            f"{type(error).__name__}: {error}" if isinstance(error, Exception) else str(error)
        )
        data.fail_path = str(pt_path)
        return data

    def __getitem__(self, idx):
        pt_path, label, sid, year, time_id = self.samples[idx]
        try:
            dex_list = self._load_dex_list(pt_path)
        except Exception as e:
            return self._get_dummy_with_error(label, sid, year, time_id, pt_path, e)

        aggregated_data = self._aggregate_dex_data(dex_list)
        if aggregated_data is None:
            return self._get_dummy_with_error(label, sid, year, time_id, pt_path, "empty valid sample")

        if self.robust_aug and self.is_train:
            s_min, s_max = self._get_temporal_aug_strength(year)
            aggregated_data = self._apply_training_augmentation(aggregated_data, s_min, s_max)
        elif not self.is_train and self.eval_perturb_type:
            aggregated_data = self._apply_eval_perturbation_wrapper(aggregated_data)

        return self._build_data_object(aggregated_data, label, sid, year, time_id)


def hierarchical_collate_fn(data_list):
    failed_items = []
    valid_items = []

    for d in data_list:
        if d is None:
            failed_items.append({"sid": None, "path": None, "reason": "data is None"})
            continue
        if getattr(d, "is_dummy", False):
            failed_items.append({
                "sid": getattr(d, "sid", None),
                "path": getattr(d, "fail_path", None),
                "reason": getattr(d, "fail_reason", "dummy sample"),
            })
            continue
        valid_items.append(d)

    if not valid_items:
        return {
            "graph_batch": None,
            "masks": None,
            "labels": None,
            "sids": None,
            "years": None,
            "time_ids": None,
            "quality": None,
            "failed_items": failed_items,
            "num_failed": len(failed_items),
            "num_valid": 0,
        }

    sids = [d.sid for d in valid_items]
    masks = [d.attn_mask for d in valid_items]
    years = torch.stack([d.year for d in valid_items]).view(-1)
    time_ids = torch.stack([d.time_id for d in valid_items]).view(-1)
    labels = torch.stack([d.y for d in valid_items])

    q_apis = torch.stack([d.q_api for d in valid_items])
    q_graphs = torch.stack([d.q_graph for d in valid_items])
    q_aligns = torch.stack([d.q_align for d in valid_items])
    pert_apis = torch.stack([d.pert_api for d in valid_items])
    pert_graphs = torch.stack([d.pert_graph for d in valid_items])
    align_penalties = torch.stack([d.align_penalty for d in valid_items])

    base_device = valid_items[0].x.device
    api_aug_strengths = torch.stack([
        getattr(d, "api_aug_strength", torch.tensor([0.0], dtype=torch.float32, device=base_device))
        for d in valid_items
    ])
    graph_aug_strengths = torch.stack([
        getattr(d, "graph_aug_strength", torch.tensor([0.0], dtype=torch.float32, device=base_device))
        for d in valid_items
    ])
    overall_aug_strengths = torch.stack([
        getattr(d, "overall_aug_strength", torch.tensor([0.0], dtype=torch.float32, device=base_device))
        for d in valid_items
    ])
    node_masks = [getattr(d, "node_mask", None) for d in valid_items]
    api_event_masks = [getattr(d, "api_event_mask", None) for d in valid_items]

    graph_list = []
    api_ids_all, api_type_all, api_sensitive_all = [], [], []
    api_method_index_all, api_in_graph_all, api_batch_all = [], [], []
    method_api_edges_all = []
    api_category_counts_all = []
    node_offset, api_offset = 0, 0

    for sample_idx, d in enumerate(valid_items):
        gd = Data(x=d.x, edge_index=d.edge_index, y=d.y)
        gd.sensitive_mask = d.sensitive_mask
        graph_list.append(gd)

        api_ids = getattr(d, "api_ids", torch.empty((0,), dtype=torch.long))
        api_type_ids = getattr(d, "api_type_ids", torch.empty((0,), dtype=torch.long))
        api_sensitive_mask = getattr(d, "api_sensitive_mask", torch.empty((0,), dtype=torch.float32))
        api_method_index = getattr(d, "api_method_index", torch.empty((0,), dtype=torch.long))
        api_in_graph_mask = getattr(d, "api_in_graph_mask", torch.empty((0,), dtype=torch.float32))
        n_api = int(api_ids.numel())

        if n_api > 0:
            api_ids_all.append(api_ids.long())
            api_type_all.append(api_type_ids.long())
            api_sensitive_all.append(api_sensitive_mask.float())

            api_method_index = api_method_index.long().clone()
            valid_method = api_method_index >= 0
            api_method_index[valid_method] += node_offset
            api_method_index_all.append(api_method_index)

            api_in_graph_all.append(api_in_graph_mask.float())
            api_batch_all.append(torch.full((n_api,), sample_idx, dtype=torch.long, device=api_ids.device))

            edge = getattr(d, "method_api_edge_index", torch.empty((2, 0), dtype=torch.long, device=api_ids.device))
            if isinstance(edge, torch.Tensor) and edge.numel() > 0:
                edge = edge.long().clone()
                edge[0] += node_offset
                edge[1] += api_offset
                method_api_edges_all.append(edge)

        api_category_counts_all.append(
            getattr(d, "api_category_counts", torch.zeros((16,), dtype=torch.float32, device=d.x.device)).float()
        )
        node_offset += int(d.x.size(0))
        api_offset += n_api

    graph_batch = Batch.from_data_list(graph_list)
    batch_device = graph_batch.x.device
    graph_batch.api_ids = (
        torch.cat(api_ids_all, dim=0).long()
        if api_ids_all else torch.empty((0,), dtype=torch.long, device=batch_device)
    )
    graph_batch.api_type_ids = (
        torch.cat(api_type_all, dim=0).long()
        if api_type_all else torch.empty((0,), dtype=torch.long, device=batch_device)
    )
    graph_batch.api_sensitive_mask = (
        torch.cat(api_sensitive_all, dim=0).float()
        if api_sensitive_all else torch.empty((0,), dtype=torch.float32, device=batch_device)
    )
    graph_batch.api_method_index = (
        torch.cat(api_method_index_all, dim=0).long()
        if api_method_index_all else torch.empty((0,), dtype=torch.long, device=batch_device)
    )
    graph_batch.api_in_graph_mask = (
        torch.cat(api_in_graph_all, dim=0).float()
        if api_in_graph_all else torch.empty((0,), dtype=torch.float32, device=batch_device)
    )
    graph_batch.api_batch = (
        torch.cat(api_batch_all, dim=0).long()
        if api_batch_all else torch.empty((0,), dtype=torch.long, device=batch_device)
    )
    graph_batch.method_api_edge_index = (
        torch.cat(method_api_edges_all, dim=1).long()
        if method_api_edges_all else torch.empty((2, 0), dtype=torch.long, device=batch_device)
    )
    graph_batch.api_category_counts = torch.stack(api_category_counts_all).float()
    graph_batch.q_api = q_apis
    graph_batch.q_graph = q_graphs
    graph_batch.q_align = q_aligns
    graph_batch.pert_api = pert_apis
    graph_batch.pert_graph = pert_graphs
    graph_batch.years = years
    graph_batch.time_ids = time_ids
    graph_batch.align_penalty = align_penalties
    graph_batch.api_aug_strength = api_aug_strengths
    graph_batch.graph_aug_strength = graph_aug_strengths
    graph_batch.overall_aug_strength = overall_aug_strengths

    api_aug_types = [getattr(d, "api_aug_type", "none") for d in valid_items]
    graph_aug_types = [getattr(d, "graph_aug_type", "none") for d in valid_items]

    return {
        "graph_batch": graph_batch,
        "masks": masks,
        "labels": labels,
        "sids": sids,
        "years": years,
        "time_ids": time_ids,
        "quality": {
            "q_api": q_apis,
            "q_graph": q_graphs,
            "q_align": q_aligns,
            "pert_api": pert_apis,
            "pert_graph": pert_graphs,
            "align_penalty": align_penalties,
        },
        "aug_types": {
            "api": api_aug_types,
            "graph": graph_aug_types,
        },
        "aux_masks": {
            "node": node_masks,
            "api_event": api_event_masks,
        },
        "failed_items": failed_items,
        "num_failed": len(failed_items),
        "num_valid": len(valid_items),
        "api_aug_strength": api_aug_strengths,
        "graph_aug_strength": graph_aug_strengths,
        "overall_aug_strength": overall_aug_strengths,
    }
