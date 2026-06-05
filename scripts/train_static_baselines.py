#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.dataset import RobustTriModalDataset
from fusion.perturbations import EVAL_PERTURB_TYPES
from fusion.train import build_dataset, load_config


REFERENCE_METHODS = (
    "drebin_lr",
    "drebin_svm",
    "mamadroid_lr",
    "api_bow_lr",
    "manifest_lr",
    "tri_static_lr",
)


def _tensor(value: Any) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.empty((0,))


def _scalar(value: Any, default: float = 0.0) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().float().view(-1)[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _vec(value: Any, length: int | None = None) -> np.ndarray:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return np.zeros((length or 0,), dtype=np.float32)
    arr = value.detach().float().view(-1).cpu().numpy().astype(np.float32)
    if length is None:
        return arr
    if arr.size >= length:
        return arr[:length]
    out = np.zeros((length,), dtype=np.float32)
    out[: arr.size] = arr
    return out


def _safe_log_counts(arr: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(arr.astype(np.float32), 0.0))


def _markov_type_features(data) -> np.ndarray:
    type_ids = _tensor(getattr(data, "api_type_ids", None)).long().view(-1).cpu().numpy()
    type_ids = np.clip(type_ids, 0, 15)
    matrix = np.zeros((16, 16), dtype=np.float32)
    if type_ids.size >= 2:
        for left, right in zip(type_ids[:-1], type_ids[1:]):
            matrix[int(left), int(right)] += 1.0
        row_sum = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, np.maximum(row_sum, 1.0), out=np.zeros_like(matrix), where=row_sum > 0)
    hist = np.bincount(type_ids, minlength=16).astype(np.float32) if type_ids.size else np.zeros(16, dtype=np.float32)
    hist = hist / max(float(hist.sum()), 1.0)
    return np.concatenate([matrix.reshape(-1), hist], axis=0)


def _dense_semantic_features(data, *, include_manifest: bool = True) -> np.ndarray:
    parts = [
        _safe_log_counts(_vec(getattr(data, "api_semantic_category_counts", None), 12)),
        _safe_log_counts(_vec(getattr(data, "graph_semantic_category_counts", None), 12)),
    ]
    if include_manifest:
        parts.extend(
            [
                _safe_log_counts(_vec(getattr(data, "manifest_category_counts", None), 12)),
                _vec(getattr(data, "manifest_stats", None), 11),
            ]
        )
    x = _tensor(getattr(data, "x", None))
    edge_index = _tensor(getattr(data, "edge_index", None))
    num_nodes = int(x.size(0)) if x.ndim == 2 else 0
    num_edges = int(edge_index.size(1)) if edge_index.ndim == 2 else 0
    parts.append(
        np.asarray(
            [
                _scalar(getattr(data, "q_api", None)),
                _scalar(getattr(data, "q_graph", None)),
                _scalar(getattr(data, "q_manifest", None)),
                _scalar(getattr(data, "q_align", None)),
                _scalar(getattr(data, "pert_api", None)),
                _scalar(getattr(data, "pert_graph", None)),
                _scalar(getattr(data, "pert_manifest", None)),
                float(_tensor(getattr(data, "api_ids", None)).numel()),
                float(num_nodes),
                float(num_edges),
            ],
            dtype=np.float32,
        )
    )
    return np.concatenate(parts, axis=0).astype(np.float32)


def _sparse_feature_dict(data, method: str) -> dict[str, float]:
    feats: dict[str, float] = {}
    api_ids = _tensor(getattr(data, "api_ids", None)).long().view(-1).cpu().tolist()
    api_types = _tensor(getattr(data, "api_type_ids", None)).long().view(-1).cpu().tolist()
    perm_ids = _tensor(getattr(data, "manifest_permission_ids", None)).long().view(-1).cpu().tolist()
    intent_ids = _tensor(getattr(data, "manifest_intent_ids", None)).long().view(-1).cpu().tolist()

    if method in {"drebin_lr", "drebin_svm", "api_bow_lr"}:
        for item in set(api_ids):
            feats[f"api:{int(item)}"] = 1.0
        for item in api_types:
            feats[f"api_type_count:{int(item)}"] = feats.get(f"api_type_count:{int(item)}", 0.0) + 1.0

    if method in {"drebin_lr", "drebin_svm"}:
        for item in set(perm_ids):
            feats[f"perm:{int(item)}"] = 1.0
        for item in set(intent_ids):
            feats[f"intent:{int(item)}"] = 1.0
        for prefix, arr in (
            ("api_sem", _vec(getattr(data, "api_semantic_category_counts", None), 12)),
            ("graph_sem", _vec(getattr(data, "graph_semantic_category_counts", None), 12)),
            ("manifest_sem", _vec(getattr(data, "manifest_category_counts", None), 12)),
            ("manifest_stat", _vec(getattr(data, "manifest_stats", None), 11)),
        ):
            for idx, value in enumerate(arr):
                if float(value) != 0.0:
                    feats[f"{prefix}:{idx}"] = float(value)
        for key in ("q_api", "q_graph", "q_manifest", "q_align"):
            feats[key] = _scalar(getattr(data, key, None))

    return feats


def _load_split_features(
    cfg: dict[str, Any],
    split: str,
    method: str,
    *,
    perturb_type: str | None = None,
    perturb_strength: float = 0.0,
) -> tuple[list[str], list[int], list[Any]]:
    dataset: RobustTriModalDataset = build_dataset(
        cfg,
        split,
        is_train=False,
        perturb_type=perturb_type,
        perturb_strength=perturb_strength,
    )
    sids: list[str] = []
    labels: list[int] = []
    features: list[Any] = []
    for idx in tqdm(range(len(dataset)), desc=f"{split}:{method}", leave=False):
        data = dataset[idx]
        if bool(getattr(data, "is_dummy", False)):
            continue
        sids.append(str(getattr(data, "sid", f"{split}_{idx}")))
        labels.append(int(data.y.detach().cpu().view(-1)[0].item()))
        if method in {"drebin_lr", "drebin_svm", "api_bow_lr"}:
            features.append(_sparse_feature_dict(data, method))
        elif method == "mamadroid_lr":
            features.append(_markov_type_features(data))
        elif method == "manifest_lr":
            parts = [
                _safe_log_counts(_vec(getattr(data, "manifest_category_counts", None), 12)),
                _vec(getattr(data, "manifest_stats", None), 11),
                np.asarray([_scalar(getattr(data, "q_manifest", None))], dtype=np.float32),
            ]
            features.append(np.concatenate(parts, axis=0).astype(np.float32))
        elif method == "tri_static_lr":
            features.append(_dense_semantic_features(data, include_manifest=True))
        else:
            raise ValueError(f"Unsupported reference method: {method}")
    return sids, labels, features


def _binary_metrics(labels: list[int], probs: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    if not labels:
        return {"acc": 0.0, "macro_f1": 0.0, "f1_pos": 0.0, "macro_recall": 0.0, "auc": 0.0, "ap": 0.0, "brier": 0.0, "ece_10": 0.0}
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    pred = np.asarray(preds, dtype=np.int64)
    confidence = np.maximum(p, 1.0 - p)
    correct = (pred == y).astype(np.float64)
    ece = 0.0
    for lo, hi in zip(np.linspace(0.0, 1.0, 11)[:-1], np.linspace(0.0, 1.0, 11)[1:]):
        mask = (confidence >= lo) & (confidence <= hi) if hi >= 1.0 else (confidence >= lo) & (confidence < hi)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_pos": float(f1_score(y, pred, average="binary", pos_label=1, zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else 0.0,
        "ap": float(average_precision_score(y, p)) if len(set(y.tolist())) > 1 else 0.0,
        "brier": float(np.mean((p - y.astype(np.float64)) ** 2)),
        "ece_10": float(ece),
    }


def _fit_reference_model(method: str, train_features: list[Any], train_labels: list[int]) -> dict[str, Any]:
    if method in {"drebin_lr", "drebin_svm", "api_bow_lr"}:
        vectorizer = DictVectorizer(sparse=True)
        x_train = vectorizer.fit_transform(train_features)
        if method == "drebin_svm":
            model = LinearSVC(C=1.0, max_iter=5000, class_weight="balanced")
            model.fit(x_train, train_labels)
            return {"kind": "sparse_svm", "vectorizer": vectorizer, "model": model}
        model = LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
        )
        model.fit(x_train, train_labels)
        return {"kind": "sparse_lr", "vectorizer": vectorizer, "model": model}

    x_train = np.asarray(train_features, dtype=np.float32)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(x_train, train_labels)
    return {"kind": "dense_lr", "scaler": scaler, "model": model}


def _predict_reference_model(fitted: dict[str, Any], eval_features: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    kind = str(fitted["kind"])
    model = fitted["model"]
    if kind in {"sparse_lr", "sparse_svm"}:
        x_eval = fitted["vectorizer"].transform(eval_features)
        if kind == "sparse_svm":
            score = model.decision_function(x_eval)
            probs = 1.0 / (1.0 + np.exp(-score))
            preds = (score >= 0.0).astype(np.int64)
            return probs, preds
        probs = model.predict_proba(x_eval)[:, 1]
        preds = model.predict(x_eval)
        return probs, preds

    x_eval = np.asarray(eval_features, dtype=np.float32)
    x_eval = fitted["scaler"].transform(x_eval)
    probs = model.predict_proba(x_eval)[:, 1]
    preds = model.predict(x_eval)
    return probs, preds


def _write_prediction_rows(path: Path, split: str, sids: list[str], labels: list[int], probs: np.ndarray, preds: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "sid", "label", "prob_malware", "pred", "correct"])
        writer.writeheader()
        for sid, label, prob, pred in zip(sids, labels, probs, preds):
            writer.writerow(
                {
                    "split": split,
                    "sid": sid,
                    "label": int(label),
                    "prob_malware": float(prob),
                    "pred": int(pred),
                    "correct": int(int(pred) == int(label)),
                }
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    methods = args.methods or list(REFERENCE_METHODS)
    unknown = sorted(set(methods) - set(REFERENCE_METHODS))
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}; choose from {REFERENCE_METHODS}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}
    perturb_tests = list(args.perturb_tests or [])
    perturb_strengths = [float(v) for v in (args.perturb_strengths or [0.5])]

    for method in methods:
        _train_sids, train_labels, train_features = _load_split_features(cfg, "train", method)
        fitted = _fit_reference_model(method, train_features, train_labels)
        method_results: dict[str, Any] = {}
        for split in ("val", "test"):
            sids, labels, features = _load_split_features(cfg, split, method)
            probs, preds = _predict_reference_model(fitted, features)
            metrics = _binary_metrics(labels, probs, preds)
            metrics["num_eval"] = len(labels)
            method_results[f"{split}_clean"] = metrics
            if args.write_predictions:
                _write_prediction_rows(out_dir / method / f"{split}_clean_predictions.csv", split, sids, labels, probs, preds)

        if args.robust_test:
            for perturb in perturb_tests:
                if perturb == "clean":
                    continue
                if perturb not in EVAL_PERTURB_TYPES:
                    raise ValueError(f"Unsupported perturbation: {perturb}")
                strengths = [1.0] if perturb.endswith("_missing") or perturb.startswith("modality_dropout_") else perturb_strengths
                for strength in strengths:
                    key = perturb if len(strengths) == 1 else f"{perturb}_s{strength:g}"
                    sids, labels, features = _load_split_features(
                        cfg,
                        "test",
                        method,
                        perturb_type=perturb,
                        perturb_strength=float(strength),
                    )
                    probs, preds = _predict_reference_model(fitted, features)
                    metrics = _binary_metrics(labels, probs, preds)
                    metrics["num_eval"] = len(labels)
                    method_results[f"test_{key}"] = metrics

        all_results[method] = method_results
        (out_dir / method / "metrics.json").parent.mkdir(parents=True, exist_ok=True)
        (out_dir / method / "metrics.json").write_text(
            json.dumps(method_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    (out_dir / "metrics_all.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(all_results, indent=2, ensure_ascii=False))
    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight static reference baselines on tri-modal PTs.")
    parser.add_argument("--config", nargs="+", default=["config/experiments/tri_modal_robust/base_tri_modal_robust.yaml"])
    parser.add_argument("--methods", nargs="+", choices=REFERENCE_METHODS, default=list(REFERENCE_METHODS))
    parser.add_argument("--out-dir", default="results/static_reference_baselines")
    parser.add_argument("--robust-test", action="store_true", help="Evaluate selected synthetic perturbation tests on test split.")
    parser.add_argument("--perturb-tests", nargs="+", default=["api_graph_degraded", "manifest_degraded", "all_degraded", "api_missing", "graph_missing", "manifest_missing"])
    parser.add_argument("--perturb-strengths", nargs="+", type=float, default=[0.5])
    parser.add_argument("--write-predictions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
