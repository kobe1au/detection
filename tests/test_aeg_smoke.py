from __future__ import annotations

import csv
import argparse
import json
import logging
import shutil
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from fusion.aeg_builder import _compress_method_feature, build_aeg_payload
from fusion.constants import AEG_SCHEMA_VERSION, EDGE_TYPES, NODE_TYPES, VIEW_TYPES
from fusion.dataset import AEGDataset, AEGDatasetConfigError, aeg_collate_fn, payload_to_data
from fusion.io_utils import load_aeg_payload, load_checkpoint
from fusion.losses import compute_aeg_loss
from fusion.manifest_features import build_manifest_vocab, extract_manifest_record, vectorize_manifest_record
from fusion.model import AEGModel, build_model
from fusion.perturbations import apply_aeg_view
from fusion.payload_contract import validate_aeg_payload
from fusion.train import _validate_split_isolation, load_config, run
import fusion.train as train_module


def _manifest_record() -> dict:
    return {
        "sid": "a" * 64,
        "sha256": "a" * 64,
        "apk_name": "sample.apk",
        "package_name": "com.example.sample",
        "permissions": ["android.permission.INTERNET", "android.permission.READ_SMS"],
        "activities": ["com.example.MainActivity"],
        "services": [],
        "receivers": ["com.example.BootReceiver"],
        "providers": [],
        "intent_actions": ["android.intent.action.BOOT_COMPLETED"],
        "intent_categories": [],
        "uses_features": [],
        "min_sdk": 21,
        "target_sdk": 34,
        "debuggable": False,
        "exported_component_count": 1,
        "component_count": 2,
        "parse_error": "",
    }


def _payload() -> dict:
    record = _manifest_record()
    vocab = build_manifest_vocab([record], max_permissions=8, max_intents=8, max_features=4)
    manifest_payload = vectorize_manifest_record(record, vocab, manifest_dim=32)
    dex_item = {
        "call_x": torch.randn(3, 16),
        "call_edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "call_sensitive_mask": torch.tensor([0, 1, 0], dtype=torch.float32),
        "api_ids": torch.tensor([10, 11, 12, 13], dtype=torch.long),
        "api_type_ids": torch.tensor([6, 2, 8, 12], dtype=torch.long),
        "api_sensitive_mask": torch.tensor([1, 1, 0, 1], dtype=torch.float32),
        "api_method_index": torch.tensor([0, 1, 1, 2], dtype=torch.long),
        "api_in_graph_mask": torch.ones(4),
        "method_api_edge_index": torch.tensor([[0, 1, 1, 2], [0, 1, 2, 3]], dtype=torch.long),
        "call_method_names": ["Lcom/example/Main;->a", "Lcom/example/BootReceiver;->onReceive", "Lcom/example/C;->reflect"],
        "api_tokens": ["urlconnection", "smsmanager", "class#forname", "cipher"],
    }
    return build_aeg_payload(
        sid=record["sha256"],
        apk_name=record["apk_name"],
        split="train",
        dex_list=[dex_item],
        manifest_payload=manifest_payload,
        manifest_record=record,
        direct_meta={
            "dex_success_ratio": 1.0,
            "num_dex_total": 1,
            "num_dex_success": 1,
        },
        node_feature_dim=32,
    )


def _variant_payload(
    sid: str,
    *,
    package_name: str,
    split: str = "train",
) -> dict:
    payload = dict(_payload())
    payload["sid"] = sid
    payload["sha256"] = sid
    payload["package_name"] = package_name
    payload["split"] = split
    return payload


def _write_split(tmp_path: Path, split: str, rows: list[tuple[dict, int]]) -> tuple[Path, Path]:
    pt_dir = tmp_path / split
    pt_dir.mkdir()
    csv_path = tmp_path / f"{split}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        for payload, label in rows:
            torch.save(payload, pt_dir / f"{payload['sid']}.pt")
            writer.writerow({"id": payload["sid"], "label": label})
    return pt_dir, csv_path


def test_aeg_builder_dataset_model_loss(tmp_path: Path):
    payload = _payload()
    assert payload["schema_version"] == AEG_SCHEMA_VERSION
    assert payload["node_x"].ndim == 2
    assert payload["edge_index"].size(0) == 2
    risk_mask = payload["node_type"] == NODE_TYPES["RISK_SEMANTIC"]
    assert bool(risk_mask.any())
    risk_quality = payload["node_quality"][risk_mask]
    assert bool((risk_quality > 0).any())
    assert bool((risk_quality == 0).any())

    pt_dir = tmp_path / "train"
    pt_dir.mkdir()
    torch.save(payload, pt_dir / f"{payload['sid']}.pt")
    csv_path = tmp_path / "train.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerow({"id": payload["sid"], "label": 1})

    ds = AEGDataset(pt_dir, csv_path, train_aug=True, aug_views=["api_graph_degraded"], aug_strengths=[0.5], aug_prob=1.0)
    batch = next(iter(DataLoader(ds, batch_size=1, collate_fn=aeg_collate_fn)))
    model = AEGModel(node_input_dim=32, hidden_dim=32, layers=1, num_latents=4)
    clean_logits, clean_extra = model(batch["clean"])
    aug_logits, aug_extra = model(batch["aug"])
    loss, parts = compute_aeg_loss(clean_logits, batch["clean"].y.view(-1), clean_extra, aug_logits=aug_logits, aug_extra=aug_extra)

    assert clean_logits.shape == (1, 2)
    assert torch.isfinite(loss)
    assert parts["consistency"] >= 0.0
    assert parts["weighted_consistency"] >= 0.0
    assert parts["loss_mode"] == "compact_kl"
    assert "attention_mass" in clean_extra


def test_aeg_perturbation_updates_reliability():
    data = payload_to_data(_payload(), label=1)
    degraded = apply_aeg_view(data, view="api_missing", strength=1.0)
    assert degraded.view_type_id.item() == VIEW_TYPES["api_missing"]
    assert degraded.q_api.item() == data.q_api.item()
    assert degraded.pert_api.item() == 1.0
    assert degraded.q_align.item() == 0.0
    method_mask = degraded.node_type == NODE_TYPES["METHOD"]
    assert torch.all(degraded.node_semantic[method_mask] == 0)
    api_related = torch.isin(
        degraded.edge_type,
        torch.tensor(
            [
                EDGE_TYPES["METHOD_INVOKES_API_FAMILY"],
                EDGE_TYPES["API_FAMILY_INVOKED_BY_METHOD"],
                EDGE_TYPES["PERMISSION_RELATED_TO_API_FAMILY"],
                EDGE_TYPES["API_FAMILY_RELATED_TO_PERMISSION"],
                EDGE_TYPES["METHOD_HAS_RISK"],
                EDGE_TYPES["RISK_OBSERVED_IN_METHOD"],
            ],
            dtype=torch.long,
        ),
    )
    assert bool(api_related.any())
    assert torch.all(degraded.edge_quality[api_related] == 0)
    apk_mask = degraded.node_type == NODE_TYPES["APK"]
    assert degraded.node_quality[apk_mask].item() == max(
        degraded.q_api.item() * (1.0 - degraded.pert_api.item()),
        degraded.q_graph.item() * (1.0 - degraded.pert_graph.item()),
        degraded.q_manifest.item() * (1.0 - degraded.pert_manifest.item()),
    )
    string_hint_mask = degraded.node_type == NODE_TYPES["STRING_HINT"]
    assert torch.all(degraded.node_quality[string_hint_mask] == 0)
    assert torch.all(degraded.x[string_hint_mask] == 0)
    _, extra = AEGModel(node_input_dim=data.x.size(1), hidden_dim=16, layers=1, num_latents=2)(
        Batch.from_data_list([data])
    )
    expected = (data.q_api * data.q_graph).sqrt() * (0.5 + 0.5 * data.q_align)
    assert torch.allclose(extra["code_reliability"].cpu(), expected.view(-1), atol=1e-6)


def test_api_missing_clears_graph_behavior_hint_features():
    payload = _payload()
    payload["graph_behavior_hints"] = True
    payload["graph_behavior_hint_start"] = 1
    payload["graph_behavior_hint_dim"] = 4
    method_mask = payload["node_type"] == NODE_TYPES["METHOD"]
    payload["node_x"][method_mask, 1:5] = 1.0
    data = payload_to_data(payload, label=1)
    degraded = apply_aeg_view(data, view="api_missing", strength=1.0)
    degraded_method_mask = degraded.node_type == NODE_TYPES["METHOD"]
    assert torch.all(degraded.x[degraded_method_mask, 1:5] == 0)


def test_manifest_missing_clears_aggregate_semantic_and_refreshes_risk():
    degraded = apply_aeg_view(payload_to_data(_payload(), label=1), view="manifest_missing", strength=1.0)
    apk_mask = degraded.node_type == NODE_TYPES["APK"]
    assert torch.all(degraded.node_semantic[apk_mask] == 0)
    assert degraded.node_quality[apk_mask].item() == max(
        degraded.q_api.item() * (1.0 - degraded.pert_api.item()),
        degraded.q_graph.item() * (1.0 - degraded.pert_graph.item()),
        degraded.q_manifest.item() * (1.0 - degraded.pert_manifest.item()),
    )
    risk_mask = degraded.node_type == NODE_TYPES["RISK_SEMANTIC"]
    src, dst = degraded.edge_index
    risk_edge_types = torch.tensor(
        [
            EDGE_TYPES["METHOD_HAS_RISK"],
            EDGE_TYPES["RISK_OBSERVED_IN_METHOD"],
            EDGE_TYPES["MANIFEST_HAS_RISK"],
            EDGE_TYPES["RISK_DECLARED_BY_MANIFEST"],
        ],
        dtype=torch.long,
    )
    risk_edge_mask = torch.isin(degraded.edge_type, risk_edge_types) & (degraded.edge_quality > 0)
    for idx in torch.where(risk_mask)[0].tolist():
        has_live = bool((((src == idx) | (dst == idx)) & risk_edge_mask).any())
        if not has_live:
            assert degraded.node_quality[idx].item() == 0.0
            assert torch.all(degraded.node_semantic[idx] == 0)


def test_manifest_shuffle_is_type_aware(tmp_path: Path):
    payload_a = _payload()
    payload_b = _payload()
    payload_b["sid"] = "b" * 64
    payload_b["sha256"] = "b" * 64
    data_a = payload_to_data(payload_a, label=1)
    data_b = payload_to_data(payload_b, label=0)
    aug = apply_aeg_view(data_a, view="manifest_shuffled", strength=1.0)
    before_type = aug.node_type.clone()
    batch = aeg_collate_fn(
        [
            {"clean": data_a, "aug": aug, "manifest_donor": data_b},
            {"clean": data_b, "aug": apply_aeg_view(data_b, view="clean", strength=0.0)},
        ]
    )
    shuffled = batch["aug"].to_data_list()[0]
    assert torch.equal(shuffled.node_type, before_type)
    apk_mask = shuffled.node_type == NODE_TYPES["APK"]
    assert torch.all(shuffled.node_semantic[apk_mask] == 0)
    manifest_types = {NODE_TYPES["PERMISSION"], NODE_TYPES["INTENT"], NODE_TYPES["COMPONENT"]}
    for node_type in manifest_types:
        mask = (shuffled.node_source == 1) & (shuffled.node_type == node_type)
        if bool(mask.any()):
            assert torch.all(shuffled.node_type[mask] == node_type)


def test_manifest_shuffle_uses_dataset_donor_with_batch_size_one(tmp_path: Path):
    payload_a = _payload()
    payload_b = _payload()
    payload_b["sid"] = "b" * 64
    payload_b["sha256"] = "b" * 64
    payload_b["q_manifest"] = torch.tensor([0.25], dtype=torch.float32)
    pt_dir = tmp_path / "train"
    pt_dir.mkdir()
    torch.save(payload_a, pt_dir / f"{payload_a['sid']}.pt")
    torch.save(payload_b, pt_dir / f"{payload_b['sid']}.pt")
    csv_path = tmp_path / "train.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerow({"id": payload_a["sid"], "label": 1})
        writer.writerow({"id": payload_b["sid"], "label": 0})

    ds = AEGDataset(pt_dir, csv_path, train_aug=True, aug_views=["manifest_shuffled"], aug_strengths=[1.0], aug_prob=1.0)
    item = ds[0]
    assert getattr(item["manifest_donor"], "sid", "") == payload_b["sid"]
    batch = aeg_collate_fn([item])
    aug = batch["aug"].to_data_list()[0]
    assert batch["manifest_donor_sid"][0] == payload_b["sid"]
    assert batch["requested_view_type_id"][0] == VIEW_TYPES["manifest_shuffled"]
    assert batch["effective_view_type_id"][0] == VIEW_TYPES["manifest_shuffled"]
    assert batch["manifest_shuffle_fallback"][0] == 0
    assert aug.requested_view_type_id.item() == VIEW_TYPES["manifest_shuffled"]
    assert aug.effective_view_type_id.item() == VIEW_TYPES["manifest_shuffled"]
    assert aug.manifest_shuffle_fallback.item() == 0
    assert abs(float(aug.q_manifest.item()) - 0.25) < 1e-6
    assert aug.pert_manifest.item() == 1.0
    _, extra = AEGModel(node_input_dim=aug.x.size(1), hidden_dim=16, layers=1, num_latents=2)(
        Batch.from_data_list([aug])
    )
    assert extra["r_manifest"].item() == 0.0
    assert extra["manifest_reliability"].item() == 0.0


def test_manifest_shuffle_falls_back_to_missing_without_donor(tmp_path: Path):
    payload = _payload()
    pt_dir, csv_path = _write_split(tmp_path, "train", [(payload, 1)])
    ds = AEGDataset(
        pt_dir,
        csv_path,
        train_aug=True,
        aug_views=["manifest_shuffled"],
        aug_strengths=[1.0],
        aug_prob=1.0,
    )
    batch = aeg_collate_fn([ds[0]])
    aug = batch["aug"].to_data_list()[0]
    assert batch["manifest_donor_sid"][0] == ""
    assert batch["requested_view_type_id"][0] == VIEW_TYPES["manifest_shuffled"]
    assert batch["effective_view_type_id"][0] == VIEW_TYPES["manifest_missing"]
    assert batch["manifest_shuffle_fallback"][0] == 1
    assert aug.requested_view_type_id.item() == VIEW_TYPES["manifest_shuffled"]
    assert aug.effective_view_type_id.item() == VIEW_TYPES["manifest_missing"]
    assert aug.manifest_shuffle_fallback.item() == 1
    assert aug.pert_manifest.item() == 1.0


def test_manifest_shuffle_blind_keeps_reliability_scalars(tmp_path: Path):
    payload_a = _payload()
    payload_b = _payload()
    payload_b["sid"] = "b" * 64
    payload_b["sha256"] = "b" * 64
    payload_b["q_manifest"] = torch.tensor([0.25], dtype=torch.float32)
    data_a = payload_to_data(payload_a, label=1)
    data_b = payload_to_data(payload_b, label=0)
    aug = apply_aeg_view(data_a, view="manifest_shuffled_blind", strength=1.0)
    batch = aeg_collate_fn([{"clean": data_a, "aug": aug, "manifest_donor": data_b}])
    shuffled = batch["aug"].to_data_list()[0]
    assert shuffled.view_type_id.item() == VIEW_TYPES["manifest_shuffled_blind"]
    assert torch.allclose(shuffled.q_manifest.cpu(), data_a.q_manifest.cpu())
    assert shuffled.pert_manifest.item() == data_a.pert_manifest.item()
    _, extra = AEGModel(node_input_dim=shuffled.x.size(1), hidden_dim=16, layers=1, num_latents=2)(
        Batch.from_data_list([shuffled])
    )
    assert extra["r_manifest"].item() > 0.0


def test_manifest_noisy_blind_does_not_change_reliability_scalars():
    data = payload_to_data(_payload(), label=1)
    aug = apply_aeg_view(data, view="manifest_noisy_blind", strength=0.7)
    assert aug.view_type_id.item() == VIEW_TYPES["manifest_noisy_blind"]
    assert torch.allclose(aug.q_manifest.cpu(), data.q_manifest.cpu())
    assert torch.allclose(aug.pert_manifest.cpu(), data.pert_manifest.cpu())


def test_manifest_shuffle_default_donor_is_label_agnostic():
    from fusion.dataset import _build_manifest_donor_indices

    samples = [
        (Path("a.pt"), 0),
        (Path("b.pt"), 0),
        (Path("c.pt"), 1),
    ]
    assert _build_manifest_donor_indices(samples, mode="cyclic") == [1, 2, 0]
    assert _build_manifest_donor_indices(samples, mode="opposite_label") == [2, 2, 0]
    with pytest.raises(AEGDatasetConfigError, match="manifest_donor_mode"):
        _build_manifest_donor_indices(samples, mode="invalid")


def test_fixed_evaluation_perturbation_is_deterministic(tmp_path: Path):
    payload = _payload()
    pt_dir = tmp_path / "test"
    pt_dir.mkdir()
    torch.save(payload, pt_dir / f"{payload['sid']}.pt")
    csv_path = tmp_path / "test.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerow({"id": payload["sid"], "label": 1})
    ds = AEGDataset(
        pt_dir,
        csv_path,
        train_aug=True,
        aug_views=["manifest_noisy"],
        aug_strengths=[0.5],
        aug_prob=1.0,
        seed=42,
        deterministic_aug=True,
    )
    assert torch.equal(ds[0]["aug"].x, ds[0]["aug"].x)


def test_missing_source_is_not_treated_as_semantic_conflict():
    clean = payload_to_data(_payload(), label=1)
    manifest_missing = apply_aeg_view(clean, view="manifest_missing", strength=1.0)
    model = AEGModel(node_input_dim=clean.x.size(1), hidden_dim=16, layers=1, num_latents=2)
    _, extra = model(Batch.from_data_list([manifest_missing]))
    assert extra["code_manifest_conflict"].item() == 0.0


def test_reliability_uses_raw_quality_and_single_perturbation_factor():
    clean = payload_to_data(_payload(), label=1)
    degraded = apply_aeg_view(clean, view="api_degraded", strength=0.5)
    assert degraded.q_api.item() == clean.q_api.item()
    model = AEGModel(node_input_dim=clean.x.size(1), hidden_dim=16, layers=1, num_latents=2)
    _, extra = model(Batch.from_data_list([degraded]))
    assert torch.allclose(extra["r_api"], clean.q_api.view(-1) * 0.5, atol=1e-6)


def test_model_ablation_switches_keep_missing_evidence_masked():
    data = apply_aeg_view(payload_to_data(_payload(), label=1), view="api_missing", strength=1.0)
    model = AEGModel(
        node_input_dim=data.x.size(1),
        hidden_dim=16,
        layers=1,
        num_latents=2,
        use_relation_types=False,
        use_node_types=False,
        use_node_source=False,
        use_edge_source=False,
        use_node_quality=False,
        use_edge_quality=False,
        source_bias_weight=0.0,
        reliability_bias_weight=0.0,
        conflict_bias_weight=0.0,
        fusion_mode="mean_pool",
    )
    logits, extra = model(Batch.from_data_list([data]))
    assert torch.isfinite(logits).all()
    assert torch.isfinite(extra["fused_emb"]).all()


def test_model_structural_ablation_masks_configured_nodes_and_edges():
    data = payload_to_data(_payload(), label=1)
    model = AEGModel(
        node_input_dim=data.x.size(1),
        hidden_dim=16,
        layers=1,
        num_latents=2,
        masked_node_types=["RISK_SEMANTIC"],
        masked_edge_types=["PERMISSION_RELATED_TO_API_FAMILY", "API_FAMILY_RELATED_TO_PERMISSION"],
    )
    assert NODE_TYPES["RISK_SEMANTIC"] in set(model.masked_node_type_ids.cpu().tolist())
    assert EDGE_TYPES["PERMISSION_RELATED_TO_API_FAMILY"] in set(model.masked_edge_type_ids.cpu().tolist())
    _, extra = model(Batch.from_data_list([data]))
    assert torch.allclose(extra["risk_emb"].cpu(), torch.zeros_like(extra["risk_emb"].cpu()), atol=1e-6)


def test_manifest_record_extracts_package_name(monkeypatch, tmp_path: Path):
    class FakeAPK:
        def __init__(self, _path: str):
            pass

        def get_package(self):
            return "Com.Example.App"

        def get_permissions(self):
            return []

        def get_activities(self):
            return []

        def get_services(self):
            return []

        def get_receivers(self):
            return []

        def get_providers(self):
            return []

        def get_features(self):
            return []

        def get_min_sdk_version(self):
            return "21"

        def get_target_sdk_version(self):
            return "34"

        def is_debuggable(self):
            return False

        def get_android_manifest_xml(self):
            return ET.fromstring('<manifest package="fallback.package"><application /></manifest>')

    androguard_mod = types.ModuleType("androguard")
    core_mod = types.ModuleType("androguard.core")
    apk_mod = types.ModuleType("androguard.core.apk")
    apk_mod.APK = FakeAPK
    monkeypatch.setitem(sys.modules, "androguard", androguard_mod)
    monkeypatch.setitem(sys.modules, "androguard.core", core_mod)
    monkeypatch.setitem(sys.modules, "androguard.core.apk", apk_mod)

    apk_path = tmp_path / "sample.apk"
    apk_path.write_bytes(b"not-a-real-apk")
    record = extract_manifest_record(apk_path, sid="abc").to_json()
    assert record["package_name"] == "com.example.app"


def _prepare_local_aeg_experiment_configs(tmp_path: Path) -> Path:
    src = Path("config/experiments/aeg_robust")
    dst = tmp_path / "aeg_robust"
    shutil.copytree(src, dst)
    return dst


def test_loss_modes_are_supported():
    payload = _payload()
    batch = Batch.from_data_list(
        [
            payload_to_data(payload, label=1),
            payload_to_data({**payload, "sid": "b" * 64, "sha256": "b" * 64}, label=0),
        ]
    )
    model = AEGModel(node_input_dim=batch.x.size(1), hidden_dim=16, layers=1, num_latents=2)
    clean_logits, clean_extra = model(batch)
    aug = Batch.from_data_list(
        [apply_aeg_view(item, view="api_degraded", strength=0.5) for item in batch.to_data_list()]
    )
    aug_logits, aug_extra = model(aug)

    for mode in ["ce_only", "plain_kl", "compact_kl"]:
        loss, parts = compute_aeg_loss(
            clean_logits,
            batch.y.view(-1),
            clean_extra,
            aug_logits=aug_logits,
            aug_extra=aug_extra,
            loss_cfg={"mode": mode, "ce_weight": 1.0, "consistency_weight": 0.05},
        )
        assert torch.isfinite(loss)
        assert parts["loss_mode"] == mode


def test_kl_modes_require_augmented_view():
    payload = _payload()
    batch = Batch.from_data_list([payload_to_data(payload, label=1)])
    model = AEGModel(node_input_dim=batch.x.size(1), hidden_dim=16, layers=1, num_latents=2)
    clean_logits, clean_extra = model(batch)

    with pytest.raises(ValueError, match="requires augmented logits"):
        compute_aeg_loss(
            clean_logits,
            batch.y.view(-1),
            clean_extra,
            loss_cfg={"mode": "compact_kl"},
        )


def test_aeg_config_loads(tmp_path: Path):
    cfg_root = _prepare_local_aeg_experiment_configs(tmp_path)
    cfg = load_config(cfg_root / "main/full_compact_kl_seed42.yaml")
    assert cfg["loss"]["mode"] == "compact_kl"
    assert float(cfg["loss"]["consistency_weight"]) > 0.0
    assert cfg["robust"]["train_aug"] is True


def test_all_aeg_experiment_configs_load_and_have_unique_outputs(tmp_path: Path):
    cfg_root = _prepare_local_aeg_experiment_configs(tmp_path)
    outputs = set()
    for path in sorted(p for p in cfg_root.rglob("*.yaml") if not p.name.endswith(".example.yaml")):
        cfg = load_config(path)
        output = str(cfg["train"]["output_dir"])
        assert output not in outputs, f"duplicate output_dir: {output}"
        outputs.add(output)
        assert cfg["robust"]["manifest_donor_mode"] == "cyclic"
        assert int(cfg["eval"]["seed"]) == 2026


def test_aeg_configs_require_tracked_base_yaml(tmp_path: Path):
    cfg_root = _prepare_local_aeg_experiment_configs(tmp_path)
    base_path = cfg_root / "base.yaml"
    assert base_path.exists()

    cfg = load_config(cfg_root / "main/full_compact_kl_seed42.yaml")
    assert cfg["train"]["output_dir"] == "results/aeg_robust/main/full_compact_kl_seed42"
    assert cfg["loss"]["mode"] == "compact_kl"


def test_diagnostic_slice_summarizer(tmp_path: Path):
    from scripts.summarize_aeg_diagnostics import run

    input_dir = tmp_path / "run"
    input_dir.mkdir()
    path = input_dir / "diagnostics_test_clean.csv"
    fields = [
        "label",
        "pred",
        "prob_malware",
        "year",
        "manifest_parse_ok",
        "dex_success_ratio",
        "multi_dex_total",
        "has_reflection",
        "has_dynamic_loading",
        "has_native",
        "has_string_encryption_hint",
        "code_reliability",
        "manifest_reliability",
        "code_manifest_conflict",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "label": 1,
                "pred": 1,
                "prob_malware": 0.9,
                "year": 2024,
                "manifest_parse_ok": 1,
                "dex_success_ratio": 0.5,
                "multi_dex_total": 2,
                "has_reflection": 1,
                "has_dynamic_loading": 0,
                "has_native": 0,
                "has_string_encryption_hint": 0,
                "code_reliability": 0.4,
                "manifest_reliability": 0.8,
                "code_manifest_conflict": 0.7,
            }
        )
    output = input_dir / "slice_metrics.csv"
    run(input_dir, output, min_count=1)
    text = output.read_text(encoding="utf-8")
    assert "reflection_hint" in text
    assert "dex_partial_failed" in text


def test_diagnostic_slice_summarizer_handles_empty_selected_slices(tmp_path: Path):
    from scripts.summarize_aeg_diagnostics import run

    input_dir = tmp_path / "run"
    input_dir.mkdir()
    path = input_dir / "diagnostics_test_clean.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "pred", "prob_malware"])
        writer.writeheader()
        writer.writerow({"label": 1, "pred": 1, "prob_malware": 0.9})
    output = input_dir / "slice_metrics.csv"
    run(input_dir, output, min_count=10)
    text = output.read_text(encoding="utf-8")
    assert text.startswith("scenario,slice,num_samples")


def test_run_groups_expose_current_experiments():
    from run import resolve_target_specs

    expected = {
        "main": 1,
        "loss": 5,
        "r1_graph": 7,
        "r3_fusion": 3,
        "all": 14,
    }

    for group, minimum in expected.items():
        paths = resolve_target_specs([group])
        assert len(paths) >= minimum
        assert all(path.exists() for path in paths)


def test_build_model_uses_payload_node_width_over_yaml_hint():
    cfg = {"model": {"node_input_dim": 128, "hidden_dim": 16, "layers": 1, "num_latents": 2}}
    model = build_model(cfg, node_input_dim=519)
    assert model.input_proj.in_features == 519


def test_model_rejects_node_dim_mismatch_by_default():
    data = payload_to_data(_payload(), label=1)
    model = AEGModel(node_input_dim=data.x.size(1) + 1, hidden_dim=16, layers=1, num_latents=2)
    with pytest.raises(ValueError, match="node feature dimension mismatch"):
        model(Batch.from_data_list([data]))


def test_model_can_explicitly_adapt_node_dim_mismatch():
    data = payload_to_data(_payload(), label=1)
    model = AEGModel(
        node_input_dim=data.x.size(1) + 1,
        hidden_dim=16,
        layers=1,
        num_latents=2,
        allow_node_dim_adapt=True,
    )
    logits, extra = model(Batch.from_data_list([data]))
    assert logits.shape == (1, 2)
    assert torch.isfinite(extra["fused_emb"]).all()


def test_extract_behavior_hint_config_is_explicit_ablation():
    from scripts.build_aeg_pts_direct import _load_config, _parse_config

    base = _load_config(Path("config/extract/extract_aeg.yaml"))
    ablation = _load_config(Path("config/extract/extract_aeg_behavior_hints.yaml"))
    train_only = _load_config(Path("config/extract/extract_aeg_train_only.yaml"))
    assert base["graph"]["use_behavior_hints"] is False
    assert base["data"]["require_all_label_ids"] is True
    assert train_only["data"]["require_all_label_ids"] is True
    assert int(base["graph"]["max_methods_per_apk"]) > 0
    assert int(base["api"]["max_events_per_apk"]) > 0
    assert base["aeg"]["retain_intermediate_features"] is False
    assert ablation["graph"]["use_behavior_hints"] is True
    required_dim = int(ablation["graph"]["vocab_size"]) * 2 + 3 + 4
    assert int(ablation["aeg"]["node_feature_dim"]) >= required_dim

    broken = _load_config(Path("config/extract/extract_aeg.yaml"))
    broken["graph"]["use_behavior_hints"] = True
    broken["aeg"]["node_feature_dim"] = 128
    args = argparse.Namespace(workers=1, resume=False, rebuild_vocab=False)
    with pytest.raises(ValueError, match="use_behavior_hints=true"):
        _parse_config(broken, args)


def test_val_test_extraction_keeps_train_vocab_guard_path():
    from scripts.build_aeg_pts_direct import _load_config, _parse_config

    args = argparse.Namespace(workers=1, resume=False, rebuild_vocab=False)
    cfg = _parse_config(_load_config(Path("config/extract/extract_aeg_val_test.yaml")), args)
    assert cfg["splits"] == ["val", "test"]
    assert str(cfg["train_label_csv_for_vocab"]).replace("\\", "/").endswith("results/labels/train.csv")


def test_manifest_vocab_fingerprint_must_match_train_csv(tmp_path: Path):
    from fusion.constants import stable_table_hash
    from scripts.build_aeg_pts_direct import _validate_vocab_matches_train_csv

    train_ids = ["a" * 64, "b" * 64]
    csv_path = tmp_path / "train.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        for sid in train_ids:
            writer.writerow({"id": sid, "label": 1})

    cfg = {"train_label_csv_for_vocab": csv_path}
    bad_vocab = {
        "metadata": {
            "source_sample_count": 1,
            "source_id_fingerprint": "bad",
        }
    }
    with pytest.raises(ValueError, match="does not match the configured train CSV"):
        _validate_vocab_matches_train_csv(bad_vocab, cfg)

    good_vocab = {
        "metadata": {
            "source_sample_count": len(train_ids),
            "source_id_fingerprint": stable_table_hash(sorted(train_ids)),
        }
    }
    _validate_vocab_matches_train_csv(good_vocab, cfg)


def test_aeg_builder_index_merge_preserves_other_splits(tmp_path: Path):
    from scripts.build_aeg_pts_direct import _merge_index_rows, _write_index

    index_path = tmp_path / "aeg_pt_index.csv"
    _write_index(
        [
            {"split": "train", "sha256": "a", "apk_name": "a.apk", "apk_path": "a", "pt_path": "a.pt", "status": "ok", "reason": ""},
            {"split": "test", "sha256": "b", "apk_name": "b.apk", "apk_path": "b", "pt_path": "b.pt", "status": "ok", "reason": ""},
        ],
        index_path,
    )
    merged = _merge_index_rows(
        index_path,
        [{"split": "test", "sha256": "c", "apk_name": "c.apk", "apk_path": "c", "pt_path": "c.pt", "status": "ok", "reason": ""}],
        ["test"],
    )
    assert {(row["split"], row["sha256"]) for row in merged} == {("train", "a"), ("test", "c")}


def test_aeg_builder_filters_jobs_to_label_csv(tmp_path: Path):
    from scripts.build_aeg_pts_direct import _filter_jobs_to_labels

    label_csv = tmp_path / "test.csv"
    with label_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerows([{"id": "a", "label": 0}, {"id": "b", "label": 1}])
    jobs = [
        {"split": "test", "sha256": "a", "apk_name": "a.apk", "apk_path": "a.apk"},
        {"split": "test", "sha256": "c", "apk_name": "c.apk", "apk_path": "c.apk"},
    ]
    filtered, ignored = _filter_jobs_to_labels(
        jobs,
        {"test": label_csv},
        ["test"],
        require_all_label_ids=False,
    )
    assert [job["sha256"] for job in filtered] == ["a"]
    assert [job["sha256"] for job in ignored] == ["c"]
    with pytest.raises(RuntimeError, match="do not cover all label CSV ids"):
        _filter_jobs_to_labels(
            jobs,
            {"test": label_csv},
            ["test"],
            require_all_label_ids=True,
        )


def test_vocab_only_automatically_scans_train_only():
    from scripts.build_aeg_pts_direct import _load_config, _parse_config

    raw = _load_config(Path("config/extract/extract_aeg.yaml"))
    args = argparse.Namespace(workers=1, resume=False, rebuild_vocab=True, vocab_only=True)
    cfg = _parse_config(raw, args)
    assert cfg["splits"] == ["train"]
    assert set(cfg["split_dirs"]) == {"train"}
    assert set(cfg["label_csvs"]) == {"train"}


def test_aeg_apk_hash_cache_reuses_unchanged_files(tmp_path: Path):
    from scripts.build_aeg_pts_direct import _collect_apks

    apk_dir = tmp_path / "train"
    apk_dir.mkdir()
    apk = apk_dir / "sample.apk"
    apk.write_bytes(b"first")
    cache_path = tmp_path / "hash_cache.csv"

    first = _collect_apks({"train": apk_dir}, ["train"], hash_cache_path=cache_path)
    second = _collect_apks({"train": apk_dir}, ["train"], hash_cache_path=cache_path)
    assert first[0]["hash_cache_hit"] is False
    assert second[0]["hash_cache_hit"] is True
    assert first[0]["sha256"] == second[0]["sha256"]

    apk.write_bytes(b"changed-content")
    third = _collect_apks({"train": apk_dir}, ["train"], hash_cache_path=cache_path)
    assert third[0]["hash_cache_hit"] is False
    assert third[0]["sha256"] != first[0]["sha256"]


def test_aeg_apk_container_preflight_classifies_invalid_files(tmp_path: Path):
    import zipfile

    from scripts.build_aeg_pts_direct import _scan_invalid_apk_containers

    apk_dir = tmp_path / "train"
    apk_dir.mkdir()
    (apk_dir / "empty.apk").write_bytes(b"")
    (apk_dir / "invalid.apk").write_bytes(b"\0" * 64)
    with zipfile.ZipFile(apk_dir / "valid.apk", "w") as archive:
        archive.writestr("AndroidManifest.xml", "manifest")

    invalid = _scan_invalid_apk_containers({"train": apk_dir}, ["train"])
    assert {(row["apk_name"], row["status"]) for row in invalid} == {
        ("empty.apk", "zero_byte"),
        ("invalid.apk", "non_zip_content"),
    }


def test_invalid_extra_apk_is_not_required_by_labels(tmp_path: Path):
    from scripts.build_aeg_pts_direct import _invalid_apks_required_by_labels

    label_csv = tmp_path / "train.csv"
    with label_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerow({"id": "required", "label": 0})
    rows = [
        {"split": "train", "filename_id": "required", "status": "zero_byte"},
        {"split": "train", "filename_id": "extra", "status": "zero_byte"},
    ]
    blocking = _invalid_apks_required_by_labels(rows, {"train": label_csv})
    assert [row["filename_id"] for row in blocking] == ["required"]


def test_aeg_apk_scan_index_marks_zero_and_content_mismatch(tmp_path: Path):
    from scripts.build_aeg_pts_direct import _write_apk_scan_index

    label_csv = tmp_path / "train.csv"
    with label_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerow({"id": "expected", "label": 0})
    report = tmp_path / "scan.csv"
    _write_apk_scan_index(
        [
            {
                "split": "train",
                "apk_name": "expected.apk",
                "apk_path": "expected.apk",
                "filename_id": "expected",
                "sha256": "different",
                "size_bytes": 10,
            },
            {
                "split": "train",
                "apk_name": "empty.apk",
                "apk_path": "empty.apk",
                "filename_id": "empty",
                "sha256": "e3b0",
                "size_bytes": 0,
            },
        ],
        {"train": label_csv},
        report,
    )
    rows = list(csv.DictReader(report.open("r", encoding="utf-8")))
    assert [row["status"] for row in rows] == ["content_hash_mismatch", "zero_byte"]


def test_aeg_payload_omits_large_intermediate_features_by_default():
    payload = _payload()
    for key in (
        "call_x",
        "api_ids",
        "method_names",
        "api_tokens",
        "manifest_x",
        "method_nodes",
        "method_api_edges",
        "method_call_edges",
    ):
        assert key not in payload
    assert "node_x" in payload
    assert "manifest_category_counts" in payload


def test_method_feature_compression_preserves_structural_stats():
    feature = torch.cat([torch.arange(512, dtype=torch.float32), torch.tensor([0.25, 0.5, 0.75])])
    compressed = _compress_method_feature(feature, 128)
    assert compressed.shape == (128,)
    assert torch.equal(compressed[-3:], feature[-3:])
    assert not torch.equal(compressed, feature[:128])


def test_quality_is_not_duplicated_inside_node_content_features():
    payload = _payload()
    node_x = payload["node_x"].float()
    node_type = payload["node_type"].long()
    apk = node_type == NODE_TYPES["APK"]
    api_family = node_type == NODE_TYPES["API_FAMILY"]
    permission = node_type == NODE_TYPES["PERMISSION"]
    assert torch.all(node_x[apk] == 0)
    assert torch.all(node_x[api_family, 2:] == 0)
    assert torch.all(node_x[permission, 1:] == 0)


def test_generated_payload_contract_validation():
    from scripts.build_aeg_pts_direct import _validate_payload_for_save

    payload = _payload()
    cfg = {"node_feature_dim": int(payload["node_x"].size(1))}
    job = {"sha256": payload["sid"]}
    _validate_payload_for_save(payload, cfg, job)
    broken = dict(payload)
    broken["edge_quality"] = torch.empty((0,), dtype=torch.float32)
    with pytest.raises(ValueError, match="edge_quality"):
        _validate_payload_for_save(broken, cfg, job)


def test_dataset_rejects_malformed_payload_instead_of_padding():
    payload = _payload()
    payload["edge_quality"] = payload["edge_quality"][:-1]
    with pytest.raises(AEGDatasetConfigError, match="edge_quality"):
        payload_to_data(payload, label=1)


def test_dataset_can_skip_redundant_payload_validation_after_preflight():
    payload = _payload()
    payload["edge_quality"] = payload["edge_quality"][:-1]
    data = payload_to_data(payload, label=1, validate_payload=False)
    assert data.edge_quality.numel() + 1 == data.edge_index.size(1)


def test_payload_exposes_real_failure_slice_metadata():
    payload = _payload()
    payload["year"] = 2024
    payload["has_reflection"] = True
    payload["has_dynamic_loading"] = True
    data = payload_to_data(payload, label=1)
    assert data.year.item() == 2024
    assert data.has_reflection.item() == 1.0
    assert data.has_dynamic_loading.item() == 1.0
    assert data.multi_dex_total.item() == 1


def test_shared_payload_contract_accepts_generated_payload():
    validate_aeg_payload(_payload())


def test_compact_storage_round_trips_to_training_dtypes():
    record = _manifest_record()
    vocab = build_manifest_vocab([record], max_permissions=8, max_intents=8, max_features=4)
    manifest_payload = vectorize_manifest_record(record, vocab, manifest_dim=32)
    compact = build_aeg_payload(
        sid=record["sha256"],
        apk_name=record["apk_name"],
        split="train",
        dex_list=[
            {
                "call_x": torch.randn(2, 16),
                "call_edge_index": torch.tensor([[0], [1]], dtype=torch.long),
                "api_ids": torch.tensor([1], dtype=torch.long),
                "api_type_ids": torch.tensor([6], dtype=torch.long),
                "api_method_index": torch.tensor([0], dtype=torch.long),
                "api_in_graph_mask": torch.ones(1),
                "method_api_edge_index": torch.tensor([[0], [0]], dtype=torch.long),
                "call_method_names": ["a", "b"],
                "api_tokens": ["urlconnection"],
            }
        ],
        manifest_payload=manifest_payload,
        manifest_record=record,
        direct_meta={
            "dex_success_ratio": 1.0,
            "num_dex_total": 1,
            "num_dex_success": 1,
        },
        node_feature_dim=32,
        storage_dtype="float16",
    )
    validate_aeg_payload(compact)
    assert compact["node_x"].dtype == torch.float16
    assert compact["edge_index"].dtype == torch.int32
    data = payload_to_data(compact, label=1)
    assert data.x.dtype == torch.float32
    assert data.edge_index.dtype == torch.long


def test_package_name_overlap_across_splits_is_rejected(tmp_path: Path):
    payload_train = _payload()
    payload_val = _payload()
    payload_val["sid"] = "b" * 64
    payload_val["sha256"] = "b" * 64
    payload_train["package_name"] = "com.example.same"
    payload_val["package_name"] = "com.example.same"

    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()
    torch.save(payload_train, train_dir / f"{payload_train['sid']}.pt")
    torch.save(payload_val, val_dir / f"{payload_val['sid']}.pt")

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    for csv_path, payload, label in ((train_csv, payload_train, 1), (val_csv, payload_val, 0)):
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "label"])
            writer.writeheader()
            writer.writerow({"id": payload["sid"], "label": label})

    train_ds = AEGDataset(train_dir, train_csv, split="train")
    val_ds = AEGDataset(val_dir, val_csv, split="val")
    with pytest.raises(ValueError, match="Package name overlap"):
        _validate_split_isolation(train_ds, val_ds, check_package=True)


def test_run_rejects_contrastive_batch_size_one(tmp_path: Path):
    payload_train = _variant_payload("a" * 64, package_name="com.example.train", split="train")
    payload_val = _variant_payload("b" * 64, package_name="com.example.val", split="val")
    payload_test = _variant_payload("c" * 64, package_name="com.example.test", split="test")

    train_dir, train_csv = _write_split(tmp_path, "train", [(payload_train, 1)])
    val_dir, val_csv = _write_split(tmp_path, "val", [(payload_val, 0)])
    test_dir, test_csv = _write_split(tmp_path, "test", [(payload_test, 1)])

    cfg = {
        "data": {
            "strict_integrity": True,
            "enforce_package_isolation": True,
            "train": {"pt_dir": str(train_dir), "csv": str(train_csv)},
            "val": {"pt_dir": str(val_dir), "csv": str(val_csv)},
            "test": {"pt_dir": str(test_dir), "csv": str(test_csv)},
        },
        "train": {
            "output_dir": str(tmp_path / "run_batch1"),
            "batch_size": 1,
            "eval_batch_size": 1,
            "epochs": 1,
            "patience": 1,
            "num_workers": 0,
            "device": "cpu",
        },
        "model": {
            "node_input_dim": 32,
            "hidden_dim": 16,
            "layers": 1,
            "num_latents": 2,
            "dropout": 0.0,
            "num_classes": 2,
        },
        "loss": {
            "clean_degraded_contrast_weight": 0.1,
            "source_degraded_contrast_weight": 0.0,
            "cross_source_contrast_weight": 0.0,
        },
        "robust": {"train_aug": False},
        "eval": {"robust_eval": False},
    }

    with pytest.raises(ValueError, match="batch_size >= 2"):
        run(cfg)



def test_run_writes_failed_metadata_before_early_validation_errors(tmp_path: Path):
    payload_train = _variant_payload("a" * 64, package_name="com.example.train", split="train")
    payload_val = _variant_payload("b" * 64, package_name="com.example.val", split="val")
    payload_test = _variant_payload("c" * 64, package_name="com.example.test", split="test")

    train_dir, train_csv = _write_split(tmp_path, "train", [(payload_train, 1)])
    val_dir, val_csv = _write_split(tmp_path, "val", [(payload_val, 0)])
    test_dir, test_csv = _write_split(tmp_path, "test", [(payload_test, 1)])
    config_path = tmp_path / "config.yaml"
    config_path.write_text("train: {}\n", encoding="utf-8")
    output_dir = tmp_path / "run_failed_metadata"

    cfg = {
        "data": {
            "strict_integrity": True,
            "enforce_package_isolation": True,
            "train": {"pt_dir": str(train_dir), "csv": str(train_csv)},
            "val": {"pt_dir": str(val_dir), "csv": str(val_csv)},
            "test": {"pt_dir": str(test_dir), "csv": str(test_csv)},
        },
        "train": {
            "output_dir": str(output_dir),
            "batch_size": 1,
            "eval_batch_size": 1,
            "epochs": 1,
            "patience": 1,
            "num_workers": 0,
            "device": "cpu",
        },
        "model": {
            "node_input_dim": 32,
            "hidden_dim": 16,
            "layers": 1,
            "num_latents": 2,
            "dropout": 0.0,
            "num_classes": 2,
        },
        "loss": {
            "clean_degraded_contrast_weight": 0.1,
            "source_degraded_contrast_weight": 0.0,
            "cross_source_contrast_weight": 0.0,
        },
        "robust": {"train_aug": False},
        "eval": {"robust_eval": False},
    }

    with pytest.raises(ValueError, match="batch_size >= 2"):
        run(cfg, config_path=config_path)

    metadata = json.loads((output_dir / "experiment_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["config_path"] == str(config_path.resolve())
    assert metadata["output_dir"] == str(output_dir.resolve())
    assert metadata["error"]["type"] == "ValueError"
    assert "batch_size >= 2" in metadata["error"]["message"]



def test_run_writes_completed_metadata_and_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class FakeDataset:
        def __init__(self, split: str, sample_path: Path):
            self.split = split
            self.samples = [(sample_path, 1)]

        def __len__(self) -> int:
            return 1

    sample_path = tmp_path / "sample.pt"
    sample_path.write_bytes(b"placeholder")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("train: {}\n", encoding="utf-8")
    output_dir = tmp_path / "run_completed_metadata"
    datasets = {
        split: FakeDataset(split, sample_path) for split in ("train", "val", "test")
    }

    cfg = {
        "data": {
            "strict_integrity": True,
            "enforce_package_isolation": True,
            "train": {"pt_dir": str(tmp_path / "train"), "csv": str(tmp_path / "train.csv")},
            "val": {"pt_dir": str(tmp_path / "val"), "csv": str(tmp_path / "val.csv")},
            "test": {"pt_dir": str(tmp_path / "test"), "csv": str(tmp_path / "test.csv")},
        },
        "train": {
            "output_dir": str(output_dir),
            "batch_size": 1,
            "eval_batch_size": 1,
            "epochs": 1,
            "patience": 1,
            "num_workers": 0,
            "device": "cpu",
            "seed": 7,
        },
        "model": {
            "hidden_dim": 16,
            "layers": 1,
            "num_latents": 2,
            "dropout": 0.0,
            "num_classes": 2,
        },
        "loss": {
            "clean_degraded_contrast_weight": 0.0,
            "source_degraded_contrast_weight": 0.0,
            "cross_source_contrast_weight": 0.0,
        },
        "robust": {"train_aug": False},
        "eval": {"robust_eval": False},
    }

    def fake_make_dataset(_cfg: dict, split: str, **_kwargs):
        return datasets[split]

    def fake_loader(_cfg: dict, dataset: FakeDataset, *, train: bool):
        return {"split": dataset.split, "train": train}

    def fake_train_one_epoch(*_args, **_kwargs):
        return {"loss": 0.25}

    def fake_evaluate(_model, loader_obj, _device, *, split_name: str, batch_key: str = "clean", dump_rows: bool = False):
        metrics = {
            "macro_f1": 0.75 if split_name == "val" else 0.8,
            "ece": 0.1,
        }
        rows = []
        if dump_rows:
            rows = [
                {
                    "requested_view": "clean",
                    "effective_view": "clean",
                    "manifest_shuffle_fallback": 0,
                    "sid": f"{split_name}-{batch_key}",
                }
            ]
        return metrics, rows

    monkeypatch.setattr(train_module, "_make_dataset", fake_make_dataset)
    monkeypatch.setattr(train_module, "_validate_split_isolation", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "split_label_stats", lambda dataset: {"samples": len(dataset)})
    monkeypatch.setattr(train_module, "_loader", fake_loader)
    monkeypatch.setattr(train_module, "load_aeg_payload", lambda *args, **kwargs: {"node_x": torch.zeros(3, 4)})
    monkeypatch.setattr(train_module, "build_model", lambda _cfg, node_input_dim: torch.nn.Linear(node_input_dim, 2))
    monkeypatch.setattr(train_module, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(train_module, "evaluate", fake_evaluate)

    summary = run(cfg, config_path=config_path)

    metadata = json.loads((output_dir / "experiment_metadata.json").read_text(encoding="utf-8"))
    assert summary["best_epoch"] == 1
    assert metadata["status"] == "completed"
    assert metadata["config_path"] == str(config_path.resolve())
    assert metadata["output_dir"] == str(output_dir.resolve())
    assert metadata["seed"] == 7
    assert metadata["node_input_dim"] == 4
    assert metadata["datasets"]["train"] == {"samples": 1}
    assert metadata["outputs"]["checkpoint"] == "best.pt"
    assert metadata["outputs"]["history"] == "history.csv"
    assert metadata["outputs"]["summary"] == "summary.json"
    assert "diagnostics_val.csv" in metadata["outputs"]["diagnostics"]
    assert "diagnostics_test_clean.csv" in metadata["outputs"]["diagnostics"]
    assert metadata["results"]["test"]["macro_f1"] == 0.8
    assert metadata["shuffle_fallback"]["diagnostics_test_clean.csv"]["fallback_rate"] == 0.0
    assert (output_dir / "best.pt").exists()
    assert (output_dir / "history.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "diagnostics_val.csv").exists()
    assert (output_dir / "diagnostics_test_clean.csv").exists()



def test_run_rejects_zero_training_batches_with_drop_last(tmp_path: Path):
    payload_train = _variant_payload("a" * 64, package_name="com.example.train", split="train")
    payload_val = _variant_payload("b" * 64, package_name="com.example.val", split="val")
    payload_test = _variant_payload("c" * 64, package_name="com.example.test", split="test")

    train_dir, train_csv = _write_split(tmp_path, "train", [(payload_train, 1)])
    val_dir, val_csv = _write_split(tmp_path, "val", [(payload_val, 0)])
    test_dir, test_csv = _write_split(tmp_path, "test", [(payload_test, 1)])

    cfg = {
        "data": {
            "strict_integrity": True,
            "enforce_package_isolation": True,
            "train": {"pt_dir": str(train_dir), "csv": str(train_csv)},
            "val": {"pt_dir": str(val_dir), "csv": str(val_csv)},
            "test": {"pt_dir": str(test_dir), "csv": str(test_csv)},
        },
        "train": {
            "output_dir": str(tmp_path / "run_drop_last"),
            "batch_size": 2,
            "eval_batch_size": 1,
            "epochs": 1,
            "patience": 1,
            "num_workers": 0,
            "device": "cpu",
        },
        "model": {
            "node_input_dim": 32,
            "hidden_dim": 16,
            "layers": 1,
            "num_latents": 2,
            "dropout": 0.0,
            "num_classes": 2,
        },
        "loss": {
            "clean_degraded_contrast_weight": 0.1,
            "source_degraded_contrast_weight": 0.0,
            "cross_source_contrast_weight": 0.0,
        },
        "robust": {"train_aug": False},
        "eval": {"robust_eval": False},
    }

    with pytest.raises(ValueError, match="would produce 0 training batches"):
        run(cfg)


def test_load_checkpoint_round_trips_saved_training_artifact(tmp_path: Path):
    checkpoint_path = tmp_path / "best.pt"
    payload = {
        "model": {"weight": torch.tensor([1.0])},
        "cfg": {"train": {"device": "cpu"}},
        "node_input_dim": 32,
        "aeg_payload_contract_fingerprint": "contract",
    }
    torch.save(payload, checkpoint_path)

    loaded = load_checkpoint(checkpoint_path, map_location="cpu")
    assert loaded["node_input_dim"] == 32
    assert torch.equal(loaded["model"]["weight"], torch.tensor([1.0]))


def test_load_aeg_payload_retries_without_mmap_for_compatibility(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pt_path = tmp_path / "sample.pt"
    pt_path.write_bytes(b"placeholder")
    expected = _payload()
    calls: list[dict[str, object]] = []

    def fake_load(path: str, **kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise TypeError("torch.load() got an unexpected keyword argument 'mmap'")
        return expected

    monkeypatch.setattr(torch, "load", fake_load)
    loaded = load_aeg_payload(pt_path, validate=False)
    assert loaded["sid"] == expected["sid"]
    assert len(calls) == 2
    assert calls[0]["weights_only"] is True
    assert calls[0]["mmap"] is True
    assert calls[1]["weights_only"] is True
    assert "mmap" not in calls[1]



def test_load_aeg_payload_errors_when_weights_only_is_unsupported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pt_path = tmp_path / "legacy_payload.pt"
    pt_path.write_bytes(b"placeholder")

    def fake_load(path: str, **kwargs):
        raise TypeError("torch.load() got an unexpected keyword argument 'weights_only'")

    monkeypatch.setattr(torch, "load", fake_load)
    with pytest.raises(RuntimeError, match="does not support weights_only safe loading"):
        load_aeg_payload(pt_path, validate=False)



def test_load_aeg_payload_fails_closed_on_weights_only_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pt_path = tmp_path / "unsafe.pt"
    pt_path.write_bytes(b"placeholder")

    def fake_load(path: str, **kwargs):
        raise RuntimeError(
            "Weights only load failed. Unsupported global: GLOBAL __main__.DangerousClass "
            "was not an allowed global by default."
        )

    monkeypatch.setattr(torch, "load", fake_load)
    with pytest.raises(RuntimeError, match="Weights only load failed"):
        load_aeg_payload(pt_path, validate=False)


def test_load_checkpoint_falls_back_to_legacy_for_weights_only_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    checkpoint_path = tmp_path / "legacy.pt"
    checkpoint_path.write_bytes(b"placeholder")
    calls: list[dict[str, object]] = []

    def fake_load(path: str, **kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise RuntimeError(
                "Weights only load failed. Unsupported global: GLOBAL __main__.LegacyCheckpoint "
                "was not an allowed global by default."
            )
        return {"model": {}, "cfg": {}, "node_input_dim": 32}

    monkeypatch.setattr(torch, "load", fake_load)
    with caplog.at_level(logging.WARNING):
        loaded = load_checkpoint(checkpoint_path, map_location="cpu")
    assert loaded["node_input_dim"] == 32
    assert len(calls) == 2
    assert calls[0]["weights_only"] is True
    assert calls[0]["mmap"] is True
    assert calls[1] == {"map_location": "cpu"}
    assert "Falling back to legacy torch.load for checkpoint legacy.pt" in caplog.text



def test_load_checkpoint_falls_back_to_legacy_when_weights_only_is_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    checkpoint_path = tmp_path / "old_torch.pt"
    checkpoint_path.write_bytes(b"placeholder")
    calls: list[dict[str, object]] = []

    def fake_load(path: str, **kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise TypeError("torch.load() got an unexpected keyword argument 'weights_only'")
        return {"model": {}, "cfg": {}, "node_input_dim": 24}

    monkeypatch.setattr(torch, "load", fake_load)
    with caplog.at_level(logging.WARNING):
        loaded = load_checkpoint(checkpoint_path, map_location="cpu")
    assert loaded["node_input_dim"] == 24
    assert len(calls) == 2
    assert calls[0]["weights_only"] is True
    assert calls[0]["mmap"] is True
    assert calls[1] == {"map_location": "cpu"}
    assert "Falling back to legacy torch.load for checkpoint old_torch.pt" in caplog.text



def test_load_checkpoint_retries_without_mmap_before_legacy_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    checkpoint_path = tmp_path / "compat.pt"
    checkpoint_path.write_bytes(b"placeholder")
    calls: list[dict[str, object]] = []

    def fake_load(path: str, **kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise TypeError("torch.load() got an unexpected keyword argument 'mmap'")
        if len(calls) == 2:
            raise RuntimeError(
                "Weights only load failed. Unsupported global: GLOBAL __main__.LegacyCheckpoint "
                "was not an allowed global by default."
            )
        return {"model": {}, "cfg": {}, "node_input_dim": 16}

    monkeypatch.setattr(torch, "load", fake_load)
    with caplog.at_level(logging.WARNING):
        loaded = load_checkpoint(checkpoint_path, map_location="cpu")
    assert loaded["node_input_dim"] == 16
    assert len(calls) == 3
    assert calls[0]["weights_only"] is True and calls[0]["mmap"] is True
    assert calls[1]["weights_only"] is True and "mmap" not in calls[1]
    assert calls[2] == {"map_location": "cpu"}
    assert "Falling back to legacy torch.load for checkpoint compat.pt" in caplog.text


def test_validate_aeg_pt_reports_schema_and_node_dimensions(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    from scripts import validate_aeg_pts as script

    train_payload = _variant_payload("a" * 64, package_name="com.example.train", split="train")
    val_payload = _variant_payload("b" * 64, package_name="com.example.val", split="val")

    train_dir, train_csv = _write_split(tmp_path, "train", [(train_payload, 1)])
    val_dir, val_csv = _write_split(tmp_path, "val", [(val_payload, 0)])
    config_path = tmp_path / "extract.yaml"
    config_path.write_text("dummy: true\n", encoding="utf-8")

    monkeypatch.setattr(
        script,
        "_load_config",
        lambda _path: {"dummy": True},
    )
    monkeypatch.setattr(
        script,
        "_parse_config",
        lambda _raw_cfg, _args: {
            "splits": ["train", "val"],
            "out_dirs": {"train": train_dir, "val": val_dir},
            "label_csvs": {"train": train_csv, "val": val_csv},
            "node_feature_dim": 32,
        },
    )
    monkeypatch.setattr(sys, "argv", ["validate_aeg_pts.py", "--config", str(config_path), "--all"])

    code = script.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "Node feature dims observed:" in out
    assert "dim=32: 2" in out
    assert "Schema versions observed:" in out
    assert str(AEG_SCHEMA_VERSION) in out
    assert "RESULT: PASS" in out
