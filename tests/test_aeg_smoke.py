from __future__ import annotations

import csv
import argparse
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
from fusion.losses import _fused_contrast_weight, _source_contrast_weights, compute_aeg_loss
from fusion.manifest_features import build_manifest_vocab, extract_manifest_record, vectorize_manifest_record
from fusion.model import AEGModel, build_model
from fusion.perturbations import apply_aeg_view
from fusion.payload_contract import validate_aeg_payload
from fusion.train import _validate_split_isolation, load_config


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
            "aeg_build_fingerprint": "test-build",
        },
        node_feature_dim=32,
    )


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
    assert parts["clean_degraded_contrast"] >= 0.0
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
    assert abs(float(aug.q_manifest.item()) - 0.25) < 1e-6
    assert aug.pert_manifest.item() == 1.0
    _, extra = AEGModel(node_input_dim=aug.x.size(1), hidden_dim=16, layers=1, num_latents=2)(
        Batch.from_data_list([aug])
    )
    assert extra["r_manifest"].item() == 0.0
    assert extra["manifest_reliability"].item() == 0.0


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


def test_corrupted_source_contrast_weights_drop_untrusted_source():
    ref = torch.zeros((2, 2))
    clean = {
        "code_reliability": torch.ones(2),
        "manifest_reliability": torch.ones(2),
    }
    aug = {
        "code_reliability": torch.ones(2),
        "manifest_reliability": torch.ones(2),
        "view_type_id": torch.full((2,), VIEW_TYPES["manifest_shuffled"], dtype=torch.long),
    }
    code_weight, manifest_weight, risk_weight = _source_contrast_weights(clean, aug, ref)
    assert torch.all(code_weight == 1)
    assert torch.all(manifest_weight == 0)
    assert torch.all(risk_weight == 1)


def test_fused_contrast_weight_drops_when_all_sources_are_unreliable():
    ref = torch.zeros((2, 2))
    clean = {
        "code_reliability": torch.ones(2),
        "manifest_reliability": torch.ones(2),
    }
    aug = {
        "code_reliability": torch.tensor([0.0, 0.2]),
        "manifest_reliability": torch.tensor([0.0, 0.3]),
    }
    weight = _fused_contrast_weight(clean, aug, ref)
    assert torch.allclose(weight, torch.tensor([0.0, 0.3]))


def test_aeg_config_loads():
    cfg = load_config("config/experiments/aeg_robust/full/ours.yaml")
    assert cfg["model"]["hidden_dim"] == 128
    assert cfg["loss"]["clean_degraded_contrast_weight"] > 0.0


def test_all_aeg_experiment_configs_load_and_have_unique_outputs():
    outputs = set()
    for path in sorted(Path("config/experiments/aeg_robust").rglob("*.yaml")):
        cfg = load_config(path)
        output = str(cfg["train"]["output_dir"])
        assert output not in outputs, f"duplicate output_dir: {output}"
        outputs.add(output)
        assert cfg["robust"]["manifest_donor_mode"] == "cyclic"


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


def test_run_groups_expose_isolated_innovation_experiments():
    from run import resolve_target_specs

    for group, minimum in (("i1", 5), ("i2", 5), ("i3", 6), ("full_seeds", 3)):
        paths = resolve_target_specs([group])
        assert len(paths) >= minimum
        assert all(path.exists() for path in paths)


def test_build_model_uses_payload_node_width_over_yaml_hint():
    cfg = {"model": {"node_input_dim": 128, "hidden_dim": 16, "layers": 1, "num_latents": 2}}
    model = build_model(cfg, node_input_dim=519)
    assert model.input_proj.in_features == 519


def test_extract_behavior_hint_config_is_explicit_ablation():
    from scripts.build_aeg_pts_direct import _load_config, _parse_config

    base = _load_config(Path("config/extract_aeg.yaml"))
    ablation = _load_config(Path("config/extract_aeg_behavior_hints.yaml"))
    train_only = _load_config(Path("config/extract_aeg_train_only.yaml"))
    assert base["graph"]["use_behavior_hints"] is False
    assert base["data"]["require_all_label_ids"] is True
    assert train_only["data"]["require_all_label_ids"] is True
    assert int(base["graph"]["max_methods_per_apk"]) > 0
    assert int(base["api"]["max_events_per_apk"]) > 0
    assert base["aeg"]["retain_intermediate_features"] is False
    assert ablation["graph"]["use_behavior_hints"] is True
    required_dim = int(ablation["graph"]["vocab_size"]) * 2 + 3 + 4
    assert int(ablation["aeg"]["node_feature_dim"]) >= required_dim

    broken = _load_config(Path("config/extract_aeg.yaml"))
    broken["graph"]["use_behavior_hints"] = True
    broken["aeg"]["node_feature_dim"] = 128
    args = argparse.Namespace(workers=1, resume=False, rebuild_vocab=False)
    with pytest.raises(ValueError, match="use_behavior_hints=true"):
        _parse_config(broken, args)


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


def test_source_hash_is_stable_across_line_endings(tmp_path: Path):
    from scripts.build_aeg_pts_direct import _sha256_text_file

    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"a = 1\nb = 2\n")
    crlf.write_bytes(b"a = 1\r\nb = 2\r\n")
    assert _sha256_text_file(lf) == _sha256_text_file(crlf)


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
    payload["aeg_build_fingerprint"] = "build"
    cfg = {"node_feature_dim": int(payload["node_x"].size(1)), "build_fingerprint": "build"}
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
    base = _payload()
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
            "aeg_build_fingerprint": base["aeg_build_fingerprint"],
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


def test_mixed_build_fingerprints_across_splits_are_rejected(tmp_path: Path):
    payload_train = _payload()
    payload_val = _payload()
    payload_val["sid"] = "b" * 64
    payload_val["sha256"] = "b" * 64
    payload_val["package_name"] = "com.example.other"
    payload_val["aeg_build_fingerprint"] = "different-build"

    datasets = []
    for split, payload, label in (("train", payload_train, 1), ("val", payload_val, 0)):
        pt_dir = tmp_path / split
        pt_dir.mkdir()
        torch.save(payload, pt_dir / f"{payload['sid']}.pt")
        csv_path = tmp_path / f"{split}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "label"])
            writer.writeheader()
            writer.writerow({"id": payload["sid"], "label": label})
        datasets.append(AEGDataset(pt_dir, csv_path, split=split))

    with pytest.raises(ValueError, match="Mixed AEG build fingerprints"):
        _validate_split_isolation(*datasets, check_package=True)
