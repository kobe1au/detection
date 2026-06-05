from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fusion.aeg_builder import build_aeg_payload
from fusion.constants import AEG_SCHEMA_VERSION, EDGE_TYPES, NODE_TYPES, VIEW_TYPES
from fusion.dataset import AEGDataset, aeg_collate_fn, payload_to_data
from fusion.losses import _source_contrast_weights, compute_aeg_loss
from fusion.manifest_features import build_manifest_vocab, vectorize_manifest_record
from fusion.model import AEGModel
from fusion.perturbations import apply_aeg_view
from fusion.train import load_config


def _manifest_record() -> dict:
    return {
        "sid": "a" * 64,
        "sha256": "a" * 64,
        "apk_name": "sample.apk",
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
        direct_meta={"dex_success_ratio": 1.0, "num_dex_total": 1, "num_dex_success": 1},
        node_feature_dim=32,
    )


def test_aeg_builder_dataset_model_loss(tmp_path: Path):
    payload = _payload()
    assert payload["schema_version"] == AEG_SCHEMA_VERSION
    assert payload["node_x"].ndim == 2
    assert payload["edge_index"].size(0) == 2

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
    assert degraded.q_api.item() == 0.0
    assert degraded.q_align.item() == 0.0
    api_related = torch.isin(
        degraded.edge_type,
        torch.tensor(
            [
                EDGE_TYPES["METHOD_INVOKES_API_FAMILY"],
                EDGE_TYPES["API_FAMILY_INVOKED_BY_METHOD"],
                EDGE_TYPES["PERMISSION_RELATED_TO_API_FAMILY"],
                EDGE_TYPES["API_FAMILY_RELATED_TO_PERMISSION"],
            ],
            dtype=torch.long,
        ),
    )
    assert bool(api_related.any())
    assert torch.all(degraded.edge_quality[api_related] == 0)


def test_manifest_shuffle_is_type_aware(tmp_path: Path):
    payload_a = _payload()
    payload_b = _payload()
    payload_b["sid"] = "b" * 64
    payload_b["sha256"] = "b" * 64
    data_a = payload_to_data(payload_a, label=1)
    data_b = payload_to_data(payload_b, label=0)
    aug = apply_aeg_view(data_a, view="manifest_shuffled", strength=1.0)
    before_type = aug.node_type.clone()
    batch = aeg_collate_fn([{"clean": data_a, "aug": aug}, {"clean": data_b, "aug": apply_aeg_view(data_b, view="clean", strength=0.0)}])
    shuffled = batch["aug"].to_data_list()[0]
    assert torch.equal(shuffled.node_type, before_type)
    manifest_types = {NODE_TYPES["PERMISSION"], NODE_TYPES["INTENT"], NODE_TYPES["COMPONENT"]}
    for node_type in manifest_types:
        mask = (shuffled.node_source == 1) & (shuffled.node_type == node_type)
        if bool(mask.any()):
            assert torch.all(shuffled.node_type[mask] == node_type)


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


def test_aeg_config_loads():
    cfg = load_config("config/experiments/aeg_robust/full/ours.yaml")
    assert cfg["model"]["hidden_dim"] == 128
    assert cfg["loss"]["clean_degraded_contrast_weight"] > 0.0
