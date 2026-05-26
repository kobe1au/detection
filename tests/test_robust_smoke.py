from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from fusion.robust.dataset import RobustTriModalDataset, robust_collate_fn
from fusion.robust.losses import compute_robust_loss
from fusion.robust.model import TriModalRobustModel


def test_robust_model_forward_and_loss():
    items = []
    for i in range(2):
        data = Data(
            x=torch.randn(4, 16),
            edge_index=torch.tensor([[0, 1, 2, 2], [1, 2, 3, 0]], dtype=torch.long),
            y=torch.tensor(i % 2),
        )
        data.sensitive_mask = torch.zeros(4, dtype=torch.uint8)
        items.append(data)

    batch = Batch.from_data_list(items)
    batch.api_ids = torch.randint(1, 32, (12,), dtype=torch.long)
    batch.api_type_ids = torch.randint(0, 4, (12,), dtype=torch.long)
    batch.api_sensitive_mask = torch.zeros(12)
    batch.api_batch = torch.cat([torch.full((6,), i, dtype=torch.long) for i in range(2)])
    batch.method_api_edge_index = torch.empty((2, 0), dtype=torch.long)
    batch.api_category_counts = torch.rand(2, 12)
    batch.graph_category_counts = torch.rand(2, 12)
    batch.manifest_x = torch.rand(2, 32)
    batch.manifest_category_counts = torch.rand(2, 12)
    batch.manifest_stats = torch.rand(2, 11)
    batch.q_api = torch.ones(2, 1)
    batch.q_graph = torch.ones(2, 1)
    batch.q_manifest = torch.ones(2, 1)
    batch.q_align = torch.ones(2, 1) * 0.8
    batch.pert_api = torch.zeros(2, 1)
    batch.pert_graph = torch.zeros(2, 1)
    batch.pert_manifest = torch.zeros(2, 1)

    model = TriModalRobustModel(
        in_feat_dim=16,
        fusion_mode="tri_modal_ours",
        api_num_hash_buckets=64,
        api_type_vocab_size=16,
        api_emb_dim=32,
        api_hidden_dim=64,
        api_layers=1,
        api_heads=4,
        api_max_seq_len=16,
        graph_emb_dim=32,
        graph_hidden=32,
        graph_heads=4,
        graph_layers=1,
        max_nodes_gnn=64,
        manifest_in_dim=32,
        manifest_emb_dim=32,
        manifest_hidden_dim=64,
        joint_emb_dim=32,
    )
    logits, extra = model(batch, return_features=True)
    loss, parts = compute_robust_loss(logits, torch.tensor([0, 1]), extra, {"branch_aux_weight": 0.05})
    assert logits.shape == (2, 2)
    assert extra["gate_weights"].shape == (2, 4)
    assert torch.isfinite(loss)
    assert parts["branch_aux_weight"] == 0.05


def test_robust_dataset_collate(tmp_path: Path):
    pt_dir = tmp_path / "pts"
    pt_dir.mkdir()
    sid = "sample1"
    torch.save(
        [
            {
                "call_x": torch.randn(3, 8),
                "call_edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                "call_sensitive_mask": torch.tensor([0, 1, 0], dtype=torch.uint8),
                "api_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "api_type_ids": torch.tensor([1, 2, 0], dtype=torch.long),
                "api_sensitive_mask": torch.tensor([1.0, 0.0, 0.0]),
                "api_method_index": torch.tensor([0, 1, 2], dtype=torch.long),
                "api_in_graph_mask": torch.tensor([1.0, 1.0, 1.0]),
                "method_api_edge_index": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
                "manifest_x": torch.ones(16),
                "manifest_category_counts": torch.ones(12),
                "manifest_stats": torch.ones(11),
                "q_manifest": torch.tensor([1.0]),
                "pert_manifest": torch.tensor([0.0]),
            }
        ],
        pt_dir / f"{sid}.pt",
    )
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "year"])
        writer.writeheader()
        writer.writerow({"id": sid, "label": 1, "year": 2024})

    dataset = RobustTriModalDataset(
        str(pt_dir),
        str(csv_path),
        is_train=True,
        manifest_dim=16,
        manifest_category_dim=12,
        manifest_stats_dim=11,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=robust_collate_fn)))
    graph = batch["graph_batch"]
    assert graph.manifest_x.shape == (1, 16)
    assert graph.q_manifest.shape == (1, 1)
    assert graph.api_ids.numel() == 3
