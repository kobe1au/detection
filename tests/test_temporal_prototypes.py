from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion.prototypes import TemporalPrototypeMemory


def test_forecast_next_prototypes_does_not_mutate_memory():
    memory = TemporalPrototypeMemory(
        num_domains=3,
        num_classes=2,
        feature_dim=3,
        momentum=0.5,
        num_clusters=2,
    )

    feats = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])
    time_ids = torch.tensor([0, 1, 0, 1])
    memory.update_weighted(feats, labels, time_ids)

    before = {
        "prototypes": memory.prototypes.clone(),
        "seen": memory.seen.clone(),
        "spread": memory.spread.clone(),
    }
    if hasattr(memory, "initialized"):
        before["initialized"] = memory.initialized.clone()
    if hasattr(memory, "inherited"):
        before["inherited"] = memory.inherited.clone()

    future, valid = memory.forecast_next_prototypes(torch.tensor([1, 2]))
    assert future.shape == (2, 2, 2, 3)
    assert valid.shape == (2, 2, 2)

    for name, tensor in before.items():
        assert torch.equal(getattr(memory, name), tensor), name


def test_future_year_forecast_uses_history_without_observing_future_year():
    memory = TemporalPrototypeMemory(
        num_domains=3,
        num_classes=1,
        feature_dim=2,
        momentum=0.0,
        num_clusters=1,
    )

    memory.update_weighted(
        feats=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        labels=torch.tensor([0, 0]),
        time_ids=torch.tensor([0, 1]),
    )

    future, valid = memory.forecast_next_prototypes(torch.tensor([2]))
    assert future.shape == (1, 1, 1, 2)
    assert bool(valid[0, 0, 0].item())
    assert int(memory.seen[2, 0, 0].item()) == 0
    if hasattr(memory, "initialized"):
        assert not bool(memory.initialized[2, 0, 0].item())


def test_inherited_initialization_does_not_add_fake_seen_count_when_supported():
    memory = TemporalPrototypeMemory(
        num_domains=2,
        num_classes=1,
        feature_dim=2,
        momentum=0.0,
        num_clusters=1,
    )

    if not hasattr(memory, "initialized") or not hasattr(memory, "inherited"):
        raise AssertionError("TemporalPrototypeMemory should separate initialized/inherited from observed seen count")

    memory.update_weighted(
        feats=torch.tensor([[1.0, 0.0], [0.9, 0.1]]),
        labels=torch.tensor([0, 0]),
        time_ids=torch.tensor([0, 0]),
    )
    assert int(memory.seen[0, 0, 0].item()) == 2
    assert bool(memory.initialized[0, 0, 0].item())

    memory.update_weighted(
        feats=torch.tensor([[0.0, 1.0]]),
        labels=torch.tensor([0]),
        time_ids=torch.tensor([1]),
    )

    assert int(memory.seen[1, 0, 0].item()) == 1
    assert bool(memory.initialized[1, 0, 0].item())
    assert not bool(memory.inherited[1, 0, 0].item())
