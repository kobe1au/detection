import unittest
from unittest.mock import patch

import numpy as np

from fusion import train as train_mod


class TinyDataset:
    def __init__(self, labels, years=None):
        years = years if years is not None else [2023] * len(labels)
        self.samples = [
            (None, int(label), f"s{i}", int(year))
            for i, (label, year) in enumerate(zip(labels, years))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _cfg(**train_overrides):
    train = {
        "adaptation_ratio": 0.5,
        "replay_strategy": "static",
        "replay_budget_mode": "adapt_relative",
        "replay_budget_ratio": 0.5,
        "adaptation_selection": "random_class_balanced",
        "seed": 7,
    }
    train.update(train_overrides)
    return {"train": train, "model": {"num_classes": 2}}


class AdaptationSelectionAndReplayTest(unittest.TestCase):
    def test_replay_budget_selected_adapt_relative_and_alias(self):
        cfg = _cfg(replay_budget_mode="adapt_relative", replay_budget_ratio=0.5)
        self.assertEqual(train_mod._get_replay_budget_mode(cfg["train"]), "selected_adapt_relative")
        self.assertEqual(train_mod._compute_replay_target_count(cfg["train"], 300), 150)

    def test_replay_budget_historical_relative_keeps_legacy_semantics(self):
        hist = TinyDataset([0, 1] * 50)
        cfg = _cfg(replay_budget_mode="historical_relative", replay_budget_ratio=0.25)
        self.assertEqual(
            train_mod._compute_replay_target_count(cfg["train"], 300, historical_dataset=hist),
            25,
        )

    def test_replay_budget_rejects_conflicting_legacy_ratio(self):
        cfg = _cfg(replay_budget_ratio=0.5)
        cfg["train"]["replay_ratio"] = 0.25
        with self.assertRaises(ValueError):
            train_mod._get_replay_budget_ratio(cfg["train"])

    def test_balanced_target_sampling_returns_exact_target_for_small_budget(self):
        dataset = TinyDataset(
            labels=[0, 1, 0, 1, 0, 1],
            years=[2020, 2020, 2021, 2021, 2022, 2022],
        )
        selected = train_mod._balanced_sample_indices_to_target(
            dataset,
            target_count=2,
            seed=3,
            min_per_group=1,
            group_by_year=True,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(set(selected)), 2)
        self.assertTrue(all(0 <= idx < len(dataset) for idx in selected))

    def test_balanced_target_sampling_preserves_minority_when_budget_allows(self):
        dataset = TinyDataset(labels=[0] * 9 + [1])
        selected = train_mod._balanced_sample_indices_to_target(
            dataset,
            target_count=5,
            seed=5,
            min_per_group=1,
            group_by_year=False,
        )
        labels = [dataset.samples[idx][1] for idx in selected]
        self.assertEqual(len(selected), 5)
        self.assertIn(1, labels)

    def test_build_adaptation_loader_dispatches_random_pure(self):
        hist = TinyDataset([0, 1])
        adapt = TinyDataset([0, 0, 1, 1])
        cfg = _cfg(adaptation_selection="random_pure")

        with patch.object(train_mod, "_sample_random_indices", return_value=[1, 3]) as pure, \
             patch.object(train_mod, "_balanced_sample_indices") as balanced, \
             patch.object(train_mod, "_build_continual_adaptation_loader", return_value=("loader", 2, 1)) as build:
            result = train_mod.build_adaptation_loader(hist, adapt, cfg, 2, {})

        self.assertEqual(result, ("loader", 2, 1))
        pure.assert_called_once()
        balanced.assert_not_called()
        self.assertEqual(build.call_args.kwargs["adaptation_indices"], [1, 3])

    def test_build_adaptation_loader_dispatches_random_alias_to_class_balanced(self):
        hist = TinyDataset([0, 1])
        adapt = TinyDataset([0, 0, 1, 1])
        cfg = _cfg(adaptation_selection="random")

        with patch.object(train_mod, "_sample_random_indices") as pure, \
             patch.object(train_mod, "_balanced_sample_indices", return_value=[0, 2]) as balanced, \
             patch.object(train_mod, "_build_continual_adaptation_loader", return_value=("loader", 2, 1)) as build:
            result = train_mod.build_adaptation_loader(hist, adapt, cfg, 2, {})

        self.assertEqual(result, ("loader", 2, 1))
        pure.assert_not_called()
        balanced.assert_called_once()
        self.assertEqual(build.call_args.kwargs["adaptation_indices"], [0, 2])

    def test_dynamic_replay_uses_full_selected_adapt_subset_and_budgeted_replay(self):
        hist = TinyDataset(
            labels=[0, 1, 0, 1, 0, 1],
            years=[2020, 2020, 2021, 2021, 2022, 2022],
        )
        adapt = TinyDataset([0, 1, 0, 1])
        cfg = _cfg(replay_strategy="dynamic_year_class", replay_budget_ratio=0.5)

        loader, n_adapt, n_replay = train_mod._build_continual_adaptation_loader(
            hist,
            adapt,
            cfg,
            batch_size=2,
            loader_kwargs={},
            adaptation_indices=[0, 1, 2, 3],
        )

        sampled = list(loader.sampler)
        adapt_positions = [idx for idx in sampled if idx < n_adapt]
        replay_positions = [idx for idx in sampled if idx >= n_adapt]
        self.assertEqual(n_adapt, 4)
        self.assertEqual(n_replay, 2)
        self.assertCountEqual(adapt_positions, [0, 1, 2, 3])
        self.assertEqual(len(replay_positions), 2)

    def test_drift_matched_replay_uses_selected_adapt_relative_target(self):
        hist = TinyDataset(
            labels=[0, 1] * 6,
            years=[2020, 2020, 2021, 2021, 2022, 2022] * 2,
        )
        hist_records = [
            {"dataset_index": i, "embedding": np.asarray([float(i), 1.0], dtype=np.float32)}
            for i in range(len(hist))
        ]
        selected_adapt_records = [
            {"dataset_index": i, "embedding": np.asarray([float(i), 1.0], dtype=np.float32)}
            for i in range(4)
        ]
        cfg = _cfg(replay_strategy="drift_matched", replay_budget_ratio=0.5)

        replay_indices = train_mod._build_drift_matched_replay_indices(
            hist,
            hist_records,
            selected_adapt_records,
            cfg,
        )

        self.assertEqual(len(replay_indices), 2)
        self.assertEqual(len(set(replay_indices)), 2)

    def test_dbta_drift_first_then_representative_selection(self):
        embeddings = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.99],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
        uncertainty = [1.0, 0.9, 0.8, 0.1, 0.1, 0.1]
        records = [
            {
                "dataset_index": i,
                "pred_label": 0,
                "embedding": np.asarray(emb, dtype=np.float32),
                "uncertainty": uncertainty[i],
                "branch_disagreement": 0.0,
                "prototype_distance": 0.0,
            }
            for i, emb in enumerate(embeddings)
        ]
        cfg = _cfg(
            adaptation_ratio=0.34,
            dbta_balance="none",
            dbta_candidate_top_p=0.5,
            dbta_representative_k=2,
            dbta_representative_weight=1.0,
            dbta_diversity_weight=0.0,
            dbta_selection_mode="diversity_aware",
        )

        selected = train_mod._score_and_select_dbta_records(records, cfg)
        selected_ids = {int(record["dataset_index"]) for record in selected}

        self.assertEqual(len(selected), 2)
        self.assertTrue(selected_ids.issubset({0, 1, 2}))
        self.assertNotIn(0, selected_ids)
        for record in selected:
            self.assertIn("selection_score", record)
            self.assertIn("representativeness_score", record)
            self.assertIn("density_bucket", record)


if __name__ == "__main__":
    unittest.main()
