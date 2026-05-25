import csv
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fusion import train as train_mod
from scripts import make_ablation_configs


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
        "replay_budget_mode": "selected_adapt_relative",
        "replay_budget_ratio": 0.5,
        "adaptation_selection": "random_class_balanced",
        "seed": 7,
    }
    train.update(train_overrides)
    return {"train": train, "model": {"num_classes": 2}}


class AdaptationSelectionAndReplayTest(unittest.TestCase):
    def test_replay_budget_selected_adapt_relative_official_mode(self):
        cfg = _cfg(replay_budget_mode="selected_adapt_relative", replay_budget_ratio=0.5)
        self.assertEqual(train_mod._get_replay_budget_mode(cfg["train"]), "selected_adapt_relative")
        self.assertEqual(train_mod._compute_replay_target_count(cfg["train"], 300), 150)

    def test_replay_budget_rejects_removed_adapt_relative_alias(self):
        cfg = _cfg(replay_budget_mode="adapt_relative", replay_budget_ratio=0.5)
        with self.assertRaises(ValueError):
            train_mod._compute_replay_target_count(cfg["train"], 300)

    def test_config_schema_rejects_removed_replay_ratio(self):
        cfg = train_mod.deep_update(
            train_mod.load_yaml_file("config/base.yaml"),
            {"train": {"replay_ratio": 0.25}},
        )

        with self.assertRaises(ValueError):
            train_mod.validate_full_config(cfg)

    def test_config_schema_accepts_representative_scope_values(self):
        cfg = train_mod.deep_update(
            train_mod.load_yaml_file("config/base.yaml"),
            {"train": {"dbta_representative_scope": "candidate_pool"}},
        )
        train_mod.validate_full_config(cfg)

        cfg["train"]["dbta_representative_scope"] = "recent_pool"
        train_mod.validate_full_config(cfg)

    def test_config_schema_rejects_invalid_representative_scope(self):
        cfg = train_mod.deep_update(
            train_mod.load_yaml_file("config/base.yaml"),
            {"train": {"dbta_representative_scope": "batch"}},
        )
        with self.assertRaises(ValueError):
            train_mod.validate_full_config(cfg)

    def test_replay_budget_allows_only_selected_adapt_relative_mode(self):
        cfg = _cfg(replay_budget_mode="historical_relative", replay_budget_ratio=0.25)
        with self.assertRaises(ValueError):
            train_mod._compute_replay_target_count(cfg["train"], 300)

    def test_config_schema_rejects_removed_temporal_soft_weight(self):
        cfg = train_mod.deep_update(
            train_mod.load_yaml_file("config/base.yaml"),
            {"loss": {"alignment_use_temporal_soft_weight": False}},
        )

        with self.assertRaises(ValueError):
            train_mod.validate_full_config(cfg)

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

    def test_build_adaptation_loader_dispatches_random_class_balanced(self):
        hist = TinyDataset([0, 1])
        adapt = TinyDataset([0, 0, 1, 1])
        cfg = _cfg(adaptation_selection="random_class_balanced")

        with patch.object(train_mod, "_sample_random_indices") as pure, \
             patch.object(train_mod, "_balanced_sample_indices", return_value=[0, 2]) as balanced, \
             patch.object(train_mod, "_build_continual_adaptation_loader", return_value=("loader", 2, 1)) as build:
            result = train_mod.build_adaptation_loader(hist, adapt, cfg, 2, {})

        self.assertEqual(result, ("loader", 2, 1))
        pure.assert_not_called()
        balanced.assert_called_once()
        self.assertEqual(build.call_args.kwargs["adaptation_indices"], [0, 2])

    def test_build_adaptation_loader_rejects_random_alias(self):
        hist = TinyDataset([0, 1])
        adapt = TinyDataset([0, 0, 1, 1])
        cfg = _cfg(adaptation_selection="random")

        with self.assertRaises(ValueError):
            train_mod.build_adaptation_loader(hist, adapt, cfg, 2, {})

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

    def test_dynamic_replay_rejects_removed_dynamic_alias(self):
        hist = TinyDataset([0, 1, 0, 1])
        adapt = TinyDataset([0, 1])
        cfg = _cfg(replay_strategy="dynamic")

        with self.assertRaises(ValueError):
            train_mod._build_continual_adaptation_loader(
                hist,
                adapt,
                cfg,
                batch_size=2,
                loader_kwargs={},
                adaptation_indices=[0, 1],
            )

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
            self.assertIn("final_selection_score", record)
            self.assertNotIn("selection_score", record)
            self.assertIn("representativeness_score", record)
            self.assertIn("density_bucket", record)

    def test_dbta_candidate_pool_representativeness_scores_only_shortlist(self):
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
            dbta_representative_scope="candidate_pool",
            dbta_representative_weight=1.0,
            dbta_diversity_weight=0.0,
            dbta_selection_mode="diversity_aware",
        )

        selected = train_mod._score_and_select_dbta_records(records, cfg)
        selected_ids = {int(record["dataset_index"]) for record in selected}

        self.assertEqual(len(selected), 2)
        self.assertTrue(selected_ids.issubset({0, 1, 2}))
        for record in records[3:]:
            self.assertEqual(record["density_bucket"], "not_candidate")
            self.assertEqual(record["representativeness_score"], 0.0)
        for record in selected:
            self.assertNotEqual(record["density_bucket"], "not_candidate")

    def test_dbta_dumps_use_final_selection_score_only(self):
        records = [
            {
                "dataset_index": 0,
                "sid": "a",
                "year": 2023,
                "time_id": 5,
                "label": 1,
                "pred_label": 1,
                "drift_score": 0.7,
                "final_selection_score": 0.8,
                "uncertainty": 0.1,
                "branch_disagreement": 0.2,
                "prototype_distance": 0.3,
                "predicted_prototype_distance": 0.4,
                "nearest_prototype_distance": 0.3,
                "representativeness_score": 0.9,
                "density_bucket": "high",
                "diversity_gain": 0.5,
            },
            {
                "dataset_index": 1,
                "sid": "b",
                "year": 2023,
                "time_id": 5,
                "label": 0,
                "pred_label": 0,
                "drift_score": 0.2,
                "final_selection_score": 0.0,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            selection_path = os.path.join(tmpdir, "dbta_selection.csv")
            train_mod._write_dbta_selection_dump(selection_path, [records[0]])
            train_mod._write_dbta_recent_pool_scores_dump(selection_path, records, [records[0]])

            with open(selection_path, newline="", encoding="utf-8") as f:
                selection_fields = csv.DictReader(f).fieldnames
            pool_path = os.path.join(tmpdir, "dbta_recent_pool_scores.csv")
            with open(pool_path, newline="", encoding="utf-8") as f:
                pool_reader = csv.DictReader(f)
                pool_fields = pool_reader.fieldnames
                pool_rows = list(pool_reader)

            self.assertIn("final_selection_score", selection_fields)
            self.assertNotIn("selection_score", selection_fields)
            self.assertIn("final_selection_score", pool_fields)
            self.assertNotIn("selection_score", pool_fields)
            self.assertTrue(os.path.exists(pool_path))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "dbta_candidates.csv")))
            self.assertEqual([row["selected"] for row in pool_rows], ["1", "0"])

    def test_generator_uses_no_refinement_i1_name(self):
        configs, groups = make_ablation_configs.build_configs()
        i1_paths = groups["i1"]
        no_refinement_path = "i1_dbta/I1_03_dbta_no_refinement_020_dynamic_replay.yaml"

        self.assertIn(no_refinement_path, i1_paths)
        self.assertFalse(any("dbta_drift_only" in path for path in i1_paths))
        no_refinement = configs[no_refinement_path]["train"]
        self.assertEqual(no_refinement["dbta_candidate_top_p"], 1.0)
        self.assertEqual(no_refinement["dbta_representative_weight"], 0.0)
        self.assertEqual(no_refinement["dbta_selection_mode"], "topk")
        self.assertEqual(no_refinement["dbta_diversity_weight"], 0.0)
        self.assertEqual(make_ablation_configs.BASE_DEFAULTS["train"]["dbta_representative_scope"], "recent_pool")

    def test_generator_adds_tuned_performance_group(self):
        configs, groups = make_ablation_configs.build_configs()
        tuned_path = "tuned_performance/TP2_full_dbta_v2_020_candidate_pool_drift020.yaml"

        self.assertIn("tuned", groups)
        self.assertIn(tuned_path, groups["tuned"])
        tuned_train = configs[tuned_path]["train"]
        self.assertEqual(tuned_train["dbta_representative_scope"], "candidate_pool")
        self.assertEqual(tuned_train["dbta_final_drift_weight"], 0.2)

    def test_i3_manifest_tracks_availability_inputs(self):
        configs, groups = make_ablation_configs.build_configs()
        signals = make_ablation_configs.build_i3_gate_signal_manifest(configs, groups)

        self.assertFalse(signals["I3_01_learned_emb_only"]["availability_inputs"])
        self.assertFalse(signals["I3_02_quality_only"]["availability_inputs"])
        self.assertTrue(signals["I3_05_uncertainty_only"]["availability_inputs"])
        self.assertTrue(signals["I3_07_full_gate"]["availability_inputs"])


if __name__ == "__main__":
    unittest.main()
