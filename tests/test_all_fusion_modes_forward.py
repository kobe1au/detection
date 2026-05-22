import unittest

import torch

from tests._helpers import make_explicit_qs, make_graph_batch, make_masks, make_model


class AllFusionModesForwardTest(unittest.TestCase):
    def test_all_fusion_modes_forward(self):
        graph = make_graph_batch()
        explicit_qs = make_explicit_qs()
        time_ids = torch.tensor([0, 1], dtype=torch.long)

        for mode in ("api", "graph", "concat", "late_fusion", "cross_attention", "ours"):
            with self.subTest(mode=mode):
                model = make_model(mode)
                model.eval()
                masks = make_masks() if mode == "ours" else None
                logits, extra = model(
                    graph_data=graph,
                    explicit_qs=explicit_qs,
                    time_ids=time_ids,
                    masks=masks,
                )
                self.assertEqual(tuple(logits.shape), (2, 2))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertIsInstance(extra, dict)

    def test_ours_forward_with_time_gate_inputs(self):
        model = make_model("ours", use_time_gate_inputs=True)
        model.eval()
        logits, extra = model(
            graph_data=make_graph_batch(),
            explicit_qs=make_explicit_qs(),
            time_ids=torch.tensor([0, 1], dtype=torch.long),
            masks=make_masks(),
        )
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIn("time_gate_features", extra)
        self.assertEqual(tuple(extra["time_gate_features"].shape), (2, 4))
        self.assertIn("gate_weights", extra)
        self.assertEqual(tuple(extra["gate_weights"].shape), (2, 3))

    def test_ours_forward_with_temporal_reliability(self):
        model = make_model("ours", use_temporal_reliability=True)
        model.eval()
        logits, extra = model(
            graph_data=make_graph_batch(),
            explicit_qs=make_explicit_qs(),
            time_ids=torch.tensor([0, 1], dtype=torch.long),
            masks=make_masks(),
        )
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIn("q_time", extra)
        self.assertEqual(tuple(extra["q_time"].shape), (2,))
        self.assertIn("gate_weights", extra)

    def test_ours_forward_with_temporal_drift_and_time_gate(self):
        model = make_model(
            "ours",
            use_time_gate_inputs=True,
            use_temporal_reliability=True,
            use_drift_reliability=True,
            num_time_domains=4,
            historical_time_id_max=1,
        )
        model.eval()
        logits, extra = model(
            graph_data=make_graph_batch(),
            explicit_qs=make_explicit_qs(),
            time_ids=torch.tensor([0, 1], dtype=torch.long),
            masks=make_masks(),
        )
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIn("q_time", extra)
        self.assertIn("q_drift", extra)
        self.assertIn("time_gate_features", extra)
        self.assertIn("gate_weights", extra)
        self.assertEqual(tuple(extra["q_drift"].shape), (2,))
        self.assertEqual(tuple(extra["time_gate_features"].shape), (2, 4))
        self.assertEqual(tuple(extra["gate_weights"].shape), (2, 3))

    def test_temporal_features_are_batch_invariant(self):
        model = make_model(
            "ours",
            use_time_gate_inputs=True,
            use_temporal_reliability=True,
            use_drift_reliability=True,
            num_time_domains=5,
            historical_time_id_max=2,
        )
        q_time_single, q_drift_single = model._build_temporal_reliability(
            torch.tensor([3], dtype=torch.long),
            1,
            torch.device("cpu"),
            torch.float32,
        )
        q_time_mixed, q_drift_mixed = model._build_temporal_reliability(
            torch.tensor([0, 3, 4], dtype=torch.long),
            3,
            torch.device("cpu"),
            torch.float32,
        )
        self.assertTrue(torch.allclose(q_time_single.view(-1), q_time_mixed[1].view(-1)))
        self.assertTrue(torch.allclose(q_drift_single.view(-1), q_drift_mixed[1].view(-1)))


if __name__ == "__main__":
    unittest.main()
