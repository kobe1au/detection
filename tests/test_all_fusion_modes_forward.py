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


if __name__ == "__main__":
    unittest.main()
