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
        model = make_model(
            "ours",
            use_temporal_reliability=True,
            use_gate_temporal_reliability_inputs=True,
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
        self.assertEqual(tuple(extra["q_time"].shape), (2,))
        self.assertIn("gate_weights", extra)

    def test_ours_forward_with_temporal_drift_and_time_gate(self):
        model = make_model(
            "ours",
            use_time_gate_inputs=True,
            use_gate_temporal_reliability_inputs=True,
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

    def test_gate_temporal_reliability_inputs_are_independent_of_quality_inputs(self):
        model = make_model(
            "ours",
            use_quality_gate_inputs=False,
            use_uncertainty_gate=False,
            use_time_gate_inputs=False,
            use_gate_temporal_reliability_inputs=True,
            use_temporal_reliability=True,
            use_drift_reliability=True,
            confidence_inputs=False,
            num_time_domains=4,
            historical_time_id_max=1,
        )
        model.eval()
        self.assertEqual(model.gate_net.q_dim, 12)
        logits, extra = model(
            graph_data=make_graph_batch(),
            explicit_qs=make_explicit_qs(),
            time_ids=torch.tensor([0, 2], dtype=torch.long),
            masks=make_masks(),
        )
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIn("q_time", extra)
        self.assertIn("q_drift", extra)
        self.assertNotIn("api_confidence", extra)
        self.assertNotIn("graph_confidence", extra)
        self.assertNotIn("joint_confidence", extra)

    def test_quality_only_gate_excludes_temporal_reliability_inputs(self):
        model = make_model(
            "ours",
            use_quality_gate_inputs=True,
            use_uncertainty_gate=False,
            use_time_gate_inputs=False,
            use_gate_temporal_reliability_inputs=False,
            use_temporal_reliability=True,
            use_drift_reliability=True,
            confidence_inputs=False,
            num_time_domains=4,
            historical_time_id_max=1,
        )
        model.eval()
        self.assertEqual(model.gate_net.q_dim, 10)
        logits, extra = model(
            graph_data=make_graph_batch(),
            explicit_qs=make_explicit_qs(),
            time_ids=torch.tensor([0, 2], dtype=torch.long),
            masks=make_masks(),
        )
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIn("q_time", extra)
        self.assertIn("q_drift", extra)

    def test_confidence_inputs_add_three_gate_dimensions(self):
        model = make_model(
            "ours",
            use_quality_gate_inputs=False,
            use_uncertainty_gate=False,
            use_time_gate_inputs=False,
            use_gate_temporal_reliability_inputs=True,
            use_temporal_reliability=True,
            use_drift_reliability=True,
            confidence_inputs=True,
            num_time_domains=4,
            historical_time_id_max=1,
        )
        self.assertEqual(model.gate_net.q_dim, 15)

    def test_gate_confidence_source_raw_ignores_temperatures(self):
        model = make_model(
            "ours",
            confidence_source="raw",
        )
        model.branch_temperatures = {"api": 9.0, "graph": 7.0, "joint": 5.0}
        api_logits = torch.tensor([[4.0, 1.0]], dtype=torch.float32)
        graph_logits = torch.tensor([[2.5, 0.5]], dtype=torch.float32)
        joint_logits = torch.tensor([[3.0, 0.0]], dtype=torch.float32)

        api_conf, graph_conf, joint_conf = model._compute_branch_confidences(
            api_logits,
            graph_logits,
            joint_logits,
            torch.float32,
        )

        expected_api = torch.softmax(api_logits, dim=-1).amax(dim=-1, keepdim=True)
        expected_graph = torch.softmax(graph_logits, dim=-1).amax(dim=-1, keepdim=True)
        expected_joint = torch.softmax(joint_logits, dim=-1).amax(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(api_conf, expected_api, atol=1e-6))
        self.assertTrue(torch.allclose(graph_conf, expected_graph, atol=1e-6))
        self.assertTrue(torch.allclose(joint_conf, expected_joint, atol=1e-6))

    def test_gate_confidence_source_calibrated_uses_temperatures(self):
        model = make_model(
            "ours",
            confidence_source="calibrated",
        )
        model.branch_temperatures = {"api": 2.0, "graph": 4.0, "joint": 8.0}
        api_logits = torch.tensor([[4.0, 1.0]], dtype=torch.float32)
        graph_logits = torch.tensor([[2.5, 0.5]], dtype=torch.float32)
        joint_logits = torch.tensor([[3.0, 0.0]], dtype=torch.float32)

        api_conf, graph_conf, joint_conf = model._compute_branch_confidences(
            api_logits,
            graph_logits,
            joint_logits,
            torch.float32,
        )

        expected_api = torch.softmax(api_logits / 2.0, dim=-1).amax(dim=-1, keepdim=True)
        expected_graph = torch.softmax(graph_logits / 4.0, dim=-1).amax(dim=-1, keepdim=True)
        expected_joint = torch.softmax(joint_logits / 8.0, dim=-1).amax(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(api_conf, expected_api, atol=1e-6))
        self.assertTrue(torch.allclose(graph_conf, expected_graph, atol=1e-6))
        self.assertTrue(torch.allclose(joint_conf, expected_joint, atol=1e-6))

        model = make_model(
            "ours",
            use_time_gate_inputs=True,
            use_temporal_reliability=True,
            use_drift_reliability=True,
            num_time_domains=5,
            historical_time_id_max=2,
        )
        q_time_single, q_drift_single, _, _, _ = model._build_temporal_reliability(
            torch.tensor([3], dtype=torch.long),
            1,
            torch.device("cpu"),
            torch.float32,
            disagreement=torch.tensor([0.25], dtype=torch.float32),
            entropy=torch.tensor([0.40], dtype=torch.float32),
            alignment_coverage=torch.tensor([0.60], dtype=torch.float32),
            alignment_density=torch.tensor([0.81], dtype=torch.float32),
        )
        q_time_mixed, q_drift_mixed, _, _, _ = model._build_temporal_reliability(
            torch.tensor([0, 3, 4], dtype=torch.long),
            3,
            torch.device("cpu"),
            torch.float32,
            disagreement=torch.tensor([0.10, 0.25, 0.90], dtype=torch.float32),
            entropy=torch.tensor([0.20, 0.40, 0.95], dtype=torch.float32),
            alignment_coverage=torch.tensor([0.90, 0.60, 0.10], dtype=torch.float32),
            alignment_density=torch.tensor([0.90, 0.81, 0.05], dtype=torch.float32),
        )
        self.assertTrue(torch.allclose(q_time_single.view(-1), q_time_mixed[1].view(-1)))
        self.assertTrue(torch.allclose(q_drift_single.view(-1), q_drift_mixed[1].view(-1)))

    def test_q_drift_changes_with_sample_evidence_at_fixed_time(self):
        model = make_model(
            "ours",
            use_temporal_reliability=True,
            use_drift_reliability=True,
            num_time_domains=5,
            historical_time_id_max=2,
        )
        q_time, q_drift, drift_prior, drift_evidence, drift_score = model._build_temporal_reliability(
            torch.tensor([3, 3], dtype=torch.long),
            2,
            torch.device("cpu"),
            torch.float32,
            disagreement=torch.tensor([0.05, 0.95], dtype=torch.float32),
            entropy=torch.tensor([0.10, 0.90], dtype=torch.float32),
            alignment_coverage=torch.tensor([0.95, 0.15], dtype=torch.float32),
            alignment_density=torch.tensor([0.90, 0.10], dtype=torch.float32),
        )
        self.assertTrue(torch.allclose(q_time[0].view(-1), q_time[1].view(-1)))
        self.assertTrue(torch.allclose(drift_prior[0].view(-1), drift_prior[1].view(-1)))
        self.assertGreater(float(drift_evidence[1].item()), float(drift_evidence[0].item()))
        self.assertGreater(float(drift_score[1].item()), float(drift_score[0].item()))
        self.assertLess(float(q_drift[1].item()), float(q_drift[0].item()))


if __name__ == "__main__":
    unittest.main()
