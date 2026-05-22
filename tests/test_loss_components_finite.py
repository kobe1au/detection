import unittest

import torch
import torch.nn as nn

from fusion.losses import compute_total_loss
from tests._helpers import make_explicit_qs, make_graph_batch, make_loss_cfg, make_masks, make_model


class LossComponentsFiniteTest(unittest.TestCase):
    def test_loss_components_are_finite(self):
        graph = make_graph_batch()
        model = make_model("ours")
        model.train()
        y = torch.tensor([0, 1], dtype=torch.long)

        logits, extra = model(
            graph_data=graph,
            y=y,
            explicit_qs=make_explicit_qs(),
            time_ids=torch.tensor([0, 1], dtype=torch.long),
            masks=make_masks(),
        )
        loss, loss_cls, loss_alignment = compute_total_loss(
            logits,
            extra,
            y,
            nn.CrossEntropyLoss(),
            make_loss_cfg(),
        )

        self.assertTrue(torch.isfinite(loss).all())
        for value in (loss_cls, loss_alignment):
            self.assertTrue(torch.isfinite(value).all())
        for value in extra.get("loss_components", {}).values():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.isfinite(value).all())

    def test_enhanced_loss_components_are_finite(self):
        graph = make_graph_batch()
        model = make_model(
            "ours",
            use_time_gate_inputs=True,
            use_temporal_reliability=True,
            use_drift_reliability=True,
        )
        model.train()
        y = torch.tensor([0, 1], dtype=torch.long)

        logits, extra = model(
            graph_data=graph,
            y=y,
            explicit_qs=make_explicit_qs(),
            time_ids=torch.tensor([0, 1], dtype=torch.long),
            masks=make_masks(),
        )
        loss, loss_cls, loss_alignment = compute_total_loss(
            logits,
            extra,
            y,
            nn.CrossEntropyLoss(),
            make_loss_cfg(),
        )

        for value in (loss, loss_cls, loss_alignment):
            self.assertTrue(torch.isfinite(value).all())

    def test_local_alignment_loss_finite_with_limits(self):
        torch.manual_seed(0)
        batch_size, num_nodes, num_tokens, dim = 2, 4, 5, 4
        logits = torch.randn(batch_size, 2, requires_grad=True)
        y = torch.tensor([0, 1], dtype=torch.long)

        masks = torch.zeros(batch_size, num_nodes, num_tokens)
        masks[0, 0, 0] = 1.0
        masks[0, 2, 4] = 0.5
        masks[1, 3, 1] = 1.0
        extra = {
            "local_alignment_node": torch.randn(batch_size, num_nodes, dim),
            "local_alignment_api": torch.randn(batch_size, num_tokens, dim),
            "local_alignment_masks": masks,
            "local_alignment_node_valid": torch.ones(batch_size, num_nodes),
            "local_alignment_api_valid": torch.ones(batch_size, num_tokens),
            "local_alignment_quality": torch.tensor([1.0, 0.8]),
            "local_alignment_time_weight": torch.tensor([1.0, 0.7]),
        }
        loss_cfg = make_loss_cfg()
        loss_cfg.update(
            {
                "semantic_alignment_weight": 0.0,
                "local_alignment_weight": 0.5,
                "max_local_align_nodes": 2,
                "max_local_align_tokens": 3,
                "branch_aux_weight": 0.0,
                "stage1_branch_aux_weight": 0.0,
            }
        )

        loss, loss_cls, loss_alignment = compute_total_loss(
            logits,
            extra,
            y,
            nn.CrossEntropyLoss(),
            loss_cfg,
        )

        for value in (loss, loss_cls, loss_alignment):
            self.assertTrue(torch.isfinite(value).all())
        self.assertIn("local_align", extra["loss_components"])
        self.assertTrue(torch.isfinite(extra["loss_components"]["local_align"]).all())


if __name__ == "__main__":
    unittest.main()
