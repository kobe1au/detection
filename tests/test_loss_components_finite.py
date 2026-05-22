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


if __name__ == "__main__":
    unittest.main()
