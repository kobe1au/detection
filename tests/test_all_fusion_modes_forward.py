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


if __name__ == "__main__":
    unittest.main()
