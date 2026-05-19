import unittest

import torch

from fusion.mm_dataset import MultiModalMalwareDataset


class AlignmentMaskShapesTest(unittest.TestCase):
    def test_weighted_alignment_mask_shape_and_values(self):
        edge = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
        strong = torch.tensor([True, False, False])
        weak = torch.tensor([False, True, False])

        mask, q_align = MultiModalMalwareDataset._build_method_api_mask(
            edge,
            num_nodes=3,
            num_api=3,
            api_strong_mask=strong,
            api_weak_mask=weak,
        )

        self.assertEqual(tuple(mask.shape), (3, 3))
        self.assertEqual(mask.dtype, torch.float32)
        self.assertAlmostEqual(float(mask[0, 0].item()), 1.0)
        self.assertAlmostEqual(float(mask[1, 1].item()), 0.5)
        self.assertAlmostEqual(float(mask[2, 2].item()), 0.0)
        self.assertGreater(q_align, 0.0)


if __name__ == "__main__":
    unittest.main()
