import unittest

import torch

from fusion.prototypes import TemporalPrototypeMemory


class TemporalMemoryInitializedSeenTest(unittest.TestCase):
    def test_inherited_prototype_is_initialized_but_not_seen_until_observed(self):
        memory = TemporalPrototypeMemory(
            num_domains=3,
            num_classes=2,
            feature_dim=4,
            momentum=0.0,
            num_clusters=2,
        )

        year0 = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0],
            ]
        )
        memory.update_weighted(
            year0,
            torch.zeros(4, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
        )
        self.assertGreaterEqual(int((memory.seen[0, 0] > 0).sum().item()), 2)

        year1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4)
        memory.update_weighted(
            year1,
            torch.zeros(4, dtype=torch.long),
            torch.ones(4, dtype=torch.long),
        )

        inherited_only = memory.inherited[1, 0] & (memory.seen[1, 0] == 0)
        self.assertTrue(bool(inherited_only.any().item()))
        self.assertTrue(bool(memory.initialized[1, 0][inherited_only].all().item()))


if __name__ == "__main__":
    unittest.main()
