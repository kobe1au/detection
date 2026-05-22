import unittest

import torch

try:
    from fusion.prototypes import TemporalPrototypeMemory
except ModuleNotFoundError:
    TemporalPrototypeMemory = None


class TemporalMemoryInitializedSeenTest(unittest.TestCase):
    @unittest.skipIf(TemporalPrototypeMemory is None, "legacy temporal prototype module is not in the current DBTA mainline")
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

    @unittest.skipIf(TemporalPrototypeMemory is None, "legacy temporal prototype module is not in the current DBTA mainline")
    def test_first_observation_overwrites_inherited_prior_before_ema(self):
        memory = TemporalPrototypeMemory(
            num_domains=2,
            num_classes=1,
            feature_dim=2,
            momentum=0.99,
            num_clusters=1,
        )

        memory.update_weighted(
            torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
        )

        memory.update_weighted(
            torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            torch.zeros(2, dtype=torch.long),
            torch.ones(2, dtype=torch.long),
        )

        proto = memory.prototypes[1, 0, 0]
        expected = torch.tensor([0.0, 1.0], dtype=proto.dtype, device=proto.device)

        self.assertTrue(torch.allclose(proto, expected, atol=1e-4))
        self.assertEqual(int(memory.seen[1, 0, 0].item()), 2)
        self.assertFalse(bool(memory.inherited[1, 0, 0].item()))

if __name__ == "__main__":
    unittest.main()
