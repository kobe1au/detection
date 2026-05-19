import unittest

import torch

from fusion.prototypes import TemporalPrototypeMemory


class TemporalMemoryNoEvalUpdateTest(unittest.TestCase):
    def test_drift_estimation_does_not_update_memory(self):
        memory = TemporalPrototypeMemory(3, 2, 4, momentum=0.5, num_clusters=2)
        memory.update_weighted(
            torch.randn(8, 4),
            torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        )
        before_seen = memory.seen.clone()
        before_initialized = memory.initialized.clone()
        before_proto = memory.prototypes.clone()

        memory.estimate_drift(
            torch.randn(5, 4),
            torch.tensor([1, 1, 2, 2, 2]),
            class_probs=torch.full((5, 2), 0.5),
        )

        self.assertTrue(torch.equal(memory.seen, before_seen))
        self.assertTrue(torch.equal(memory.initialized, before_initialized))
        self.assertTrue(torch.allclose(memory.prototypes, before_proto))


if __name__ == "__main__":
    unittest.main()
