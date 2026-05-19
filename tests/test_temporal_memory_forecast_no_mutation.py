import unittest

import torch

from fusion.prototypes import TemporalPrototypeMemory


class TemporalMemoryForecastNoMutationTest(unittest.TestCase):
    def test_forecast_next_prototypes_does_not_mutate_memory(self):
        memory = TemporalPrototypeMemory(3, 2, 4, momentum=0.5, num_clusters=2)
        feats = torch.randn(8, 4)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        time_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        memory.update_weighted(feats, labels, time_ids)

        before = {
            "prototypes": memory.prototypes.clone(),
            "seen": memory.seen.clone(),
            "spread": memory.spread.clone(),
            "initialized": memory.initialized.clone(),
            "inherited": memory.inherited.clone(),
        }
        memory.forecast_next_prototypes(torch.tensor([0, 1, 2]), velocity_scale=0.5)

        for name, tensor in before.items():
            current = getattr(memory, name)
            if tensor.dtype == torch.bool:
                self.assertTrue(torch.equal(current, tensor), name)
            else:
                self.assertTrue(torch.allclose(current, tensor), name)


if __name__ == "__main__":
    unittest.main()
