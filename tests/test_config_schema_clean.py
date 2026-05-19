import unittest
from pathlib import Path

import yaml

from fusion.train import deep_update, load_yaml_file, validate_full_config


class ConfigSchemaCleanTest(unittest.TestCase):
    def test_base_and_train_2026_overrides_validate(self):
        root = Path(__file__).resolve().parents[1]
        base_path = root / "config" / "base.yaml"
        base = load_yaml_file(str(base_path))
        validate_full_config(dict(base))

        override_root = root / "config" / "train_2026"
        for override_path in sorted(override_root.rglob("*.yaml")):
            with self.subTest(config=str(override_path.relative_to(root))):
                override = load_yaml_file(str(override_path))
                validate_full_config(deep_update(load_yaml_file(str(base_path)), override))


if __name__ == "__main__":
    unittest.main()
