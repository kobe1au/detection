import unittest
from pathlib import Path

from fusion.train import deep_update, load_yaml_file, validate_full_config


class ConfigSchemaCleanTest(unittest.TestCase):
    def test_base_and_current_overrides_validate(self):
        root = Path(__file__).resolve().parents[1]
        base_path = root / "config" / "base.yaml"
        base = load_yaml_file(str(base_path))
        validate_full_config(dict(base))

        override_paths = []
        override_paths.extend(sorted((root / "config" / "chapters").rglob("*.yaml")))
        override_paths.extend(sorted(root.glob("config/exp_*.yaml")))

        for override_path in override_paths:
            with self.subTest(config=str(override_path.relative_to(root))):
                override = load_yaml_file(str(override_path))
                validate_full_config(deep_update(load_yaml_file(str(base_path)), override))


if __name__ == "__main__":
    unittest.main()
