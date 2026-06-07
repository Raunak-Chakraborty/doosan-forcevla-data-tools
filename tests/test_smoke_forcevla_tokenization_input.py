import tempfile
import unittest
from pathlib import Path


class SmokeForceVLATokenizationInputTests(unittest.TestCase):
    def test_module_imports_without_forcevla_import_side_effects(self):
        from doosan_forcevla_data.inspect import smoke_forcevla_tokenization_input as module

        self.assertTrue(hasattr(module, "build_forcevla_tokenization_input_report"))
        self.assertTrue(hasattr(module, "main"))

    def test_missing_forcevla_root_fails_before_dataset_processing(self):
        from doosan_forcevla_data.inspect.smoke_forcevla_tokenization_input import (
            build_forcevla_tokenization_input_report,
        )

        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp) / "dataset"
            forcevla_root = Path(tmp) / "missing_forcevla"
            dataset_root.mkdir()

            with self.assertRaises(FileNotFoundError):
                build_forcevla_tokenization_input_report(
                    dataset_root=dataset_root,
                    forcevla_root=forcevla_root,
                )


if __name__ == "__main__":
    unittest.main()
