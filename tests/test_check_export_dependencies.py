import unittest

from doosan_forcevla_data.inspect.check_export_dependencies import check_export_dependencies


class CheckExportDependenciesTests(unittest.TestCase):
    def test_dependency_report_shape(self):
        results = check_export_dependencies()

        self.assertIsInstance(results, dict)
        for key in ["python", "pyarrow", "pandas", "lerobot", "cv2", "imageio", "PIL", "ffmpeg"]:
            self.assertIn(key, results)
            self.assertIn("available", results[key])
            self.assertIn("version", results[key])
            self.assertIn("detail", results[key])
            self.assertIsInstance(results[key]["available"], bool)
            self.assertTrue(results[key]["version"] is None or isinstance(results[key]["version"], str))
            self.assertIsInstance(results[key]["detail"], str)


if __name__ == "__main__":
    unittest.main()
