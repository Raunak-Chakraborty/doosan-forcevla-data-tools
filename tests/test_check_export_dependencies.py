import unittest
from unittest import mock

import doosan_forcevla_data.inspect.check_export_dependencies as dependencies_module
from doosan_forcevla_data.inspect.check_export_dependencies import (
    check_export_dependencies,
    implemented_video_backend_ready,
)


class CheckExportDependenciesTests(unittest.TestCase):
    def test_dependency_report_shape(self):
        results = check_export_dependencies()

        self.assertIsInstance(results, dict)
        for key in [
            "python",
            "pyarrow",
            "pandas",
            "lerobot",
            "cv2",
            "imageio",
            "imageio_ffmpeg",
            "PIL",
            "ffmpeg",
        ]:
            self.assertIn(key, results)
            self.assertIn("available", results[key])
            self.assertIn("version", results[key])
            self.assertIn("detail", results[key])
            self.assertIsInstance(results[key]["available"], bool)
            self.assertTrue(results[key]["version"] is None or isinstance(results[key]["version"], str))
            self.assertIsInstance(results[key]["detail"], str)

    def test_imageio_ffmpeg_dependency_reports_get_ffmpeg_exe(self):
        class FakeImageioFfmpeg:
            __version__ = "9.9.9"

            @staticmethod
            def get_ffmpeg_exe() -> str:
                return "/bundled/ffmpeg"

        real_import_module = dependencies_module.importlib.import_module

        def fake_import_module(module_name: str):
            if module_name == "imageio_ffmpeg":
                return FakeImageioFfmpeg
            return real_import_module(module_name)

        with mock.patch.object(dependencies_module.importlib, "import_module", side_effect=fake_import_module):
            results = check_export_dependencies()

        self.assertTrue(results["imageio_ffmpeg"]["available"])
        self.assertEqual(results["imageio_ffmpeg"]["version"], "9.9.9")
        self.assertIn("/bundled/ffmpeg", results["imageio_ffmpeg"]["detail"])

    def test_video_backend_ready_ignores_pil_plus_ffmpeg_without_encoder(self):
        dependencies = {
            "imageio_ffmpeg": {"available": False},
            "imageio": {"available": False},
            "cv2": {"available": False},
            "PIL": {"available": True},
            "ffmpeg": {"available": True},
        }

        self.assertFalse(implemented_video_backend_ready(dependencies))

        for key in ["imageio_ffmpeg", "imageio", "cv2"]:
            with self.subTest(key=key):
                backend_dependencies = dict(dependencies)
                backend_dependencies[key] = {"available": True}
                self.assertTrue(implemented_video_backend_ready(backend_dependencies))


if __name__ == "__main__":
    unittest.main()
