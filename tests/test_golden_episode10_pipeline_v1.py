from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from doosan_forcevla_data.smoke import golden_episode10_end_to_end as module


class OutputSafetyTests(unittest.TestCase):
    def test_output_must_be_separate_from_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            episode = root / "episode"
            converter = root / "converter"
            forcevla = root / "forcevla"
            for path in (episode, converter, forcevla):
                path.mkdir()

            with self.assertRaises(module.GoldenPipelineError):
                module._prepare_output_root(
                    converter / "output",
                    overwrite=False,
                    protected_paths=(episode, converter, forcevla),
                )

            output = root / "golden_output"
            result = module._prepare_output_root(
                output,
                overwrite=False,
                protected_paths=(episode, converter, forcevla),
            )
            self.assertEqual(result, output)
            self.assertTrue(result.is_dir())


class ForceVlaEnvironmentTests(unittest.TestCase):
    def test_forcevla_environment_replaces_ros_pythonpath(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            converter = root / "converter"
            forcevla = root / "forcevla"
            for path in (
                converter / "src",
                forcevla / "src",
                forcevla / "lerobot",
                forcevla / "dlimp",
            ):
                path.mkdir(parents=True, exist_ok=True)

            with mock.patch.dict(
                os.environ,
                {
                    "PYTHONPATH": "/opt/ros/jazzy/lib/python3.12:/bad",
                    "PYTHONHOME": "/bad-home",
                },
                clear=False,
            ):
                env = module._forcevla_environment(converter, forcevla)

            self.assertNotIn("/opt/ros/jazzy/lib/python3.12", env["PYTHONPATH"])
            self.assertNotIn("PYTHONHOME", env)
            self.assertEqual(env["PYTHONNOUSERSITE"], "1")
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertIn(str((converter / "src").resolve()), env["PYTHONPATH"])
            self.assertIn(str((forcevla / "lerobot").resolve()), env["PYTHONPATH"])


class PipelineOrchestrationTests(unittest.TestCase):
    def test_public_pipeline_uses_existing_patch8_layers_and_merges_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            episode = root / "episode"
            converter = root / "converter"
            forcevla = root / "forcevla"
            output = root / "output"
            fv_python = root / "python"
            for path in (episode, converter, forcevla):
                path.mkdir()
            fv_python.write_text("", encoding="utf-8")

            source_report = {"forcevla_commit": "fv"}
            raw_report = {"episode_index": 10}
            processed_report = {"frame_count": 1008}
            forcevla_report = {"accepted": True, "lerobot_dataset": {"frame_count": 1008}}

            def fake_build(raw, processed, *, overwrite=False):
                self.assertEqual(Path(raw), episode.resolve())
                self.assertFalse(overwrite)
                Path(processed).mkdir(parents=True)
                return Path(processed)

            def fake_forcevla_stage(**kwargs):
                Path(kwargs["dataset_root"]).mkdir(parents=True)
                kwargs["fragment_path"].write_text(json.dumps(forcevla_report), encoding="utf-8")
                kwargs["log_path"].write_text("PASS\n", encoding="utf-8")
                return forcevla_report

            with (
                mock.patch.object(module, "validate_golden_source_repositories", return_value=source_report),
                mock.patch.object(module, "validate_golden_raw_episode", return_value=raw_report),
                mock.patch.object(module, "build_doosan_processed_episode", side_effect=fake_build),
                mock.patch.object(module, "validate_golden_processed_episode", return_value=processed_report),
                mock.patch.object(module, "_run_forcevla_stage", side_effect=fake_forcevla_stage),
            ):
                report_path = module.run_golden_episode10_pipeline(
                    episode,
                    output,
                    converter_root=converter,
                    forcevla_root=forcevla,
                    forcevla_python=fv_python,
                )

            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["accepted"])
            self.assertEqual(payload["raw_episode"], raw_report)
            self.assertEqual(payload["processed_episode"], processed_report)
            self.assertEqual(payload["forcevla_stage"], forcevla_report)
            self.assertEqual(Path(payload["artifacts"]["processed_episode"]), output / module.PROCESSED_DIRNAME)
            self.assertEqual(Path(payload["artifacts"]["lerobot_dataset"]), output / module.DATASET_DIRNAME)

    def test_forcevla_stage_exports_then_runs_file_and_runtime_acceptance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            processed = root / "processed"
            dataset = root / "dataset"
            fragment = root / "fragment.json"
            converter = root / "converter"
            forcevla = root / "forcevla"
            processed.mkdir()
            converter.mkdir()
            forcevla.mkdir()

            sequence = []

            def fake_export(source, target, *, overwrite=False):
                sequence.append("export")
                self.assertEqual(Path(source), processed)
                self.assertEqual(Path(target), dataset)
                self.assertFalse(overwrite)
                dataset.mkdir()
                return dataset

            with (
                mock.patch.object(module, "validate_golden_source_repositories", side_effect=lambda _: sequence.append("sources") or {"ok": True}),
                mock.patch.object(module, "validate_golden_processed_episode", side_effect=lambda _: sequence.append("processed") or {"frame_count": 1008}),
                mock.patch.object(module, "export_doosan_processed_to_lerobot_v21", side_effect=fake_export),
                mock.patch.object(module, "validate_golden_lerobot_dataset", side_effect=lambda *_: sequence.append("dataset") or {"frame_count": 1008}),
                mock.patch.object(module, "validate_golden_forcevla_runtime", side_effect=lambda *_, **__: sequence.append("runtime") or {"accepted": True}),
            ):
                result = module._forcevla_stage(
                    processed=processed,
                    dataset=dataset,
                    fragment=fragment,
                    converter_root=converter,
                    forcevla_root=forcevla,
                )

            self.assertEqual(sequence, ["sources", "processed", "export", "dataset", "runtime"])
            self.assertEqual(result, fragment.resolve())
            payload = json.loads(fragment.read_text(encoding="utf-8"))
            self.assertTrue(payload["accepted"])


if __name__ == "__main__":
    unittest.main()
