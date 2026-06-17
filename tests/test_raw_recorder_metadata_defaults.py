import re
import unittest
from pathlib import Path


class RawRecorderMetadataDefaultsTests(unittest.TestCase):
    def test_episode_metadata_defaults_do_not_generate_false_null_failure_label(self):
        text = Path("configs/raw_recorder/episode_metadata_defaults.example.yaml").read_text()

        invalid_pair = re.compile(
            r"(?m)^success:\s*false\s*$\n^failure_reason:\s*null\s*$"
        )
        self.assertIsNone(
            invalid_pair.search(text),
            "metadata defaults must not use success:false with failure_reason:null",
        )
        self.assertIn(
            "failure_reason: unannotated_episode_replace_before_training",
            text,
        )

    def test_schema_plan_example_matches_success_failure_policy(self):
        text = Path("docs/raw_recorder_schema_plan.md").read_text()

        invalid_pair = re.compile(
            r"(?m)^success:\s*false\s*$\n^failure_reason:\s*null\s*$"
        )
        self.assertIsNone(
            invalid_pair.search(text),
            "schema plan example must not show success:false with failure_reason:null",
        )
        self.assertIn(
            "Must be non-empty when `success` is `false`",
            text,
        )

    def test_episode_metadata_defaults_enable_strict_lab_provenance(self):
        text = Path("configs/raw_recorder/episode_metadata_defaults.example.yaml").read_text()

        self.assertIn("lab_provenance_required: true", text)
        self.assertIn("source_workspace:", text)
        self.assertIn("live_graph_verification:", text)

    def test_schema_plan_example_enables_strict_lab_provenance(self):
        text = Path("docs/raw_recorder_schema_plan.md").read_text()

        self.assertIn("lab_provenance_required: true", text)
        self.assertIn("source_workspace:", text)
        self.assertIn("Strict lab/source provenance readiness", text)

    def test_doosan_euler_orientation_markers_are_documented_as_unsupported(self):
        config_text = Path("configs/raw_recorder/unit_frame_conventions.example.yaml").read_text()
        schema_plan_text = Path("docs/raw_recorder_schema_plan.md").read_text()

        for text in [config_text, schema_plan_text]:
            self.assertIn("recognized_but_unsupported_tcp_orientation_conventions", text)
            self.assertIn("doosan_posx_euler_zyz_degrees", text)
            self.assertIn("doosan_robotstate_actual_tcp_position_euler_zyz_degrees", text)
            self.assertIn("euler_zyz_degrees", text)
            self.assertIn("Do not label native Doosan Euler pose values as rotation vectors", text)



    def test_camera_robot_source_stamp_tolerance_policy_is_documented(self):
        config_text = Path("configs/raw_recorder/unit_frame_conventions.example.yaml").read_text()
        schema_plan_text = Path("docs/raw_recorder_schema_plan.md").read_text()

        for text in [config_text, schema_plan_text]:
            self.assertIn("max_camera_robot_source_stamp_offset_sec", text)
            self.assertIn("0.5 / fps", text)
            self.assertIn("0.02", text)
            self.assertIn("2.0 / fps", text)

if __name__ == "__main__":
    unittest.main()
