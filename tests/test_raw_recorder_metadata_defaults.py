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


if __name__ == "__main__":
    unittest.main()
