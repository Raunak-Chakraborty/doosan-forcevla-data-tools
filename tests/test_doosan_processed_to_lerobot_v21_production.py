from __future__ import annotations

import unittest

from doosan_forcevla_data.convert.doosan_force_proprio_v1 import OBSERVATION_STATE_FIELDS
from doosan_forcevla_data.convert.doosan_measured_action_v1 import ACTION_FIELDS
from doosan_forcevla_data.convert import doosan_processed_to_lerobot_v21 as module


class DoosanProcessedToLeRobotV21ProductionTests(unittest.TestCase):
    def test_features_are_exact_two_camera_forcevla_contract(self):
        features = module._features()
        self.assertEqual(
            set(features),
            {
                "observation.state",
                "action",
                "observation.images.tcp_camera",
                "observation.images.external_camera_2",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            },
        )
        self.assertNotIn("observation.images.external_camera_1", features)
        self.assertNotIn("observation.images.right_wrist_0_rgb", features)
        self.assertEqual(features["observation.state"]["shape"], [25])
        self.assertEqual(features["observation.state"]["names"], list(OBSERVATION_STATE_FIELDS))
        self.assertEqual(features["action"]["shape"], [7])
        self.assertEqual(features["action"]["names"], list(ACTION_FIELDS))
        self.assertEqual(features["observation.images.tcp_camera"]["shape"], [480, 640, 3])
        self.assertEqual(
            features["observation.images.external_camera_2"]["shape"], [480, 848, 3]
        )

    def test_dataset_rows_preserve_25d_7d_and_regularize_timestamp_only(self):
        processed = []
        for index in range(3):
            processed.append(
                {
                    "frame_index": index,
                    "reference_index": index,
                    "lerobot_timestamp": index / 30,
                    "observation_state_25d": [float(index + j) for j in range(25)],
                    "action_7d": [float(index + j) for j in range(7)],
                }
            )
        rows = module._dataset_rows(processed)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[2]["observation.state"], processed[2]["observation_state_25d"])
        self.assertEqual(rows[2]["action"], processed[2]["action_7d"])
        self.assertEqual(rows[2]["timestamp"], 2 / 30)
        self.assertEqual(rows[2]["frame_index"], 2)
        self.assertEqual(rows[2]["index"], 2)
        self.assertEqual(rows[2]["episode_index"], 0)
        self.assertEqual(rows[2]["task_index"], 0)

    def test_feature_stats_are_population_statistics_with_lerobot_count_shape(self):
        stats = module._feature_stats([[0.0, 2.0], [2.0, 4.0]])
        self.assertEqual(stats["min"], [0.0, 2.0])
        self.assertEqual(stats["max"], [2.0, 4.0])
        self.assertEqual(stats["mean"], [1.0, 3.0])
        self.assertEqual(stats["std"], [1.0, 1.0])
        self.assertEqual(stats["count"], [2])

    def test_export_provenance_pins_actual_frozen_dependency_commits(self):
        self.assertEqual(
            module.EXPECTED_FORCEVLA_COMMIT,
            "9b61abef116f207d587d10aaf30170b73757c3e0",
        )
        self.assertEqual(
            module.EXPECTED_LEROBOT_COMMIT,
            "e7aea92dd833f83d163820dcf2e58250307697a4",
        )
        self.assertEqual(
            module.EXPECTED_DLIMP_COMMIT,
            "5edaa4691567873d495633f2708982b42edf1972",
        )


if __name__ == "__main__":
    unittest.main()
