import math
import unittest

from doosan_forcevla_data.convert.compute_actions import compute_measured_tcp_delta_action


class ComputeMeasuredTcpDeltaActionTests(unittest.TestCase):
    def test_zero_motion_gives_all_zeros(self):
        action = compute_measured_tcp_delta_action(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
        self.assertEqual(len(action), 7)
        for value in action:
            self.assertAlmostEqual(value, 0.0)

    def test_pure_x_translation_gives_dx_only(self):
        action = compute_measured_tcp_delta_action(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.25, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
        self.assertAlmostEqual(action[0], 0.25)
        for value in action[1:]:
            self.assertAlmostEqual(value, 0.0)

    def test_pure_z_rotation_90_degrees(self):
        q_z_90 = [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)]
        action = compute_measured_tcp_delta_action(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            q_z_90,
        )
        self.assertAlmostEqual(action[3], 0.0, places=12)
        self.assertAlmostEqual(action[4], 0.0, places=12)
        self.assertAlmostEqual(action[5], math.pi / 2.0, places=12)

    def test_non_normalized_quaternion_input_still_works(self):
        q_z_90_scaled = [
            0.0,
            0.0,
            2.0 * math.sin(math.pi / 4.0),
            2.0 * math.cos(math.pi / 4.0),
        ]
        action = compute_measured_tcp_delta_action(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
            q_z_90_scaled,
        )
        self.assertAlmostEqual(action[5], math.pi / 2.0, places=12)

    def test_invalid_input_sizes_raise_value_error(self):
        with self.assertRaises(ValueError):
            compute_measured_tcp_delta_action(
                [0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            )
        with self.assertRaises(ValueError):
            compute_measured_tcp_delta_action(
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            )

    def test_output_length_7_and_no_nans(self):
        action = compute_measured_tcp_delta_action(
            (1.0, 2.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
            (1.1, 2.2, 3.3),
            (0.0, 0.0, 0.0, 1.0),
            gripper_t=[0.2],
            gripper_t1=[0.4],
        )
        self.assertEqual(len(action), 7)
        self.assertTrue(all(math.isfinite(value) for value in action))
        self.assertAlmostEqual(action[6], 0.2)


if __name__ == "__main__":
    unittest.main()
