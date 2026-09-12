from __future__ import annotations

import math
import unittest

from doosan_forcevla_data.convert.model_state_profile_v1 import (
    ModelStateProfileError,
    StateMode,
    encode_legacy_observation_states,
    model_state_layout,
)
from doosan_forcevla_data.convert.orientation_representation_v1 import (
    OrientationRepresentation,
    decode_orientation,
    rotvec_to_matrix,
)


def _legacy_state(rotvec=(0.0, 0.0, 0.0)):
    return (
        0.1,
        -0.2,
        0.3,
        *rotvec,
        1.0,
        *[0.01 * index for index in range(1, 7)],
        *[0.02 * index for index in range(1, 7)],
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    )


def _matrix_max_error(left, right):
    return max(
        abs(float(left[row][col]) - float(right[row][col]))
        for row in range(3)
        for col in range(3)
    )


class ModelStateProfileV1Tests(unittest.TestCase):
    def test_all_expected_dimensions_and_wrench_tail(self):
        expected = {
            OrientationRepresentation.ROTVEC_PRINCIPAL: (19, 25),
            OrientationRepresentation.ROTVEC_CONTINUOUS: (19, 25),
            OrientationRepresentation.QUATERNION: (20, 26),
            OrientationRepresentation.ROTATION6D: (22, 28),
        }
        for representation, (no_wrench, full) in expected.items():
            with self.subTest(representation=representation):
                compact = model_state_layout(representation, StateMode.NO_WRENCH)
                complete = model_state_layout(representation, StateMode.FULL)
                self.assertEqual(compact.state_dim, no_wrench)
                self.assertEqual(complete.state_dim, full)
                self.assertIsNone(compact.wrench_slice)
                self.assertIsNotNone(complete.wrench_slice)
                self.assertEqual(
                    complete.state_fields[-6:],
                    (
                        "force_x_n",
                        "force_y_n",
                        "force_z_n",
                        "torque_x_nm",
                        "torque_y_nm",
                        "torque_z_nm",
                    ),
                )

    def test_legacy_default_is_exact_value_pass_through(self):
        state = _legacy_state((0.2, -0.3, 0.4))
        layout = model_state_layout()
        encoded = encode_legacy_observation_states([state], layout=layout)
        self.assertEqual(encoded, [state])
        self.assertTrue(layout.is_legacy_default)

    def test_no_wrench_principal_removes_only_final_six(self):
        state = _legacy_state((0.2, -0.3, 0.4))
        layout = model_state_layout(
            OrientationRepresentation.ROTVEC_PRINCIPAL,
            StateMode.NO_WRENCH,
        )
        encoded = encode_legacy_observation_states([state], layout=layout)[0]
        self.assertEqual(encoded, state[:19])

    def test_quaternion_and_rotation6d_preserve_physical_rotation(self):
        source_rotvec = (0.8, -0.4, 0.3)
        source_matrix = rotvec_to_matrix(source_rotvec)
        for representation in (
            OrientationRepresentation.QUATERNION,
            OrientationRepresentation.ROTATION6D,
        ):
            with self.subTest(representation=representation):
                layout = model_state_layout(representation, StateMode.FULL)
                encoded_state = encode_legacy_observation_states(
                    [_legacy_state(source_rotvec)],
                    layout=layout,
                )[0]
                encoded_orientation = encoded_state[layout.orientation_slice]
                reconstructed = decode_orientation(encoded_orientation, representation)
                self.assertLess(_matrix_max_error(source_matrix, reconstructed), 1e-12)
                self.assertEqual(encoded_state[-6:], (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    def test_rotation6d_order_is_explicit_two_columns(self):
        source_rotvec = (0.4, -0.2, 0.7)
        matrix = rotvec_to_matrix(source_rotvec)
        layout = model_state_layout(
            OrientationRepresentation.ROTATION6D,
            StateMode.NO_WRENCH,
        )
        state = encode_legacy_observation_states(
            [_legacy_state(source_rotvec)],
            layout=layout,
        )[0]
        self.assertEqual(
            state[layout.orientation_slice],
            (
                matrix[0][0],
                matrix[1][0],
                matrix[2][0],
                matrix[0][1],
                matrix[1][1],
                matrix[2][1],
            ),
        )

    def test_continuous_rotvec_removes_principal_pi_branch_jump(self):
        eps = 1e-3
        first = (math.pi - eps, 0.0, 0.0)
        # Same physical forward continuation represented on the opposite
        # principal branch just beyond pi.
        second_matrix = rotvec_to_matrix((math.pi + eps, 0.0, 0.0))
        # Re-express through the principal log by decoding/encoding using the
        # canonical processed convention.
        from doosan_forcevla_data.convert.orientation_representation_v1 import (
            matrix_to_principal_rotvec,
        )

        second = matrix_to_principal_rotvec(second_matrix)
        self.assertGreater(abs(first[0] - second[0]), 6.0)

        layout = model_state_layout(
            OrientationRepresentation.ROTVEC_CONTINUOUS,
            StateMode.NO_WRENCH,
        )
        converted = encode_legacy_observation_states(
            [_legacy_state(first), _legacy_state(second)],
            layout=layout,
        )
        a = converted[0][layout.orientation_slice]
        b = converted[1][layout.orientation_slice]
        jump = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
        self.assertLess(jump, 0.01)

    def test_metadata_is_representation_explicit(self):
        layout = model_state_layout(
            OrientationRepresentation.QUATERNION,
            StateMode.FULL,
        )
        metadata = layout.to_metadata()
        self.assertEqual(metadata["schema_version"], "doosan_model_state_profile_v1")
        self.assertEqual(metadata["orientation_representation"], "quaternion")
        self.assertEqual(metadata["state_mode"], "full")
        self.assertEqual(metadata["state_dim"], 26)
        self.assertEqual(metadata["orientation_convention"]["ordering"], "wxyz")
        self.assertEqual(metadata["wrench_policy"], "final_six_channels")

    def test_rejects_non_25d_source(self):
        layout = model_state_layout()
        with self.assertRaises(ModelStateProfileError):
            encode_legacy_observation_states([[0.0] * 24], layout=layout)


if __name__ == "__main__":
    unittest.main()
