import math
import unittest

from doosan_forcevla_data.convert.doosan_force_proprio_v1 import (
    rotation_matrix_to_rotvec as legacy_rotation_matrix_to_rotvec,
    rotation_vector_to_matrix as legacy_rotation_vector_to_matrix,
)
from doosan_forcevla_data.convert.orientation_representation_v1 import (
    OrientationRepresentation,
    OrientationRepresentationError,
    decode_orientation,
    encode_orientation,
    lift_continuous_rotvec,
    matrix_to_principal_rotvec,
    matrix_to_quaternion_wxyz,
    matrix_to_rotation6d,
    model_state_dimension,
    orientation_dimension,
    quaternion_wxyz_to_matrix,
    resolve_orientation_representation,
    rotation6d_to_matrix,
    rotvec_to_matrix,
    validate_rotation_matrix,
)


def _assert_matrix_close(testcase, actual, expected, places=12):
    testcase.assertEqual(len(actual), 3)
    testcase.assertEqual(len(expected), 3)
    for row in range(3):
        for col in range(3):
            testcase.assertAlmostEqual(actual[row][col], expected[row][col], places=places)


def _rz(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return (
        (c, -s, 0.0),
        (s, c, 0.0),
        (0.0, 0.0, 1.0),
    )


def _rx(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return (
        (1.0, 0.0, 0.0),
        (0.0, c, -s),
        (0.0, s, c),
    )


def _matmul(left, right):
    return tuple(
        tuple(
            sum(left[row][k] * right[k][col] for k in range(3))
            for col in range(3)
        )
        for row in range(3)
    )


class RepresentationContractTests(unittest.TestCase):
    def test_enum_values_are_frozen(self):
        self.assertEqual(
            [item.value for item in OrientationRepresentation],
            [
                "rotvec_principal",
                "rotvec_continuous",
                "quaternion",
                "rotation6d",
            ],
        )

    def test_representation_dimensions(self):
        self.assertEqual(orientation_dimension("rotvec_principal"), 3)
        self.assertEqual(orientation_dimension("rotvec_continuous"), 3)
        self.assertEqual(orientation_dimension("quaternion"), 4)
        self.assertEqual(orientation_dimension("rotation6d"), 6)

    def test_model_state_dimensions_match_planned_profiles(self):
        expected = {
            "rotvec_principal": (19, 25),
            "rotvec_continuous": (19, 25),
            "quaternion": (20, 26),
            "rotation6d": (22, 28),
        }
        for representation, (no_wrench, full) in expected.items():
            with self.subTest(representation=representation):
                self.assertEqual(
                    model_state_dimension(representation, include_wrench=False),
                    no_wrench,
                )
                self.assertEqual(
                    model_state_dimension(representation, include_wrench=True),
                    full,
                )

    def test_invalid_representation_rejected(self):
        with self.assertRaises(OrientationRepresentationError):
            resolve_orientation_representation("euler")
        with self.assertRaises(OrientationRepresentationError):
            model_state_dimension("quaternion", include_wrench=1)


class PrincipalRotvecCompatibilityTests(unittest.TestCase):
    def test_new_principal_rotvec_matches_existing_patch5_wrapper(self):
        matrices = [
            _rz(0.0),
            _rz(0.3),
            _rz(-1.7),
            _rx(math.pi),
            _matmul(_rz(0.7), _rx(-0.4)),
            _matmul(_rz(-2.9), _rx(1.2)),
        ]
        for matrix in matrices:
            with self.subTest(matrix=matrix):
                self.assertEqual(
                    matrix_to_principal_rotvec(matrix),
                    legacy_rotation_matrix_to_rotvec(matrix),
                )

    def test_new_rotvec_exponential_matches_existing_patch5_wrapper(self):
        rotvecs = [
            (0.0, 0.0, 0.0),
            (0.1, -0.2, 0.3),
            (math.pi, 0.0, 0.0),
            (0.0, 0.0, 2.0 * math.pi),
            (-0.3, 1.2, -2.4),
        ]
        for rotvec in rotvecs:
            with self.subTest(rotvec=rotvec):
                self.assertEqual(
                    rotvec_to_matrix(rotvec),
                    legacy_rotation_vector_to_matrix(rotvec),
                )

    def test_principal_angle_never_exceeds_pi(self):
        for degrees in range(-720, 721, 5):
            rotvec = matrix_to_principal_rotvec(_rz(math.radians(degrees)))
            self.assertLessEqual(
                math.sqrt(sum(value * value for value in rotvec)),
                math.pi + 1e-12,
            )


class QuaternionTests(unittest.TestCase):
    def test_identity_is_explicit_wxyz(self):
        self.assertEqual(
            matrix_to_quaternion_wxyz(_rz(0.0)),
            (1.0, 0.0, 0.0, 0.0),
        )

    def test_quaternion_round_trip_preserves_rotation(self):
        matrices = [
            _rz(0.0),
            _rz(math.pi / 2.0),
            _rx(math.pi),
            _matmul(_rz(0.7), _rx(-0.4)),
            _matmul(_rz(-2.8), _rx(1.3)),
        ]
        for matrix in matrices:
            with self.subTest(matrix=matrix):
                quat = matrix_to_quaternion_wxyz(matrix)
                self.assertEqual(len(quat), 4)
                self.assertAlmostEqual(sum(value * value for value in quat), 1.0, places=12)
                _assert_matrix_close(
                    self,
                    quaternion_wxyz_to_matrix(quat),
                    matrix,
                    places=12,
                )

    def test_exact_pi_sign_is_deterministic(self):
        positive = matrix_to_quaternion_wxyz(rotvec_to_matrix((math.pi, 0.0, 0.0)))
        negative = matrix_to_quaternion_wxyz(rotvec_to_matrix((-math.pi, 0.0, 0.0)))
        for actual in (positive, negative):
            self.assertAlmostEqual(actual[0], 0.0, places=12)
            self.assertAlmostEqual(actual[1], 1.0, places=12)
            self.assertAlmostEqual(actual[2], 0.0, places=12)
            self.assertAlmostEqual(actual[3], 0.0, places=12)

    def test_zero_and_two_pi_have_same_quaternion(self):
        q0 = matrix_to_quaternion_wxyz(rotvec_to_matrix((0.0, 0.0, 0.0)))
        q2pi = matrix_to_quaternion_wxyz(rotvec_to_matrix((0.0, 0.0, 2.0 * math.pi)))
        for left, right in zip(q0, q2pi):
            self.assertAlmostEqual(left, right, places=12)

    def test_zero_quaternion_rejected(self):
        with self.assertRaises(OrientationRepresentationError):
            quaternion_wxyz_to_matrix((0.0, 0.0, 0.0, 0.0))


class Rotation6DTests(unittest.TestCase):
    def test_flattening_is_first_two_columns_not_rows(self):
        representation = matrix_to_rotation6d(_rz(math.pi / 2.0))
        expected = (0.0, 1.0, 0.0, -1.0, 0.0, 0.0)
        for actual, wanted in zip(representation, expected):
            self.assertAlmostEqual(actual, wanted, places=12)

    def test_rotation6d_round_trip_preserves_rotation(self):
        matrices = [
            _rz(0.0),
            _rz(math.pi / 2.0),
            _rx(math.pi),
            _matmul(_rz(0.7), _rx(-0.4)),
            _matmul(_rz(-2.8), _rx(1.3)),
        ]
        for matrix in matrices:
            with self.subTest(matrix=matrix):
                encoded = matrix_to_rotation6d(matrix)
                self.assertEqual(len(encoded), 6)
                _assert_matrix_close(
                    self,
                    rotation6d_to_matrix(encoded),
                    matrix,
                    places=12,
                )

    def test_zero_and_two_pi_have_same_rotation6d(self):
        r0 = matrix_to_rotation6d(rotvec_to_matrix((0.0, 0.0, 0.0)))
        r2pi = matrix_to_rotation6d(rotvec_to_matrix((0.0, 0.0, 2.0 * math.pi)))
        for left, right in zip(r0, r2pi):
            self.assertAlmostEqual(left, right, places=12)

    def test_degenerate_rotation6d_rejected(self):
        with self.assertRaises(OrientationRepresentationError):
            rotation6d_to_matrix((0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        with self.assertRaises(OrientationRepresentationError):
            rotation6d_to_matrix((1.0, 0.0, 0.0, 2.0, 0.0, 0.0))


class ContinuousRotvecTests(unittest.TestCase):
    def test_first_frame_is_principal(self):
        matrix = _rz(math.radians(179.0))
        self.assertEqual(
            lift_continuous_rotvec(matrix, None),
            matrix_to_principal_rotvec(matrix),
        )

    def test_pi_branch_crossing_is_lifted_without_two_pi_jump(self):
        first = lift_continuous_rotvec(_rz(math.radians(179.0)), None)
        second = lift_continuous_rotvec(_rz(math.radians(181.0)), first)

        self.assertAlmostEqual(first[2], math.radians(179.0), places=12)
        self.assertAlmostEqual(second[2], math.radians(181.0), places=12)
        self.assertAlmostEqual(second[2] - first[2], math.radians(2.0), places=12)
        _assert_matrix_close(self, rotvec_to_matrix(second), _rz(math.radians(181.0)))

    def test_exact_identity_preserves_nearest_winding(self):
        previous = (0.0, 0.0, math.radians(359.0))
        identity_lift = lift_continuous_rotvec(_rz(0.0), previous)
        after = lift_continuous_rotvec(_rz(math.radians(1.0)), identity_lift)

        self.assertAlmostEqual(identity_lift[2], 2.0 * math.pi, places=12)
        self.assertAlmostEqual(after[2], math.radians(361.0), places=12)
        _assert_matrix_close(self, rotvec_to_matrix(identity_lift), _rz(0.0))
        _assert_matrix_close(self, rotvec_to_matrix(after), _rz(math.radians(1.0)))

    def test_continuous_lift_preserves_physical_rotation_across_long_sequence(self):
        previous = None
        lifted_values = []
        for degrees in range(0, 721, 5):
            matrix = _rz(math.radians(degrees))
            lifted = lift_continuous_rotvec(matrix, previous)
            _assert_matrix_close(self, rotvec_to_matrix(lifted), matrix, places=11)
            lifted_values.append(lifted[2])
            previous = lifted

        for before, after in zip(lifted_values, lifted_values[1:]):
            self.assertLess(abs(after - before), math.radians(6.0))
        self.assertAlmostEqual(lifted_values[-1], math.radians(720.0), places=10)


class GenericEncodeDecodeTests(unittest.TestCase):
    def test_all_representations_decode_to_same_physical_rotation(self):
        matrix = _matmul(_rz(1.2), _rx(-0.8))
        for representation in OrientationRepresentation:
            with self.subTest(representation=representation.value):
                encoded = encode_orientation(
                    matrix,
                    representation,
                    previous_continuous_rotvec_rad=None,
                )
                self.assertEqual(len(encoded), orientation_dimension(representation))
                decoded = decode_orientation(encoded, representation)
                _assert_matrix_close(self, decoded, matrix, places=11)

    def test_rotation_matrix_validation_is_fail_closed(self):
        with self.assertRaises(OrientationRepresentationError):
            validate_rotation_matrix(((1.0, 0.0), (0.0, 1.0)))
        with self.assertRaises(OrientationRepresentationError):
            validate_rotation_matrix(
                (
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, -1.0),
                )
            )


if __name__ == "__main__":
    unittest.main()
