"""Constants for the v0 processed dataset layout."""

MODEL_STATE_DIM = 25
ACTION_DIM = 7
JOINT_COUNT = 6
WRENCH_DIM = 6
QUATERNION_CONVENTION = "xyzw"

EE_POS_FIELDS = ["ee_pos_x", "ee_pos_y", "ee_pos_z"]
EE_QUAT_FIELDS = ["ee_quat_x", "ee_quat_y", "ee_quat_z", "ee_quat_w"]
EE_AXIS_ANGLE_FIELDS = ["ee_axis_angle_x", "ee_axis_angle_y", "ee_axis_angle_z"]
GRIPPER_POS_FIELDS = ["gripper_pos"]
WRENCH_FIELDS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
JOINT_POS_FIELDS = [f"joint_pos_{idx}" for idx in range(JOINT_COUNT)]
JOINT_VEL_FIELDS = [f"joint_vel_{idx}" for idx in range(JOINT_COUNT)]

MODEL_STATE_FIELDS = (
    EE_POS_FIELDS
    + EE_AXIS_ANGLE_FIELDS
    + GRIPPER_POS_FIELDS
    + WRENCH_FIELDS
    + JOINT_POS_FIELDS
    + JOINT_VEL_FIELDS
)

ACTION_FIELDS = [
    "dx",
    "dy",
    "dz",
    "dRx",
    "dRy",
    "dRz",
    "gripper_delta_or_zero",
]

STATE_FIELDS = (
    EE_POS_FIELDS
    + EE_QUAT_FIELDS
    + GRIPPER_POS_FIELDS
    + WRENCH_FIELDS
    + JOINT_POS_FIELDS
    + JOINT_VEL_FIELDS
)

assert len(MODEL_STATE_FIELDS) == MODEL_STATE_DIM
assert len(ACTION_FIELDS) == ACTION_DIM
