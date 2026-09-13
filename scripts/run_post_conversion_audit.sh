#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/robotics_thesis/envs/data_conv_thesis_env/bin/python}"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 INPUT_ROOT OUTPUT_DIR [audit options...]" >&2
    echo "Example: $0 /path/to/converted_population /path/to/audit --expected-bundles 184" >&2
    exit 2
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "POSTCONVERSION_RUNTIME_GATE=FAIL: Python not executable: $PYTHON_BIN" >&2
    exit 3
fi

unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION || true
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/src"

"$PYTHON_BIN" -c 'import pyarrow' >/dev/null 2>&1 || {
    echo "POSTCONVERSION_RUNTIME_GATE=FAIL: PyArrow missing in $PYTHON_BIN" >&2
    exit 4
}

echo "POSTCONVERSION_RUNTIME_GATE=PASS"
exec "$PYTHON_BIN" -m doosan_forcevla_data.audit.post_conversion_population_v1 "$@"
