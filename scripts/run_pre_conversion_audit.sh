#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
ROS_BASE_SETUP="${ROS_BASE_SETUP:-/opt/ros/jazzy/setup.bash}"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 DATASET_ROOT OUTPUT_DIR [audit options...]" >&2
    echo "Example: $0 /path/to/cranfield_assembly /path/to/audit_logs --workers auto" >&2
    exit 2
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "PRECONVERSION_RUNTIME_GATE=FAIL: Python not executable: $PYTHON_BIN" >&2
    exit 3
fi

if [ ! -f "$ROS_BASE_SETUP" ]; then
    echo "PRECONVERSION_RUNTIME_GATE=FAIL: ROS base setup missing: $ROS_BASE_SETUP" >&2
    exit 4
fi

# Probe the complete raw-MCAP ROS message contract in an isolated subshell.
# This is intentionally stronger than merely checking that setup.bash exists:
# stale/incomplete workspaces can exist on the same machine and must not be
# selected if they cannot resolve all message definitions used by the dataset.
probe_ros_contract() (
    candidate="${1:-}"

    set +u
    source "$ROS_BASE_SETUP"
    if [ -n "$candidate" ]; then
        source "$candidate"
    fi
    set -u

    unset PYTHONHOME 2>/dev/null || true
    export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

    "$PYTHON_BIN" -c '
from rosidl_runtime_py.utilities import get_message
from doosan_forcevla_data.ingest.doosan_raw_v1 import DOOSAN_RAW_V1_TOPICS

errors = []
seen = set()
for contract in DOOSAN_RAW_V1_TOPICS:
    if contract.type_name in seen:
        continue
    seen.add(contract.type_name)
    try:
        get_message(contract.type_name)
    except Exception as exc:
        errors.append(f"{contract.type_name}: {type(exc).__name__}: {exc}")

if errors:
    for error in errors:
        print("PRECONVERSION_ROS_TYPE_MISSING=" + error)
    raise SystemExit(1)

print("PRECONVERSION_ROS_CONTRACT_PROBE=PASS")
'
)

SELECTED_WORKSPACE_SETUP=""

if [ -n "${ROS_WORKSPACE_SETUP:-}" ]; then
    if [ ! -f "$ROS_WORKSPACE_SETUP" ]; then
        echo "PRECONVERSION_RUNTIME_GATE=FAIL: ROS_WORKSPACE_SETUP missing: $ROS_WORKSPACE_SETUP" >&2
        exit 5
    fi

    echo "Probing explicit ROS workspace: $ROS_WORKSPACE_SETUP"
    if ! probe_ros_contract "$ROS_WORKSPACE_SETUP"; then
        echo "PRECONVERSION_RUNTIME_GATE=FAIL: explicit ROS workspace does not satisfy the raw-MCAP message contract: $ROS_WORKSPACE_SETUP" >&2
        exit 6
    fi
    SELECTED_WORKSPACE_SETUP="$ROS_WORKSPACE_SETUP"
else
    if [ -n "${ROS_WORKSPACE_CANDIDATES:-}" ]; then
        IFS=':' read -r -a candidates <<< "$ROS_WORKSPACE_CANDIDATES"
    else
        candidates=(
            "$HOME/robotics_thesis/lab_myros2_ws/install/setup.bash"
            "$HOME/robotics_thesis/lab_myros2_git_clone/install/setup.bash"
        )
    fi

    for candidate in "${candidates[@]}"; do
        [ -n "$candidate" ] || continue
        [ -f "$candidate" ] || continue

        echo "Probing ROS workspace candidate: $candidate"
        if probe_ros_contract "$candidate"; then
            SELECTED_WORKSPACE_SETUP="$candidate"
            break
        fi
        echo "Rejected ROS workspace candidate (message contract incomplete): $candidate" >&2
    done

    # Support systems where all required message packages are installed into
    # the base ROS environment and no thesis overlay is necessary.
    if [ -z "$SELECTED_WORKSPACE_SETUP" ]; then
        echo "Probing base ROS environment without a workspace overlay"
        if ! probe_ros_contract ""; then
            echo "PRECONVERSION_RUNTIME_GATE=FAIL: no ROS workspace candidate satisfies the raw-MCAP message contract." >&2
            echo "Set ROS_WORKSPACE_SETUP=/full/path/to/install/setup.bash to select the correct built workspace explicitly." >&2
            exit 7
        fi
    fi
fi

# Source only the selected environment into the production process. Candidate
# probes above ran in subshells, so rejected overlays cannot pollute this shell.
set +u
source "$ROS_BASE_SETUP"
if [ -n "$SELECTED_WORKSPACE_SETUP" ]; then
    source "$SELECTED_WORKSPACE_SETUP"
fi
set -u

unset PYTHONHOME 2>/dev/null || true
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# Final fail-closed check in the exact environment inherited by spawn workers.
if ! probe_ros_contract "$SELECTED_WORKSPACE_SETUP"; then
    echo "PRECONVERSION_RUNTIME_GATE=FAIL: final ROS environment failed the raw-MCAP message contract probe." >&2
    exit 8
fi

echo "PRECONVERSION_ROS_BASE_SETUP=$ROS_BASE_SETUP"
if [ -n "$SELECTED_WORKSPACE_SETUP" ]; then
    echo "PRECONVERSION_ROS_WORKSPACE_SETUP=$SELECTED_WORKSPACE_SETUP"
else
    echo "PRECONVERSION_ROS_WORKSPACE_SETUP=BASE_ONLY"
fi
echo "PRECONVERSION_RUNTIME_GATE=PASS"

# One process per episode is already the parallelism boundary. Prevent BLAS
# libraries inside each worker from multiplying the thread count again.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec "$PYTHON_BIN" -m doosan_forcevla_data.audit.pre_conversion_population_v1 "$@"
