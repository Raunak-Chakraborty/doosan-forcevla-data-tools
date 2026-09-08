#!/usr/bin/env bash
set -euo pipefail

REPO="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.." &&
    pwd
)"

VENV="${DOOSAN_FORCEVLA_DATA_VENV:-$HOME/.venvs/doosan-forcevla-data-tools}"

MODE="${1:-core}"

case "$MODE" in
    core)
        ;;
    lerobot-v21)
        ;;
    check-only)
        ;;
    *)
        echo "Usage:"
        echo "  $0"
        echo "  $0 core"
        echo "  $0 lerobot-v21"
        echo "  $0 check-only"
        exit 2
        ;;
esac

echo "============================================================"
echo "DOOSAN FORCEVLA DATA TOOLS — LAPTOP BOOTSTRAP"
echo "============================================================"

echo "repo=$REPO"
echo "venv=$VENV"
echo "mode=$MODE"

command -v python3 >/dev/null 2>&1 || {
    echo "PYTHON_GATE=FAIL"
    exit 1
}

python3 - <<'PY'
import sys

print("python_executable =", sys.executable)
print("python_version =", sys.version.split()[0])

if sys.version_info < (3, 10):
    raise SystemExit("Python >=3.10 is required.")

print("PYTHON_GATE=PASS")
PY

command -v git >/dev/null 2>&1 || {
    echo "GIT_GATE=FAIL"
    exit 1
}

if [[ ! -d "$REPO/.git" ]]; then
    echo "REPOSITORY_GATE=FAIL"
    exit 1
fi

echo "REPOSITORY_GATE=PASS"

echo "ffmpeg=$(command -v ffmpeg 2>/dev/null || echo NOT_FOUND)"
echo "ffprobe=$(command -v ffprobe 2>/dev/null || echo NOT_FOUND)"

if [[ "$MODE" == "check-only" ]]; then
    echo
    echo "BOOTSTRAP_CHECK_ONLY=PASS"
    exit 0
fi

if [[ ! -d "$VENV" ]]; then
    mkdir -p "$(dirname "$VENV")"

    python3 -m venv "$VENV"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "VENV_GATE=FAIL"
    echo
    echo "If Ubuntu reports that ensurepip/venv is unavailable,"
    echo "install the Python venv package appropriate for the laptop."
    exit 1
fi

echo "VENV_GATE=PASS"

source "$VENV/bin/activate"

python -m pip install \
    --upgrade \
    pip \
    setuptools \
    wheel

if [[ "$MODE" == "lerobot-v21" ]]; then

    python -m pip install \
        -e "${REPO}[lerobot-v21]"

else

    python -m pip install \
        -e "$REPO"

fi

echo "EDITABLE_INSTALL_GATE=PASS"

PYTHON_BIN=python \
"$REPO/scripts/run_laptop_tests.sh"

echo
echo "===== OPTIONAL DEPENDENCY REPORT ====="

set +e

PYTHONPATH="$REPO/src" \
python -m doosan_forcevla_data.inspect.check_export_dependencies

DOCTOR_RC=$?

set -e

echo "dependency_doctor_rc=$DOCTOR_RC"

echo
echo "============================================================"
echo "DATA_TOOLS_LAPTOP_BOOTSTRAP=PASS"
echo "============================================================"

echo
echo "Activate later with:"
echo "  source \"$VENV/bin/activate\""
