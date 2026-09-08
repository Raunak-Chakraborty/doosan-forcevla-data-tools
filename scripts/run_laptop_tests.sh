#!/usr/bin/env bash
set -euo pipefail

REPO="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.." &&
    pwd
)"

PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO"

echo "============================================================"
echo "DOOSAN FORCEVLA DATA TOOLS — PORTABLE TEST SUITE"
echo "============================================================"

echo
echo "===== ISOLATED DEPENDENCY-DOCTOR TEST ====="

(
    cd tests

    PYTHONPATH="../src" \
    "$PYTHON_BIN" -m unittest \
        -v \
        test_processed_to_lerobot_v21.ProcessedToLeRobotV21Tests.test_doctor_reports_without_forbidden_imports
)

echo
echo "ISOLATED_DOCTOR_TEST_GATE=PASS"

echo
echo "===== REMAINING TESTS ====="

PYTHONPATH=src "$PYTHON_BIN" - <<'PY'
import sys
import unittest

EXCLUDED_SUFFIX = (
    "test_processed_to_lerobot_v21."
    "ProcessedToLeRobotV21Tests."
    "test_doctor_reports_without_forbidden_imports"
)

loader = unittest.TestLoader()
discovered = loader.discover("tests")

def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item

all_tests = list(flatten(discovered))

excluded = [
    test
    for test in all_tests
    if test.id().endswith(EXCLUDED_SUFFIX)
]

selected = [
    test
    for test in all_tests
    if not test.id().endswith(EXCLUDED_SUFFIX)
]

print("discovered_test_count =", len(all_tests))
print("isolated_test_count =", len(excluded))
print("remaining_test_count =", len(selected))

if len(excluded) != 1:
    print("TEST_SELECTION_GATE=FAIL")
    raise SystemExit(1)

result = unittest.TextTestRunner(
    verbosity=1,
    stream=sys.stdout,
).run(
    unittest.TestSuite(selected)
)

print("remaining_tests_run =", result.testsRun)
print("failures =", len(result.failures))
print("errors =", len(result.errors))
print("skipped =", len(result.skipped))

if not result.wasSuccessful():
    print("REMAINING_TEST_SUITE_GATE=FAIL")
    raise SystemExit(1)

if result.testsRun != len(selected):
    print("TEST_COUNT_GATE=FAIL")
    raise SystemExit(1)

print("REMAINING_TEST_SUITE_GATE=PASS")
PY

echo
echo "============================================================"
echo "DATA_TOOLS_PORTABLE_TESTS=PASS"
echo "============================================================"
