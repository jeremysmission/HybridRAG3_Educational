# ===========================================================================
# test_all.py -- Test Suite Aggregator
# ===========================================================================
#
# WHAT THIS IS:
#   A lightweight aggregator that confirms all test modules are present.
#   The quality audit (tests/run_audit.py) checks for this file by name.
#
# INTERNET ACCESS: NONE
# ===========================================================================

from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent

EXPECTED_TEST_FILES = [
    "virtual_test_framework.py",
    "virtual_test_phase1_foundation.py",
    "virtual_test_phase2_exhaustive.py",
    "virtual_test_phase4_exhaustive.py",
]


def test_all_test_files_present():
    """Verify all expected test modules exist on disk."""
    missing = []
    for filename in EXPECTED_TEST_FILES:
        path = TEST_DIR / filename
        if not path.exists():
            missing.append(filename)
    assert len(missing) == 0, f"Missing test files: {missing}"