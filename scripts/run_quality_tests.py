#!/usr/bin/env python3
"""
Quick test runner for quality assessment integration tests.

Usage:
    python scripts/run_quality_tests.py              # Run all tests
    python scripts/run_quality_tests.py --schema     # Schema validation only
    python scripts/run_quality_tests.py --integration  # Integration tests only
    python scripts/run_quality_tests.py --quick      # Quick tests (no real data)
"""

import sys
import subprocess
from pathlib import Path
import argparse


def run_tests(test_type: str = "all", verbose: bool = True):
    """
    Run quality assessment tests.
    
    Args:
        test_type: Type of tests to run (all, schema, integration, quick)
        verbose: Show verbose output
    """
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"
    
    # Build pytest command
    cmd = ["pytest"]
    
    if test_type == "schema":
        cmd.append(str(tests_dir / "test_quality_schema_validation.py"))
        print("[TEST] Running schema validation tests...")
    elif test_type == "integration":
        cmd.append(str(tests_dir / "test_quality_integration.py"))
        cmd.extend(["-m", "integration"])
        print("[TEST] Running integration tests (requires lakehouse data)...")
    elif test_type == "quick":
        cmd.extend([
            str(tests_dir / "test_quality_schema_validation.py"),
            str(tests_dir / "test_quality_integration.py"),
        ])
        cmd.extend(["-m", "not integration"])
        print("[TEST] Running quick tests (no real data required)...")
    else:  # all
        cmd.extend([
            str(tests_dir / "test_quality_integration.py"),
            str(tests_dir / "test_quality_schema_validation.py"),
        ])
        print("[TEST] Running all quality assessment tests...")
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend([
        "--tb=short",  # Shorter tracebacks
        "--color=yes",  # Colored output
    ])
    
    print(f"Command: {' '.join(cmd)}\n")
    
    # Run tests
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode == 0:
        print("\n[OK] All tests passed!")
    else:
        print("\n[FAIL] Some tests failed!")
        print("\nCommon fixes:")
        print("- Missing key: Update calculator to return expected keys")
        print("- Wrong key: Update reporter to use correct keys from calculator")
        print("- Non-zero assertion: Check column names and data loading")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run quality assessment integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_quality_tests.py                 # Run all tests
  python scripts/run_quality_tests.py --schema        # Schema validation only
  python scripts/run_quality_tests.py --integration   # Integration tests only
  python scripts/run_quality_tests.py --quick         # Quick tests (no real data)
  python scripts/run_quality_tests.py --quiet         # Less verbose output

Test Types:
  --schema       : Validate calculator output schemas (fast, no data required)
  --integration  : Test full pipeline with real data (requires lakehouse)
  --quick        : Run tests that don't need real lakehouse data
  (default)      : Run all tests
        """
    )
    
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Run schema validation tests only"
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only (requires lakehouse data)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests (no real data required)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose output"
    )
    
    args = parser.parse_args()
    
    # Determine test type
    if args.schema:
        test_type = "schema"
    elif args.integration:
        test_type = "integration"
    elif args.quick:
        test_type = "quick"
    else:
        test_type = "all"
    
    verbose = not args.quiet
    
    # Run tests
    exit_code = run_tests(test_type, verbose)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

