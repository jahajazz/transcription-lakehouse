"""
Snapshot validation for lakehouse snapshots.

Validates snapshot structure, manifest integrity, file inventory, and checksums.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lakehouse.snapshot.artifacts import calculate_sha256
from lakehouse.logger import get_default_logger


logger = get_default_logger()


class SnapshotValidator:
    """
    Validates lakehouse snapshot integrity.
    
    Performs multi-stage validation including existence checks, manifest
    structure validation, file inventory verification, and QA status reporting.
    """
    
    def __init__(self, snapshot_path: Path):
        """
        Initialize snapshot validator.
        
        Args:
            snapshot_path: Path to snapshot root directory
        """
        self.snapshot_path = Path(snapshot_path)
        logger.info(f"Initialized snapshot validator for: {self.snapshot_path}")
    
    def validate(self) -> Dict[str, Any]:
        """
        Perform complete snapshot validation.
        
        Orchestrates all validation checks and returns comprehensive results.
        
        Returns:
            Validation result dictionary with status, checks, warnings, and errors
        
        Example:
            >>> validator = SnapshotValidator(Path("snapshots/v1.0.0"))
            >>> result = validator.validate()
            >>> result['status']
            'PASS'
        """
        return validate_snapshot(self.snapshot_path)


def validate_existence(snapshot_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate snapshot directory exists and is readable.
    
    Checks that the snapshot directory exists and is accessible.
    
    Args:
        snapshot_path: Path to snapshot root directory
    
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    
    Example:
        >>> success, error = validate_existence(Path("snapshots/v1.0.0"))
        >>> success
        True
    """
    if not snapshot_path.exists():
        error_msg = f"Snapshot directory does not exist: {snapshot_path}"
        logger.error(error_msg)
        return (False, error_msg)
    
    if not snapshot_path.is_dir():
        error_msg = f"Snapshot path is not a directory: {snapshot_path}"
        logger.error(error_msg)
        return (False, error_msg)
    
    # Check if directory is readable by attempting to list contents
    try:
        list(snapshot_path.iterdir())
        logger.debug(f"Snapshot directory exists and is readable: {snapshot_path}")
        return (True, None)
    
    except PermissionError as e:
        error_msg = f"Snapshot directory is not readable: {snapshot_path}"
        logger.error(f"{error_msg}: {e}")
        return (False, error_msg)
    
    except Exception as e:
        error_msg = f"Failed to access snapshot directory: {snapshot_path}"
        logger.error(f"{error_msg}: {e}")
        return (False, error_msg)


def validate_manifest_structure(
    manifest_path: Path,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Validate manifest structure and required fields.
    
    Loads manifest JSON and validates that all required keys are present
    with correct types.
    
    Args:
        manifest_path: Path to lake_manifest.json file
    
    Returns:
        Tuple of (success: bool, manifest_dict: Dict, error_message: Optional[str])
    
    Example:
        >>> success, manifest, error = validate_manifest_structure(Path("lake_manifest.json"))
        >>> success
        True
        >>> 'lake_version' in manifest
        True
    """
    if not manifest_path.exists():
        error_msg = f"Manifest file not found: {manifest_path}"
        logger.error(error_msg)
        return (False, {}, error_msg)
    
    try:
        # Load manifest JSON
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        logger.debug(f"Loaded manifest from {manifest_path}")
        
        # Required top-level keys
        required_keys = [
            "lake_version",
            "created_at",
            "producer",
            "contracts",
            "files",
            "qa_status",
        ]
        
        # Check for missing keys
        missing_keys = [key for key in required_keys if key not in manifest]
        if missing_keys:
            error_msg = f"Manifest missing required keys: {', '.join(missing_keys)}"
            logger.error(error_msg)
            return (False, manifest, error_msg)
        
        # Validate types
        type_errors = []
        
        if not isinstance(manifest.get("lake_version"), str):
            type_errors.append("lake_version must be string")
        
        if not isinstance(manifest.get("created_at"), str):
            type_errors.append("created_at must be string")
        
        if not isinstance(manifest.get("producer"), dict):
            type_errors.append("producer must be object")
        else:
            # Check producer sub-keys
            producer = manifest["producer"]
            if "repo" not in producer or "commit" not in producer or "task" not in producer:
                type_errors.append("producer missing required fields (repo, commit, task)")
        
        if not isinstance(manifest.get("contracts"), dict):
            type_errors.append("contracts must be object")
        else:
            # Check contracts sub-keys
            contracts = manifest["contracts"]
            if "schemas" not in contracts:
                type_errors.append("contracts missing required field: schemas")
        
        if not isinstance(manifest.get("files"), list):
            type_errors.append("files must be array")
        
        if not isinstance(manifest.get("qa_status"), dict):
            type_errors.append("qa_status must be object")
        else:
            # Check qa_status sub-keys
            qa_status = manifest["qa_status"]
            required_qa_keys = ["state", "summary", "provisional"]
            missing_qa_keys = [key for key in required_qa_keys if key not in qa_status]
            if missing_qa_keys:
                type_errors.append(f"qa_status missing required fields: {', '.join(missing_qa_keys)}")
        
        if type_errors:
            error_msg = f"Manifest type validation errors: {'; '.join(type_errors)}"
            logger.error(error_msg)
            return (False, manifest, error_msg)
        
        logger.info("Manifest structure validation passed")
        return (True, manifest, None)
    
    except json.JSONDecodeError as e:
        error_msg = f"Manifest is not valid JSON: {e}"
        logger.error(error_msg)
        return (False, {}, error_msg)
    
    except Exception as e:
        error_msg = f"Failed to validate manifest: {e}"
        logger.error(error_msg)
        return (False, {}, error_msg)


def validate_inventory(
    snapshot_path: Path,
    manifest: Dict[str, Any],
) -> List[Tuple[str, bool, Optional[str]]]:
    """
    Validate file inventory matches manifest.
    
    Checks that each file listed in manifest exists, matches recorded byte
    size, and matches recorded SHA-256 checksum.
    
    Args:
        snapshot_path: Path to snapshot root directory
        manifest: Loaded manifest dictionary
    
    Returns:
        List of tuples (file_path, success, error_message) for each file
    
    Example:
        >>> results = validate_inventory(Path("snapshot"), manifest)
        >>> all(success for _, success, _ in results)
        True
    """
    results = []
    files = manifest.get("files", [])
    
    logger.info(f"Validating inventory of {len(files)} files")
    
    for file_entry in files:
        file_path_str = file_entry.get("path", "")
        expected_bytes = file_entry.get("bytes", 0)
        expected_sha256 = file_entry.get("sha256", "")
        
        # Construct full path
        file_path = snapshot_path / file_path_str
        
        # Check file exists
        if not file_path.exists():
            error_msg = f"File not found: {file_path_str}"
            logger.error(error_msg)
            results.append((file_path_str, False, error_msg))
            continue
        
        # Check file size
        actual_bytes = file_path.stat().st_size
        if actual_bytes != expected_bytes:
            error_msg = f"File size mismatch: {file_path_str} (expected {expected_bytes}, got {actual_bytes})"
            logger.error(error_msg)
            results.append((file_path_str, False, error_msg))
            continue
        
        # Check SHA-256 checksum
        try:
            actual_sha256 = calculate_sha256(file_path)
            if actual_sha256 != expected_sha256:
                error_msg = f"Checksum mismatch: {file_path_str}"
                logger.error(error_msg)
                results.append((file_path_str, False, error_msg))
                continue
            
            # All checks passed
            logger.debug(f"Validated: {file_path_str}")
            results.append((file_path_str, True, None))
        
        except Exception as e:
            error_msg = f"Failed to validate {file_path_str}: {e}"
            logger.error(error_msg)
            results.append((file_path_str, False, error_msg))
    
    successful = sum(1 for _, success, _ in results if success)
    logger.info(f"Inventory validation: {successful}/{len(files)} files passed")
    
    return results


def validate_snapshot(snapshot_path: Path) -> Dict[str, Any]:
    """
    Orchestrate complete snapshot validation.
    
    Performs all validation checks:
    1. Existence check
    2. Manifest structure validation
    3. File inventory verification
    4. QA status reporting (warnings only, not failures)
    
    Args:
        snapshot_path: Path to snapshot root directory
    
    Returns:
        Dictionary with validation results:
        - status: "PASS" or "FAIL"
        - checks: List of check results
        - warnings: List of warning messages
        - errors: List of error messages
        - manifest: Loaded manifest (if available)
        - files_validated: Number of files validated
        - files_passed: Number of files that passed
    
    Example:
        >>> result = validate_snapshot(Path("snapshots/v1.0.0"))
        >>> result['status']
        'PASS'
    """
    logger.info(f"Starting validation of snapshot: {snapshot_path}")
    
    checks = []
    warnings = []
    errors = []
    manifest = {}
    files_validated = 0
    files_passed = 0
    
    # Check 1: Existence
    logger.info("Check 1: Validating snapshot existence")
    exists_ok, exists_error = validate_existence(snapshot_path)
    checks.append({
        "name": "Existence Check",
        "passed": exists_ok,
        "error": exists_error,
    })
    
    if not exists_ok:
        logger.error("Existence check failed, aborting validation")
        return {
            "status": "FAIL",
            "checks": checks,
            "warnings": warnings,
            "errors": [exists_error],
            "manifest": {},
            "files_validated": 0,
            "files_passed": 0,
        }
    
    # Check 2: Manifest structure
    logger.info("Check 2: Validating manifest structure")
    manifest_path = snapshot_path / "lake_manifest.json"
    manifest_ok, manifest, manifest_error = validate_manifest_structure(manifest_path)
    checks.append({
        "name": "Manifest Structure",
        "passed": manifest_ok,
        "error": manifest_error,
    })
    
    if not manifest_ok:
        logger.error("Manifest validation failed, aborting validation")
        return {
            "status": "FAIL",
            "checks": checks,
            "warnings": warnings,
            "errors": [manifest_error],
            "manifest": manifest,
            "files_validated": 0,
            "files_passed": 0,
        }
    
    # Check 3: File inventory
    logger.info("Check 3: Validating file inventory")
    inventory_results = validate_inventory(snapshot_path, manifest)
    files_validated = len(inventory_results)
    files_passed = sum(1 for _, success, _ in inventory_results if success)
    
    # Collect inventory errors
    inventory_errors = [error for _, success, error in inventory_results if not success and error]
    
    checks.append({
        "name": "File Inventory",
        "passed": len(inventory_errors) == 0,
        "files_validated": files_validated,
        "files_passed": files_passed,
    })
    
    if inventory_errors:
        errors.extend(inventory_errors)
    
    # Check 4: QA Status (warning only, per FR-25)
    logger.info("Check 4: Checking QA status")
    qa_status = manifest.get("qa_status", {})
    qa_state = qa_status.get("state", "UNKNOWN")
    qa_summary = qa_status.get("summary", "")
    
    checks.append({
        "name": "QA Status",
        "state": qa_state,
        "summary": qa_summary,
    })
    
    # Generate warnings for non-PASS QA states (FR-25)
    if qa_state == "UNKNOWN":
        warning_msg = f"QA Status is UNKNOWN: {qa_summary}"
        warnings.append(warning_msg)
        logger.warning(warning_msg)
    elif qa_state == "FAIL":
        warning_msg = f"QA Status is FAIL: {qa_summary}"
        warnings.append(warning_msg)
        logger.warning(warning_msg)
    else:
        logger.info(f"QA Status is {qa_state}")
    
    # Determine overall status
    # Only structural/inventory errors cause FAIL (FR-25)
    status = "FAIL" if errors else "PASS"
    
    result = {
        "status": status,
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
        "manifest": manifest,
        "files_validated": files_validated,
        "files_passed": files_passed,
    }
    
    logger.info(f"Validation complete: {status}")
    
    return result


def generate_validation_report(validation_result: Dict[str, Any]) -> str:
    """
    Generate human-readable validation report.
    
    Formats validation results as a clear text report with checks performed,
    warnings, errors, and summary statistics.
    
    Args:
        validation_result: Result dictionary from validate_snapshot()
    
    Returns:
        Formatted validation report as string
    
    Example:
        >>> report = generate_validation_report(result)
        >>> 'PASS' in report or 'FAIL' in report
        True
    """
    lines = []
    
    # Header
    lines.append("=" * 60)
    lines.append("SNAPSHOT VALIDATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Overall status
    status = validation_result.get("status", "UNKNOWN")
    status_symbol = "✓" if status == "PASS" else "✗"
    lines.append(f"Overall Status: {status_symbol} {status}")
    lines.append("")
    
    # Summary statistics
    lines.append("Summary Statistics:")
    lines.append(f"  Files Validated: {validation_result.get('files_validated', 0)}")
    lines.append(f"  Files Passed: {validation_result.get('files_passed', 0)}")
    
    total_bytes = 0
    manifest = validation_result.get("manifest", {})
    files = manifest.get("files", [])
    for file_entry in files:
        total_bytes += file_entry.get("bytes", 0)
    
    lines.append(f"  Total Bytes Verified: {total_bytes:,}")
    lines.append("")
    
    # Checks performed
    lines.append("Checks Performed:")
    checks = validation_result.get("checks", [])
    for i, check in enumerate(checks, 1):
        check_name = check.get("name", "Unknown")
        
        if "passed" in check:
            check_passed = check.get("passed", False)
            symbol = "✓" if check_passed else "✗"
            status_text = "PASS" if check_passed else "FAIL"
            lines.append(f"  {i}. {check_name}: {symbol} {status_text}")
            
            if check_name == "File Inventory":
                lines.append(f"     - Validated: {check.get('files_validated', 0)} files")
                lines.append(f"     - Passed: {check.get('files_passed', 0)} files")
        
        elif "state" in check:
            # QA Status check
            qa_state = check.get("state", "UNKNOWN")
            qa_summary = check.get("summary", "")
            lines.append(f"  {i}. {check_name}: {qa_state}")
            if qa_summary:
                lines.append(f"     Summary: {qa_summary}")
    
    lines.append("")
    
    # Warnings
    warnings = validation_result.get("warnings", [])
    if warnings:
        lines.append("Warnings:")
        for warning in warnings:
            lines.append(f"  ⚠ {warning}")
        lines.append("")
    
    # Errors
    errors = validation_result.get("errors", [])
    if errors:
        lines.append("Errors:")
        for error in errors:
            lines.append(f"  ✗ {error}")
        lines.append("")
    
    # Footer
    if status == "PASS" and warnings:
        lines.append("Validation PASSED with warnings.")
    elif status == "PASS":
        lines.append("Validation PASSED - All checks successful.")
    else:
        lines.append("Validation FAILED - See errors above.")
    
    lines.append("=" * 60)
    
    report = "\n".join(lines)
    
    logger.debug("Generated validation report")
    
    return report

