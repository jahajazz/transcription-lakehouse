"""
Tests for snapshot validation.

Tests existence checks, manifest validation, inventory verification, and report generation.
"""

import json
from pathlib import Path

import pytest

from lakehouse.snapshot.validator import (
    SnapshotValidator,
    validate_existence,
    validate_manifest_structure,
    validate_inventory,
    validate_snapshot,
    generate_validation_report,
)


class TestExistenceValidation:
    """Tests for snapshot existence validation."""
    
    def test_validate_existence_success(self, tmp_path):
        """Test validating existing snapshot directory."""
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        
        success, error = validate_existence(snapshot)
        
        assert success is True
        assert error is None
    
    def test_validate_existence_not_exists(self, tmp_path):
        """Test validating non-existent directory."""
        snapshot = tmp_path / "nonexistent"
        
        success, error = validate_existence(snapshot)
        
        assert success is False
        assert error is not None
        assert "does not exist" in error
    
    def test_validate_existence_not_directory(self, tmp_path):
        """Test validating file instead of directory."""
        snapshot = tmp_path / "file.txt"
        snapshot.write_text("not a directory")
        
        success, error = validate_existence(snapshot)
        
        assert success is False
        assert error is not None
        assert "not a directory" in error


class TestManifestValidation:
    """Tests for manifest structure validation."""
    
    def test_validate_manifest_structure_success(self, tmp_path):
        """Test validating valid manifest."""
        manifest = {
            "lake_version": "1.0.0-provisional",
            "created_at": "2025-10-27T12:00:00Z",
            "producer": {
                "repo": "test-repo",
                "commit": "abc123",
                "task": "SNAPSHOT",
                "report_path": None,
            },
            "contracts": {
                "schemas": {"spans": "1.0"},
                "compatibility": {},
            },
            "files": [],
            "qa_status": {
                "state": "PASS",
                "summary": "All good",
                "invariants": {},
                "provisional": True,
            },
        }
        
        manifest_path = tmp_path / "lake_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        success, loaded_manifest, error = validate_manifest_structure(manifest_path)
        
        assert success is True
        assert error is None
        assert loaded_manifest["lake_version"] == "1.0.0-provisional"
    
    def test_validate_manifest_not_found(self, tmp_path):
        """Test validating non-existent manifest."""
        manifest_path = tmp_path / "lake_manifest.json"
        
        success, loaded_manifest, error = validate_manifest_structure(manifest_path)
        
        assert success is False
        assert error is not None
        assert "not found" in error
    
    def test_validate_manifest_invalid_json(self, tmp_path):
        """Test validating malformed JSON."""
        manifest_path = tmp_path / "lake_manifest.json"
        manifest_path.write_text("{ invalid json }")
        
        success, loaded_manifest, error = validate_manifest_structure(manifest_path)
        
        assert success is False
        assert error is not None
        assert "not valid JSON" in error
    
    def test_validate_manifest_missing_keys(self, tmp_path):
        """Test validating manifest with missing required keys."""
        manifest = {
            "lake_version": "1.0.0",
            # Missing other required keys
        }
        
        manifest_path = tmp_path / "lake_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        success, loaded_manifest, error = validate_manifest_structure(manifest_path)
        
        assert success is False
        assert error is not None
        assert "missing required keys" in error.lower()
    
    def test_validate_manifest_wrong_types(self, tmp_path):
        """Test validating manifest with wrong field types."""
        manifest = {
            "lake_version": 123,  # Should be string
            "created_at": "2025-10-27T12:00:00Z",
            "producer": {},
            "contracts": {},
            "files": [],
            "qa_status": {},
        }
        
        manifest_path = tmp_path / "lake_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        success, loaded_manifest, error = validate_manifest_structure(manifest_path)
        
        assert success is False
        assert error is not None
    
    def test_validate_manifest_incomplete_producer(self, tmp_path):
        """Test validating manifest with incomplete producer."""
        manifest = {
            "lake_version": "1.0.0",
            "created_at": "2025-10-27T12:00:00Z",
            "producer": {
                "repo": "test",
                # Missing commit and task
            },
            "contracts": {"schemas": {}},
            "files": [],
            "qa_status": {"state": "PASS", "summary": "", "provisional": True},
        }
        
        manifest_path = tmp_path / "lake_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        success, loaded_manifest, error = validate_manifest_structure(manifest_path)
        
        assert success is False
        assert "producer" in error.lower()


class TestInventoryValidation:
    """Tests for file inventory validation."""
    
    def test_validate_inventory_success(self, tmp_path):
        """Test validating complete file inventory."""
        # Create snapshot structure
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        
        # Create test file
        test_file = snapshot / "test.txt"
        test_file.write_text("test content")
        
        # Calculate actual checksum
        import hashlib
        checksum = hashlib.sha256(b"test content").hexdigest()
        
        # Create manifest
        manifest = {
            "files": [
                {
                    "path": "test.txt",
                    "bytes": len("test content"),
                    "sha256": checksum,
                }
            ]
        }
        
        results = validate_inventory(snapshot, manifest)
        
        assert len(results) == 1
        path, success, error = results[0]
        assert success is True
        assert error is None
    
    def test_validate_inventory_missing_file(self, tmp_path):
        """Test validating inventory with missing file."""
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        
        manifest = {
            "files": [
                {
                    "path": "missing.txt",
                    "bytes": 100,
                    "sha256": "abc123",
                }
            ]
        }
        
        results = validate_inventory(snapshot, manifest)
        
        assert len(results) == 1
        path, success, error = results[0]
        assert success is False
        assert "not found" in error.lower()
    
    def test_validate_inventory_size_mismatch(self, tmp_path):
        """Test validating inventory with file size mismatch."""
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        
        test_file = snapshot / "test.txt"
        test_file.write_text("test content")
        
        manifest = {
            "files": [
                {
                    "path": "test.txt",
                    "bytes": 999,  # Wrong size
                    "sha256": "abc123",
                }
            ]
        }
        
        results = validate_inventory(snapshot, manifest)
        
        assert len(results) == 1
        path, success, error = results[0]
        assert success is False
        assert "size mismatch" in error.lower()
    
    def test_validate_inventory_checksum_mismatch(self, tmp_path):
        """Test validating inventory with checksum mismatch."""
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        
        test_file = snapshot / "test.txt"
        test_file.write_text("test content")
        
        manifest = {
            "files": [
                {
                    "path": "test.txt",
                    "bytes": len("test content"),
                    "sha256": "wrongchecksum123",
                }
            ]
        }
        
        results = validate_inventory(snapshot, manifest)
        
        assert len(results) == 1
        path, success, error = results[0]
        assert success is False
        assert "checksum mismatch" in error.lower()
    
    def test_validate_inventory_multiple_files(self, tmp_path):
        """Test validating inventory with multiple files."""
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        
        # Create files
        file1 = snapshot / "file1.txt"
        file1.write_text("content1")
        file2 = snapshot / "file2.txt"
        file2.write_text("content2")
        
        import hashlib
        checksum1 = hashlib.sha256(b"content1").hexdigest()
        checksum2 = hashlib.sha256(b"content2").hexdigest()
        
        manifest = {
            "files": [
                {"path": "file1.txt", "bytes": len("content1"), "sha256": checksum1},
                {"path": "file2.txt", "bytes": len("content2"), "sha256": checksum2},
            ]
        }
        
        results = validate_inventory(snapshot, manifest)
        
        assert len(results) == 2
        assert all(success for _, success, _ in results)


class TestCompleteValidation:
    """Tests for complete snapshot validation."""
    
    def test_validate_snapshot_success(self, tmp_path):
        """Test complete validation of valid snapshot."""
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        
        # Create test file
        test_file = snapshot / "test.txt"
        test_file.write_text("test")
        
        import hashlib
        checksum = hashlib.sha256(b"test").hexdigest()
        
        # Create manifest
        manifest = {
            "lake_version": "1.0.0",
            "created_at": "2025-10-27T12:00:00Z",
            "producer": {"repo": "test", "commit": "abc", "task": "SNAPSHOT", "report_path": None},
            "contracts": {"schemas": {}},
            "files": [{"path": "test.txt", "bytes": 4, "sha256": checksum}],
            "qa_status": {"state": "PASS", "summary": "OK", "provisional": True},
        }
        
        with open(snapshot / "lake_manifest.json", "w") as f:
            json.dump(manifest, f)
        
        result = validate_snapshot(snapshot)
        
        assert result['status'] == 'PASS'
        assert len(result['checks']) > 0
        assert result['files_validated'] == 1
        assert result['files_passed'] == 1
        assert len(result['errors']) == 0
    
    def test_validate_snapshot_qa_warning(self, tmp_path):
        """Test validation with QA warning (FR-25)."""
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        
        manifest = {
            "lake_version": "1.0.0",
            "created_at": "2025-10-27T12:00:00Z",
            "producer": {"repo": "test", "commit": "abc", "task": "SNAPSHOT", "report_path": None},
            "contracts": {"schemas": {}},
            "files": [],
            "qa_status": {"state": "FAIL", "summary": "Issues detected", "provisional": True},
        }
        
        with open(snapshot / "lake_manifest.json", "w") as f:
            json.dump(manifest, f)
        
        result = validate_snapshot(snapshot)
        
        # Should pass validation despite QA FAIL (FR-25)
        assert result['status'] == 'PASS'
        # Should have warning about QA status
        assert len(result['warnings']) > 0
        assert any('FAIL' in w for w in result['warnings'])
    
    def test_validate_snapshot_qa_unknown_warning(self, tmp_path):
        """Test validation with unknown QA status."""
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        
        manifest = {
            "lake_version": "1.0.0",
            "created_at": "2025-10-27T12:00:00Z",
            "producer": {"repo": "test", "commit": "abc", "task": "SNAPSHOT", "report_path": None},
            "contracts": {"schemas": {}},
            "files": [],
            "qa_status": {"state": "UNKNOWN", "summary": "No QA", "provisional": True},
        }
        
        with open(snapshot / "lake_manifest.json", "w") as f:
            json.dump(manifest, f)
        
        result = validate_snapshot(snapshot)
        
        assert result['status'] == 'PASS'
        assert len(result['warnings']) > 0
        assert any('UNKNOWN' in w for w in result['warnings'])
    
    def test_validate_snapshot_missing_directory(self, tmp_path):
        """Test validation of non-existent snapshot."""
        snapshot = tmp_path / "nonexistent"
        
        result = validate_snapshot(snapshot)
        
        assert result['status'] == 'FAIL'
        assert len(result['errors']) > 0
    
    def test_validate_snapshot_invalid_manifest(self, tmp_path):
        """Test validation with invalid manifest."""
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        
        # Create invalid manifest
        (snapshot / "lake_manifest.json").write_text("{ invalid }")
        
        result = validate_snapshot(snapshot)
        
        assert result['status'] == 'FAIL'
        assert len(result['errors']) > 0


class TestValidationReport:
    """Tests for validation report generation."""
    
    def test_generate_validation_report_pass(self):
        """Test generating report for successful validation."""
        result = {
            "status": "PASS",
            "checks": [
                {"name": "Existence Check", "passed": True, "error": None},
                {"name": "Manifest Structure", "passed": True, "error": None},
            ],
            "warnings": [],
            "errors": [],
            "manifest": {"files": [{"bytes": 100}]},
            "files_validated": 5,
            "files_passed": 5,
        }
        
        report = generate_validation_report(result)
        
        assert "PASS" in report
        assert "Files Validated: 5" in report
        assert "Files Passed: 5" in report
        assert "Total Bytes Verified: 100" in report
    
    def test_generate_validation_report_fail(self):
        """Test generating report for failed validation."""
        result = {
            "status": "FAIL",
            "checks": [
                {"name": "Existence Check", "passed": False, "error": "Not found"},
            ],
            "warnings": [],
            "errors": ["Snapshot not found"],
            "manifest": {},
            "files_validated": 0,
            "files_passed": 0,
        }
        
        report = generate_validation_report(result)
        
        assert "FAIL" in report
        assert "Snapshot not found" in report
    
    def test_generate_validation_report_with_warnings(self):
        """Test generating report with warnings."""
        result = {
            "status": "PASS",
            "checks": [
                {"name": "QA Status", "state": "FAIL", "summary": "Issues found"},
            ],
            "warnings": ["QA Status is FAIL: Issues found"],
            "errors": [],
            "manifest": {"files": []},
            "files_validated": 0,
            "files_passed": 0,
        }
        
        report = generate_validation_report(result)
        
        assert "PASS" in report
        assert "Warnings:" in report
        assert "QA Status is FAIL" in report


class TestSnapshotValidator:
    """Tests for SnapshotValidator class."""
    
    def test_snapshot_validator_initialization(self, tmp_path):
        """Test SnapshotValidator initialization."""
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        
        validator = SnapshotValidator(snapshot)
        
        assert validator.snapshot_path == snapshot
    
    def test_snapshot_validator_validate(self, tmp_path):
        """Test SnapshotValidator.validate() method."""
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        
        manifest = {
            "lake_version": "1.0.0",
            "created_at": "2025-10-27T12:00:00Z",
            "producer": {"repo": "test", "commit": "abc", "task": "SNAPSHOT", "report_path": None},
            "contracts": {"schemas": {}},
            "files": [],
            "qa_status": {"state": "PASS", "summary": "", "provisional": True},
        }
        
        with open(snapshot / "lake_manifest.json", "w") as f:
            json.dump(manifest, f)
        
        validator = SnapshotValidator(snapshot)
        result = validator.validate()
        
        assert result['status'] in ['PASS', 'FAIL']
        assert 'checks' in result

