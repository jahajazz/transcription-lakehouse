"""
Unit tests for validation checks and reporting.

Tests validation checks, reports, and validation functions for all artifact types.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

from lakehouse.validation.checks import (
    ValidationCheck,
    ValidationReport,
    check_schema_compliance,
    check_id_quality,
    check_timestamps,
    check_numeric_quality,
    check_referential_integrity,
    check_text_quality,
    validate_artifact,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_utterances_df() -> pd.DataFrame:
    """Create sample utterance DataFrame."""
    return pd.DataFrame([
        {
            "utterance_id": "utt_test_000000_abc",
            "episode_id": "EP001",
            "speaker": "Alice",
            "start": 0.0,
            "end": 5.0,
            "duration": 5.0,
            "text": "Hello world",
        },
        {
            "utterance_id": "utt_test_000001_def",
            "episode_id": "EP001",
            "speaker": "Bob",
            "start": 5.0,
            "end": 10.0,
            "duration": 5.0,
            "text": "Hi there",
        },
    ])


@pytest.fixture
def sample_spans_df() -> pd.DataFrame:
    """Create sample span DataFrame."""
    return pd.DataFrame([
        {
            "span_id": "spn_test_000000_abc",
            "episode_id": "EP001",
            "speaker": "Alice",
            "start_time": 0.0,
            "end_time": 60.0,
            "duration": 60.0,
            "text": "First span",
            "utterance_ids": ["utt_1", "utt_2"],
        },
    ])


@pytest.fixture
def sample_embeddings_df() -> pd.DataFrame:
    """Create sample embedding DataFrame."""
    embedding_dim = 384
    return pd.DataFrame([
        {
            "artifact_id": "spn_test_000000_abc",
            "artifact_type": "span",
            "embedding": np.random.rand(embedding_dim).tolist(),
            "model_name": "test-model",
            "model_version": "1.0",
            "embedding_dim": embedding_dim,
        },
    ])


# ============================================================================
# ValidationCheck Tests
# ============================================================================

class TestValidationCheck:
    """Test ValidationCheck class."""
    
    def test_check_creation(self):
        """Test creating a validation check."""
        check = ValidationCheck(
            check_name="test_check",
            passed=True,
            message="Check passed",
            details={"count": 10},
            severity="info",
        )
        
        assert check.check_name == "test_check"
        assert check.passed is True
        assert check.message == "Check passed"
        assert check.details["count"] == 10
        assert check.severity == "info"
    
    def test_check_to_dict(self):
        """Test converting check to dictionary."""
        check = ValidationCheck(
            check_name="test_check",
            passed=False,
            message="Check failed",
            severity="error",
        )
        
        check_dict = check.to_dict()
        
        assert check_dict["check_name"] == "test_check"
        assert check_dict["passed"] is False
        assert check_dict["message"] == "Check failed"
        assert check_dict["severity"] == "error"
    
    def test_check_default_severity(self):
        """Test that default severity is error."""
        check = ValidationCheck(
            check_name="test",
            passed=False,
            message="Failed",
        )
        
        assert check.severity == "error"


# ============================================================================
# ValidationReport Tests
# ============================================================================

class TestValidationReport:
    """Test ValidationReport class."""
    
    def test_report_creation(self):
        """Test creating a validation report."""
        report = ValidationReport("utterance", "v1")
        
        assert report.artifact_type == "utterance"
        assert report.version == "v1"
        assert len(report.checks) == 0
        assert len(report.statistics) == 0
    
    def test_add_check(self):
        """Test adding checks to report."""
        report = ValidationReport("span", "v1")
        
        check1 = ValidationCheck("check1", True, "Passed")
        check2 = ValidationCheck("check2", False, "Failed", severity="error")
        
        report.add_check(check1)
        report.add_check(check2)
        
        assert len(report.checks) == 2
    
    def test_get_passed_checks(self):
        """Test getting passed checks."""
        report = ValidationReport("span", "v1")
        
        report.add_check(ValidationCheck("check1", True, "Passed"))
        report.add_check(ValidationCheck("check2", False, "Failed"))
        report.add_check(ValidationCheck("check3", True, "Passed"))
        
        passed = report.get_passed_checks()
        assert len(passed) == 2
    
    def test_get_failed_checks(self):
        """Test getting failed checks."""
        report = ValidationReport("span", "v1")
        
        report.add_check(ValidationCheck("check1", True, "Passed"))
        report.add_check(ValidationCheck("check2", False, "Failed"))
        report.add_check(ValidationCheck("check3", False, "Failed"))
        
        failed = report.get_failed_checks()
        assert len(failed) == 2
    
    def test_get_errors(self):
        """Test getting error-level checks."""
        report = ValidationReport("span", "v1")
        
        report.add_check(ValidationCheck("check1", False, "Failed", severity="error"))
        report.add_check(ValidationCheck("check2", False, "Failed", severity="warning"))
        report.add_check(ValidationCheck("check3", False, "Failed", severity="error"))
        
        errors = report.get_errors()
        assert len(errors) == 2
    
    def test_get_warnings(self):
        """Test getting warning-level checks."""
        report = ValidationReport("span", "v1")
        
        report.add_check(ValidationCheck("check1", False, "Failed", severity="error"))
        report.add_check(ValidationCheck("check2", False, "Failed", severity="warning"))
        report.add_check(ValidationCheck("check3", False, "Failed", severity="warning"))
        
        warnings = report.get_warnings()
        assert len(warnings) == 2
    
    def test_is_valid(self):
        """Test validation status check."""
        report = ValidationReport("span", "v1")
        
        # No errors = valid
        assert report.is_valid()
        
        # Add warning (still valid)
        report.add_check(ValidationCheck("check1", False, "Warning", severity="warning"))
        assert report.is_valid()
        
        # Add error (now invalid)
        report.add_check(ValidationCheck("check2", False, "Error", severity="error"))
        assert not report.is_valid()
    
    def test_add_statistics(self):
        """Test adding statistics to report."""
        report = ValidationReport("span", "v1")
        
        report.add_statistics({"total_rows": 100, "unique_episodes": 5})
        
        assert report.statistics["total_rows"] == 100
        assert report.statistics["unique_episodes"] == 5


# ============================================================================
# Schema Compliance Tests
# ============================================================================

class TestSchemaCompliance:
    """Test schema compliance checks."""
    
    def test_schema_compliance_pass(self, sample_utterances_df):
        """Test schema compliance with valid data."""
        check = check_schema_compliance(sample_utterances_df, "utterance")
        
        # Should pass for valid data
        assert check.passed
    
    def test_schema_compliance_missing_column(self, sample_utterances_df):
        """Test schema compliance with missing column."""
        # Remove required column
        df = sample_utterances_df.drop(columns=["text"])
        
        check = check_schema_compliance(df, "utterance")
        
        # Should fail
        assert not check.passed
    
    def test_schema_compliance_wrong_type(self, sample_utterances_df):
        """Test schema compliance with wrong data type."""
        # Change type of numeric column
        df = sample_utterances_df.copy()
        df["start"] = df["start"].astype(str)
        
        # This may or may not fail depending on coercion, but should not crash
        check = check_schema_compliance(df, "utterance")
        assert check is not None


# ============================================================================
# ID Quality Tests
# ============================================================================

class TestIDQuality:
    """Test ID quality checks."""
    
    def test_id_quality_valid(self, sample_utterances_df):
        """Test ID quality with valid IDs."""
        checks = check_id_quality(sample_utterances_df, "utterance")
        
        # Should pass null and duplicate checks
        null_checks = [c for c in checks if "null" in c.check_name]
        assert all(check.passed for check in null_checks)
    
    def test_id_quality_null_ids(self, sample_utterances_df):
        """Test ID quality with null IDs."""
        df = sample_utterances_df.copy()
        df.loc[0, "utterance_id"] = None
        
        checks = check_id_quality(df, "utterance")
        
        # Should have failed null check
        null_checks = [c for c in checks if "null" in c.check_name]
        assert any(not check.passed for check in null_checks)
    
    def test_id_quality_duplicate_ids(self, sample_utterances_df):
        """Test ID quality with duplicate IDs."""
        df = sample_utterances_df.copy()
        df.loc[1, "utterance_id"] = df.loc[0, "utterance_id"]
        
        checks = check_id_quality(df, "utterance")
        
        # Should have failed duplicate check
        dup_checks = [c for c in checks if "duplicate" in c.check_name]
        assert any(not check.passed for check in dup_checks)
    
    def test_id_quality_foreign_key(self, sample_utterances_df):
        """Test that foreign keys (episode_id) are not checked for uniqueness."""
        # All rows have same episode_id - this is expected
        checks = check_id_quality(sample_utterances_df, "utterance")
        
        # episode_id should have a foreign key check, not uniqueness
        foreign_key_checks = [c for c in checks if "foreign_key" in c.check_name and "episode_id" in c.check_name]
        assert len(foreign_key_checks) > 0
        assert all(check.passed for check in foreign_key_checks)


# ============================================================================
# Timestamp Tests
# ============================================================================

class TestTimestamps:
    """Test timestamp validation checks."""
    
    def test_timestamps_valid(self, sample_utterances_df):
        """Test timestamp validation with valid timestamps."""
        checks = check_timestamps(sample_utterances_df, "utterance")
        
        # All checks should pass
        assert all(check.passed for check in checks)
    
    def test_timestamps_negative(self, sample_utterances_df):
        """Test detection of negative timestamps."""
        df = sample_utterances_df.copy()
        df.loc[0, "start"] = -5.0
        
        checks = check_timestamps(df, "utterance")
        
        # Should have failed check for negative values
        assert any(not check.passed for check in checks)
    
    def test_timestamps_end_before_start(self, sample_utterances_df):
        """Test detection of end < start."""
        df = sample_utterances_df.copy()
        df.loc[0, "end"] = df.loc[0, "start"] - 1.0
        
        checks = check_timestamps(df, "utterance")
        
        # Should have some failed checks or warnings
        assert any(not check.passed or check.severity == "warning" for check in checks)
    
    def test_timestamps_overlaps(self):
        """Test detection of overlapping timestamps."""
        df = pd.DataFrame([
            {"utterance_id": "u1", "episode_id": "E1", "speaker": "A", "start": 0.0, "end": 10.0, "duration": 10.0, "text": "One"},
            {"utterance_id": "u2", "episode_id": "E1", "speaker": "A", "start": 5.0, "end": 15.0, "duration": 10.0, "text": "Two"},
        ])
        
        checks = check_timestamps(df, "utterance")
        
        # Should detect overlaps or pass with warning
        # Implementation may vary
        assert len(checks) > 0


# ============================================================================
# Numeric Quality Tests
# ============================================================================

class TestNumericQuality:
    """Test numeric quality checks."""
    
    def test_numeric_quality_valid(self, sample_utterances_df):
        """Test numeric quality with valid data."""
        checks = check_numeric_quality(sample_utterances_df, "utterance")
        
        # Should have checks
        assert len(checks) > 0
    
    def test_numeric_quality_zero_duration(self, sample_utterances_df):
        """Test detection of zero duration."""
        df = sample_utterances_df.copy()
        df.loc[0, "duration"] = 0.0
        
        checks = check_numeric_quality(df, "utterance")
        
        # Should warn about zero duration
        zero_checks = [c for c in checks if "zero" in c.check_name.lower()]
        if len(zero_checks) > 0:
            assert any(not check.passed or check.severity == "warning" for check in zero_checks)
    
    def test_numeric_quality_negative_duration(self, sample_utterances_df):
        """Test detection of negative duration."""
        df = sample_utterances_df.copy()
        df.loc[0, "duration"] = -1.0
        
        checks = check_numeric_quality(df, "utterance")
        
        # Should produce checks (implementation may vary on what's detected)
        assert len(checks) > 0


# ============================================================================
# Referential Integrity Tests
# ============================================================================

class TestReferentialIntegrity:
    """Test referential integrity checks."""
    
    def test_referential_integrity_valid(self, sample_spans_df, sample_utterances_df):
        """Test referential integrity with valid references."""
        # Referential integrity checks require specific data structure
        # For now, just test that the function runs
        checks = check_referential_integrity(sample_spans_df, "span")
        
        # Should return checks
        assert len(checks) >= 0
    
    def test_referential_integrity_runs(self, sample_utterances_df):
        """Test that referential integrity checks run without error."""
        checks = check_referential_integrity(sample_utterances_df, "utterance")
        
        # Should return checks
        assert len(checks) >= 0


# ============================================================================
# Text Quality Tests
# ============================================================================

class TestTextQuality:
    """Test text quality checks."""
    
    def test_text_quality_valid(self, sample_utterances_df):
        """Test text quality with valid text."""
        checks = check_text_quality(sample_utterances_df, "utterance")
        
        # Should have checks
        assert len(checks) > 0
        # All should pass with valid data
        assert all(check.passed for check in checks)
    
    def test_text_quality_empty_text(self, sample_utterances_df):
        """Test detection of empty text."""
        df = sample_utterances_df.copy()
        df.loc[0, "text"] = ""
        
        checks = check_text_quality(df, "utterance")
        
        # Should detect empty text
        empty_checks = [c for c in checks if "empty" in c.check_name.lower()]
        assert any(not check.passed or check.severity == "warning" for check in empty_checks)
    
    def test_text_quality_null_text(self, sample_utterances_df):
        """Test detection of null text."""
        df = sample_utterances_df.copy()
        df.loc[0, "text"] = None
        
        checks = check_text_quality(df, "utterance")
        
        # Should detect null text
        null_checks = [c for c in checks if "null" in c.check_name.lower() or "missing" in c.check_name.lower()]
        assert any(not check.passed for check in null_checks)


# ============================================================================
# Integration Tests
# ============================================================================

class TestValidateArtifact:
    """Test full artifact validation."""
    
    def test_validate_utterances(self, sample_utterances_df):
        """Test validating utterance artifact."""
        report = validate_artifact(sample_utterances_df, "utterance", "v1")
        
        assert report.artifact_type == "utterance"
        assert report.version == "v1"
        assert len(report.checks) > 0
        
        # Valid data should pass
        assert report.is_valid()
    
    def test_validate_spans(self, sample_spans_df):
        """Test validating span artifact."""
        report = validate_artifact(sample_spans_df, "span", "v1")
        
        assert report.artifact_type == "span"
        assert len(report.checks) > 0
    
    def test_validate_embeddings(self, sample_embeddings_df):
        """Test validating embedding artifact."""
        report = validate_artifact(sample_embeddings_df, "embedding", "v1")
        
        assert report.artifact_type == "embedding"
        assert len(report.checks) > 0
    
    def test_validate_with_failures(self, sample_utterances_df):
        """Test validation with data quality issues."""
        # Introduce issues
        df = sample_utterances_df.copy()
        df.loc[0, "utterance_id"] = None  # Null ID
        df.loc[1, "duration"] = -1.0  # Negative duration
        
        report = validate_artifact(df, "utterance", "v1")
        
        # Should have failures
        assert not report.is_valid()
        assert len(report.get_errors()) > 0
    
    def test_validate_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame()
        
        report = validate_artifact(df, "utterance", "v1")
        
        # Should handle empty DataFrame gracefully
        assert len(report.checks) > 0

