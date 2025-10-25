"""
Data quality validation checks for lakehouse artifacts.

Provides comprehensive validation functions for utterances, spans, beats, sections,
and embeddings to ensure data integrity and quality.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from lakehouse.logger import get_default_logger
from lakehouse.schemas import get_schema, validate_dataframe_schema


logger = get_default_logger()


class ValidationCheck:
    """Represents a single validation check result."""
    
    def __init__(
        self,
        check_name: str,
        passed: bool,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "error"
    ):
        """
        Initialize validation check result.
        
        Args:
            check_name: Name of the check performed
            passed: Whether the check passed
            message: Human-readable result message
            details: Additional details about the check
            severity: Severity level (error, warning, info)
        """
        self.check_name = check_name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.severity = severity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "severity": self.severity,
        }


class ValidationReport:
    """Container for validation results."""
    
    def __init__(self, artifact_type: str, version: str):
        """Initialize validation report."""
        self.artifact_type = artifact_type
        self.version = version
        self.checks: List[ValidationCheck] = []
        self.statistics: Dict[str, Any] = {}
        self.timestamp = pd.Timestamp.now()
    
    def add_check(self, check: ValidationCheck):
        """Add a validation check result."""
        self.checks.append(check)
    
    def add_statistics(self, stats: Dict[str, Any]):
        """Add statistics to the report."""
        self.statistics.update(stats)
    
    def get_passed_checks(self) -> List[ValidationCheck]:
        """Get all passed checks."""
        return [c for c in self.checks if c.passed]
    
    def get_failed_checks(self) -> List[ValidationCheck]:
        """Get all failed checks."""
        return [c for c in self.checks if not c.passed]
    
    def get_errors(self) -> List[ValidationCheck]:
        """Get all error-level failed checks."""
        return [c for c in self.checks if not c.passed and c.severity == "error"]
    
    def get_warnings(self) -> List[ValidationCheck]:
        """Get all warning-level failed checks."""
        return [c for c in self.checks if not c.passed and c.severity == "warning"]
    
    def is_valid(self) -> bool:
        """Check if all critical checks passed."""
        return len(self.get_errors()) == 0
    
    def summary(self) -> str:
        """Generate a summary of validation results."""
        total_checks = len(self.checks)
        passed_checks = len(self.get_passed_checks())
        failed_checks = len(self.get_failed_checks())
        errors = len(self.get_errors())
        warnings = len(self.get_warnings())
        
        lines = [
            f"Validation Report for {self.artifact_type} (v{self.version})",
            f"Timestamp: {self.timestamp}",
            f"Total checks: {total_checks}",
            f"Passed: {passed_checks}",
            f"Failed: {failed_checks} (errors: {errors}, warnings: {warnings})",
        ]
        
        if self.statistics:
            lines.append("\nStatistics:")
            for key, value in self.statistics.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


def check_non_empty_tables(df: pd.DataFrame, artifact_type: str) -> ValidationCheck:
    """
    Check that tables are not empty.
    
    Args:
        df: DataFrame to check
        artifact_type: Type of artifact being validated
    
    Returns:
        ValidationCheck result
    """
    if len(df) == 0:
        return ValidationCheck(
            check_name="non_empty_table",
            passed=False,
            message=f"{artifact_type} table is empty",
            details={"row_count": 0},
            severity="error"
        )
    
    return ValidationCheck(
        check_name="non_empty_table",
        passed=True,
        message=f"{artifact_type} table contains {len(df)} rows",
        details={"row_count": len(df)},
    )


def check_required_fields(df: pd.DataFrame, artifact_type: str) -> ValidationCheck:
    """
    Check that all required fields are present.
    
    Args:
        df: DataFrame to check
        artifact_type: Type of artifact being validated
    
    Returns:
        ValidationCheck result
    """
    try:
        schema = get_schema(artifact_type)
        required_fields = set(schema.names)
        actual_fields = set(df.columns)
        missing_fields = required_fields - actual_fields
        
        if missing_fields:
            return ValidationCheck(
                check_name="required_fields",
                passed=False,
                message=f"Missing required fields: {sorted(missing_fields)}",
                details={"missing_fields": list(missing_fields)},
                severity="error"
            )
        
        return ValidationCheck(
            check_name="required_fields",
            passed=True,
            message=f"All required fields present",
            details={"field_count": len(required_fields)},
        )
    
    except Exception as e:
        return ValidationCheck(
            check_name="required_fields",
            passed=False,
            message=f"Could not validate required fields: {e}",
            details={"error": str(e)},
            severity="error"
        )


def check_schema_compliance(df: pd.DataFrame, artifact_type: str) -> ValidationCheck:
    """
    Check that data conforms to the expected schema.
    
    Args:
        df: DataFrame to check
        artifact_type: Type of artifact being validated
    
    Returns:
        ValidationCheck result
    """
    try:
        is_valid = validate_dataframe_schema(df, artifact_type)
        if is_valid:
            return ValidationCheck(
                check_name="schema_compliance",
                passed=True,
                message="Data conforms to schema",
                details={"schema_type": artifact_type},
            )
        else:
            return ValidationCheck(
                check_name="schema_compliance",
                passed=False,
                message="Data does not conform to schema",
                details={"schema_type": artifact_type},
                severity="error"
            )
    except Exception as e:
        return ValidationCheck(
            check_name="schema_compliance",
            passed=False,
            message=f"Schema validation failed: {e}",
            details={"error": str(e)},
            severity="error"
        )


def check_timestamps(df: pd.DataFrame, artifact_type: str) -> List[ValidationCheck]:
    """
    Check timestamp-related data quality.
    
    Args:
        df: DataFrame to check
        artifact_type: Type of artifact being validated
    
    Returns:
        List of ValidationCheck results
    """
    checks = []
    
    # Check for timestamp columns
    time_columns = []
    if 'start_time' in df.columns:
        time_columns.append('start_time')
    if 'end_time' in df.columns:
        time_columns.append('end_time')
    if 'start' in df.columns:
        time_columns.append('start')
    if 'end' in df.columns:
        time_columns.append('end')
    
    if not time_columns:
        checks.append(ValidationCheck(
            check_name="timestamp_columns",
            passed=False,
            message="No timestamp columns found",
            severity="warning"
        ))
        return checks
    
    # Check for non-null timestamps
    for col in time_columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                checks.append(ValidationCheck(
                    check_name=f"timestamp_nulls_{col}",
                    passed=False,
                    message=f"Found {null_count} null values in {col}",
                    details={"null_count": int(null_count), "column": col},
                    severity="error"
                ))
            else:
                checks.append(ValidationCheck(
                    check_name=f"timestamp_nulls_{col}",
                    passed=True,
                    message=f"No null values in {col}",
                    details={"column": col},
                ))
    
    # Check monotonic timestamps (start <= end)
    if 'start_time' in df.columns and 'end_time' in df.columns:
        invalid_pairs = df[df['start_time'] > df['end_time']]
        if len(invalid_pairs) > 0:
            checks.append(ValidationCheck(
                check_name="timestamp_monotonic",
                passed=False,
                message=f"Found {len(invalid_pairs)} records where start_time > end_time",
                details={"invalid_count": len(invalid_pairs)},
                severity="error"
            ))
        else:
            checks.append(ValidationCheck(
                check_name="timestamp_monotonic",
                passed=True,
                message="All timestamps are monotonic",
            ))
    
    # Check for negative timestamps
    for col in time_columns:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                checks.append(ValidationCheck(
                    check_name=f"timestamp_negative_{col}",
                    passed=False,
                    message=f"Found {negative_count} negative values in {col}",
                    details={"negative_count": int(negative_count), "column": col},
                    severity="error"
                ))
            else:
                checks.append(ValidationCheck(
                    check_name=f"timestamp_negative_{col}",
                    passed=True,
                    message=f"No negative values in {col}",
                    details={"column": col},
                ))
    
    return checks


def check_text_quality(df: pd.DataFrame, artifact_type: str) -> List[ValidationCheck]:
    """
    Check text-related data quality.
    
    Args:
        df: DataFrame to check
        artifact_type: Type of artifact being validated
    
    Returns:
        List of ValidationCheck results
    """
    checks = []
    
    if 'text' not in df.columns:
        checks.append(ValidationCheck(
            check_name="text_column",
            passed=False,
            message="No 'text' column found",
            severity="warning"
        ))
        return checks
    
    # Check for null text
    null_text_count = df['text'].isnull().sum()
    if null_text_count > 0:
        checks.append(ValidationCheck(
            check_name="text_nulls",
            passed=False,
            message=f"Found {null_text_count} null text values",
            details={"null_count": int(null_text_count)},
            severity="error"
        ))
    else:
        checks.append(ValidationCheck(
            check_name="text_nulls",
            passed=True,
            message="No null text values",
        ))
    
    # Check for empty text
    empty_text_count = (df['text'].str.len() == 0).sum()
    if empty_text_count > 0:
        checks.append(ValidationCheck(
            check_name="text_empty",
            passed=False,
            message=f"Found {empty_text_count} empty text values",
            details={"empty_count": int(empty_text_count)},
            severity="warning"
        ))
    else:
        checks.append(ValidationCheck(
            check_name="text_empty",
            passed=True,
            message="No empty text values",
        ))
    
    # Check text length statistics
    text_lengths = df['text'].str.len()
    min_length = text_lengths.min()
    max_length = text_lengths.max()
    mean_length = text_lengths.mean()
    
    checks.append(ValidationCheck(
        check_name="text_length_stats",
        passed=True,
        message=f"Text length: min={min_length}, max={max_length}, mean={mean_length:.1f}",
        details={
            "min_length": int(min_length),
            "max_length": int(max_length),
            "mean_length": float(mean_length),
        },
    ))
    
    return checks


def check_id_quality(df: pd.DataFrame, artifact_type: str) -> List[ValidationCheck]:
    """
    Check ID-related data quality.
    
    Args:
        df: DataFrame to check
        artifact_type: Type of artifact being validated
    
    Returns:
        List of ValidationCheck results
    """
    checks = []
    
    # Find ID columns
    id_columns = [col for col in df.columns if col.endswith('_id')]
    
    if not id_columns:
        checks.append(ValidationCheck(
            check_name="id_columns",
            passed=False,
            message="No ID columns found",
            severity="warning"
        ))
        return checks
    
    # Define primary keys (must be unique) vs foreign keys (can have duplicates)
    primary_key_columns = [
        'utterance_id', 'span_id', 'beat_id', 'section_id', 'artifact_id'
    ]
    foreign_key_columns = ['episode_id']
    
    for col in id_columns:
        # Check for null IDs (applies to all ID columns)
        null_count = df[col].isnull().sum()
        if null_count > 0:
            checks.append(ValidationCheck(
                check_name=f"id_nulls_{col}",
                passed=False,
                message=f"Found {null_count} null values in {col}",
                details={"null_count": int(null_count), "column": col},
                severity="error"
            ))
        else:
            checks.append(ValidationCheck(
                check_name=f"id_nulls_{col}",
                passed=True,
                message=f"No null values in {col}",
                details={"column": col},
            ))
        
        # Check for duplicate IDs (only for primary keys)
        if col in primary_key_columns:
            duplicate_count = df[col].duplicated().sum()
            if duplicate_count > 0:
                checks.append(ValidationCheck(
                    check_name=f"id_duplicates_{col}",
                    passed=False,
                    message=f"Found {duplicate_count} duplicate values in {col}",
                    details={"duplicate_count": int(duplicate_count), "column": col},
                    severity="error"
                ))
            else:
                checks.append(ValidationCheck(
                    check_name=f"id_duplicates_{col}",
                    passed=True,
                    message=f"No duplicate values in {col}",
                    details={"column": col},
                ))
        elif col in foreign_key_columns:
            # Foreign keys are expected to have duplicates - just report the count
            unique_count = df[col].nunique()
            checks.append(ValidationCheck(
                check_name=f"id_foreign_key_{col}",
                passed=True,
                message=f"Foreign key {col} has {unique_count} unique values across {len(df)} rows",
                details={"unique_count": int(unique_count), "total_rows": len(df), "column": col},
            ))
    
    return checks


def check_referential_integrity(df: pd.DataFrame, artifact_type: str) -> List[ValidationCheck]:
    """
    Check referential integrity between artifacts.
    
    Args:
        df: DataFrame to check
        artifact_type: Type of artifact being validated
    
    Returns:
        List of ValidationCheck results
    """
    checks = []
    
    # Check utterance_ids in spans
    if artifact_type == "span" and 'utterance_ids' in df.columns:
        # This would need to be checked against the utterances table
        # For now, just check that the column exists and is not null
        null_count = df['utterance_ids'].isnull().sum()
        if null_count > 0:
            checks.append(ValidationCheck(
                check_name="referential_integrity_utterance_ids",
                passed=False,
                message=f"Found {null_count} null utterance_ids in spans",
                details={"null_count": int(null_count)},
                severity="error"
            ))
        else:
            checks.append(ValidationCheck(
                check_name="referential_integrity_utterance_ids",
                passed=True,
                message="No null utterance_ids in spans",
            ))
    
    # Check span_ids in beats
    if artifact_type == "beat" and 'span_ids' in df.columns:
        null_count = df['span_ids'].isnull().sum()
        if null_count > 0:
            checks.append(ValidationCheck(
                check_name="referential_integrity_span_ids",
                passed=False,
                message=f"Found {null_count} null span_ids in beats",
                details={"null_count": int(null_count)},
                severity="error"
            ))
        else:
            checks.append(ValidationCheck(
                check_name="referential_integrity_span_ids",
                passed=True,
                message="No null span_ids in beats",
            ))
    
    # Check beat_ids in sections
    if artifact_type == "section" and 'beat_ids' in df.columns:
        null_count = df['beat_ids'].isnull().sum()
        if null_count > 0:
            checks.append(ValidationCheck(
                check_name="referential_integrity_beat_ids",
                passed=False,
                message=f"Found {null_count} null beat_ids in sections",
                details={"null_count": int(null_count)},
                severity="error"
            ))
        else:
            checks.append(ValidationCheck(
                check_name="referential_integrity_beat_ids",
                passed=True,
                message="No null beat_ids in sections",
            ))
    
    return checks


def check_numeric_quality(df: pd.DataFrame, artifact_type: str) -> List[ValidationCheck]:
    """
    Check numeric data quality.
    
    Args:
        df: DataFrame to check
        artifact_type: Type of artifact being validated
    
    Returns:
        List of ValidationCheck results
    """
    checks = []
    
    # Find numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        # Check for NaN values
        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            checks.append(ValidationCheck(
                check_name=f"numeric_nans_{col}",
                passed=False,
                message=f"Found {nan_count} NaN values in {col}",
                details={"nan_count": int(nan_count), "column": col},
                severity="error"
            ))
        else:
            checks.append(ValidationCheck(
                check_name=f"numeric_nans_{col}",
                passed=True,
                message=f"No NaN values in {col}",
                details={"column": col},
            ))
        
        # Check for infinite values
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            checks.append(ValidationCheck(
                check_name=f"numeric_inf_{col}",
                passed=False,
                message=f"Found {inf_count} infinite values in {col}",
                details={"inf_count": int(inf_count), "column": col},
                severity="error"
            ))
        else:
            checks.append(ValidationCheck(
                check_name=f"numeric_inf_{col}",
                passed=True,
                message=f"No infinite values in {col}",
                details={"column": col},
            ))
    
    return checks


def validate_artifact(
    df: pd.DataFrame,
    artifact_type: str,
    version: str = "v1",
    config: Optional[Dict[str, Any]] = None
) -> ValidationReport:
    """
    Perform comprehensive validation on an artifact DataFrame.
    
    Args:
        df: DataFrame to validate
        artifact_type: Type of artifact (utterance, span, beat, section, embedding)
        version: Version of the artifact
        config: Validation configuration (optional)
    
    Returns:
        ValidationReport with all check results
    """
    report = ValidationReport(artifact_type, version)
    
    logger.info(f"Validating {artifact_type} artifact (v{version}) with {len(df)} rows")
    
    # Basic statistics
    stats = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
    }
    report.add_statistics(stats)
    
    # Run all validation checks
    checks_to_run = [
        check_non_empty_tables,
        check_required_fields,
        check_schema_compliance,
    ]
    
    for check_func in checks_to_run:
        try:
            result = check_func(df, artifact_type)
            if isinstance(result, list):
                for check in result:
                    report.add_check(check)
            else:
                report.add_check(result)
        except Exception as e:
            logger.error(f"Error running {check_func.__name__}: {e}")
            report.add_check(ValidationCheck(
                check_name=check_func.__name__,
                passed=False,
                message=f"Check failed with error: {e}",
                details={"error": str(e)},
                severity="error"
            ))
    
    # Run specialized checks
    specialized_checks = [
        check_timestamps,
        check_text_quality,
        check_id_quality,
        check_referential_integrity,
        check_numeric_quality,
    ]
    
    for check_func in specialized_checks:
        try:
            results = check_func(df, artifact_type)
            for result in results:
                report.add_check(result)
        except Exception as e:
            logger.error(f"Error running {check_func.__name__}: {e}")
            report.add_check(ValidationCheck(
                check_name=check_func.__name__,
                passed=False,
                message=f"Check failed with error: {e}",
                details={"error": str(e)},
                severity="error"
            ))
    
    logger.info(f"Validation complete: {len(report.get_passed_checks())} passed, {len(report.get_failed_checks())} failed")
    return report


def validate_lakehouse(
    lakehouse_path: Path,
    version: str = "v1",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, ValidationReport]:
    """
    Validate all artifacts in a lakehouse.
    
    Args:
        lakehouse_path: Path to lakehouse directory
        version: Version to validate
        config: Validation configuration (optional)
    
    Returns:
        Dictionary mapping artifact types to ValidationReport objects
    """
    reports = {}
    
    # Define artifact types and their paths
    artifact_paths = {
        "utterance": lakehouse_path / "normalized" / version,
        "span": lakehouse_path / "spans" / version,
        "beat": lakehouse_path / "beats" / version,
        "section": lakehouse_path / "sections" / version,
        "embedding": lakehouse_path / "embeddings" / version,
    }
    
    for artifact_type, base_path in artifact_paths.items():
        if not base_path.exists():
            logger.warning(f"No {artifact_type} data found at {base_path}")
            continue
        
        # Find parquet files
        parquet_files = list(base_path.glob("*.parquet"))
        if not parquet_files:
            logger.warning(f"No parquet files found in {base_path}")
            continue
        
        # Load and validate each file
        for parquet_file in parquet_files:
            try:
                df = pd.read_parquet(parquet_file)
                report = validate_artifact(df, artifact_type, version, config)
                reports[f"{artifact_type}_{parquet_file.stem}"] = report
            except Exception as e:
                logger.error(f"Error validating {parquet_file}: {e}")
                # Create error report
                error_report = ValidationReport(artifact_type, version)
                error_report.add_check(ValidationCheck(
                    check_name="file_loading",
                    passed=False,
                    message=f"Could not load file: {e}",
                    details={"file": str(parquet_file), "error": str(e)},
                    severity="error"
                ))
                reports[f"{artifact_type}_{parquet_file.stem}_error"] = error_report
    
    return reports

