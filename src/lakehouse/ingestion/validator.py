"""
Input validation for transcript utterances.

Validates that raw transcript data contains required fields and meets
basic quality requirements before normalization.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from lakehouse.logger import get_default_logger


logger = get_default_logger()


# Required fields for utterance records
REQUIRED_UTTERANCE_FIELDS = {
    "episode_id",
    "start",
    "end",
    "speaker",
    "text",
}


class ValidationError:
    """Represents a validation error for an utterance."""
    
    def __init__(
        self,
        error_type: str,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        utterance_index: Optional[int] = None,
    ):
        """
        Initialize validation error.
        
        Args:
            error_type: Type of error (e.g., "missing_field", "invalid_type")
            message: Human-readable error message
            field: Field name that caused the error (optional)
            value: Invalid value (optional)
            utterance_index: Index of utterance in list (optional)
        """
        self.error_type = error_type
        self.message = message
        self.field = field
        self.value = value
        self.utterance_index = utterance_index
    
    def __repr__(self) -> str:
        parts = [f"ValidationError(type={self.error_type}"]
        if self.utterance_index is not None:
            parts.append(f"index={self.utterance_index}")
        if self.field:
            parts.append(f"field={self.field}")
        parts.append(f"message={self.message})")
        return ", ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "field": self.field,
            "value": self.value,
            "utterance_index": self.utterance_index,
        }


class ValidationResult:
    """Result of validating a list of utterances."""
    
    def __init__(self):
        """Initialize empty validation result."""
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.valid_count: int = 0
        self.invalid_count: int = 0
    
    def add_error(self, error: ValidationError) -> None:
        """Add an error to the result."""
        self.errors.append(error)
        self.invalid_count += 1
    
    def add_warning(self, warning: ValidationError) -> None:
        """Add a warning to the result."""
        self.warnings.append(warning)
    
    def increment_valid(self) -> None:
        """Increment valid utterance count."""
        self.valid_count += 1
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0
    
    @property
    def total_count(self) -> int:
        """Total number of utterances validated."""
        return self.valid_count + self.invalid_count
    
    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"Validation Summary:",
            f"  Total: {self.total_count}",
            f"  Valid: {self.valid_count}",
            f"  Invalid: {self.invalid_count}",
            f"  Errors: {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
        ]
        return "\n".join(lines)


def validate_utterance(
    utterance: Dict[str, Any],
    index: Optional[int] = None,
    required_fields: Optional[Set[str]] = None,
) -> List[ValidationError]:
    """
    Validate a single utterance record.
    
    Args:
        utterance: Utterance dictionary to validate
        index: Index of utterance in list (for error reporting)
        required_fields: Set of required field names (default: REQUIRED_UTTERANCE_FIELDS)
    
    Returns:
        List of ValidationError objects (empty if valid)
    
    Example:
        >>> utterance = {"episode_id": "EP1", "start": 0.0, "end": 1.5, "speaker": "A", "text": "Hi"}
        >>> errors = validate_utterance(utterance)
        >>> len(errors)
        0
    """
    if required_fields is None:
        required_fields = REQUIRED_UTTERANCE_FIELDS
    
    errors = []
    
    # Check if utterance is a dictionary
    if not isinstance(utterance, dict):
        errors.append(ValidationError(
            error_type="invalid_type",
            message=f"Utterance must be a dictionary, got {type(utterance).__name__}",
            utterance_index=index,
        ))
        return errors
    
    # Check for missing required fields
    utterance_fields = set(utterance.keys())
    missing_fields = required_fields - utterance_fields
    
    for field in missing_fields:
        errors.append(ValidationError(
            error_type="missing_field",
            message=f"Required field '{field}' is missing",
            field=field,
            utterance_index=index,
        ))
    
    # If missing critical fields, skip further validation
    if missing_fields:
        return errors
    
    # Validate episode_id
    episode_id = utterance.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id.strip():
        errors.append(ValidationError(
            error_type="invalid_episode_id",
            message="episode_id must be a non-empty string",
            field="episode_id",
            value=episode_id,
            utterance_index=index,
        ))
    
    # Validate start and end times
    start = utterance.get("start")
    end = utterance.get("end")
    
    if not isinstance(start, (int, float)):
        errors.append(ValidationError(
            error_type="invalid_timestamp",
            message=f"start must be a number, got {type(start).__name__}",
            field="start",
            value=start,
            utterance_index=index,
        ))
    elif start < 0:
        errors.append(ValidationError(
            error_type="negative_timestamp",
            message=f"start must be non-negative, got {start}",
            field="start",
            value=start,
            utterance_index=index,
        ))
    
    if not isinstance(end, (int, float)):
        errors.append(ValidationError(
            error_type="invalid_timestamp",
            message=f"end must be a number, got {type(end).__name__}",
            field="end",
            value=end,
            utterance_index=index,
        ))
    elif end < 0:
        errors.append(ValidationError(
            error_type="negative_timestamp",
            message=f"end must be non-negative, got {end}",
            field="end",
            value=end,
            utterance_index=index,
        ))
    
    # Check that end > start
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        if end <= start:
            errors.append(ValidationError(
                error_type="invalid_time_range",
                message=f"end ({end}) must be greater than start ({start})",
                field="end",
                value=end,
                utterance_index=index,
            ))
    
    # Validate speaker
    speaker = utterance.get("speaker")
    if not isinstance(speaker, str) or not speaker.strip():
        errors.append(ValidationError(
            error_type="invalid_speaker",
            message="speaker must be a non-empty string",
            field="speaker",
            value=speaker,
            utterance_index=index,
        ))
    
    # Validate text
    text = utterance.get("text")
    if not isinstance(text, str):
        errors.append(ValidationError(
            error_type="invalid_text",
            message=f"text must be a string, got {type(text).__name__}",
            field="text",
            value=text,
            utterance_index=index,
        ))
    elif not text.strip():
        errors.append(ValidationError(
            error_type="empty_text",
            message="text must not be empty",
            field="text",
            value=text,
            utterance_index=index,
        ))
    
    return errors


def validate_utterances(
    utterances: List[Dict[str, Any]],
    required_fields: Optional[Set[str]] = None,
    fail_fast: bool = False,
) -> ValidationResult:
    """
    Validate a list of utterance records.
    
    Args:
        utterances: List of utterance dictionaries
        required_fields: Set of required field names (default: REQUIRED_UTTERANCE_FIELDS)
        fail_fast: If True, stop validation on first error
    
    Returns:
        ValidationResult with errors, warnings, and counts
    
    Example:
        >>> utterances = [{"episode_id": "EP1", ...}, {"episode_id": "EP1", ...}]
        >>> result = validate_utterances(utterances)
        >>> result.is_valid
        True
    """
    result = ValidationResult()
    
    for index, utterance in enumerate(utterances):
        errors = validate_utterance(utterance, index=index, required_fields=required_fields)
        
        if errors:
            for error in errors:
                result.add_error(error)
                logger.warning(f"Validation error at index {index}: {error.message}")
            
            if fail_fast:
                break
        else:
            result.increment_valid()
    
    logger.info(result.summary())
    return result


def filter_valid_utterances(
    utterances: List[Dict[str, Any]],
    required_fields: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], ValidationResult]:
    """
    Filter utterances to keep only valid ones.
    
    Invalid utterances are logged but excluded from the output.
    
    Args:
        utterances: List of utterance dictionaries
        required_fields: Set of required field names
    
    Returns:
        Tuple of (valid_utterances, validation_result)
    
    Example:
        >>> utterances = [valid_utt, invalid_utt, valid_utt]
        >>> valid, result = filter_valid_utterances(utterances)
        >>> len(valid)
        2
    """
    valid_utterances = []
    result = ValidationResult()
    
    for index, utterance in enumerate(utterances):
        errors = validate_utterance(utterance, index=index, required_fields=required_fields)
        
        if errors:
            for error in errors:
                result.add_error(error)
                logger.warning(f"Skipping invalid utterance at index {index}: {error.message}")
        else:
            valid_utterances.append(utterance)
            result.increment_valid()
    
    logger.info(f"Filtered to {len(valid_utterances)} valid utterances out of {len(utterances)} total")
    return valid_utterances, result

