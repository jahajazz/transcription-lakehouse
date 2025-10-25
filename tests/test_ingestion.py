"""
Unit tests for ingestion modules: reader, validator, normalizer, and writer.

Tests the pipeline for reading, validating, normalizing, and writing transcript data.
"""

import json
import pytest
import pandas as pd
from pathlib import Path
from typing import Dict, List

from lakehouse.ingestion.reader import (
    TranscriptReader,
    read_transcript_file,
    read_transcript_directory,
    extract_episode_id,
)
from lakehouse.ingestion.validator import (
    ValidationError,
    ValidationResult,
    validate_utterance,
    validate_utterances,
    filter_valid_utterances,
    REQUIRED_UTTERANCE_FIELDS,
)
from lakehouse.ingestion.normalizer import (
    normalize_utterance,
    normalize_utterances,
    sort_utterances_by_time,
    compute_utterance_statistics,
    group_utterances_by_episode,
    normalize_episode,
)
from lakehouse.ingestion.writer import (
    ParquetWriter,
    write_parquet,
    read_parquet,
    create_versioned_directory,
    write_versioned_parquet,
    get_parquet_file_info,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_utterance() -> Dict:
    """Create a valid sample utterance."""
    return {
        "episode_id": "TEST-001",
        "start": 0.0,
        "end": 5.0,
        "speaker": "Alice",
        "text": "Hello, world!",
    }


@pytest.fixture
def sample_utterances() -> List[Dict]:
    """Create a list of valid sample utterances."""
    return [
        {
            "episode_id": "TEST-001",
            "start": 0.0,
            "end": 5.0,
            "speaker": "Alice",
            "text": "Hello, world!",
        },
        {
            "episode_id": "TEST-001",
            "start": 5.0,
            "end": 10.0,
            "speaker": "Bob",
            "text": "How are you?",
        },
        {
            "episode_id": "TEST-001",
            "start": 10.0,
            "end": 15.0,
            "speaker": "Alice",
            "text": "I'm doing great!",
        },
    ]


@pytest.fixture
def temp_jsonl_file(tmp_path, sample_utterances) -> Path:
    """Create a temporary JSONL file with sample data."""
    file_path = tmp_path / "test_transcript.jsonl"
    
    with open(file_path, "w", encoding="utf-8") as f:
        for utterance in sample_utterances:
            f.write(json.dumps(utterance) + "\n")
    
    return file_path


@pytest.fixture
def temp_json_file(tmp_path, sample_utterances) -> Path:
    """Create a temporary JSON file with sample data."""
    file_path = tmp_path / "test_transcript.json"
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sample_utterances, f)
    
    return file_path


# ============================================================================
# Reader Tests
# ============================================================================

class TestTranscriptReader:
    """Test TranscriptReader class."""
    
    def test_reader_init_jsonl(self, temp_jsonl_file):
        """Test reader initialization with JSONL file."""
        reader = TranscriptReader(temp_jsonl_file)
        assert reader.file_format == "jsonl"
        assert reader.file_path == temp_jsonl_file
    
    def test_reader_init_json(self, temp_json_file):
        """Test reader initialization with JSON file."""
        reader = TranscriptReader(temp_json_file)
        assert reader.file_format == "json"
    
    def test_reader_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            TranscriptReader(tmp_path / "nonexistent.jsonl")
    
    def test_read_jsonl_utterances(self, temp_jsonl_file, sample_utterances):
        """Test reading JSONL file."""
        reader = TranscriptReader(temp_jsonl_file)
        utterances = reader.read_utterances()
        
        assert len(utterances) == len(sample_utterances)
        assert utterances[0]["episode_id"] == "TEST-001"
        assert utterances[0]["speaker"] == "Alice"
    
    def test_read_json_utterances(self, temp_json_file, sample_utterances):
        """Test reading JSON file."""
        reader = TranscriptReader(temp_json_file)
        utterances = reader.read_utterances()
        
        assert len(utterances) == len(sample_utterances)
        assert utterances[0]["episode_id"] == "TEST-001"
    
    def test_iter_jsonl_utterances(self, temp_jsonl_file, sample_utterances):
        """Test iterating over JSONL file."""
        reader = TranscriptReader(temp_jsonl_file)
        utterances = list(reader.iter_utterances())
        
        assert len(utterances) == len(sample_utterances)
    
    def test_read_jsonl_with_empty_lines(self, tmp_path):
        """Test reading JSONL with empty lines."""
        file_path = tmp_path / "with_empty_lines.jsonl"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write('{"episode_id": "TEST-001", "start": 0.0, "end": 1.0, "speaker": "A", "text": "Hi"}\n')
            f.write('\n')  # Empty line
            f.write('{"episode_id": "TEST-001", "start": 1.0, "end": 2.0, "speaker": "B", "text": "Hey"}\n')
        
        reader = TranscriptReader(file_path)
        utterances = reader.read_utterances()
        
        # Should skip empty lines
        assert len(utterances) == 2
    
    def test_read_jsonl_with_malformed_line(self, tmp_path):
        """Test reading JSONL with malformed JSON line."""
        file_path = tmp_path / "with_malformed.jsonl"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write('{"episode_id": "TEST-001", "start": 0.0, "end": 1.0, "speaker": "A", "text": "Hi"}\n')
            f.write('{"invalid json\n')  # Malformed
            f.write('{"episode_id": "TEST-001", "start": 2.0, "end": 3.0, "speaker": "B", "text": "Hey"}\n')
        
        reader = TranscriptReader(file_path)
        utterances = reader.read_utterances()
        
        # Should skip malformed line
        assert len(utterances) == 2


class TestReaderUtilities:
    """Test reader utility functions."""
    
    def test_read_transcript_file(self, temp_jsonl_file):
        """Test convenience function for reading transcript file."""
        utterances = read_transcript_file(temp_jsonl_file)
        assert len(utterances) > 0
        assert "episode_id" in utterances[0]
    
    def test_read_transcript_directory(self, tmp_path, sample_utterances):
        """Test reading all transcripts from a directory."""
        # Create multiple JSONL files
        for i in range(3):
            file_path = tmp_path / f"transcript_{i}.jsonl"
            with open(file_path, "w", encoding="utf-8") as f:
                for utt in sample_utterances:
                    f.write(json.dumps(utt) + "\n")
        
        transcripts = read_transcript_directory(tmp_path, pattern="*.jsonl")
        
        assert len(transcripts) == 3
    
    def test_extract_episode_id_from_file(self, temp_jsonl_file):
        """Test extracting episode ID from file."""
        episode_id = extract_episode_id(temp_jsonl_file)
        assert episode_id == "TEST-001"  # From first utterance
    
    def test_extract_episode_id_from_filename(self, tmp_path):
        """Test extracting episode ID from filename when file is empty."""
        empty_file = tmp_path / "EP-999 - Test Episode.jsonl"
        empty_file.write_text("[]")
        
        episode_id = extract_episode_id(empty_file)
        assert episode_id == "EP-999 - Test Episode"


# ============================================================================
# Validator Tests
# ============================================================================

class TestValidationError:
    """Test ValidationError class."""
    
    def test_validation_error_creation(self):
        """Test creating a validation error."""
        error = ValidationError(
            error_type="missing_field",
            message="Field 'text' is missing",
            field="text",
            utterance_index=5,
        )
        
        assert error.error_type == "missing_field"
        assert error.field == "text"
        assert error.utterance_index == 5
    
    def test_validation_error_to_dict(self):
        """Test converting error to dictionary."""
        error = ValidationError(
            error_type="invalid_type",
            message="Invalid type",
            field="start",
            value="not a number",
        )
        
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "invalid_type"
        assert error_dict["field"] == "start"


class TestValidationResult:
    """Test ValidationResult class."""
    
    def test_validation_result_empty(self):
        """Test empty validation result."""
        result = ValidationResult()
        assert result.is_valid
        assert result.total_count == 0
        assert len(result.errors) == 0
    
    def test_validation_result_with_errors(self):
        """Test validation result with errors."""
        result = ValidationResult()
        error = ValidationError("test_error", "Test error message")
        result.add_error(error)
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.invalid_count == 1
    
    def test_validation_result_with_valid(self):
        """Test validation result with valid utterances."""
        result = ValidationResult()
        result.increment_valid()
        result.increment_valid()
        
        assert result.is_valid
        assert result.valid_count == 2


class TestUtteranceValidation:
    """Test utterance validation functions."""
    
    def test_validate_valid_utterance(self, sample_utterance):
        """Test validating a valid utterance."""
        errors = validate_utterance(sample_utterance)
        assert len(errors) == 0
    
    def test_validate_missing_field(self, sample_utterance):
        """Test validation fails for missing required field."""
        del sample_utterance["text"]
        errors = validate_utterance(sample_utterance)
        
        assert len(errors) > 0
        assert any(e.error_type == "missing_field" for e in errors)
        assert any(e.field == "text" for e in errors)
    
    def test_validate_invalid_episode_id(self, sample_utterance):
        """Test validation fails for invalid episode_id."""
        sample_utterance["episode_id"] = ""
        errors = validate_utterance(sample_utterance)
        
        assert len(errors) > 0
        assert any(e.error_type == "invalid_episode_id" for e in errors)
    
    def test_validate_invalid_timestamp_type(self, sample_utterance):
        """Test validation fails for invalid timestamp type."""
        sample_utterance["start"] = "not a number"
        errors = validate_utterance(sample_utterance)
        
        assert len(errors) > 0
        assert any(e.error_type == "invalid_timestamp" for e in errors)
    
    def test_validate_negative_timestamp(self, sample_utterance):
        """Test validation fails for negative timestamp."""
        sample_utterance["start"] = -1.0
        errors = validate_utterance(sample_utterance)
        
        assert len(errors) > 0
        assert any(e.error_type == "negative_timestamp" for e in errors)
    
    def test_validate_end_before_start(self, sample_utterance):
        """Test validation fails when end <= start."""
        sample_utterance["start"] = 10.0
        sample_utterance["end"] = 5.0
        errors = validate_utterance(sample_utterance)
        
        assert len(errors) > 0
        assert any(e.error_type == "invalid_time_range" for e in errors)
    
    def test_validate_empty_speaker(self, sample_utterance):
        """Test validation fails for empty speaker."""
        sample_utterance["speaker"] = ""
        errors = validate_utterance(sample_utterance)
        
        assert len(errors) > 0
        assert any(e.error_type == "invalid_speaker" for e in errors)
    
    def test_validate_empty_text(self, sample_utterance):
        """Test validation fails for empty text."""
        sample_utterance["text"] = ""
        errors = validate_utterance(sample_utterance)
        
        assert len(errors) > 0
        assert any(e.error_type == "empty_text" for e in errors)
    
    def test_validate_not_dict(self):
        """Test validation fails when utterance is not a dictionary."""
        errors = validate_utterance("not a dict")
        
        assert len(errors) > 0
        assert errors[0].error_type == "invalid_type"


class TestBulkValidation:
    """Test bulk validation functions."""
    
    def test_validate_utterances_all_valid(self, sample_utterances):
        """Test validating all valid utterances."""
        result = validate_utterances(sample_utterances)
        
        assert result.is_valid
        assert result.valid_count == len(sample_utterances)
        assert len(result.errors) == 0
    
    def test_validate_utterances_with_invalid(self, sample_utterances):
        """Test validating with some invalid utterances."""
        sample_utterances[1]["text"] = ""  # Make one invalid
        
        result = validate_utterances(sample_utterances)
        
        assert not result.is_valid
        assert result.valid_count == len(sample_utterances) - 1
        assert result.invalid_count == 1
    
    def test_filter_valid_utterances(self, sample_utterances):
        """Test filtering to keep only valid utterances."""
        sample_utterances[1]["speaker"] = ""  # Make one invalid
        
        valid, result = filter_valid_utterances(sample_utterances)
        
        assert len(valid) == len(sample_utterances) - 1
        assert result.valid_count == len(sample_utterances) - 1


# ============================================================================
# Normalizer Tests
# ============================================================================

class TestUtteranceNormalization:
    """Test utterance normalization functions."""
    
    def test_normalize_utterance(self, sample_utterance):
        """Test normalizing a single utterance."""
        normalized = normalize_utterance(sample_utterance, position=0)
        
        assert "utterance_id" in normalized
        assert normalized["utterance_id"].startswith("utt_")
        assert "duration" in normalized
        assert normalized["duration"] == 5.0
    
    def test_normalize_utterance_deterministic(self, sample_utterance):
        """Test that normalization is deterministic."""
        normalized1 = normalize_utterance(sample_utterance.copy(), position=0)
        normalized2 = normalize_utterance(sample_utterance.copy(), position=0)
        
        assert normalized1["utterance_id"] == normalized2["utterance_id"]
    
    def test_normalize_utterance_preserves_metadata(self, sample_utterance):
        """Test that additional metadata is preserved."""
        sample_utterance["custom_field"] = "custom_value"
        normalized = normalize_utterance(sample_utterance, position=0)
        
        assert normalized["custom_field"] == "custom_value"
    
    def test_normalize_utterances(self, sample_utterances):
        """Test normalizing a list of utterances."""
        normalized = normalize_utterances(sample_utterances)
        
        assert len(normalized) == len(sample_utterances)
        assert all("utterance_id" in u for u in normalized)
        assert all("duration" in u for u in normalized)
    
    def test_normalize_utterances_empty_list(self):
        """Test normalizing an empty list."""
        normalized = normalize_utterances([])
        assert len(normalized) == 0
    
    def test_normalize_utterances_different_episodes_raises_error(self):
        """Test that different episode IDs raise an error."""
        utterances = [
            {"episode_id": "EP1", "start": 0.0, "end": 1.0, "speaker": "A", "text": "Hi"},
            {"episode_id": "EP2", "start": 0.0, "end": 1.0, "speaker": "B", "text": "Hey"},
        ]
        
        with pytest.raises(ValueError, match="episode_id"):
            normalize_utterances(utterances)


class TestUtteranceSorting:
    """Test utterance sorting functions."""
    
    def test_sort_utterances_by_time(self):
        """Test sorting utterances by start time."""
        utterances = [
            {"start": 10.0, "end": 15.0, "text": "second"},
            {"start": 0.0, "end": 5.0, "text": "first"},
            {"start": 5.0, "end": 10.0, "text": "middle"},
        ]
        
        sorted_utts = sort_utterances_by_time(utterances)
        
        assert sorted_utts[0]["text"] == "first"
        assert sorted_utts[1]["text"] == "middle"
        assert sorted_utts[2]["text"] == "second"


class TestUtteranceStatistics:
    """Test utterance statistics functions."""
    
    def test_compute_utterance_statistics(self, sample_utterances):
        """Test computing statistics for utterances."""
        stats = compute_utterance_statistics(sample_utterances)
        
        assert stats["total_count"] == 3
        assert stats["speaker_count"] == 2  # Alice and Bob
        assert "Alice" in stats["speakers"]
        assert "Bob" in stats["speakers"]
        assert stats["episode_duration"] == 15.0
    
    def test_compute_utterance_statistics_empty(self):
        """Test computing statistics for empty list."""
        stats = compute_utterance_statistics([])
        
        assert stats["total_count"] == 0
        assert stats["speaker_count"] == 0


class TestUtteranceGrouping:
    """Test utterance grouping functions."""
    
    def test_group_utterances_by_episode(self):
        """Test grouping utterances by episode ID."""
        utterances = [
            {"episode_id": "EP1", "start": 0.0, "end": 1.0, "speaker": "A", "text": "Hi"},
            {"episode_id": "EP1", "start": 1.0, "end": 2.0, "speaker": "B", "text": "Hey"},
            {"episode_id": "EP2", "start": 0.0, "end": 1.0, "speaker": "A", "text": "Hello"},
        ]
        
        grouped = group_utterances_by_episode(utterances)
        
        assert len(grouped) == 2
        assert len(grouped["EP1"]) == 2
        assert len(grouped["EP2"]) == 1


class TestEpisodeNormalization:
    """Test episode normalization functions."""
    
    def test_normalize_episode(self, sample_utterances):
        """Test normalizing an entire episode."""
        normalized = normalize_episode(sample_utterances, episode_id="TEST-001")
        
        assert len(normalized) == len(sample_utterances)
        assert all(u["episode_id"] == "TEST-001" for u in normalized)
    
    def test_normalize_episode_with_sorting(self):
        """Test normalizing episode with time-based sorting."""
        utterances = [
            {"episode_id": "EP1", "start": 10.0, "end": 15.0, "speaker": "A", "text": "second"},
            {"episode_id": "EP1", "start": 0.0, "end": 5.0, "speaker": "B", "text": "first"},
        ]
        
        normalized = normalize_episode(utterances, sort_by_time=True)
        
        # Should be sorted by time
        assert normalized[0]["text"] == "first"
        assert normalized[1]["text"] == "second"


# ============================================================================
# Writer Tests
# ============================================================================

class TestParquetWriter:
    """Test ParquetWriter class."""
    
    def test_writer_init(self):
        """Test initializing Parquet writer."""
        writer = ParquetWriter("utterance")
        assert writer.artifact_type == "utterance"
        assert writer.compression == "snappy"
        assert writer.enforce_schema is True
    
    def test_write_list_of_dicts(self, tmp_path, sample_utterances):
        """Test writing list of dictionaries."""
        # Normalize first to add required fields
        normalized = normalize_utterances(sample_utterances)
        
        writer = ParquetWriter("utterance", enforce_schema=False)
        output_path = tmp_path / "utterances.parquet"
        
        writer.write(normalized, output_path)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_write_dataframe(self, tmp_path, sample_utterances):
        """Test writing DataFrame."""
        normalized = normalize_utterances(sample_utterances)
        df = pd.DataFrame(normalized)
        
        writer = ParquetWriter("utterance", enforce_schema=False)
        output_path = tmp_path / "utterances.parquet"
        
        writer.write(df, output_path)
        
        assert output_path.exists()
    
    def test_write_overwrite_false_raises_error(self, tmp_path, sample_utterances):
        """Test that writing to existing file without overwrite raises error."""
        normalized = normalize_utterances(sample_utterances)
        
        writer = ParquetWriter("utterance", enforce_schema=False)
        output_path = tmp_path / "utterances.parquet"
        
        # First write
        writer.write(normalized, output_path)
        
        # Second write without overwrite should fail
        with pytest.raises(FileExistsError):
            writer.write(normalized, output_path, overwrite=False)
    
    def test_write_overwrite_true(self, tmp_path, sample_utterances):
        """Test overwriting existing file."""
        normalized = normalize_utterances(sample_utterances)
        
        writer = ParquetWriter("utterance", enforce_schema=False)
        output_path = tmp_path / "utterances.parquet"
        
        # First write
        writer.write(normalized, output_path)
        original_size = output_path.stat().st_size
        
        # Overwrite
        writer.write(normalized, output_path, overwrite=True)
        
        assert output_path.exists()
    
    def test_append(self, tmp_path, sample_utterances):
        """Test appending to existing file."""
        normalized = normalize_utterances(sample_utterances)
        
        writer = ParquetWriter("utterance", enforce_schema=False)
        output_path = tmp_path / "utterances.parquet"
        
        # Write initial data
        writer.write(normalized[:2], output_path)
        
        # Append more data
        writer.append(normalized[2:], output_path)
        
        # Read and verify
        df = pd.read_parquet(output_path)
        assert len(df) == len(normalized)


class TestParquetUtilities:
    """Test Parquet utility functions."""
    
    def test_write_and_read_parquet(self, tmp_path, sample_utterances):
        """Test writing and reading Parquet files."""
        normalized = normalize_utterances(sample_utterances)
        output_path = tmp_path / "test.parquet"
        
        # Write
        write_parquet(
            normalized,
            output_path,
            artifact_type="utterance",
            enforce_schema=False,
        )
        
        # Read
        df = read_parquet(output_path)
        
        assert len(df) == len(normalized)
        assert "utterance_id" in df.columns
    
    def test_read_parquet_with_columns(self, tmp_path, sample_utterances):
        """Test reading specific columns from Parquet file."""
        normalized = normalize_utterances(sample_utterances)
        output_path = tmp_path / "test.parquet"
        
        write_parquet(
            normalized,
            output_path,
            artifact_type="utterance",
            enforce_schema=False,
        )
        
        # Read only specific columns
        df = read_parquet(output_path, columns=["utterance_id", "text"])
        
        assert len(df.columns) == 2
        assert "utterance_id" in df.columns
        assert "text" in df.columns
    
    def test_read_parquet_file_not_found(self, tmp_path):
        """Test reading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            read_parquet(tmp_path / "nonexistent.parquet")
    
    def test_create_versioned_directory(self, tmp_path):
        """Test creating versioned directory."""
        versioned_dir = create_versioned_directory(
            tmp_path,
            "normalized",
            "v1",
        )
        
        assert versioned_dir.exists()
        assert versioned_dir == tmp_path / "normalized" / "v1"
    
    def test_write_versioned_parquet(self, tmp_path, sample_utterances):
        """Test writing versioned Parquet file."""
        normalized = normalize_utterances(sample_utterances)
        
        output_path = write_versioned_parquet(
            normalized,
            tmp_path,
            "normalized",
            "test.parquet",
            version="v1",
            enforce_schema=False,
        )
        
        assert output_path.exists()
        assert "v1" in str(output_path)
        assert "normalized" in str(output_path)
    
    def test_get_parquet_file_info(self, tmp_path, sample_utterances):
        """Test getting Parquet file metadata."""
        normalized = normalize_utterances(sample_utterances)
        output_path = tmp_path / "test.parquet"
        
        write_parquet(
            normalized,
            output_path,
            artifact_type="utterance",
            enforce_schema=False,
        )
        
        info = get_parquet_file_info(output_path)
        
        assert info["num_rows"] == len(normalized)
        assert info["num_columns"] > 0
        assert "utterance_id" in info["column_names"]


# ============================================================================
# Integration Tests
# ============================================================================

class TestIngestionPipeline:
    """Test full ingestion pipeline integration."""
    
    def test_full_pipeline(self, temp_jsonl_file, tmp_path):
        """Test complete ingestion pipeline: read -> validate -> normalize -> write."""
        # 1. Read
        reader = TranscriptReader(temp_jsonl_file)
        raw_utterances = reader.read_utterances()
        
        # 2. Validate
        result = validate_utterances(raw_utterances)
        assert result.is_valid
        
        # 3. Normalize
        normalized = normalize_utterances(raw_utterances)
        
        # 4. Write
        output_path = tmp_path / "output.parquet"
        write_parquet(
            normalized,
            output_path,
            artifact_type="utterance",
            enforce_schema=False,
        )
        
        # 5. Verify
        df = read_parquet(output_path)
        assert len(df) == len(raw_utterances)
        assert all("utterance_id" in col for col in ["utterance_id"])
    
    def test_pipeline_with_filtering(self, tmp_path):
        """Test pipeline with filtering of invalid utterances."""
        # Create file with mixed valid/invalid data
        file_path = tmp_path / "mixed.jsonl"
        utterances = [
            {"episode_id": "EP1", "start": 0.0, "end": 1.0, "speaker": "A", "text": "Valid"},
            {"episode_id": "EP1", "start": 1.0, "end": 2.0, "speaker": "", "text": "Invalid"},
            {"episode_id": "EP1", "start": 2.0, "end": 3.0, "speaker": "B", "text": "Valid"},
        ]
        
        with open(file_path, "w", encoding="utf-8") as f:
            for utt in utterances:
                f.write(json.dumps(utt) + "\n")
        
        # Read
        reader = TranscriptReader(file_path)
        raw_utterances = reader.read_utterances()
        
        # Filter valid only
        valid_utterances, result = filter_valid_utterances(raw_utterances)
        
        assert len(valid_utterances) == 2
        assert result.valid_count == 2
        
        # Normalize and write
        normalized = normalize_utterances(valid_utterances)
        output_path = tmp_path / "output.parquet"
        
        write_parquet(
            normalized,
            output_path,
            artifact_type="utterance",
            enforce_schema=False,
        )
        
        # Verify
        df = read_parquet(output_path)
        assert len(df) == 2

