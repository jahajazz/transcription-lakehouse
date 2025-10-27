"""
Unit tests for speaker role configuration and enrichment.

Tests config loading, validation, role determination, and metadata enrichment
for spans and beats.
"""

import pytest
import tempfile
from pathlib import Path
import yaml

from lakehouse.speaker_roles import (
    SpeakerRoleConfig,
    determine_speaker_role,
    enrich_spans_with_speaker_metadata,
    enrich_beats_with_speaker_metadata,
    calculate_expert_coverage_pct,
)


# Test fixtures

@pytest.fixture
def valid_config_file():
    """Create a temporary valid config file."""
    config_data = {
        "experts": ["Fr Stephen De Young", "Jonathan Pageau"],
        "roles": {
            "Fr Stephen De Young": "expert",
            "Jonathan Pageau": "expert",
            "Host Name": "host",
        },
        "default_role": "other"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    temp_path.unlink()


@pytest.fixture
def missing_keys_config_file():
    """Create a config file missing required keys."""
    config_data = {
        "experts": ["Expert One"],
        # Missing "default_role"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    temp_path.unlink()


@pytest.fixture
def invalid_role_config_file():
    """Create a config file with invalid role value."""
    config_data = {
        "experts": ["Expert One"],
        "roles": {
            "Expert One": "invalid_role",  # Invalid role
        },
        "default_role": "other"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    temp_path.unlink()


@pytest.fixture
def sample_config():
    """Create a SpeakerRoleConfig instance for testing."""
    config_data = {
        "experts": ["Fr Stephen De Young", "Jonathan Pageau"],
        "roles": {
            "Fr Stephen De Young": "expert",
            "Jonathan Pageau": "expert",
            "Host Name": "host",
        },
        "default_role": "other"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = Path(f.name)
    
    config = SpeakerRoleConfig(temp_path)
    temp_path.unlink()
    
    return config


# Config Loading Tests

def test_load_valid_config(valid_config_file):
    """Test loading a valid configuration file."""
    config = SpeakerRoleConfig(valid_config_file)
    
    assert len(config.experts) == 2
    assert "Fr Stephen De Young" in config.experts
    assert "Jonathan Pageau" in config.experts
    assert config.default_role == "other"
    assert len(config.roles) == 3


def test_config_file_not_found():
    """Test that missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        SpeakerRoleConfig(Path("nonexistent/config.yaml"))
    
    assert "not found" in str(exc_info.value).lower()


def test_config_missing_required_keys(missing_keys_config_file):
    """Test that config missing required keys raises ValueError (Task 1.10)."""
    with pytest.raises(ValueError) as exc_info:
        SpeakerRoleConfig(missing_keys_config_file)
    
    error_msg = str(exc_info.value).lower()
    assert "missing required keys" in error_msg
    assert "default_role" in error_msg


def test_config_invalid_default_role():
    """Test that invalid default_role raises ValueError (Task 1.10)."""
    config_data = {
        "experts": ["Expert One"],
        "default_role": "invalid_role"  # Invalid
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = Path(f.name)
    
    try:
        with pytest.raises(ValueError) as exc_info:
            SpeakerRoleConfig(temp_path)
        
        error_msg = str(exc_info.value)
        assert "default_role" in error_msg.lower()
        assert "invalid_role" in error_msg
    finally:
        temp_path.unlink()


def test_config_invalid_role_in_map(invalid_role_config_file):
    """Test that invalid role in roles map raises ValueError (Task 1.10)."""
    with pytest.raises(ValueError) as exc_info:
        SpeakerRoleConfig(invalid_role_config_file)
    
    error_msg = str(exc_info.value)
    assert "invalid role" in error_msg.lower()
    assert "invalid_role" in error_msg


def test_config_empty_file():
    """Test that empty config file raises ValueError (Task 1.10)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")  # Empty file
        temp_path = Path(f.name)
    
    try:
        with pytest.raises(ValueError) as exc_info:
            SpeakerRoleConfig(temp_path)
        
        assert "empty" in str(exc_info.value).lower()
    finally:
        temp_path.unlink()


# Role Determination Tests

def test_is_expert_true(sample_config):
    """Test is_expert returns True for configured experts."""
    assert sample_config.is_expert("Fr Stephen De Young") is True
    assert sample_config.is_expert("Jonathan Pageau") is True


def test_is_expert_false(sample_config):
    """Test is_expert returns False for non-experts."""
    assert sample_config.is_expert("Host Name") is False
    assert sample_config.is_expert("Unknown Speaker") is False


def test_get_role_explicit_mapping(sample_config):
    """Test get_role returns explicitly mapped roles."""
    assert sample_config.get_role("Host Name") == "host"


def test_get_role_expert_default(sample_config):
    """Test get_role returns 'expert' for speakers in experts list but not in roles map."""
    # Remove Jonathan Pageau from roles map but keep in experts list
    sample_config.roles.pop("Jonathan Pageau", None)
    assert sample_config.get_role("Jonathan Pageau") == "expert"


def test_get_role_default(sample_config):
    """Test get_role returns default_role for unmapped speakers."""
    assert sample_config.get_role("Random Speaker") == "other"


def test_determine_speaker_role_expert(sample_config):
    """Test determine_speaker_role for expert speaker."""
    canonical, role, is_expert = determine_speaker_role("Fr Stephen De Young", sample_config)
    
    assert canonical == "Fr Stephen De Young"
    assert role == "expert"
    assert is_expert is True


def test_determine_speaker_role_non_expert(sample_config):
    """Test determine_speaker_role for non-expert speaker."""
    canonical, role, is_expert = determine_speaker_role("Host Name", sample_config)
    
    assert canonical == "Host Name"
    assert role == "host"
    assert is_expert is False


def test_determine_speaker_role_unknown(sample_config):
    """Test determine_speaker_role for unknown speaker."""
    canonical, role, is_expert = determine_speaker_role("Unknown Person", sample_config)
    
    assert canonical == "Unknown Person"
    assert role == "other"
    assert is_expert is False


# Span Enrichment Tests

def test_enrich_spans_basic(sample_config):
    """Test basic span enrichment with speaker metadata."""
    spans = [
        {"span_id": "s1", "speaker": "Fr Stephen De Young", "text": "Test"},
        {"span_id": "s2", "speaker": "Host Name", "text": "Test 2"},
    ]
    
    enriched = enrich_spans_with_speaker_metadata(spans, sample_config)
    
    # Check first span (expert)
    assert enriched[0]["speaker_canonical"] == "Fr Stephen De Young"
    assert enriched[0]["speaker_role"] == "expert"
    assert enriched[0]["is_expert"] is True
    
    # Check second span (host)
    assert enriched[1]["speaker_canonical"] == "Host Name"
    assert enriched[1]["speaker_role"] == "host"
    assert enriched[1]["is_expert"] is False


def test_enrich_spans_missing_speaker(sample_config):
    """Test span enrichment handles missing speaker field gracefully."""
    spans = [
        {"span_id": "s1", "text": "No speaker"},
    ]
    
    enriched = enrich_spans_with_speaker_metadata(spans, sample_config)
    
    assert enriched[0]["speaker_canonical"] == ""
    assert enriched[0]["speaker_role"] == "other"
    assert enriched[0]["is_expert"] is False


def test_enrich_spans_modifies_in_place(sample_config):
    """Test that enrichment modifies spans in place and returns them."""
    spans = [{"span_id": "s1", "speaker": "Fr Stephen De Young"}]
    
    result = enrich_spans_with_speaker_metadata(spans, sample_config)
    
    # Should be same object
    assert result is spans
    assert "is_expert" in spans[0]


# Beat Enrichment Tests

def test_enrich_beats_basic(sample_config):
    """Test basic beat enrichment with speaker metadata."""
    spans = [
        {"span_id": "s1", "speaker_canonical": "Fr Stephen De Young", "is_expert": True, "text": "a" * 100},
        {"span_id": "s2", "speaker_canonical": "Host Name", "is_expert": False, "text": "b" * 100},
        {"span_id": "s3", "speaker_canonical": "Fr Stephen De Young", "is_expert": True, "text": "c" * 50},
    ]
    
    beats = [
        {"beat_id": "b1", "span_ids": ["s1", "s2", "s3"]},
    ]
    
    enriched = enrich_beats_with_speaker_metadata(beats, spans, sample_config)
    
    # Check speakers_set (deduplicated, ordered)
    assert enriched[0]["speakers_set"] == ["Fr Stephen De Young", "Host Name"]
    
    # Check expert_span_ids
    assert set(enriched[0]["expert_span_ids"]) == {"s1", "s3"}
    
    # Check expert_coverage_pct (150 expert chars / 250 total = 60%)
    assert enriched[0]["expert_coverage_pct"] == pytest.approx(60.0, rel=0.01)


def test_enrich_beats_no_span_ids(sample_config):
    """Test beat enrichment handles empty span_ids gracefully."""
    beats = [{"beat_id": "b1", "span_ids": []}]
    spans = []
    
    enriched = enrich_beats_with_speaker_metadata(beats, spans, sample_config)
    
    assert enriched[0]["speakers_set"] == []
    assert enriched[0]["expert_span_ids"] == []
    assert enriched[0]["expert_coverage_pct"] == 0.0


def test_enrich_beats_missing_span_reference(sample_config):
    """Test beat enrichment handles missing span references gracefully."""
    beats = [{"beat_id": "b1", "span_ids": ["s1", "s999"]}]  # s999 doesn't exist
    spans = [
        {"span_id": "s1", "speaker_canonical": "Host Name", "is_expert": False, "text": "test"}
    ]
    
    enriched = enrich_beats_with_speaker_metadata(beats, spans, sample_config)
    
    # Should still work with available span
    assert enriched[0]["speakers_set"] == ["Host Name"]
    assert enriched[0]["expert_span_ids"] == []


# Expert Coverage Calculation Tests

def test_calculate_expert_coverage_token_weighted():
    """Test expert coverage calculation using token counts."""
    span_lookup = {
        "s1": {"span_id": "s1", "token_count": 100, "text": "a" * 500},
        "s2": {"span_id": "s2", "token_count": 50, "text": "b" * 250},
        "s3": {"span_id": "s3", "token_count": 50, "text": "c" * 250},
    }
    
    expert_span_ids = ["s1", "s3"]  # 150 tokens out of 200
    
    coverage = calculate_expert_coverage_pct(
        ["s1", "s2", "s3"], span_lookup, expert_span_ids
    )
    
    assert coverage == pytest.approx(75.0, rel=0.01)


def test_calculate_expert_coverage_char_weighted():
    """Test expert coverage calculation falls back to character count."""
    span_lookup = {
        "s1": {"span_id": "s1", "text": "a" * 100},  # No token_count
        "s2": {"span_id": "s2", "text": "b" * 100},
        "s3": {"span_id": "s3", "text": "c" * 200},
    }
    
    expert_span_ids = ["s1", "s3"]  # 300 chars out of 400
    
    coverage = calculate_expert_coverage_pct(
        ["s1", "s2", "s3"], span_lookup, expert_span_ids
    )
    
    assert coverage == pytest.approx(75.0, rel=0.01)


def test_calculate_expert_coverage_all_experts():
    """Test expert coverage is 100% when all spans are experts."""
    span_lookup = {
        "s1": {"span_id": "s1", "token_count": 100},
        "s2": {"span_id": "s2", "token_count": 50},
    }
    
    expert_span_ids = ["s1", "s2"]
    
    coverage = calculate_expert_coverage_pct(
        ["s1", "s2"], span_lookup, expert_span_ids
    )
    
    assert coverage == pytest.approx(100.0, rel=0.01)


def test_calculate_expert_coverage_no_experts():
    """Test expert coverage is 0% when no spans are experts."""
    span_lookup = {
        "s1": {"span_id": "s1", "token_count": 100},
        "s2": {"span_id": "s2", "token_count": 50},
    }
    
    expert_span_ids = []
    
    coverage = calculate_expert_coverage_pct(
        ["s1", "s2"], span_lookup, expert_span_ids
    )
    
    assert coverage == 0.0


def test_calculate_expert_coverage_empty_spans():
    """Test expert coverage handles empty span list."""
    coverage = calculate_expert_coverage_pct([], {}, [])
    assert coverage == 0.0


def test_calculate_expert_coverage_mixed_token_presence():
    """Test that token weighting is used if any spans have token_count."""
    span_lookup = {
        "s1": {"span_id": "s1", "token_count": 100, "text": "a" * 1000},
        "s2": {"span_id": "s2", "text": "b" * 10},  # No token_count, but has text
    }
    
    expert_span_ids = ["s1"]
    
    # Should use token weighting since s1 has token_count
    # Only s1's 100 tokens count, s2 has no tokens so contributes 0
    coverage = calculate_expert_coverage_pct(
        ["s1", "s2"], span_lookup, expert_span_ids
    )
    
    assert coverage == pytest.approx(100.0, rel=0.01)


# Integration Tests

def test_full_enrichment_pipeline(sample_config):
    """Test complete enrichment pipeline from spans to beats."""
    # Step 1: Enrich spans
    spans = [
        {"span_id": "s1", "speaker": "Fr Stephen De Young", "text": "Theological discussion", "token_count": 200},
        {"span_id": "s2", "speaker": "Jonathan Pageau", "text": "Symbolic insights", "token_count": 150},
        {"span_id": "s3", "speaker": "Host Name", "text": "Introduction", "token_count": 50},
    ]
    
    enriched_spans = enrich_spans_with_speaker_metadata(spans, sample_config)
    
    # Step 2: Enrich beats
    beats = [
        {"beat_id": "b1", "span_ids": ["s1", "s2", "s3"]},
    ]
    
    enriched_beats = enrich_beats_with_speaker_metadata(beats, enriched_spans, sample_config)
    
    # Verify beat enrichment
    beat = enriched_beats[0]
    assert len(beat["speakers_set"]) == 3
    assert len(beat["expert_span_ids"]) == 2  # s1 and s2
    
    # Expert coverage: (200 + 150) / (200 + 150 + 50) = 87.5%
    assert beat["expert_coverage_pct"] == pytest.approx(87.5, rel=0.01)

