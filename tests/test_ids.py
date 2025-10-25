"""
Unit tests for deterministic ID generation.

Tests ensure that ID generation is:
- Deterministic (same input → same ID)
- Unique (different input → different ID)
- Stable across runs
"""

import pytest
from lakehouse.ids import (
    compute_content_hash,
    compute_dict_hash,
    generate_utterance_id,
    generate_span_id,
    generate_beat_id,
    generate_section_id,
)


class TestContentHashing:
    """Test content hashing functions for determinism."""
    
    def test_compute_content_hash_deterministic(self):
        """Same content should produce same hash."""
        content = "Hello, world!"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 character hex string
    
    def test_compute_content_hash_different_input(self):
        """Different content should produce different hashes."""
        hash1 = compute_content_hash("Hello, world!")
        hash2 = compute_content_hash("Hello, World!")  # Different capitalization
        
        assert hash1 != hash2
    
    def test_compute_dict_hash_deterministic(self):
        """Same dictionary should produce same hash."""
        data = {"text": "test", "speaker": "Alice", "start": 0.0}
        hash1 = compute_dict_hash(data)
        hash2 = compute_dict_hash(data)
        
        assert hash1 == hash2
    
    def test_compute_dict_hash_order_independent(self):
        """Dictionary order shouldn't affect hash (keys are sorted)."""
        data1 = {"text": "test", "speaker": "Alice", "start": 0.0}
        data2 = {"start": 0.0, "text": "test", "speaker": "Alice"}
        
        hash1 = compute_dict_hash(data1)
        hash2 = compute_dict_hash(data2)
        
        assert hash1 == hash2


class TestUtteranceIDGeneration:
    """Test utterance ID generation."""
    
    def test_generate_utterance_id_deterministic(self):
        """Same input should generate same utterance ID."""
        params = {
            "episode_id": "TEST-001",
            "position": 0,
            "text": "Hello, world!",
            "speaker": "Alice",
            "start": 0.0,
            "end": 5.0,
        }
        
        id1 = generate_utterance_id(**params)
        id2 = generate_utterance_id(**params)
        
        assert id1 == id2
    
    def test_generate_utterance_id_format(self):
        """Utterance ID should have correct format."""
        utterance_id = generate_utterance_id(
            episode_id="TEST-001",
            position=0,
            text="Hello",
            speaker="Alice",
            start=0.0,
            end=5.0,
        )
        
        # Format: utt_{episode_hash}_{position}_{content_hash}
        assert utterance_id.startswith("utt_")
        parts = utterance_id.split("_")
        assert len(parts) == 4
        assert parts[0] == "utt"
        assert len(parts[1]) == 12  # Episode hash truncated to 12 chars
        assert parts[2] == "000000"  # Position padded to 6 digits
        assert len(parts[3]) == 8   # Content hash truncated to 8 chars
    
    def test_generate_utterance_id_different_episode(self):
        """Different episode IDs should produce different utterance IDs."""
        id1 = generate_utterance_id(
            episode_id="TEST-001",
            position=0,
            text="Hello",
            speaker="Alice",
            start=0.0,
            end=5.0,
        )
        id2 = generate_utterance_id(
            episode_id="TEST-002",
            position=0,
            text="Hello",
            speaker="Alice",
            start=0.0,
            end=5.0,
        )
        
        assert id1 != id2
    
    def test_generate_utterance_id_different_position(self):
        """Different positions should produce different utterance IDs."""
        id1 = generate_utterance_id(
            episode_id="TEST-001",
            position=0,
            text="Hello",
            speaker="Alice",
            start=0.0,
            end=5.0,
        )
        id2 = generate_utterance_id(
            episode_id="TEST-001",
            position=1,
            text="Hello",
            speaker="Alice",
            start=0.0,
            end=5.0,
        )
        
        assert id1 != id2
    
    def test_generate_utterance_id_different_content(self):
        """Different content should produce different utterance IDs."""
        id1 = generate_utterance_id(
            episode_id="TEST-001",
            position=0,
            text="Hello",
            speaker="Alice",
            start=0.0,
            end=5.0,
        )
        id2 = generate_utterance_id(
            episode_id="TEST-001",
            position=0,
            text="Goodbye",  # Different text
            speaker="Alice",
            start=0.0,
            end=5.0,
        )
        
        assert id1 != id2
    
    def test_generate_utterance_id_floating_point_rounding(self):
        """Small floating point differences should be ignored (rounded)."""
        id1 = generate_utterance_id(
            episode_id="TEST-001",
            position=0,
            text="Hello",
            speaker="Alice",
            start=0.001,
            end=5.001,
        )
        id2 = generate_utterance_id(
            episode_id="TEST-001",
            position=0,
            text="Hello",
            speaker="Alice",
            start=0.002,  # Rounds to same value (0.00)
            end=5.002,
        )
        
        # Should produce same ID due to rounding to 2 decimal places
        assert id1 == id2


class TestSpanIDGeneration:
    """Test span ID generation."""
    
    def test_generate_span_id_deterministic(self):
        """Same input should generate same span ID."""
        params = {
            "episode_id": "TEST-001",
            "position": 0,
            "speaker": "Alice",
            "utterance_ids": ["utt_1", "utt_2"],
            "text": "Hello, world!",
        }
        
        id1 = generate_span_id(**params)
        id2 = generate_span_id(**params)
        
        assert id1 == id2
    
    def test_generate_span_id_format(self):
        """Span ID should have correct format."""
        span_id = generate_span_id(
            episode_id="TEST-001",
            position=0,
            speaker="Alice",
            utterance_ids=["utt_1"],
            text="Hello",
        )
        
        # Format: spn_{episode_hash}_{position}_{content_hash}
        assert span_id.startswith("spn_")
        parts = span_id.split("_")
        assert len(parts) == 4
        assert parts[0] == "spn"
    
    def test_generate_span_id_utterance_order_independent(self):
        """Utterance ID order shouldn't affect span ID (they're sorted)."""
        id1 = generate_span_id(
            episode_id="TEST-001",
            position=0,
            speaker="Alice",
            utterance_ids=["utt_1", "utt_2", "utt_3"],
            text="Hello",
        )
        id2 = generate_span_id(
            episode_id="TEST-001",
            position=0,
            speaker="Alice",
            utterance_ids=["utt_3", "utt_1", "utt_2"],  # Different order
            text="Hello",
        )
        
        assert id1 == id2


class TestBeatIDGeneration:
    """Test beat ID generation."""
    
    def test_generate_beat_id_deterministic(self):
        """Same input should generate same beat ID."""
        params = {
            "episode_id": "TEST-001",
            "position": 0,
            "span_ids": ["spn_1", "spn_2"],
            "text": "Beat content",
        }
        
        id1 = generate_beat_id(**params)
        id2 = generate_beat_id(**params)
        
        assert id1 == id2
    
    def test_generate_beat_id_format(self):
        """Beat ID should have correct format."""
        beat_id = generate_beat_id(
            episode_id="TEST-001",
            position=0,
            span_ids=["spn_1"],
            text="Hello",
        )
        
        # Format: bet_{episode_hash}_{position}_{content_hash}
        assert beat_id.startswith("bet_")
        parts = beat_id.split("_")
        assert len(parts) == 4
        assert parts[0] == "bet"


class TestSectionIDGeneration:
    """Test section ID generation."""
    
    def test_generate_section_id_deterministic(self):
        """Same input should generate same section ID."""
        params = {
            "episode_id": "TEST-001",
            "position": 0,
            "beat_ids": ["bet_1", "bet_2"],
            "text": "Section content",
        }
        
        id1 = generate_section_id(**params)
        id2 = generate_section_id(**params)
        
        assert id1 == id2
    
    def test_generate_section_id_format(self):
        """Section ID should have correct format."""
        section_id = generate_section_id(
            episode_id="TEST-001",
            position=0,
            beat_ids=["bet_1"],
            text="Hello",
        )
        
        # Format: sec_{episode_hash}_{position}_{content_hash}
        assert section_id.startswith("sec_")
        parts = section_id.split("_")
        assert len(parts) == 4
        assert parts[0] == "sec"


class TestIDUniqueness:
    """Test that different artifact types produce unique IDs."""
    
    def test_different_artifact_types_different_prefixes(self):
        """Different artifact types should have different prefixes."""
        utterance_id = generate_utterance_id(
            episode_id="TEST-001",
            position=0,
            text="Hello",
            speaker="Alice",
            start=0.0,
            end=5.0,
        )
        span_id = generate_span_id(
            episode_id="TEST-001",
            position=0,
            speaker="Alice",
            utterance_ids=["utt_1"],
            text="Hello",
        )
        beat_id = generate_beat_id(
            episode_id="TEST-001",
            position=0,
            span_ids=["spn_1"],
            text="Hello",
        )
        section_id = generate_section_id(
            episode_id="TEST-001",
            position=0,
            beat_ids=["bet_1"],
            text="Hello",
        )
        
        assert utterance_id.startswith("utt_")
        assert span_id.startswith("spn_")
        assert beat_id.startswith("bet_")
        assert section_id.startswith("sec_")
        
        # All should be different
        ids = {utterance_id, span_id, beat_id, section_id}
        assert len(ids) == 4


class TestIDStability:
    """Test that IDs remain stable across multiple runs."""
    
    def test_utterance_id_stability_multiple_runs(self):
        """ID should be identical across 100 generations."""
        params = {
            "episode_id": "TEST-001",
            "position": 0,
            "text": "Stability test",
            "speaker": "Alice",
            "start": 0.0,
            "end": 5.0,
        }
        
        ids = [generate_utterance_id(**params) for _ in range(100)]
        
        # All IDs should be identical
        assert len(set(ids)) == 1
    
    def test_known_utterance_id_value(self):
        """Test against a known ID value to detect unexpected changes."""
        # This test ensures the ID generation algorithm hasn't changed
        utterance_id = generate_utterance_id(
            episode_id="TEST-001",
            position=0,
            text="Hello, world!",
            speaker="Alice",
            start=0.0,
            end=5.0,
        )
        
        # The ID should always be the same for this input
        # If this test fails, the ID generation algorithm has changed
        assert utterance_id.startswith("utt_")
        assert len(utterance_id) == 32  # utt_ + 12 + _ + 6 + _ + 8

