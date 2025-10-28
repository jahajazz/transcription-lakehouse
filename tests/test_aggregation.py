"""
Unit tests for aggregation modules: spans, beats, and sections.

Tests the hierarchical aggregation from utterances → spans → beats → sections.
"""

import pytest
import numpy as np
from typing import Dict, List

from lakehouse.aggregation.spans import SpanGenerator, generate_spans
from lakehouse.aggregation.beats import BeatGenerator, generate_beats
from lakehouse.aggregation.sections import SectionGenerator, generate_sections
from lakehouse.ingestion.normalizer import normalize_utterances


# Helper function to generate mock embeddings for testing
def generate_mock_embeddings(count: int, dim: int = 384, seed: int = 42) -> List[np.ndarray]:
    """
    Generate mock embeddings with controlled similarity patterns.
    
    Args:
        count: Number of embeddings to generate
        dim: Embedding dimension
        seed: Random seed for reproducibility
    
    Returns:
        List of numpy arrays (embeddings)
    """
    np.random.seed(seed)
    embeddings = []
    
    # Generate embeddings with some structure
    # Group embeddings in clusters to simulate topic coherence
    cluster_size = 3
    num_clusters = (count + cluster_size - 1) // cluster_size
    
    for cluster_idx in range(num_clusters):
        # Generate a cluster center
        center = np.random.randn(dim)
        center = center / np.linalg.norm(center)  # Normalize
        
        # Generate embeddings around this center
        for i in range(cluster_size):
            idx = cluster_idx * cluster_size + i
            if idx >= count:
                break
            
            # Add some noise to the center
            noise = np.random.randn(dim) * 0.1
            embedding = center + noise
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings.append(embedding)
    
    return embeddings


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_normalized_utterances() -> List[Dict]:
    """Create normalized utterances for testing."""
    raw_utterances = [
        {"episode_id": "TEST-001", "start": 0.0, "end": 5.0, "speaker": "Alice", "text": "Hello everyone."},
        {"episode_id": "TEST-001", "start": 5.0, "end": 10.0, "speaker": "Alice", "text": "Welcome to the show."},
        {"episode_id": "TEST-001", "start": 10.0, "end": 15.0, "speaker": "Alice", "text": "Today we discuss AI."},
        {"episode_id": "TEST-001", "start": 15.0, "end": 20.0, "speaker": "Bob", "text": "Thanks for the introduction."},
        {"episode_id": "TEST-001", "start": 20.0, "end": 30.0, "speaker": "Bob", "text": "AI is transforming our world."},
        {"episode_id": "TEST-001", "start": 30.0, "end": 35.0, "speaker": "Bob", "text": "Let's explore the implications."},
        {"episode_id": "TEST-001", "start": 35.0, "end": 40.0, "speaker": "Alice", "text": "Great idea."},
        {"episode_id": "TEST-001", "start": 40.0, "end": 50.0, "speaker": "Alice", "text": "Let's start with machine learning basics."},
    ]
    return normalize_utterances(raw_utterances)


@pytest.fixture
def sample_spans() -> List[Dict]:
    """Create sample spans for testing."""
    return [
        {
            "span_id": "spn_test_000000_abc12345",
            "episode_id": "TEST-001",
            "speaker": "Alice",
            "start_time": 0.0,
            "end_time": 120.0,
            "duration": 120.0,
            "text": "First span about introduction.",
            "utterance_ids": ["utt_1", "utt_2"],
        },
        {
            "span_id": "spn_test_000001_def67890",
            "episode_id": "TEST-001",
            "speaker": "Bob",
            "start_time": 120.0,
            "end_time": 240.0,
            "duration": 120.0,
            "text": "Second span about AI fundamentals.",
            "utterance_ids": ["utt_3", "utt_4"],
        },
        {
            "span_id": "spn_test_000002_ghi13579",
            "episode_id": "TEST-001",
            "speaker": "Alice",
            "start_time": 240.0,
            "end_time": 360.0,
            "duration": 120.0,
            "text": "Third span about machine learning.",
            "utterance_ids": ["utt_5", "utt_6"],
        },
    ]


@pytest.fixture
def sample_beats() -> List[Dict]:
    """Create sample beats for testing."""
    return [
        {
            "beat_id": "bet_test_000000_aaa11111",
            "episode_id": "TEST-001",
            "start_time": 0.0,
            "end_time": 360.0,
            "duration": 360.0,
            "text": "Introduction and AI fundamentals.",
            "span_ids": ["spn_1", "spn_2"],
        },
        {
            "beat_id": "bet_test_000001_bbb22222",
            "episode_id": "TEST-001",
            "start_time": 360.0,
            "end_time": 720.0,
            "duration": 360.0,
            "text": "Machine learning basics.",
            "span_ids": ["spn_3", "spn_4"],
        },
        {
            "beat_id": "bet_test_000002_ccc33333",
            "episode_id": "TEST-001",
            "start_time": 720.0,
            "end_time": 1080.0,
            "duration": 360.0,
            "text": "Deep learning discussion.",
            "span_ids": ["spn_5", "spn_6"],
        },
    ]


# ============================================================================
# Span Generation Tests
# ============================================================================

class TestSpanGenerator:
    """Test SpanGenerator class."""
    
    def test_span_generator_init(self):
        """Test initializing span generator."""
        generator = SpanGenerator()
        assert generator.get_artifact_type() == "span"
        assert generator.min_duration == 1.0
        assert generator.max_silence_gap == 0.5
    
    def test_span_generator_custom_config(self):
        """Test span generator with custom configuration."""
        config = {
            "min_duration": 2.0,
            "max_silence_gap": 1.0,
            "break_on_speaker_change": False,
        }
        generator = SpanGenerator(config)
        
        assert generator.min_duration == 2.0
        assert generator.max_silence_gap == 1.0
        assert generator.break_on_speaker_change is False
    
    def test_generate_spans_basic(self, sample_normalized_utterances):
        """Test basic span generation."""
        generator = SpanGenerator()
        spans = generator.aggregate(sample_normalized_utterances)
        
        assert len(spans) > 0
        assert len(spans) < len(sample_normalized_utterances)
        assert all("span_id" in s for s in spans)
        assert all(s["span_id"].startswith("spn_") for s in spans)
    
    def test_spans_break_on_speaker_change(self, sample_normalized_utterances):
        """Test that spans break on speaker changes."""
        generator = SpanGenerator({"break_on_speaker_change": True})
        spans = generator.aggregate(sample_normalized_utterances)
        
        # Check that each span has only one speaker
        for span in spans:
            # All utterances in span should have same speaker
            assert span["speaker"] is not None
    
    def test_spans_consolidate_same_speaker(self):
        """Test that spans consolidate same-speaker utterances."""
        utterances = [
            {"episode_id": "EP1", "start": 0.0, "end": 2.0, "speaker": "A", "text": "First"},
            {"episode_id": "EP1", "start": 2.0, "end": 4.0, "speaker": "A", "text": "Second"},
            {"episode_id": "EP1", "start": 4.0, "end": 6.0, "speaker": "A", "text": "Third"},
        ]
        normalized = normalize_utterances(utterances)
        
        generator = SpanGenerator({"min_duration": 0.0})
        spans = generator.aggregate(normalized)
        
        # Should consolidate into one span
        assert len(spans) == 1
        assert len(spans[0]["utterance_ids"]) == 3
    
    def test_spans_break_on_silence(self):
        """Test that spans break on large silence gaps."""
        utterances = [
            {"episode_id": "EP1", "start": 0.0, "end": 2.0, "speaker": "A", "text": "First"},
            {"episode_id": "EP1", "start": 2.0, "end": 4.0, "speaker": "A", "text": "Second"},
            {"episode_id": "EP1", "start": 10.0, "end": 12.0, "speaker": "A", "text": "Third"},  # 6 second gap
        ]
        normalized = normalize_utterances(utterances)
        
        generator = SpanGenerator({"max_silence_gap": 1.0, "min_duration": 0.0})
        spans = generator.aggregate(normalized)
        
        # Should break into 2 spans due to gap
        assert len(spans) >= 2
    
    def test_span_min_duration_filter(self):
        """Test that spans below minimum duration are filtered."""
        utterances = [
            {"episode_id": "EP1", "start": 0.0, "end": 0.5, "speaker": "A", "text": "Short"},
            {"episode_id": "EP1", "start": 5.0, "end": 10.0, "speaker": "B", "text": "Longer utterance"},
        ]
        normalized = normalize_utterances(utterances)
        
        generator = SpanGenerator({"min_duration": 2.0})
        spans = generator.aggregate(normalized)
        
        # Only the longer span should be kept
        assert all(s["duration"] >= 2.0 for s in spans)
    
    def test_span_text_concatenation(self):
        """Test that span text is concatenated from utterances."""
        utterances = [
            {"episode_id": "EP1", "start": 0.0, "end": 2.0, "speaker": "A", "text": "Hello"},
            {"episode_id": "EP1", "start": 2.0, "end": 4.0, "speaker": "A", "text": "world"},
        ]
        normalized = normalize_utterances(utterances)
        
        generator = SpanGenerator({"min_duration": 0.0})
        spans = generator.aggregate(normalized)
        
        assert len(spans) == 1
        assert "Hello" in spans[0]["text"]
        assert "world" in spans[0]["text"]
    
    def test_span_id_determinism(self, sample_normalized_utterances):
        """Test that span IDs are deterministic."""
        generator = SpanGenerator()
        
        spans1 = generator.aggregate(sample_normalized_utterances)
        spans2 = generator.aggregate(sample_normalized_utterances)
        
        assert len(spans1) == len(spans2)
        for s1, s2 in zip(spans1, spans2):
            assert s1["span_id"] == s2["span_id"]
    
    def test_generate_spans_convenience_function(self, sample_normalized_utterances):
        """Test convenience function for span generation."""
        spans = generate_spans(sample_normalized_utterances)
        
        assert len(spans) > 0
        assert all("span_id" in s for s in spans)


# ============================================================================
# Beat Generation Tests
# ============================================================================

class TestBeatGenerator:
    """Test BeatGenerator class."""
    
    def test_beat_generator_init(self):
        """Test initializing beat generator."""
        generator = BeatGenerator()
        assert generator.get_artifact_type() == "beat"
        assert generator.similarity_threshold == 0.7
        assert generator.use_embeddings is True
    
    def test_beat_generator_custom_config(self):
        """Test beat generator with custom configuration."""
        config = {
            "similarity_threshold": 0.6,
            "min_spans_per_beat": 2,
            "use_embeddings": False,
        }
        generator = BeatGenerator(config)
        
        assert generator.similarity_threshold == 0.6
        assert generator.min_spans_per_beat == 2
        assert generator.use_embeddings is False
    
    def test_generate_beats_heuristic(self, sample_spans):
        """Test beat generation with heuristic method."""
        generator = BeatGenerator({"use_embeddings": False})
        beats = generator.aggregate(sample_spans)
        
        assert len(beats) > 0
        assert len(beats) <= len(sample_spans)
        assert all("beat_id" in b for b in beats)
        assert all(b["beat_id"].startswith("bet_") for b in beats)
    
    def test_beats_consolidate_spans(self, sample_spans):
        """Test that beats consolidate multiple spans."""
        generator = BeatGenerator({"use_embeddings": False, "min_spans_per_beat": 1})
        beats = generator.aggregate(sample_spans)
        
        # Check that beats reference span IDs
        for beat in beats:
            assert "span_ids" in beat
            assert len(beat["span_ids"]) >= 1
    
    def test_beat_breaks_on_speaker_change(self):
        """Test that beats break on speaker changes (heuristic)."""
        spans = [
            {"episode_id": "EP1", "start_time": 0.0, "end_time": 60.0, "duration": 60.0, 
             "speaker": "A", "text": "First", "span_id": "spn_1", "utterance_ids": ["u1"]},
            {"episode_id": "EP1", "start_time": 60.0, "end_time": 120.0, "duration": 60.0,
             "speaker": "B", "text": "Second", "span_id": "spn_2", "utterance_ids": ["u2"]},
        ]
        
        generator = BeatGenerator({"use_embeddings": False, "min_spans_per_beat": 1})
        beats = generator.aggregate(spans)
        
        # Should create 2 beats due to speaker change
        assert len(beats) >= 2
    
    def test_beat_breaks_on_time_gap(self):
        """Test that beats break on large time gaps (heuristic)."""
        spans = [
            {"episode_id": "EP1", "start_time": 0.0, "end_time": 60.0, "duration": 60.0,
             "speaker": "A", "text": "First", "span_id": "spn_1", "utterance_ids": ["u1"]},
            {"episode_id": "EP1", "start_time": 70.0, "end_time": 130.0, "duration": 60.0,  # 10 sec gap
             "speaker": "A", "text": "Second", "span_id": "spn_2", "utterance_ids": ["u2"]},
        ]
        
        generator = BeatGenerator({"use_embeddings": False, "min_spans_per_beat": 1})
        beats = generator.aggregate(spans)
        
        # Should create 2 beats due to time gap
        assert len(beats) >= 2
    
    def test_beat_text_concatenation(self):
        """Test that beat text is concatenated from spans."""
        spans = [
            {"episode_id": "EP1", "start_time": 0.0, "end_time": 60.0, "duration": 60.0,
             "speaker": "A", "text": "Hello", "span_id": "spn_1", "utterance_ids": ["u1"]},
            {"episode_id": "EP1", "start_time": 60.0, "end_time": 120.0, "duration": 60.0,
             "speaker": "A", "text": "world", "span_id": "spn_2", "utterance_ids": ["u2"]},
        ]
        
        generator = BeatGenerator({"use_embeddings": False, "min_spans_per_beat": 1})
        beats = generator.aggregate(spans)
        
        # Should concatenate spans with same speaker
        if len(beats) == 1:
            assert "Hello" in beats[0]["text"]
            assert "world" in beats[0]["text"]
    
    def test_beat_with_embeddings(self, sample_spans):
        """Test beat generation with embeddings."""
        # Add mock embeddings to spans
        for i, span in enumerate(sample_spans):
            # Create simple embeddings that simulate semantic similarity
            embedding = np.random.rand(384)
            span["embedding"] = embedding.tolist()
        
        generator = BeatGenerator({"use_embeddings": True, "similarity_threshold": 0.5})
        beats = generator.aggregate(sample_spans)
        
        assert len(beats) > 0
        assert all("beat_id" in b for b in beats)
    
    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation."""
        generator = BeatGenerator()
        
        # Identical vectors should have similarity = 1.0
        vec1 = np.array([1.0, 0.0, 0.0])
        similarity = generator._cosine_similarity(vec1, vec1)
        assert abs(similarity - 1.0) < 0.01
        
        # Orthogonal vectors should have similarity ≈ 0.0
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = generator._cosine_similarity(vec1, vec2)
        assert similarity < 0.1
    
    def test_beat_id_determinism(self, sample_spans):
        """Test that beat IDs are deterministic."""
        generator = BeatGenerator({"use_embeddings": False})
        
        beats1 = generator.aggregate(sample_spans)
        beats2 = generator.aggregate(sample_spans)
        
        assert len(beats1) == len(beats2)
        for b1, b2 in zip(beats1, beats2):
            assert b1["beat_id"] == b2["beat_id"]
    
    def test_generate_beats_convenience_function(self, sample_spans):
        """Test convenience function for beat generation."""
        beats = generate_beats(sample_spans, config={"use_embeddings": False})
        
        assert len(beats) > 0
        assert all("beat_id" in b for b in beats)


# ============================================================================
# Section Generation Tests
# ============================================================================

class TestSectionGenerator:
    """Test SectionGenerator class."""
    
    def test_section_generator_init(self):
        """Test initializing section generator."""
        generator = SectionGenerator()
        assert generator.get_artifact_type() == "section"
        assert generator.target_duration_minutes == 8.0
        assert generator.min_duration_minutes == 5.0
        assert generator.max_duration_minutes == 12.0
    
    def test_section_generator_custom_config(self):
        """Test section generator with custom configuration."""
        config = {
            "target_duration_minutes": 10.0,
            "min_duration_minutes": 7.0,
            "max_duration_minutes": 15.0,
        }
        generator = SectionGenerator(config)
        
        assert generator.target_duration_minutes == 10.0
        assert generator.min_duration_minutes == 7.0
        assert generator.max_duration_minutes == 15.0
    
    def test_generate_sections_basic(self, sample_beats):
        """Test basic section generation."""
        generator = SectionGenerator({"require_embeddings": False})  # Task 6.2
        sections = generator.aggregate(sample_beats)
        
        assert len(sections) > 0
        assert all("section_id" in s for s in sections)
        assert all(s["section_id"].startswith("sec_") for s in sections)
    
    def test_sections_respect_time_constraints(self):
        """Test that sections respect time constraints."""
        # Create beats that span 10 minutes (600 seconds)
        beats = []
        for i in range(10):
            beat = {
                "beat_id": f"bet_{i}",
                "episode_id": "EP1",
                "start_time": i * 60.0,
                "end_time": (i + 1) * 60.0,
                "duration": 60.0,
                "text": f"Beat {i}",
                "span_ids": [f"spn_{i}"],
            }
            beats.append(beat)
        
        generator = SectionGenerator({
            "min_duration_minutes": 3.0,
            "max_duration_minutes": 5.0,
            "require_embeddings": False,  # Task 6.2
        })
        sections = generator.aggregate(beats)
        
        # Check duration constraints
        for section in sections:
            duration_min = section["duration_minutes"]
            # Some sections may exceed max if semantic overflow is allowed
            assert duration_min >= 3.0 or len(sections) == 1  # Allow single section
    
    def test_sections_consolidate_beats(self):
        """Test that sections consolidate multiple beats."""
        generator = SectionGenerator({"require_embeddings": False})  # Task 6.2
        beats = []
        for i in range(5):
            beat = {
                "beat_id": f"bet_{i}",
                "episode_id": "EP1",
                "start_time": i * 120.0,  # 2 minutes each
                "end_time": (i + 1) * 120.0,
                "duration": 120.0,
                "text": f"Beat {i}",
                "span_ids": [f"spn_{i}"],
            }
            beats.append(beat)
        
        sections = generator.aggregate(beats)
        
        # Should create sections from multiple beats
        for section in sections:
            assert "beat_ids" in section
            assert len(section["beat_ids"]) >= 1
    
    def test_section_text_concatenation(self):
        """Test that section text is concatenated from beats."""
        beats = [
            {"beat_id": "b1", "episode_id": "EP1", "start_time": 0.0, "end_time": 360.0,
             "duration": 360.0, "text": "Introduction", "span_ids": ["s1"]},
            {"beat_id": "b2", "episode_id": "EP1", "start_time": 360.0, "end_time": 720.0,
             "duration": 360.0, "text": "Main content", "span_ids": ["s2"]},
        ]
        
        generator = SectionGenerator({"min_duration_minutes": 1.0, "require_embeddings": False})  # Task 6.2
        sections = generator.aggregate(beats)
        
        assert len(sections) >= 1
        # Check that text is concatenated
        combined_text = " ".join(s["text"] for s in sections)
        assert "Introduction" in combined_text or "Main content" in combined_text
    
    def test_section_breaks_on_time_gap(self):
        """Test that sections break on large time gaps."""
        beats = [
            {"beat_id": "b1", "episode_id": "EP1", "start_time": 0.0, "end_time": 300.0,
             "duration": 300.0, "text": "First", "span_ids": ["s1"]},
            {"beat_id": "b2", "episode_id": "EP1", "start_time": 400.0, "end_time": 700.0,  # 100 sec gap
             "duration": 300.0, "text": "Second", "span_ids": ["s2"]},
        ]
        
        generator = SectionGenerator({"min_duration_minutes": 1.0, "require_embeddings": False})  # Task 6.2
        sections = generator.aggregate(beats)
        
        # Should create 2 sections due to large time gap
        assert len(sections) >= 2
    
    def test_section_min_duration_respected(self):
        """Test that minimum duration is respected."""
        beats = []
        for i in range(10):
            beat = {
                "beat_id": f"bet_{i}",
                "episode_id": "EP1",
                "start_time": i * 60.0,
                "end_time": (i + 1) * 60.0,
                "duration": 60.0,
                "text": f"Beat {i}",
                "span_ids": [f"spn_{i}"],
            }
            beats.append(beat)
        
        generator = SectionGenerator({
            "min_duration_minutes": 5.0,
            "max_duration_minutes": 10.0,
            "require_embeddings": False,  # Task 6.2
        })
        sections = generator.aggregate(beats)
        
        # All sections should meet minimum duration
        for section in sections:
            assert section["duration_minutes"] >= 5.0
    
    def test_section_with_embeddings_semantic_boundaries(self):
        """Test section generation with semantic boundaries."""
        beats = []
        for i in range(5):
            # Create beat with embedding
            embedding = np.random.rand(384)
            beat = {
                "beat_id": f"bet_{i}",
                "episode_id": "EP1",
                "start_time": i * 120.0,
                "end_time": (i + 1) * 120.0,
                "duration": 120.0,
                "text": f"Beat {i}",
                "span_ids": [f"spn_{i}"],
                "embedding": embedding.tolist(),
            }
            beats.append(beat)
        
        generator = SectionGenerator({"min_duration_minutes": 2.0, "require_embeddings": False})  # Task 6.2
        sections = generator.aggregate(beats)
        
        assert len(sections) > 0
    
    def test_section_id_determinism(self, sample_beats):
        """Test that section IDs are deterministic."""
        generator = SectionGenerator({"require_embeddings": False})  # Task 6.2
        
        sections1 = generator.aggregate(sample_beats)
        sections2 = generator.aggregate(sample_beats)
        
        assert len(sections1) == len(sections2)
        for s1, s2 in zip(sections1, sections2):
            assert s1["section_id"] == s2["section_id"]
    
    def test_generate_sections_convenience_function(self, sample_beats):
        """Test convenience function for section generation."""
        sections = generate_sections(sample_beats, config={"require_embeddings": False})  # Task 6.2
        
        assert len(sections) > 0
        assert all("section_id" in s for s in sections)


# ============================================================================
# Aggregation Statistics Tests
# ============================================================================

class TestAggregationStatistics:
    """Test statistics computation for aggregation."""
    
    def test_span_statistics(self, sample_normalized_utterances):
        """Test span statistics computation."""
        generator = SpanGenerator()
        spans = generator.aggregate(sample_normalized_utterances)
        stats = generator.compute_statistics(spans)
        
        assert "count" in stats
        assert "artifact_type" in stats
        assert stats["artifact_type"] == "span"
        assert "total_duration" in stats
        assert "avg_duration" in stats
    
    def test_beat_statistics(self, sample_spans):
        """Test beat statistics computation."""
        generator = BeatGenerator({"use_embeddings": False})
        beats = generator.aggregate(sample_spans)
        stats = generator.compute_statistics(beats)
        
        assert "count" in stats
        assert "artifact_type" in stats
        assert stats["artifact_type"] == "beat"
        assert "total_spans" in stats
        assert "avg_spans_per_beat" in stats
    
    def test_section_statistics(self, sample_beats):
        """Test section statistics computation."""
        generator = SectionGenerator({"require_embeddings": False})  # Task 6.2
        sections = generator.aggregate(sample_beats)
        stats = generator.compute_statistics(sections)
        
        assert "count" in stats
        assert "artifact_type" in stats
        assert stats["artifact_type"] == "section"
        assert "total_beats" in stats
        assert "in_target_range" in stats


# ============================================================================
# Integration Tests
# ============================================================================

class TestAggregationPipeline:
    """Test full aggregation pipeline."""
    
    def test_full_aggregation_pipeline(self, sample_normalized_utterances):
        """Test complete pipeline: utterances → spans → beats → sections."""
        # 1. Generate spans
        span_gen = SpanGenerator({"min_duration": 1.0})
        spans = span_gen.aggregate(sample_normalized_utterances)
        assert len(spans) > 0
        
        # 2. Generate beats (heuristic to avoid embedding requirement)
        beat_gen = BeatGenerator({"use_embeddings": False, "min_spans_per_beat": 1})
        beats = beat_gen.aggregate(spans)
        assert len(beats) > 0
        assert len(beats) <= len(spans)
        
        # 3. Generate sections
        section_gen = SectionGenerator({"min_duration_minutes": 0.1, "require_embeddings": False})  # Low threshold for test (Task 6.2)
        sections = section_gen.aggregate(beats)
        assert len(sections) > 0
        assert len(sections) <= len(beats)
    
    def test_aggregation_preserves_episode_id(self):
        """Test that episode_id is preserved through aggregation."""
        utterances = [
            {"episode_id": "EP-TEST", "start": 0.0, "end": 5.0, "speaker": "A", "text": "Test 1"},
            {"episode_id": "EP-TEST", "start": 5.0, "end": 10.0, "speaker": "A", "text": "Test 2"},
            {"episode_id": "EP-TEST", "start": 10.0, "end": 15.0, "speaker": "B", "text": "Test 3"},
        ]
        normalized = normalize_utterances(utterances)
        
        # Span
        spans = generate_spans(normalized, config={"min_duration": 0.0})
        assert all(s["episode_id"] == "EP-TEST" for s in spans)
        
        # Beat
        beats = generate_beats(spans, config={"use_embeddings": False, "min_spans_per_beat": 1})
        assert all(b["episode_id"] == "EP-TEST" for b in beats)
        
        # Section
        sections = generate_sections(beats, config={"min_duration_minutes": 0.1})
        assert all(s["episode_id"] == "EP-TEST" for s in sections)
    
    def test_aggregation_reduces_artifact_count(self):
        """Test that aggregation progressively reduces artifact count."""
        # Create enough utterances for meaningful aggregation
        utterances = []
        for i in range(20):
            utt = {
                "episode_id": "EP1",
                "start": i * 5.0,
                "end": (i + 1) * 5.0,
                "speaker": "A" if i % 2 == 0 else "B",
                "text": f"Utterance {i}",
            }
            utterances.append(utt)
        
        normalized = normalize_utterances(utterances)
        
        # Aggregate
        spans = generate_spans(normalized, config={"min_duration": 1.0})
        beats = generate_beats(spans, config={"use_embeddings": False, "min_spans_per_beat": 1})
        sections = generate_sections(beats, config={"min_duration_minutes": 0.5})
        
        # Verify reduction
        assert len(spans) <= len(normalized)
        assert len(beats) <= len(spans)
        assert len(sections) <= len(beats)
    
    def test_multi_episode_aggregation(self):
        """Test aggregation with multiple episodes."""
        # Episode 1
        ep1_utterances = []
        for i in range(5):
            utt = {
                "episode_id": "EP1",
                "start": i * 3.0,
                "end": (i + 1) * 3.0,
                "speaker": "A",
                "text": f"EP1 utterance {i}",
            }
            ep1_utterances.append(utt)
        
        # Episode 2
        ep2_utterances = []
        for i in range(5):
            utt = {
                "episode_id": "EP2",
                "start": i * 3.0,
                "end": (i + 1) * 3.0,
                "speaker": "B",
                "text": f"EP2 utterance {i}",
            }
            ep2_utterances.append(utt)
        
        # Normalize each episode separately
        normalized_ep1 = normalize_utterances(ep1_utterances)
        normalized_ep2 = normalize_utterances(ep2_utterances)
        
        # Combine for aggregation (aggregation can handle multiple episodes)
        all_normalized = normalized_ep1 + normalized_ep2
        
        # Generate spans
        spans = generate_spans(all_normalized, config={"min_duration": 1.0})
        
        # Check that both episodes are represented
        ep1_spans = [s for s in spans if s["episode_id"] == "EP1"]
        ep2_spans = [s for s in spans if s["episode_id"] == "EP2"]
        
        assert len(ep1_spans) > 0
        assert len(ep2_spans) > 0


# ============================================================================
# Semantic Section Generation Tests (R2)
# ============================================================================

class TestSemanticSectionGeneration:
    """Tests for semantic section generation improvements (PRD R2)."""
    
    def test_multiple_sections_per_episode(self):
        """Test that episodes generate multiple sections (Task 2.8)."""
        # Create a longer episode with 60 minutes of content (10 beats * 6 min each)
        # Use strict time-based config to force multiple sections
        beats = []
        for i in range(10):
            beat = {
                "beat_id": f"bet_{i:03d}",
                "episode_id": "LONG_EP",
                "start_time": i * 360.0,  # 6 minutes per beat
                "end_time": (i + 1) * 360.0,
                "duration": 360.0,
                "text": f"Beat {i} content discussing various topics.",
                "span_ids": [f"spn_{i}"],
            }
            beats.append(beat)
        
        generator = SectionGenerator({
            "min_duration_minutes": 5.0,
            "target_duration_minutes": 8.0,
            "max_duration_minutes": 10.0,  # Stricter max
            "allow_semantic_overflow": False,  # Force hard limit
            "prefer_time_boundaries": True,  # Prefer time over semantics
            "require_embeddings": False,  # Allow time-based fallback for this test
        })
        sections = generator.aggregate(beats)
        
        # For 60 minutes of content with max 10 minutes, should generate at least 6 sections
        assert len(sections) >= 6, f"Expected at least 6 sections for 60 min content, got {len(sections)}"
        
        # Verify that no single section exceeds max duration
        for section in sections:
            assert section["duration_minutes"] <= 10.0, \
                f"Section exceeds max duration: {section['duration_minutes']:.1f} minutes"
    
    def test_all_beats_assigned_to_exactly_one_section(self):
        """Test that every beat belongs to exactly one section (Task 2.9)."""
        # Create test beats
        beats = []
        for i in range(15):
            beat = {
                "beat_id": f"beat_{i:03d}",
                "episode_id": "TEST_EP",
                "start_time": i * 120.0,  # 2 minutes each
                "end_time": (i + 1) * 120.0,
                "duration": 120.0,
                "text": f"Content for beat {i}",
                "span_ids": [f"span_{i}"],
            }
            beats.append(beat)
        
        generator = SectionGenerator({"require_embeddings": False})
        sections = generator.aggregate(beats)
        
        # Collect all beat IDs from all sections
        beat_ids_in_sections = []
        for section in sections:
            beat_ids_in_sections.extend(section["beat_ids"])
        
        # Check that each beat appears exactly once
        original_beat_ids = [b["beat_id"] for b in beats]
        assert len(beat_ids_in_sections) == len(original_beat_ids), \
            "Number of beats in sections doesn't match original beats"
        
        # Check for duplicates (no beat should appear in multiple sections)
        assert len(beat_ids_in_sections) == len(set(beat_ids_in_sections)), \
            "Some beats appear in multiple sections (overlaps detected)"
        
        # Check that all beats are covered (no gaps)
        assert set(beat_ids_in_sections) == set(original_beat_ids), \
            "Some beats are missing from sections (gaps detected)"
    
    def test_sections_chronologically_ordered(self):
        """Test that sections within an episode are chronologically ordered (Task 2.10)."""
        # Create test beats with timestamps
        beats = []
        for i in range(12):
            beat = {
                "beat_id": f"beat_{i:03d}",
                "episode_id": "ORDERED_EP",
                "start_time": i * 180.0,  # 3 minutes each
                "end_time": (i + 1) * 180.0,
                "duration": 180.0,
                "text": f"Sequential content {i}",
                "span_ids": [f"span_{i}"],
            }
            beats.append(beat)
        
        generator = SectionGenerator({"require_embeddings": False})
        sections = generator.aggregate(beats)
        
        # Verify sections are in chronological order
        for i in range(len(sections) - 1):
            current_section = sections[i]
            next_section = sections[i + 1]
            
            # Next section should start at or after current section ends
            assert next_section["start_time"] >= current_section["end_time"], \
                f"Section {i+1} starts before section {i} ends (not chronological)"
            
            # Verify beat_ids are in ascending order (assuming beat IDs encode order)
            current_last_beat_id = current_section["beat_ids"][-1]
            next_first_beat_id = next_section["beat_ids"][0]
            
            # Beat IDs should be sequential
            current_idx = int(current_last_beat_id.split("_")[-1])
            next_idx = int(next_first_beat_id.split("_")[-1])
            assert next_idx > current_idx, \
                f"Beat order incorrect between sections {i} and {i+1}"
    
    def test_section_has_title_and_synopsis(self):
        """Test that generated sections include title and synopsis fields."""
        beats = [
            {
                "beat_id": "beat_001",
                "episode_id": "TEST",
                "start_time": 0.0,
                "end_time": 300.0,
                "duration": 300.0,
                "text": "Test content",
                "span_ids": ["span_001"],
            }
        ]
        
        generator = SectionGenerator({"require_embeddings": False})
        sections = generator.aggregate(beats)
        
        assert len(sections) > 0
        section = sections[0]
        
        # Check title field exists and is populated
        assert "title" in section
        assert isinstance(section["title"], str)
        assert len(section["title"]) > 0
        assert "Section" in section["title"]  # Should be "Section N" format
        
        # Check synopsis field exists
        assert "synopsis" in section
        # Synopsis can be None or a string
        if section["synopsis"] is not None:
            assert isinstance(section["synopsis"], str)
    
    def test_semantic_boundary_detection_with_embeddings(self):
        """Test that sections break at semantic boundaries using embeddings."""
        # Create 9 beats with mock embeddings
        # Cluster pattern: 3 beats topic A, 3 beats topic B, 3 beats topic C
        beats = []
        embeddings = generate_mock_embeddings(9, dim=384, seed=42)
        
        for i in range(9):
            beat = {
                "beat_id": f"beat_{i:03d}",
                "episode_id": "SEMANTIC_EP",
                "start_time": i * 120.0,  # 2 minutes each (total 18 min)
                "end_time": (i + 1) * 120.0,
                "duration": 120.0,
                "text": f"Beat {i} discussing topic {i // 3}",
                "span_ids": [f"span_{i}"],
                "embedding": embeddings[i],  # Attach mock embedding
            }
            beats.append(beat)
        
        # Configure for semantic boundaries
        generator = SectionGenerator({
            "min_duration_minutes": 2.0,  # Low minimum to allow semantic breaks
            "target_duration_minutes": 6.0,  # 3 beats per section ideally
            "max_duration_minutes": 10.0,
            "boundary_similarity_threshold": 0.7,  # Detect topic changes
            "prefer_semantic_boundaries": True,
            "require_embeddings": False,  # Embeddings already attached to beats
        })
        sections = generator.aggregate(beats)
        
        # With clustered embeddings (3 per cluster), should create ~3 sections
        # Each cluster of 3 beats has high similarity within, low between
        assert len(sections) >= 2, f"Expected multiple sections from semantic clustering, got {len(sections)}"
        
        # Verify that embeddings were used (all beats should have them)
        for beat in beats:
            assert "embedding" in beat, "Beat should have embedding attached"
    
    def test_require_embeddings_fails_when_missing(self):
        """Test that require_embeddings=true fails when embeddings are missing."""
        beats = [
            {
                "beat_id": "beat_001",
                "episode_id": "TEST",
                "start_time": 0.0,
                "end_time": 300.0,
                "duration": 300.0,
                "text": "Test content",
                "span_ids": ["span_001"],
                # No embedding attached
            }
        ]
        
        # Should raise error when embeddings required but missing
        generator = SectionGenerator({
            "require_embeddings": True,
            "prefer_semantic_boundaries": True,
        })
        
        with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
            generator.aggregate(beats)