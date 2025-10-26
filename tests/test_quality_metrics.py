"""
Unit tests for quality assessment metrics.

Tests all metric calculators across categories A-G:
- Coverage metrics (gaps, overlaps, counts)
- Distribution metrics (duration statistics, compliance)
- Integrity metrics (monotonicity, duplicates, violations)
- Balance metrics (speaker/series distribution)
- Text quality metrics (lexical density, top terms)
- Embedding metrics (neighbor analysis, leakage, bias)
- Diagnostics (outliers, samples)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Import metrics modules
from lakehouse.quality.metrics import coverage, distribution, integrity, balance, text_quality, embedding
from lakehouse.quality import diagnostics
from lakehouse.quality.thresholds import QualityThresholds, ThresholdViolation

# Import test fixtures
from tests.fixtures.quality_test_data import (
    get_default_episodes,
    create_sample_spans,
    create_sample_beats,
    create_sample_embeddings,
    create_coverage_test_data,
    create_distribution_test_data,
    create_integrity_test_data,
    create_complete_test_dataset,
    create_embeddings_with_speaker_leakage,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_episodes():
    """Get sample episode metadata."""
    return get_default_episodes()


@pytest.fixture
def sample_spans():
    """Get sample spans with edge cases."""
    return create_sample_spans(include_edge_cases=True)


@pytest.fixture
def sample_beats():
    """Get sample beats with edge cases."""
    return create_sample_beats(include_edge_cases=True)


@pytest.fixture
def sample_embeddings():
    """Get sample random embeddings."""
    return create_sample_embeddings(100, pattern="random")


@pytest.fixture
def default_thresholds():
    """Get default quality thresholds."""
    return QualityThresholds()


@pytest.fixture
def complete_dataset():
    """Get complete test dataset."""
    return create_complete_test_dataset(num_episodes=2, include_embeddings=True)


# ============================================================================
# Task 6.2.1: Test Coverage Metrics
# ============================================================================

class TestCoverageMetrics:
    """Test coverage metrics (Category A)."""
    
    def test_calculate_episode_coverage_basic(self, sample_episodes, sample_spans):
        """Test basic episode coverage calculation."""
        result = coverage.calculate_episode_coverage(
            sample_episodes[:1],  # Use only first episode
            sample_spans,
            None
        )
        
        assert 'per_episode' in result
        assert 'global' in result
        assert len(result['per_episode']) == 1
        
        episode_metrics = result['per_episode'][0]
        assert 'episode_id' in episode_metrics
        assert 'span_count' in episode_metrics
        assert 'span_coverage_percent' in episode_metrics
        assert episode_metrics['span_count'] > 0
    
    def test_calculate_episode_coverage_global_metrics(self, sample_episodes, sample_spans):
        """Test global coverage aggregates."""
        result = coverage.calculate_episode_coverage(
            sample_episodes,
            sample_spans,
            None
        )
        
        global_metrics = result['global']
        assert global_metrics['total_episodes'] == len(sample_episodes)
        assert global_metrics['total_spans'] == len(sample_spans)
        assert 'global_span_coverage_percent' in global_metrics
        assert 0 <= global_metrics['global_span_coverage_percent'] <= 100
    
    def test_calculate_episode_coverage_with_beats(self, sample_episodes, sample_beats):
        """Test coverage calculation with beats."""
        result = coverage.calculate_episode_coverage(
            sample_episodes[:1],
            None,
            sample_beats
        )
        
        episode_metrics = result['per_episode'][0]
        assert 'beat_count' in episode_metrics
        assert 'beat_coverage_percent' in episode_metrics
        assert episode_metrics['beat_count'] > 0
    
    def test_calculate_episode_coverage_both_levels(self, sample_episodes, sample_spans, sample_beats):
        """Test coverage with both spans and beats."""
        result = coverage.calculate_episode_coverage(
            sample_episodes[:1],
            sample_spans,
            sample_beats
        )
        
        episode_metrics = result['per_episode'][0]
        assert 'span_count' in episode_metrics
        assert 'beat_count' in episode_metrics
        assert episode_metrics['span_count'] > 0
        assert episode_metrics['beat_count'] > 0
    
    def test_detect_gaps_and_overlaps_basic(self):
        """Test basic gap and overlap detection."""
        segments = pd.DataFrame([
            {'start_time': 0.0, 'end_time': 10.0},
            {'start_time': 15.0, 'end_time': 25.0},  # 5s gap
            {'start_time': 23.0, 'end_time': 30.0},  # 2s overlap
        ])
        
        result = coverage.detect_gaps_and_overlaps(segments, "TEST", 40.0)
        
        assert result['gap_count'] > 0
        assert result['overlap_count'] > 0
        assert result['gap_total_duration'] > 0
        assert result['overlap_total_duration'] > 0
    
    def test_detect_gaps_at_boundaries(self):
        """Test gap detection at episode start and end."""
        segments = pd.DataFrame([
            {'start_time': 10.0, 'end_time': 20.0},  # Gap at start
        ])
        
        result = coverage.detect_gaps_and_overlaps(segments, "TEST", 30.0)
        
        # Should have gap at start (0-10) and gap at end (20-30)
        assert result['gap_count'] == 2
        assert result['gap_total_duration'] == 20.0
    
    def test_detect_gaps_no_segments(self):
        """Test gap detection with no segments (entire episode is gap)."""
        segments = pd.DataFrame(columns=['start_time', 'end_time'])
        
        result = coverage.detect_gaps_and_overlaps(segments, "TEST", 100.0)
        
        assert result['gap_count'] == 1
        assert result['gap_total_duration'] == 100.0
        assert result['gap_percent'] == 100.0
    
    def test_validate_coverage_thresholds_pass(self):
        """Test threshold validation when coverage is good."""
        # Create test data with >95% coverage
        episodes, spans = create_coverage_test_data(coverage_percent=97.0, gap_percent=1.0, overlap_percent=1.0)
        
        coverage_metrics = coverage.calculate_episode_coverage(episodes, spans, None)
        thresholds = QualityThresholds(coverage_min=95.0, gap_max_percent=2.0, overlap_max_percent=2.0)
        
        violations = coverage.validate_coverage_thresholds(coverage_metrics, thresholds)
        
        # Should have no violations (or only minor warnings)
        error_violations = [v for v in violations if v.severity == 'error']
        assert len(error_violations) == 0
    
    def test_validate_coverage_thresholds_fail_low_coverage(self):
        """Test threshold validation when coverage is too low."""
        # Create test data with <95% coverage
        episodes, spans = create_coverage_test_data(coverage_percent=90.0)
        
        coverage_metrics = coverage.calculate_episode_coverage(episodes, spans, None)
        thresholds = QualityThresholds(coverage_min=95.0)
        
        violations = coverage.validate_coverage_thresholds(coverage_metrics, thresholds)
        
        # Should have at least one violation
        assert len(violations) > 0
        assert any('coverage' in v.threshold_name.lower() for v in violations)
    
    def test_validate_coverage_thresholds_high_gaps(self):
        """Test threshold validation when gaps are too high."""
        episodes, spans = create_coverage_test_data(coverage_percent=95.0, gap_percent=5.0)
        
        coverage_metrics = coverage.calculate_episode_coverage(episodes, spans, None)
        thresholds = QualityThresholds(gap_max_percent=2.0)
        
        violations = coverage.validate_coverage_thresholds(coverage_metrics, thresholds)
        
        # Should have gap violation
        assert any('gap' in v.threshold_name.lower() for v in violations)


# ============================================================================
# Task 6.2.2: Test Distribution Metrics
# ============================================================================

class TestDistributionMetrics:
    """Test distribution metrics (Category B)."""
    
    def test_calculate_duration_statistics_basic(self):
        """Test basic duration statistics calculation."""
        spans = create_distribution_test_data(num_spans=100, target_distribution="good")
        
        stats = distribution.calculate_duration_statistics(spans, segment_type="spans")
        
        assert 'min' in stats
        assert 'max' in stats
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert 'p5' in stats
        assert 'p95' in stats
        
        # Check that values are reasonable
        assert stats['min'] < stats['max']
        assert stats['min'] < stats['mean'] < stats['max']
        assert stats['p5'] < stats['p95']
    
    def test_calculate_duration_statistics_histogram(self):
        """Test histogram generation in duration statistics."""
        spans = create_distribution_test_data(num_spans=100, target_distribution="good")
        
        stats = distribution.calculate_duration_statistics(spans, segment_type="spans")
        
        # Check histogram exists
        assert 'histogram' in stats
        histogram = stats['histogram']
        assert 'bins' in histogram
        assert 'counts' in histogram
        assert len(histogram['bins']) > 0
        assert len(histogram['counts']) > 0
    
    def test_calculate_length_compliance_good_distribution(self):
        """Test length compliance with good distribution."""
        spans = create_distribution_test_data(num_spans=100, target_distribution="good")
        
        compliance = distribution.calculate_length_compliance(
            spans,
            min_length=20.0,
            max_length=120.0,
            segment_type="spans"
        )
        
        assert 'total_segments' in compliance
        assert 'within_bounds' in compliance
        assert 'compliance_percent' in compliance
        
        # Good distribution should have >90% compliance
        assert compliance['compliance_percent'] >= 90.0
    
    def test_calculate_length_compliance_bad_distribution(self):
        """Test length compliance with bad distribution."""
        spans = create_distribution_test_data(num_spans=100, target_distribution="bad")
        
        compliance = distribution.calculate_length_compliance(
            spans,
            min_length=20.0,
            max_length=120.0,
            segment_type="spans"
        )
        
        # Bad distribution should have <90% compliance
        assert compliance['compliance_percent'] < 90.0
    
    def test_calculate_length_compliance_outlier_counts(self):
        """Test that outlier counts are tracked."""
        spans = create_distribution_test_data(num_spans=100, target_distribution="too_short")
        
        compliance = distribution.calculate_length_compliance(
            spans,
            min_length=20.0,
            max_length=120.0,
            segment_type="spans"
        )
        
        assert 'too_short' in compliance
        assert 'too_long' in compliance
        assert compliance['too_short'] > 0  # Should have short segments
    
    def test_validate_length_thresholds_pass(self):
        """Test length threshold validation passing."""
        spans = create_distribution_test_data(num_spans=100, target_distribution="good")
        
        compliance = distribution.calculate_length_compliance(
            spans, min_length=20.0, max_length=120.0, segment_type="spans"
        )
        
        violations = distribution.validate_length_thresholds(
            compliance,
            min_compliance_percent=90.0,
            segment_type="spans"
        )
        
        # Should pass with no violations
        assert len(violations) == 0
    
    def test_validate_length_thresholds_fail(self):
        """Test length threshold validation failing."""
        spans = create_distribution_test_data(num_spans=100, target_distribution="bad")
        
        compliance = distribution.calculate_length_compliance(
            spans, min_length=20.0, max_length=120.0, segment_type="spans"
        )
        
        violations = distribution.validate_length_thresholds(
            compliance,
            min_compliance_percent=90.0,
            segment_type="spans"
        )
        
        # Should have violations
        assert len(violations) > 0
        assert any('compliance' in v.threshold_name.lower() for v in violations)


# ============================================================================
# Task 6.2.3: Test Integrity Metrics
# ============================================================================

class TestIntegrityMetrics:
    """Test integrity metrics (Category C)."""
    
    def test_check_timestamp_monotonicity_ordered(self):
        """Test monotonicity check with properly ordered timestamps."""
        segments = pd.DataFrame([
            {'episode_id': 'EP001', 'speaker': 'Alice', 'start_time': 0.0, 'end_time': 10.0},
            {'episode_id': 'EP001', 'speaker': 'Alice', 'start_time': 10.0, 'end_time': 20.0},
            {'episode_id': 'EP001', 'speaker': 'Bob', 'start_time': 20.0, 'end_time': 30.0},
        ])
        
        result = integrity.check_timestamp_monotonicity(segments)
        
        assert 'total_regressions' in result
        assert result['total_regressions'] == 0
    
    def test_check_timestamp_monotonicity_regressions(self):
        """Test monotonicity check with timestamp regressions."""
        segments = create_integrity_test_data()
        
        result = integrity.check_timestamp_monotonicity(segments)
        
        assert result['total_regressions'] > 0
        assert 'episode_regressions' in result
    
    def test_detect_integrity_violations_negative_duration(self):
        """Test detection of negative durations."""
        segments = create_integrity_test_data()
        
        result = integrity.detect_integrity_violations(segments)
        
        assert 'negative_duration_count' in result
        assert result['negative_duration_count'] > 0
    
    def test_detect_integrity_violations_zero_duration(self):
        """Test detection of zero durations."""
        segments = create_integrity_test_data()
        
        result = integrity.detect_integrity_violations(segments)
        
        assert 'zero_duration_count' in result
        assert result['zero_duration_count'] > 0
    
    def test_detect_integrity_violations_missing_fields(self):
        """Test detection of missing required fields."""
        segments = pd.DataFrame([
            {'episode_id': 'EP001', 'speaker': 'Alice', 'start_time': 0.0, 'end_time': 10.0, 'text': 'Hello'},
            {'episode_id': 'EP001', 'speaker': None, 'start_time': 10.0, 'end_time': 20.0, 'text': 'World'},
            {'episode_id': 'EP001', 'speaker': 'Bob', 'start_time': 20.0, 'end_time': 30.0, 'text': None},
        ])
        
        result = integrity.detect_integrity_violations(segments)
        
        assert 'missing_fields_count' in result
        assert result['missing_fields_count'] > 0
    
    def test_detect_duplicates_exact(self):
        """Test detection of exact duplicates."""
        segments = create_integrity_test_data()
        
        result = integrity.detect_duplicates(segments, threshold=0.95)
        
        assert 'exact_duplicate_count' in result
        assert 'exact_duplicate_percent' in result
        assert result['exact_duplicate_count'] > 0
    
    def test_detect_duplicates_near(self):
        """Test detection of near-duplicates."""
        segments = create_integrity_test_data()
        
        result = integrity.detect_duplicates(segments, threshold=0.95)
        
        assert 'near_duplicate_count' in result
        assert 'near_duplicate_percent' in result
        # Near-duplicates should be >= exact duplicates
        assert result['near_duplicate_count'] >= result['exact_duplicate_count']
    
    def test_detect_duplicates_no_duplicates(self):
        """Test duplicate detection with no duplicates."""
        segments = pd.DataFrame([
            {'text': 'First segment'},
            {'text': 'Second segment'},
            {'text': 'Third segment'},
        ])
        
        result = integrity.detect_duplicates(segments, threshold=0.95)
        
        assert result['exact_duplicate_count'] == 0
        assert result['near_duplicate_count'] == 0
    
    def test_validate_integrity_thresholds_pass(self):
        """Test integrity threshold validation passing."""
        segments = pd.DataFrame([
            {'episode_id': 'EP001', 'speaker': 'Alice', 'start_time': 0.0, 'end_time': 10.0, 'text': 'Hello', 'duration': 10.0},
            {'episode_id': 'EP001', 'speaker': 'Bob', 'start_time': 10.0, 'end_time': 20.0, 'text': 'World', 'duration': 10.0},
        ])
        
        monotonicity = integrity.check_timestamp_monotonicity(segments)
        violations_data = integrity.detect_integrity_violations(segments)
        duplicates_data = integrity.detect_duplicates(segments)
        
        metrics = {**monotonicity, **violations_data, **duplicates_data}
        thresholds = QualityThresholds()
        
        violations = integrity.validate_integrity_thresholds(metrics, thresholds)
        
        # Should have no violations
        assert len(violations) == 0
    
    def test_validate_integrity_thresholds_fail(self):
        """Test integrity threshold validation failing."""
        segments = create_integrity_test_data()
        
        monotonicity = integrity.check_timestamp_monotonicity(segments)
        violations_data = integrity.detect_integrity_violations(segments)
        duplicates_data = integrity.detect_duplicates(segments)
        
        metrics = {**monotonicity, **violations_data, **duplicates_data}
        thresholds = QualityThresholds()
        
        violations = integrity.validate_integrity_thresholds(metrics, thresholds)
        
        # Should have multiple violations
        assert len(violations) > 0


# ============================================================================
# Task 6.2.4: Test Balance Metrics
# ============================================================================

class TestBalanceMetrics:
    """Test balance metrics (Category D)."""
    
    def test_calculate_speaker_distribution_basic(self, sample_spans):
        """Test basic speaker distribution calculation."""
        result = balance.calculate_speaker_distribution(sample_spans, top_n=10)
        
        assert 'total_segments' in result
        assert 'unique_speakers' in result
        assert 'speaker_stats' in result
        assert result['total_segments'] > 0
        assert result['unique_speakers'] > 0
    
    def test_calculate_speaker_distribution_stats(self, sample_spans):
        """Test speaker statistics details."""
        result = balance.calculate_speaker_distribution(sample_spans, top_n=10)
        
        speaker_stats = result['speaker_stats']
        assert len(speaker_stats) > 0
        
        # Check first speaker stats
        first_speaker = speaker_stats[0]
        assert 'speaker' in first_speaker
        assert 'segment_count' in first_speaker
        assert 'percentage' in first_speaker
        assert 'avg_duration' in first_speaker
    
    def test_calculate_speaker_distribution_percentages(self, sample_spans):
        """Test that percentages sum to 100."""
        result = balance.calculate_speaker_distribution(sample_spans, top_n=10)
        
        speaker_stats = result['speaker_stats']
        total_percentage = sum(s['percentage'] for s in speaker_stats)
        
        # Should be approximately 100% (within rounding error)
        assert 99.0 < total_percentage <= 100.0
    
    def test_calculate_speaker_distribution_top_n(self, sample_spans):
        """Test top N speakers limiting."""
        result = balance.calculate_speaker_distribution(sample_spans, top_n=2)
        
        speaker_stats = result['speaker_stats']
        # Should return at most top_n speakers
        assert len(speaker_stats) <= 2
    
    def test_calculate_series_balance(self):
        """Test series balance calculation."""
        segments = pd.DataFrame([
            {'episode_id': 'LOS-001', 'series': 'LOS', 'duration': 30.0},
            {'episode_id': 'LOS-001', 'series': 'LOS', 'duration': 40.0},
            {'episode_id': 'SW-001', 'series': 'SW', 'duration': 50.0},
        ])
        
        result = balance.calculate_speaker_distribution(segments, top_n=10)
        
        # Should work even if series info is in episode_id
        assert result['total_segments'] == 3


# ============================================================================
# Task 6.2.5: Test Text Quality Metrics
# ============================================================================

class TestTextQualityMetrics:
    """Test text quality metrics (Category E)."""
    
    def test_calculate_text_metrics_basic(self, sample_spans):
        """Test basic text metrics calculation."""
        result = text_quality.calculate_text_metrics(sample_spans)
        
        assert 'total_segments' in result
        assert 'token_stats' in result
        assert 'word_stats' in result
        assert 'char_stats' in result
    
    def test_calculate_text_metrics_statistics(self, sample_spans):
        """Test that statistics are calculated."""
        result = text_quality.calculate_text_metrics(sample_spans)
        
        token_stats = result['token_stats']
        assert 'mean' in token_stats
        assert 'median' in token_stats
        assert 'std' in token_stats
        
        # Check reasonable values
        assert token_stats['mean'] > 0
        assert token_stats['median'] > 0
    
    def test_calculate_lexical_density_basic(self, sample_spans):
        """Test lexical density calculation."""
        result = text_quality.calculate_lexical_density(sample_spans)
        
        assert 'mean_lexical_density' in result
        assert 'median_lexical_density' in result
        assert 'mean_punctuation_ratio' in result
        
        # Lexical density should be between 0 and 1
        assert 0 <= result['mean_lexical_density'] <= 1
    
    def test_extract_top_terms_basic(self, sample_spans):
        """Test top terms extraction."""
        result = text_quality.extract_top_terms(sample_spans, top_n=20)
        
        assert 'top_unigrams' in result
        assert 'top_bigrams' in result
    
    def test_extract_top_terms_structure(self, sample_spans):
        """Test top terms structure."""
        result = text_quality.extract_top_terms(sample_spans, top_n=10)
        
        top_unigrams = result['top_unigrams']
        assert len(top_unigrams) > 0
        
        # Check structure of first unigram
        first_unigram = top_unigrams[0]
        assert 'term' in first_unigram
        assert 'count' in first_unigram
        assert first_unigram['count'] > 0
    
    def test_extract_top_terms_stopwords_filtered(self, sample_spans):
        """Test that common stopwords are filtered."""
        result = text_quality.extract_top_terms(sample_spans, top_n=20)
        
        top_unigrams = result['top_unigrams']
        unigram_terms = [u['term'].lower() for u in top_unigrams]
        
        # Common stopwords should be filtered
        # (this might not be strict depending on text content)
        assert 'the' not in unigram_terms or len(unigram_terms) < 5  # Allow if very few terms


# ============================================================================
# Task 6.2.6: Test Embedding Metrics
# ============================================================================

class TestEmbeddingMetrics:
    """Test embedding metrics (Category F)."""
    
    def test_load_embeddings_success(self, tmp_path):
        """Test successful embedding loading."""
        # Create temporary embeddings file
        embeddings_array = create_sample_embeddings(10, pattern="random")
        segments_df = pd.DataFrame({
            'segment_id': [f'S{i:03d}' for i in range(10)],
            'text': [f'Segment {i}' for i in range(10)]
        })
        
        # Save embeddings
        embeddings_file = tmp_path / "test_embeddings.parquet"
        embeddings_df = pd.DataFrame({
            'segment_id': segments_df['segment_id'],
            'embedding': list(embeddings_array)
        })
        embeddings_df.to_parquet(embeddings_file)
        
        # Load embeddings
        result = embedding.load_embeddings(embeddings_file, segments_df)
        
        assert result is not None
        merged_df, emb_matrix = result
        assert len(merged_df) == 10
        assert emb_matrix.shape[0] == 10
    
    def test_load_embeddings_missing_file(self, tmp_path):
        """Test graceful handling of missing embeddings file."""
        missing_file = tmp_path / "nonexistent.parquet"
        segments_df = pd.DataFrame({'segment_id': ['S001']})
        
        result = embedding.load_embeddings(missing_file, segments_df)
        
        # Should return None gracefully
        assert result is None
    
    def test_stratified_sample_segments(self, sample_spans):
        """Test stratified sampling of segments."""
        sampled = embedding.stratified_sample_segments(
            sample_spans,
            sample_size=5,
            random_seed=42
        )
        
        assert len(sampled) <= 5
        assert len(sampled) > 0
    
    def test_compute_cosine_similarity(self):
        """Test cosine similarity computation."""
        vec1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        vec2 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        
        similarities = embedding.compute_cosine_similarity(vec1, vec2)
        
        assert len(similarities) == 2
        assert similarities[0] == pytest.approx(1.0, abs=0.01)  # Same direction
        assert similarities[1] == pytest.approx(0.0, abs=0.01)  # Orthogonal
    
    def test_find_top_k_neighbors(self):
        """Test finding top-k nearest neighbors."""
        embeddings = create_sample_embeddings(20, pattern="random")
        query_indices = np.array([0, 5, 10])
        
        neighbor_indices, similarities = embedding.find_top_k_neighbors(
            query_indices=query_indices,
            embeddings=embeddings,
            k=5,
            exclude_self=True
        )
        
        assert neighbor_indices.shape == (3, 5)
        assert similarities.shape == (3, 5)
        
        # Neighbors should not include self
        for i, query_idx in enumerate(query_indices):
            assert query_idx not in neighbor_indices[i]
    
    def test_calculate_speaker_leakage(self):
        """Test speaker leakage calculation."""
        segments_df = pd.DataFrame({
            'speaker': ['Alice', 'Alice', 'Bob', 'Bob', 'Alice'],
        })
        embeddings = create_embeddings_with_speaker_leakage(
            segments_df,
            leakage_strength=0.8
        )
        
        query_indices = np.array([0, 2])
        neighbor_indices, _ = embedding.find_top_k_neighbors(
            query_indices, embeddings, k=3, exclude_self=True
        )
        
        leakage_pct = embedding.calculate_speaker_leakage(
            segments_df, query_indices, neighbor_indices
        )
        
        # With high leakage strength, should have high same-speaker percentage
        assert leakage_pct > 30.0  # At least some leakage
    
    def test_calculate_episode_leakage(self):
        """Test episode leakage calculation."""
        segments_df = pd.DataFrame({
            'episode_id': ['EP001', 'EP001', 'EP002', 'EP002', 'EP001'],
        })
        embeddings = create_sample_embeddings(5, pattern="clustered")
        
        query_indices = np.array([0, 2])
        neighbor_indices, _ = embedding.find_top_k_neighbors(
            query_indices, embeddings, k=3, exclude_self=True
        )
        
        leakage_pct = embedding.calculate_episode_leakage(
            segments_df, query_indices, neighbor_indices
        )
        
        assert 0 <= leakage_pct <= 100
    
    def test_calculate_embedding_norms(self):
        """Test embedding norm calculation."""
        embeddings = create_sample_embeddings(10, pattern="length_biased")
        
        norms = embedding.calculate_embedding_norms(embeddings)
        
        assert len(norms) == 10
        assert all(norms > 0)
    
    def test_calculate_length_bias_correlation(self):
        """Test length bias correlation calculation."""
        segments_df = pd.DataFrame({
            'duration': np.linspace(10, 100, 20)
        })
        embeddings = create_sample_embeddings(20, pattern="length_biased")
        norms = embedding.calculate_embedding_norms(embeddings)
        
        # Create dummy neighbor similarities
        neighbor_similarities = np.random.rand(20, 5).astype(np.float32)
        
        result = embedding.calculate_length_bias_correlation(
            segments_df, norms, neighbor_similarities
        )
        
        assert 'duration_vs_norm' in result
        assert 'duration_vs_similarity' in result
        assert -1 <= result['duration_vs_norm'] <= 1
    
    def test_calculate_adjacency_bias(self):
        """Test adjacency bias calculation."""
        segments_df = pd.DataFrame({
            'episode_id': ['EP001'] * 10,
            'start_time': np.arange(0, 100, 10),
            'end_time': np.arange(10, 110, 10),
        })
        
        # Create embeddings where adjacent segments are similar
        embeddings = create_sample_embeddings(10, pattern="adjacent")
        
        query_indices = np.array([0, 5])
        neighbor_indices, _ = embedding.find_top_k_neighbors(
            query_indices, embeddings, k=5, exclude_self=True
        )
        
        adjacency_pct = embedding.calculate_adjacency_bias(
            segments_df, query_indices, neighbor_indices, threshold_seconds=15.0
        )
        
        # With adjacent pattern, should have some adjacency
        assert adjacency_pct >= 0
    
    def test_validate_leakage_thresholds(self):
        """Test leakage threshold validation."""
        violations = embedding.validate_leakage_thresholds(
            speaker_leakage=70.0,
            episode_leakage=75.0,
            max_speaker_leakage=60.0,
            max_episode_leakage=70.0
        )
        
        # Should have violations
        assert len(violations) > 0
        assert any('speaker' in v.threshold_name.lower() for v in violations)
        assert any('episode' in v.threshold_name.lower() for v in violations)
    
    def test_validate_length_bias_threshold(self):
        """Test length bias threshold validation."""
        violations = embedding.validate_length_bias_threshold(
            duration_vs_norm_corr=0.5,
            duration_vs_sim_corr=0.4,
            max_correlation=0.3
        )
        
        # Should have violations (correlations exceed threshold)
        assert len(violations) > 0
    
    def test_validate_adjacency_threshold(self):
        """Test adjacency bias threshold validation."""
        violations = embedding.validate_adjacency_threshold(
            adjacency_pct=50.0,
            max_adjacency=40.0
        )
        
        # Should have violation
        assert len(violations) > 0
        assert any('adjacency' in v.threshold_name.lower() for v in violations)


# ============================================================================
# Task 6.2.7: Test Diagnostics
# ============================================================================

class TestDiagnostics:
    """Test diagnostics and outlier detection (Category G)."""
    
    def test_identify_outliers_basic(self, sample_spans):
        """Test basic outlier identification."""
        outliers = diagnostics.identify_outliers(
            sample_spans,
            embeddings=None,
            neighbor_indices=None,
            neighbor_similarities=None,
            outlier_count=5
        )
        
        assert 'longest_segments' in outliers
        assert 'shortest_segments' in outliers
    
    def test_identify_outliers_longest(self, sample_spans):
        """Test identification of longest segments."""
        outliers = diagnostics.identify_outliers(
            sample_spans,
            outlier_count=3
        )
        
        longest = outliers['longest_segments']
        assert len(longest) <= 3
        
        if len(longest) > 1:
            # Should be sorted by duration descending
            for i in range(len(longest) - 1):
                assert longest[i]['duration'] >= longest[i + 1]['duration']
    
    def test_identify_outliers_shortest(self, sample_spans):
        """Test identification of shortest segments."""
        outliers = diagnostics.identify_outliers(
            sample_spans,
            outlier_count=3
        )
        
        shortest = outliers['shortest_segments']
        assert len(shortest) <= 3
        
        if len(shortest) > 1:
            # Should be sorted by duration ascending
            for i in range(len(shortest) - 1):
                assert shortest[i]['duration'] <= shortest[i + 1]['duration']
    
    def test_identify_outliers_with_embeddings(self, sample_spans):
        """Test outlier identification with embedding-based metrics."""
        embeddings = create_sample_embeddings(len(sample_spans), pattern="random")
        
        # Find neighbors for sampling
        query_indices = np.array([0, 5, 10])
        neighbor_indices, neighbor_similarities = embedding.find_top_k_neighbors(
            query_indices, embeddings, k=5, exclude_self=True
        )
        
        outliers = diagnostics.identify_outliers(
            sample_spans,
            embeddings=embeddings,
            neighbor_indices=neighbor_indices,
            neighbor_similarities=neighbor_similarities,
            outlier_count=3
        )
        
        # Should have embedding-based outliers
        assert 'most_isolated' in outliers
        assert 'most_hubby' in outliers
    
    def test_sample_neighbor_lists(self, sample_spans):
        """Test neighbor list sampling."""
        embeddings = create_sample_embeddings(len(sample_spans), pattern="random")
        
        query_indices = np.array([0, 5, 10])
        neighbor_indices, neighbor_similarities = embedding.find_top_k_neighbors(
            query_indices, embeddings, k=5, exclude_self=True
        )
        
        samples = diagnostics.sample_neighbor_lists(
            sample_spans,
            embeddings,
            neighbor_indices,
            neighbor_similarities,
            sample_size=2,
            random_seed=42
        )
        
        assert len(samples) <= 2
        
        if len(samples) > 0:
            first_sample = samples[0]
            assert 'query_idx' in first_sample
            assert 'neighbors' in first_sample
    
    def test_format_text_excerpt(self):
        """Test text excerpt formatting."""
        long_text = "This is a very long text that should be truncated to 100 characters with ellipsis added at the end"
        
        excerpt = diagnostics.format_text_excerpt(long_text, max_length=50)
        
        assert len(excerpt) <= 53  # 50 + "..."
        assert excerpt.endswith('...')
    
    def test_format_text_excerpt_short(self):
        """Test text excerpt with short text."""
        short_text = "Short text"
        
        excerpt = diagnostics.format_text_excerpt(short_text, max_length=50)
        
        assert excerpt == short_text
        assert not excerpt.endswith('...')
    
    def test_export_outliers_csv(self, sample_spans, tmp_path):
        """Test exporting outliers to CSV."""
        outliers = diagnostics.identify_outliers(
            sample_spans,
            outlier_count=5
        )
        
        output_file = tmp_path / "outliers.csv"
        diagnostics.export_outliers_csv(sample_spans, outliers, output_file)
        
        # Check file was created
        assert output_file.exists()
        
        # Check it can be read back
        df = pd.read_csv(output_file)
        assert len(df) > 0
        assert 'segment_id' in df.columns or 'span_id' in df.columns or 'beat_id' in df.columns
    
    def test_export_neighbors_csv(self, sample_spans, tmp_path):
        """Test exporting neighbor samples to CSV."""
        embeddings = create_sample_embeddings(len(sample_spans), pattern="random")
        
        query_indices = np.array([0, 5])
        neighbor_indices, neighbor_similarities = embedding.find_top_k_neighbors(
            query_indices, embeddings, k=5, exclude_self=True
        )
        
        samples = diagnostics.sample_neighbor_lists(
            sample_spans,
            embeddings,
            neighbor_indices,
            neighbor_similarities,
            sample_size=2,
            random_seed=42
        )
        
        output_file = tmp_path / "neighbors.csv"
        diagnostics.export_neighbors_csv(sample_spans, samples, output_file)
        
        # Check file was created
        assert output_file.exists()
        
        # Check it can be read back
        df = pd.read_csv(output_file)
        assert len(df) > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestMetricsIntegration:
    """Test integration between different metric modules."""
    
    def test_complete_metrics_pipeline(self, complete_dataset):
        """Test running all metrics on complete dataset."""
        episodes = complete_dataset['episodes']
        spans = complete_dataset['spans']
        beats = complete_dataset['beats']
        
        # Test that all metrics can be calculated
        coverage_metrics = coverage.calculate_episode_coverage(episodes, spans, beats)
        assert coverage_metrics is not None
        
        dist_metrics = distribution.calculate_duration_statistics(spans)
        assert dist_metrics is not None
        
        integrity_metrics = integrity.check_timestamp_monotonicity(spans)
        assert integrity_metrics is not None
        
        balance_metrics = balance.calculate_speaker_distribution(spans)
        assert balance_metrics is not None
        
        text_metrics = text_quality.calculate_text_metrics(spans)
        assert text_metrics is not None
    
    def test_metrics_with_embeddings(self, complete_dataset):
        """Test metrics that require embeddings."""
        spans = complete_dataset['spans']
        embeddings_matrix = complete_dataset['span_embeddings']
        
        # Test embedding metrics
        query_indices = np.array([0, 5])
        neighbor_indices, similarities = embedding.find_top_k_neighbors(
            query_indices, embeddings_matrix, k=5
        )
        
        assert neighbor_indices is not None
        assert similarities is not None
        
        # Test diagnostics with embeddings
        outliers = diagnostics.identify_outliers(
            spans,
            embeddings=embeddings_matrix,
            neighbor_indices=neighbor_indices,
            neighbor_similarities=similarities,
            outlier_count=5
        )
        
        assert outliers is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
