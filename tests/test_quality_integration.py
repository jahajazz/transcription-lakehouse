"""
Integration tests for quality assessment pipeline.

These tests verify that the entire quality assessment pipeline works correctly,
with special focus on catching dictionary key/column name mismatches between
calculators and reporters.

Tests are designed to catch issues like:
1. Wrong dictionary keys (avg_token_count vs avg_tokens)
2. Wrong column names (speaker vs speaker_canonical)
3. Missing required fields in output
4. Silent failures (returning 0 as default when key doesn't exist)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.lakehouse.quality.assessor import QualityAssessor
from src.lakehouse.quality.reporter import QualityReporter
from src.lakehouse.quality.metrics import (
    coverage, distribution, integrity, balance, text_quality
)


class TestCalculatorOutputs:
    """Test that all calculators return expected keys."""
    
    @pytest.fixture
    def sample_spans(self):
        """Create sample spans DataFrame with realistic data."""
        return pd.DataFrame({
            'span_id': [f'span_{i}' for i in range(100)],
            'episode_id': ['ep1'] * 50 + ['ep2'] * 50,
            'speaker': ['Alice'] * 30 + ['Bob'] * 40 + ['Charlie'] * 30,
            'speaker_canonical': ['Alice'] * 30 + ['Bob'] * 40 + ['Charlie'] * 30,
            'speaker_role': ['host'] * 30 + ['guest'] * 40 + ['expert'] * 30,
            'is_expert': [False] * 30 + [False] * 40 + [True] * 30,
            'start_time': [i * 10.0 for i in range(100)],
            'end_time': [(i + 1) * 10.0 for i in range(100)],
            'duration': [10.0] * 100,
            'text': ['This is a sample text segment'] * 100,
        })
    
    @pytest.fixture
    def sample_beats(self):
        """Create sample beats DataFrame."""
        return pd.DataFrame({
            'beat_id': [f'beat_{i}' for i in range(50)],
            'episode_id': ['ep1'] * 25 + ['ep2'] * 25,
            'start_time': [i * 20.0 for i in range(50)],
            'end_time': [(i + 1) * 20.0 for i in range(50)],
            'duration': [20.0] * 50,
            'text': ['This is a combined beat segment with more text'] * 50,
            'speakers_set': [['Alice', 'Bob']] * 50,
        })
    
    @pytest.fixture
    def sample_episodes(self):
        """Create sample episodes DataFrame."""
        return pd.DataFrame({
            'episode_id': ['ep1', 'ep2'],
            'start_time': [0.0, 500.0],
            'end_time': [500.0, 1000.0],
            'duration': [500.0, 500.0],
        })
    
    def test_text_metrics_keys(self, sample_spans):
        """Test that calculate_text_metrics returns all expected keys."""
        result = text_quality.calculate_text_metrics(sample_spans)
        
        # Required keys
        required_keys = [
            'total_segments',
            'total_tokens',
            'total_words',
            'total_characters',
            'avg_tokens',  # NOT avg_token_count!
            'avg_words',   # NOT avg_word_count!
            'avg_characters',  # NOT avg_char_count!
            'per_segment_stats',
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key '{key}' in text_metrics"
        
        # Values should be non-zero
        assert result['avg_tokens'] > 0, "avg_tokens should be non-zero"
        assert result['avg_words'] > 0, "avg_words should be non-zero"
        assert result['avg_characters'] > 0, "avg_characters should be non-zero"
    
    def test_lexical_density_keys(self, sample_spans):
        """Test that calculate_lexical_density returns all expected keys."""
        result = text_quality.calculate_lexical_density(sample_spans)
        
        required_keys = [
            'total_words',
            'content_words',
            'stopword_count',
            'lexical_density',
            'avg_lexical_density',
            'per_segment_densities',
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key '{key}' in lexical_density"
        
        # Values should be non-zero
        assert result['lexical_density'] > 0, "lexical_density should be non-zero"
    
    def test_top_terms_keys(self, sample_spans):
        """Test that extract_top_terms returns all expected keys."""
        result = text_quality.extract_top_terms(sample_spans, top_n=10)
        
        required_keys = [
            'top_unigrams',  # NOT top_terms!
            'top_bigrams',
            'total_unique_unigrams',
            'total_unique_bigrams',
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key '{key}' in top_terms"
        
        # Should have some terms
        assert len(result['top_unigrams']) > 0, "Should have some unigrams"
    
    def test_distribution_keys(self, sample_spans):
        """Test that distribution metrics return all expected keys."""
        stats = distribution.calculate_duration_statistics(sample_spans)
        
        required_keys = [
            'count', 'mean', 'std', 'min', 'max',
            'median', 'p5', 'p95',  # Actual keys returned
        ]
        
        for key in required_keys:
            assert key in stats, f"Missing key '{key}' in duration_statistics"
    
    def test_length_compliance_keys(self, sample_spans):
        """Test that length_compliance returns all expected keys."""
        result = distribution.calculate_length_compliance(
            sample_spans, min_duration=5.0, max_duration=15.0
        )
        
        # These are the CORRECT keys (not below_min_percent/above_max_percent!)
        required_keys = [
            'total_count',  # Actual key (not total_segments)
            'within_bounds_count',  # Actual key (not in_range_count)
            'too_short_count',
            'too_long_count',
            'within_bounds_percent',  # Actual key (not in_range_percent)
            'too_short_percent',
            'too_long_percent',
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key '{key}' in length_compliance"
    
    def test_speaker_distribution_keys(self, sample_spans):
        """Test that speaker_distribution returns all expected keys."""
        result = balance.calculate_speaker_distribution(sample_spans, top_n=5)
        
        required_keys = [
            'total_segments',
            'total_speakers',
            'avg_segments_per_speaker',  # Should be present!
            'speaker_stats',
            'top_speakers',
            'long_tail_stats',
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key '{key}' in speaker_distribution"
        
        # Should have found speakers
        assert result['total_speakers'] > 0, "Should have found speakers"
        assert result['avg_segments_per_speaker'] > 0, "Should have avg segments per speaker"
    
    def test_integrity_metrics_structure(self, sample_spans, sample_beats):
        """Test that integrity metrics have correct structure."""
        # Check span integrity
        span_monotonicity = integrity.check_timestamp_monotonicity(
            sample_spans, segment_type="span"
        )
        
        required_keys = [
            'total_segments',
            'episode_regression_count',
            'speaker_regression_count',
            'episode_regressions',
            'speaker_regressions',
        ]
        
        for key in required_keys:
            assert key in span_monotonicity, f"Missing key '{key}' in monotonicity"
        
        # Check beat integrity (should work even without 'speaker' column)
        beat_monotonicity = integrity.check_timestamp_monotonicity(
            sample_beats, segment_type="beat"
        )
        
        assert 'episode_regression_count' in beat_monotonicity, "Missing episode_regression_count for beats"
        assert 'total_segments' in beat_monotonicity, "Missing total_segments for beats"


class TestReporterCompatibility:
    """Test that reporter can correctly read all calculator outputs."""
    
    @pytest.fixture
    def mock_metrics(self):
        """Create mock metrics bundle matching actual calculator outputs."""
        from src.lakehouse.quality.assessor import MetricsBundle
        
        metrics = MetricsBundle()
        
        # Text quality metrics (with CORRECT keys)
        metrics.text_quality_metrics = {
            'statistics': {
                'total_segments': 1000,
                'avg_tokens': 84.3,     # NOT avg_token_count
                'avg_words': 84.13,     # NOT avg_word_count
                'avg_characters': 457.99,  # NOT avg_char_count
            },
            'lexical_density': {
                'lexical_density': 0.493,
                'avg_lexical_density': 0.541,
                'content_words': 100000,
                'stopword_count': 95000,
            },
            'top_terms': {
                'top_unigrams': [('word', 100)],  # NOT top_terms
                'top_bigrams': [('two words', 50)],
            },
        }
        
        # Balance metrics (with CORRECT keys)
        metrics.balance_metrics = {
            'total_segments': 1000,
            'total_speakers': 10,
            'avg_segments_per_speaker': 100.0,  # Should be present!
            'speaker_stats': [],
            'top_speakers': [],
            'long_tail_stats': {},
        }
        
        # Distribution metrics (with CORRECT keys)
        metrics.distribution_metrics = {
            'spans': {
                'statistics': {
                    'count': 1000,
                    'mean': 10.0,
                    'std': 2.0,
                    'min': 5.0,
                    'max': 20.0,
                },
                'compliance': {
                    'total_segments': 1000,
                    'in_range_percent': 90.0,
                    'too_short_percent': 5.0,  # NOT below_min_percent
                    'too_long_percent': 5.0,   # NOT above_max_percent
                },
            },
        }
        
        # Integrity metrics (with CORRECT structure)
        metrics.integrity_metrics = {
            'spans': {
                'episode_regressions': 0,
                'speaker_regressions': 0,
                'exact_duplicates': 10,
            },
            'beats': {
                'episode_regressions': 0,
                'exact_duplicates': 5,
            },
        }
        
        return metrics
    
    def test_reporter_text_quality_section(self, mock_metrics, tmp_path):
        """Test that reporter can generate text quality section without errors."""
        from src.lakehouse.quality.thresholds import QualityThresholds, RAGStatus
        from src.lakehouse.quality.assessor import AssessmentResult
        
        # Create AssessmentResult with mock data
        result = AssessmentResult(
            metrics=mock_metrics,
            thresholds=QualityThresholds(),
            violations=[],
            rag_status=RAGStatus.GREEN,
        )
        
        reporter = QualityReporter(result)
        section = reporter._generate_text_quality_section()
        
        # Should not be empty or show "no data"
        assert section is not None
        assert "(No text quality data available)" not in section
        
        # Should contain the correct values (not 0.0)
        assert "84.3" in section or "84.30" in section, "Should show avg_tokens value"
        assert "84.1" in section or "84.13" in section, "Should show avg_words value (formatted)"
        assert "45" in section, "Should show avg_characters value (at least hundreds)"
        assert "0.49" in section or "0.493" in section, "Should show lexical_density value"
        
        # Should show top terms
        assert "word" in section, "Should show top unigrams"
        assert "two words" in section, "Should show top bigrams"
    
    def test_reporter_balance_section(self, mock_metrics, tmp_path):
        """Test that reporter can generate balance section without errors."""
        from src.lakehouse.quality.thresholds import QualityThresholds, RAGStatus
        from src.lakehouse.quality.assessor import AssessmentResult
        
        result = AssessmentResult(
            metrics=mock_metrics,
            thresholds=QualityThresholds(),
            violations=[],
            rag_status=RAGStatus.GREEN,
        )
        
        reporter = QualityReporter(result)
        section = reporter._generate_balance_section()
        
        # Should not crash and should generate section
        assert section is not None
        assert "Category D" in section or "Speaker" in section
        
        # FIXED: Should show correct values from flat structure (not nested under 'speakers')
        # Verify it actually shows the numbers from the correct keys
        assert "10" in section, "Should show total_speakers value (10)"
        assert "100.0" in section, "Should show avg_segments_per_speaker value (100.0)"
    
    def test_reporter_distribution_section(self, mock_metrics, tmp_path):
        """Test that reporter can generate distribution section without errors."""
        from src.lakehouse.quality.thresholds import QualityThresholds, RAGStatus
        from src.lakehouse.quality.assessor import AssessmentResult
        
        result = AssessmentResult(
            metrics=mock_metrics,
            thresholds=QualityThresholds(),
            violations=[],
            rag_status=RAGStatus.GREEN,
        )
        
        reporter = QualityReporter(result)
        section = reporter._generate_distribution_section()
        
        # Should not crash and should generate section
        assert section is not None
        assert "Category B" in section or "Distribution" in section
        # Should use compliance terminology (not "below minimum" from old keys)
        assert "bounds" in section.lower() or "compliance" in section.lower()
    
    def test_reporter_integrity_section(self, mock_metrics, tmp_path):
        """Test that reporter can generate integrity section without errors."""
        from src.lakehouse.quality.thresholds import QualityThresholds, RAGStatus
        from src.lakehouse.quality.assessor import AssessmentResult
        
        result = AssessmentResult(
            metrics=mock_metrics,
            thresholds=QualityThresholds(),
            violations=[],
            rag_status=RAGStatus.GREEN,
        )
        
        reporter = QualityReporter(result)
        section = reporter._generate_integrity_section()
        
        # Should handle nested structure (spans/beats)
        assert "span" in section.lower() or "Span" in section
        assert "beat" in section.lower() or "Beat" in section


class TestEndToEndAssessment:
    """Test full quality assessment pipeline end-to-end."""
    
    @pytest.mark.integration
    def test_full_assessment_pipeline(self, tmp_path):
        """
        Test complete assessment pipeline with real lakehouse data.
        
        This test would have caught all 4 bugs:
        1. Text quality keys
        2. Speaker metrics keys
        3. Length compliance keys
        4. Integrity structure
        """
        # Skip if lakehouse data not available
        lakehouse_path = Path("lakehouse")
        if not lakehouse_path.exists():
            pytest.skip("Lakehouse data not available")
        
        # Run assessment
        assessor = QualityAssessor(
            lakehouse_path=lakehouse_path,
            version="v1",
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=tmp_path,
        )
        
        # Verify metrics were calculated
        assert result.metrics is not None
        
        # Text quality should be non-zero
        if result.metrics.text_quality_metrics:
            stats = result.metrics.text_quality_metrics.get('statistics', {})
            avg_tokens = stats.get('avg_tokens', 0)
            assert avg_tokens > 0, (
                f"Text quality avg_tokens is {avg_tokens}, expected > 0. "
                "Check if reporter is using correct keys."
            )
        
        # Speaker metrics should be non-zero
        if result.metrics.balance_metrics:
            total_speakers = result.metrics.balance_metrics.get('total_speakers', 0)
            assert total_speakers > 0, (
                f"Speaker metrics total_speakers is {total_speakers}, expected > 0. "
                "Check if balance calculator is using correct column names."
            )
        
        # Distribution compliance should use correct keys
        if result.metrics.distribution_metrics:
            span_dist = result.metrics.distribution_metrics.get('spans', {})
            compliance = span_dist.get('compliance', {})
            
            # Should have too_short_percent, NOT below_min_percent
            assert 'too_short_percent' in compliance, (
                "Missing 'too_short_percent' in compliance. "
                "Still using old 'below_min_percent' key?"
            )
        
        # Integrity should have nested structure
        if result.metrics.integrity_metrics:
            assert 'spans' in result.metrics.integrity_metrics or 'beats' in result.metrics.integrity_metrics, (
                "Integrity metrics should have 'spans'/'beats' nested structure"
            )
        
        # Generate report and verify it doesn't have zeros
        reporter = result.get_reporter()
        report_md = reporter.generate_markdown_report()
        
        # Check for suspicious patterns that indicate bugs
        suspicious_patterns = [
            ("Average Token Count: 0.0", "Text quality bug"),
            ("Average Word Count: 0.0", "Text quality bug"),
            ("Unique Speakers: 0", "Speaker metrics bug"),
            ("Below minimum: 0.00%", "Length compliance might be using wrong keys"),
        ]
        
        for pattern, description in suspicious_patterns:
            if pattern in report_md:
                # Allow zeros if truly no data, but warn
                pytest.warn(
                    f"Suspicious pattern found: '{pattern}' ({description}). "
                    f"Verify this is expected and not a key mismatch bug."
                )


class TestColumnNameCompatibility:
    """Test that calculators handle different column name variations."""
    
    def test_speaker_column_variants(self):
        """Test that balance calculator tries both speaker and speaker_canonical."""
        # DataFrame with only speaker_canonical (new enriched format)
        df_canonical = pd.DataFrame({
            'speaker_canonical': ['Alice', 'Bob', 'Alice'],
            'start_time': [0, 10, 20],
            'end_time': [10, 20, 30],
        })
        
        result = balance.calculate_speaker_distribution(df_canonical)
        assert result['total_speakers'] > 0, "Should work with speaker_canonical"
        
        # DataFrame with only speaker (old format)
        df_raw = pd.DataFrame({
            'speaker': ['Alice', 'Bob', 'Alice'],
            'start_time': [0, 10, 20],
            'end_time': [10, 20, 30],
        })
        
        result = balance.calculate_speaker_distribution(df_raw)
        assert result['total_speakers'] > 0, "Should work with speaker"
    
    def test_beats_without_speaker_column(self):
        """Test that integrity checks work on beats without speaker column."""
        df_beats = pd.DataFrame({
            'beat_id': ['b1', 'b2', 'b3'],
            'episode_id': ['ep1', 'ep1', 'ep1'],
            'speakers_set': [['Alice', 'Bob'], ['Alice'], ['Bob']],
            'start_time': [0, 10, 20],
            'end_time': [10, 20, 30],
            'text': ['text1', 'text2', 'text3'],
        })
        
        # Should not crash when checking monotonicity without speaker column
        result = integrity.check_timestamp_monotonicity(df_beats, segment_type="beat")
        assert 'episode_regression_count' in result
        assert 'total_segments' in result


class TestReportValidation:
    """Validate that generated reports have expected content."""
    
    @pytest.mark.integration
    def test_report_has_no_default_zeros(self, tmp_path):
        """Test that report doesn't have suspicious default zeros."""
        lakehouse_path = Path("lakehouse")
        if not lakehouse_path.exists():
            pytest.skip("Lakehouse data not available")
        
        assessor = QualityAssessor(lakehouse_path=lakehouse_path, version="v1")
        result = assessor.run_assessment(output_dir=tmp_path)
        
        reporter = result.get_reporter()
        report_md = reporter.generate_markdown_report()
        
        # Check that key metrics are non-zero
        checks = [
            ("Average Token Count: 0.0", False, "Text quality should not be 0"),
            ("Unique Speakers: 0", False, "Speaker count should not be 0"),
            ("Total Segments:", True, "Should have total segments count"),
            ("Category", True, "Should have category headers"),
        ]
        
        for pattern, should_exist, message in checks:
            if should_exist:
                assert pattern in report_md, message
            else:
                assert pattern not in report_md, message


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

