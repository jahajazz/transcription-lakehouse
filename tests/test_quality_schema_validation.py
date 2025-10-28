"""
Schema validation tests for quality metrics.

Tests that verify the output schema/structure of all calculators
matches what reporters and downstream consumers expect.

This prevents silent failures from dictionary key mismatches.
"""

import pytest
from typing import Dict, Any, List, Set
import pandas as pd

from src.lakehouse.quality.metrics import (
    coverage, distribution, integrity, balance, text_quality
)


class SchemaValidator:
    """Helper class to validate dictionary schemas."""
    
    @staticmethod
    def validate_keys(
        result: Dict[str, Any],
        required_keys: List[str],
        optional_keys: List[str] = None,
        context: str = ""
    ):
        """
        Validate that result dict has all required keys and no unexpected keys.
        
        Args:
            result: Dictionary to validate
            required_keys: Keys that must be present
            optional_keys: Keys that may be present
            context: Description for error messages
        """
        result_keys = set(result.keys())
        required = set(required_keys)
        optional = set(optional_keys or [])
        expected = required | optional
        
        # Check for missing required keys
        missing = required - result_keys
        if missing:
            raise AssertionError(
                f"{context}: Missing required keys: {missing}\n"
                f"Expected: {required}\n"
                f"Got: {result_keys}"
            )
        
        # Check for unexpected keys (might indicate typos)
        unexpected = result_keys - expected
        if unexpected:
            raise AssertionError(
                f"{context}: Unexpected keys found: {unexpected}\n"
                f"Expected keys: {expected}\n"
                f"Got: {result_keys}\n"
                f"These might be typos or need to be added to optional_keys."
            )
    
    @staticmethod
    def validate_non_zero(
        result: Dict[str, Any],
        keys: List[str],
        context: str = ""
    ):
        """Validate that specified keys have non-zero values."""
        for key in keys:
            value = result.get(key, 0)
            if isinstance(value, (int, float)) and value == 0:
                raise AssertionError(
                    f"{context}: Key '{key}' has value 0, expected non-zero. "
                    f"This might indicate a key mismatch bug."
                )


class TestTextQualitySchemas:
    """Validate text quality calculator output schemas."""
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'text': [
                'This is a sample sentence.',
                'Another example with more words and text.',
                'Short text.',
            ] * 10,
        })
    
    def test_calculate_text_metrics_schema(self, sample_data):
        """Validate calculate_text_metrics output schema."""
        result = text_quality.calculate_text_metrics(sample_data)
        
        SchemaValidator.validate_keys(
            result,
            required_keys=[
                'total_segments',
                'total_tokens',
                'total_words',
                'total_characters',
                'avg_tokens',        # MUST be avg_tokens, not avg_token_count
                'avg_words',         # MUST be avg_words, not avg_word_count
                'avg_characters',    # MUST be avg_characters, not avg_char_count
                'per_segment_stats',
            ],
            context="calculate_text_metrics"
        )
        
        SchemaValidator.validate_non_zero(
            result,
            keys=['avg_tokens', 'avg_words', 'avg_characters'],
            context="calculate_text_metrics"
        )
    
    def test_calculate_lexical_density_schema(self, sample_data):
        """Validate calculate_lexical_density output schema."""
        result = text_quality.calculate_lexical_density(sample_data)
        
        SchemaValidator.validate_keys(
            result,
            required_keys=[
                'total_words',
                'content_words',
                'stopword_count',
                'lexical_density',
                'avg_lexical_density',
                'per_segment_densities',
            ],
            context="calculate_lexical_density"
        )
        
        SchemaValidator.validate_non_zero(
            result,
            keys=['lexical_density', 'total_words'],
            context="calculate_lexical_density"
        )
    
    def test_extract_top_terms_schema(self, sample_data):
        """Validate extract_top_terms output schema."""
        result = text_quality.extract_top_terms(sample_data, top_n=5)
        
        SchemaValidator.validate_keys(
            result,
            required_keys=[
                'top_unigrams',          # MUST be top_unigrams, not top_terms
                'top_bigrams',           # MUST have bigrams too
                'total_unique_unigrams',
                'total_unique_bigrams',
            ],
            context="extract_top_terms"
        )
        
        # Validate list structure
        assert isinstance(result['top_unigrams'], list), "top_unigrams must be a list"
        assert isinstance(result['top_bigrams'], list), "top_bigrams must be a list"
        
        # Each item should be (term, count) tuple
        if result['top_unigrams']:
            assert isinstance(result['top_unigrams'][0], tuple), "Should be (term, count) tuples"
            assert len(result['top_unigrams'][0]) == 2, "Should have exactly 2 elements"


class TestDistributionSchemas:
    """Validate distribution calculator output schemas."""
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'start_time': [i * 10.0 for i in range(100)],
            'end_time': [(i + 1) * 10.0 for i in range(100)],
            'duration': [10.0] * 100,
        })
    
    def test_calculate_duration_statistics_schema(self, sample_data):
        """Validate calculate_duration_statistics output schema."""
        result = distribution.calculate_duration_statistics(sample_data)
        
        SchemaValidator.validate_keys(
            result,
            required_keys=[
                'count', 'mean', 'std', 'min', 'max',
                'median', 'p5', 'p95',  # Actual keys returned
            ],
            context="calculate_duration_statistics"
        )
    
    def test_calculate_length_compliance_schema(self, sample_data):
        """Validate calculate_length_compliance output schema."""
        result = distribution.calculate_length_compliance(
            sample_data,
            min_duration=5.0,
            max_duration=15.0
        )
        
        SchemaValidator.validate_keys(
            result,
            required_keys=[
                'total_count',  # Actual key (not total_segments)
                'within_bounds_count',  # Actual key (not in_range_count)
                'too_short_count',
                'too_long_count',
                'within_bounds_percent',  # Actual key (not in_range_percent)
                'too_short_percent',
                'too_long_percent',
            ],
            optional_keys=[
                'histogram_bins',  # May or may not be present
            ],
            context="calculate_length_compliance"
        )


class TestBalanceSchemas:
    """Validate balance calculator output schemas."""
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'speaker_canonical': ['Alice'] * 30 + ['Bob'] * 40 + ['Charlie'] * 30,
            'start_time': [i * 10.0 for i in range(100)],
            'end_time': [(i + 1) * 10.0 for i in range(100)],
            'duration': [10.0] * 100,
        })
    
    def test_calculate_speaker_distribution_schema(self, sample_data):
        """Validate calculate_speaker_distribution output schema."""
        result = balance.calculate_speaker_distribution(sample_data, top_n=5)
        
        SchemaValidator.validate_keys(
            result,
            required_keys=[
                'total_segments',
                'total_speakers',
                'avg_segments_per_speaker',  # MUST be present!
                'speaker_stats',
                'top_speakers',
                'long_tail_stats',
            ],
            context="calculate_speaker_distribution"
        )
        
        SchemaValidator.validate_non_zero(
            result,
            keys=['total_speakers', 'avg_segments_per_speaker'],
            context="calculate_speaker_distribution"
        )
        
        # Ensure NO 'speakers' wrapper key exists (common mistake)
        assert 'speakers' not in result, (
            "Should not have 'speakers' wrapper key - "
            "calculator returns flat structure, not nested"
        )
        
        # Validate nested structures
        assert isinstance(result['speaker_stats'], list), "speaker_stats must be a list"
        assert isinstance(result['top_speakers'], list), "top_speakers must be a list"
        assert isinstance(result['long_tail_stats'], dict), "long_tail_stats must be a dict"


class TestIntegritySchemas:
    """Validate integrity calculator output schemas."""
    
    @pytest.fixture
    def sample_spans(self):
        return pd.DataFrame({
            'episode_id': ['ep1'] * 50 + ['ep2'] * 50,
            'speaker': ['Alice'] * 50 + ['Bob'] * 50,
            'start_time': [i * 10.0 for i in range(100)],
            'end_time': [(i + 1) * 10.0 for i in range(100)],
            'text': ['text'] * 100,
        })
    
    @pytest.fixture
    def sample_beats(self):
        return pd.DataFrame({
            'episode_id': ['ep1'] * 25 + ['ep2'] * 25,
            'speakers_set': [['Alice', 'Bob']] * 50,  # Note: no 'speaker' column
            'start_time': [i * 20.0 for i in range(50)],
            'end_time': [(i + 1) * 20.0 for i in range(50)],
            'text': ['text'] * 50,
        })
    
    def test_check_timestamp_monotonicity_schema(self, sample_spans, sample_beats):
        """Validate check_timestamp_monotonicity output schema."""
        # Test with spans (has speaker column)
        result_spans = integrity.check_timestamp_monotonicity(
            sample_spans, segment_type="span"
        )
        
        SchemaValidator.validate_keys(
            result_spans,
            required_keys=[
                'total_segments',
                'episode_regression_count',
                'speaker_regression_count',
                'episode_regressions',
                'speaker_regressions',
            ],
            context="check_timestamp_monotonicity (spans)"
        )
        
        # Test with beats (no speaker column - should still work)
        result_beats = integrity.check_timestamp_monotonicity(
            sample_beats, segment_type="beat"
        )
        
        # Beats should have episode regressions but not speaker regressions
        assert 'episode_regressions' in result_beats
        assert 'episode_regression_count' in result_beats
        assert 'total_segments' in result_beats
    
    def test_detect_duplicates_schema(self, sample_spans):
        """Validate detect_duplicates output schema."""
        result = integrity.detect_duplicates(
            sample_spans,
            fuzzy_threshold=0.9,
            segment_type="span"
        )
        
        required_keys = [
            'total_segments',
            'exact_duplicate_count',  # Actual key (not exact_duplicates)
            'exact_duplicate_percent',
            'exact_duplicate_groups',
        ]
        
        # Check for required keys
        for key in required_keys:
            assert key in result, f"Missing key '{key}' in detect_duplicates"


class TestReporterCompatibility:
    """Test that reporter code matches calculator schemas."""
    
    def test_reporter_uses_correct_text_quality_keys(self):
        """Verify reporter._generate_text_quality_section uses correct keys."""
        from src.lakehouse.quality.reporter import QualityReporter
        import inspect
        
        source = inspect.getsource(QualityReporter._generate_text_quality_section)
        
        # Should use correct keys
        correct_keys = ['avg_tokens', 'avg_words', 'avg_characters']
        for key in correct_keys:
            assert key in source, (
                f"Reporter should use '{key}' (found in calculator output), "
                f"not a variant like 'avg_token_count'"
            )
        
        # Should NOT use wrong keys
        wrong_keys = ['avg_token_count', 'avg_word_count', 'avg_char_count']
        for key in wrong_keys:
            assert key not in source, (
                f"Reporter should NOT use '{key}' (this key doesn't exist in calculator output)"
            )
    
    def test_reporter_uses_correct_length_compliance_keys(self):
        """Verify reporter._generate_distribution_section uses correct keys."""
        from src.lakehouse.quality.reporter import QualityReporter
        import inspect
        
        source = inspect.getsource(QualityReporter._generate_distribution_section)
        
        # Should use correct keys
        correct_keys = ['too_short_percent', 'too_long_percent']
        for key in correct_keys:
            assert key in source, (
                f"Reporter should use '{key}' (found in calculator output)"
            )
        
        # Should NOT use wrong keys
        wrong_keys = ['below_min_percent', 'above_max_percent']
        for key in wrong_keys:
            assert key not in source, (
                f"Reporter should NOT use '{key}' (this key doesn't exist in calculator output)"
            )


class TestCalculatorConsistency:
    """Test that calculators are consistent with each other."""
    
    def test_all_calculators_accept_segment_type(self):
        """All calculators should accept segment_type parameter for logging."""
        sample_df = pd.DataFrame({
            'text': ['sample text'] * 10,
            'start_time': range(10),
            'end_time': range(1, 11),
            'speaker_canonical': ['Alice'] * 10,
        })
        
        # These should all accept segment_type without errors
        try:
            text_quality.calculate_text_metrics(sample_df, segment_type="test")
            text_quality.calculate_lexical_density(sample_df, segment_type="test")
            distribution.calculate_duration_statistics(sample_df, segment_type="test")
        except TypeError as e:
            pytest.fail(f"Calculator should accept segment_type parameter: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

