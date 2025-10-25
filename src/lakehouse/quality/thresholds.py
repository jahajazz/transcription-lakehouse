"""
Quality assessment thresholds and configuration.

Defines threshold values for quality checks and validation criteria
based on PRD requirements (FR-9, FR-13, FR-17, FR-25, FR-27, FR-31).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any


class RAGStatus(Enum):
    """
    Red/Amber/Green status for quality assessment results.
    
    - GREEN: All thresholds passed, no critical issues
    - AMBER: Minor threshold violations (1-2 non-critical failures) or warnings
    - RED: Multiple threshold failures or any critical integrity issues
    """
    GREEN = "green"
    AMBER = "amber"
    RED = "red"
    
    def __str__(self) -> str:
        return self.value.upper()


@dataclass
class QualityThresholds:
    """
    Quality assessment thresholds for span and beat validation.
    
    All thresholds can be overridden via configuration file or CLI arguments.
    Default values are based on PRD functional requirements.
    """
    
    # Category A: Coverage & Count Metrics (FR-9)
    coverage_min: float = 95.0  # Minimum coverage % (sum_durations / episode_duration)
    gap_max_percent: float = 2.0  # Maximum gap % of episode duration
    overlap_max_percent: float = 2.0  # Maximum overlap % of episode duration
    
    # Category B: Length & Distribution Metrics (FR-11, FR-13)
    span_length_min: float = 20.0  # Minimum span length in seconds
    span_length_max: float = 120.0  # Maximum span length in seconds
    span_length_compliance_min: float = 90.0  # Min % of spans within bounds
    
    beat_length_min: float = 60.0  # Minimum beat length in seconds
    beat_length_max: float = 180.0  # Maximum beat length in seconds
    beat_length_compliance_min: float = 90.0  # Min % of beats within bounds
    
    # Category C: Ordering & Integrity Metrics (FR-17)
    timestamp_regressions_max: int = 0  # Maximum allowed timestamp regressions
    negative_duration_max: int = 0  # Maximum allowed negative/zero durations
    exact_duplicate_max_percent: float = 1.0  # Maximum % exact duplicates
    near_duplicate_max_percent: float = 3.0  # Maximum % near-duplicates
    near_duplicate_threshold: float = 0.95  # Fuzzy similarity threshold for near-dupes
    
    # Category F: Embedding Sanity Checks
    # Leakage thresholds (FR-25)
    same_speaker_neighbor_max_percent: float = 60.0  # Max % neighbors with same speaker
    same_episode_neighbor_max_percent: float = 70.0  # Max % neighbors from same episode
    
    # Length bias threshold (FR-27)
    length_bias_correlation_max: float = 0.3  # Max absolute correlation for length bias
    
    # Adjacency bias threshold (FR-31)
    adjacency_bias_max_percent: float = 40.0  # Max % temporally adjacent neighbors
    adjacency_tolerance_seconds: float = 5.0  # Tolerance for adjacency detection
    
    # Sampling parameters
    neighbor_sample_size: int = 100  # Number of segments to sample for neighbor analysis
    neighbor_k: int = 10  # Number of nearest neighbors to retrieve
    random_pairs_sample_size: int = 500  # Number of random pairs for lexical similarity
    outlier_count: int = 20  # Number of outliers to report per category
    neighbor_list_sample_size: int = 30  # Number of neighbor lists for human review
    
    # Top speakers to report
    top_speakers_count: int = 10
    
    # Reproducibility
    random_seed: int = 42  # Fixed seed for reproducible sampling
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert thresholds to dictionary for serialization."""
        return {
            "coverage": {
                "coverage_min": self.coverage_min,
                "gap_max_percent": self.gap_max_percent,
                "overlap_max_percent": self.overlap_max_percent,
            },
            "length": {
                "span_length_min": self.span_length_min,
                "span_length_max": self.span_length_max,
                "span_length_compliance_min": self.span_length_compliance_min,
                "beat_length_min": self.beat_length_min,
                "beat_length_max": self.beat_length_max,
                "beat_length_compliance_min": self.beat_length_compliance_min,
            },
            "integrity": {
                "timestamp_regressions_max": self.timestamp_regressions_max,
                "negative_duration_max": self.negative_duration_max,
                "exact_duplicate_max_percent": self.exact_duplicate_max_percent,
                "near_duplicate_max_percent": self.near_duplicate_max_percent,
                "near_duplicate_threshold": self.near_duplicate_threshold,
            },
            "embedding": {
                "same_speaker_neighbor_max_percent": self.same_speaker_neighbor_max_percent,
                "same_episode_neighbor_max_percent": self.same_episode_neighbor_max_percent,
                "length_bias_correlation_max": self.length_bias_correlation_max,
                "adjacency_bias_max_percent": self.adjacency_bias_max_percent,
                "adjacency_tolerance_seconds": self.adjacency_tolerance_seconds,
            },
            "sampling": {
                "neighbor_sample_size": self.neighbor_sample_size,
                "neighbor_k": self.neighbor_k,
                "random_pairs_sample_size": self.random_pairs_sample_size,
                "outlier_count": self.outlier_count,
                "neighbor_list_sample_size": self.neighbor_list_sample_size,
                "top_speakers_count": self.top_speakers_count,
                "random_seed": self.random_seed,
            },
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "QualityThresholds":
        """
        Create QualityThresholds from configuration dictionary.
        
        Args:
            config: Configuration dictionary (nested or flat)
        
        Returns:
            QualityThresholds instance
        """
        # Handle both nested and flat dictionaries
        if "coverage" in config or "length" in config:
            # Nested format
            flat_config = {}
            for category, values in config.items():
                if isinstance(values, dict):
                    flat_config.update(values)
            config = flat_config
        
        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config.items() if k in valid_fields}
        
        return cls(**filtered_config)
    
    def apply_overrides(self, **overrides) -> "QualityThresholds":
        """
        Create a new QualityThresholds instance with overridden values.
        
        Args:
            **overrides: Threshold values to override
        
        Returns:
            New QualityThresholds instance with overrides applied
        """
        current = self.to_dict()
        
        # Flatten current config
        flat_current = {}
        for category_values in current.values():
            flat_current.update(category_values)
        
        # Apply overrides
        flat_current.update(overrides)
        
        return QualityThresholds.from_dict(flat_current)


@dataclass
class ThresholdViolation:
    """Represents a single threshold violation."""
    threshold_name: str
    expected: Any
    actual: Any
    severity: str = "error"  # "error", "warning", or "info"
    message: str = ""
    
    def __str__(self) -> str:
        return (
            f"{self.threshold_name}: expected {self.expected}, "
            f"got {self.actual} ({self.severity.upper()})"
        )

