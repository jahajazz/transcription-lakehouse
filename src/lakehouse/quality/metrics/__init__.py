"""
Quality metrics calculation submodule.

Provides metric calculators for seven quality assessment categories:
- Category A: Coverage & Count metrics
- Category B: Length & Distribution metrics
- Category C: Ordering & Integrity metrics
- Category D: Speaker & Series Balance metrics
- Category E: Text Quality Proxy metrics
- Category F: Embedding Sanity Checks
- Category G: Outliers & Diagnostics (in parent module)
"""

# Category A: Coverage & Count metrics
from lakehouse.quality.metrics.coverage import (
    calculate_episode_coverage,
    detect_gaps_and_overlaps,
    validate_coverage_thresholds,
)

# Category B: Length & Distribution metrics
from lakehouse.quality.metrics.distribution import (
    calculate_duration_statistics,
    calculate_length_compliance,
    generate_histogram_bins,
    validate_length_thresholds,
)

# Category C: Ordering & Integrity metrics
from lakehouse.quality.metrics.integrity import (
    check_timestamp_monotonicity,
    detect_integrity_violations,
    detect_duplicates,
    validate_integrity_thresholds,
)

# Category D: Speaker & Series Balance metrics
from lakehouse.quality.metrics.balance import (
    calculate_speaker_distribution,
    calculate_series_balance,
)

# Category E: Text Quality Proxy metrics
from lakehouse.quality.metrics.text_quality import (
    calculate_text_metrics,
    calculate_lexical_density,
    calculate_punctuation_ratio,
    extract_top_terms,
)

# Category F: Embedding Sanity Checks
# from lakehouse.quality.metrics.embedding import (
#     load_embeddings,
#     stratified_sample_segments,
#     compute_cosine_similarity,
#     find_top_k_neighbors,
#     extract_neighbor_themes,
#     calculate_speaker_leakage,
#     calculate_episode_leakage,
#     validate_leakage_thresholds,
#     calculate_embedding_norms,
#     calculate_length_bias_correlation,
#     validate_length_bias_threshold,
#     sample_random_pairs,
#     calculate_lexical_similarity,
#     calculate_similarity_correlation,
#     calculate_cross_series_neighbors,
#     calculate_adjacency_bias,
#     validate_adjacency_threshold,
# )

__all__ = [
    # Coverage metrics
    "calculate_episode_coverage",
    "detect_gaps_and_overlaps",
    "validate_coverage_thresholds",
    # Distribution metrics
    "calculate_duration_statistics",
    "calculate_length_compliance",
    "generate_histogram_bins",
    "validate_length_thresholds",
    # Integrity metrics
    "check_timestamp_monotonicity",
    "detect_integrity_violations",
    "detect_duplicates",
    "validate_integrity_thresholds",
    # Balance metrics
    "calculate_speaker_distribution",
    "calculate_series_balance",
    # Text quality metrics
    "calculate_text_metrics",
    "calculate_lexical_density",
    "calculate_punctuation_ratio",
    "extract_top_terms",
    # Embedding metrics (commented out until implemented)
    # "load_embeddings",
    # "stratified_sample_segments",
    # "compute_cosine_similarity",
    # "find_top_k_neighbors",
    # "extract_neighbor_themes",
    # "calculate_speaker_leakage",
    # "calculate_episode_leakage",
    # "validate_leakage_thresholds",
    # "calculate_embedding_norms",
    # "calculate_length_bias_correlation",
    # "validate_length_bias_threshold",
    # "sample_random_pairs",
    # "calculate_lexical_similarity",
    # "calculate_similarity_correlation",
    # "calculate_cross_series_neighbors",
    # "calculate_adjacency_bias",
    # "validate_adjacency_threshold",
]

