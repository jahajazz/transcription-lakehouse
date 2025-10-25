"""
Length and distribution metrics for quality assessment (Category B).

Calculates duration statistics, length compliance, and histogram bins
per PRD requirements FR-10, FR-11, FR-12, FR-13.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from lakehouse.logger import get_default_logger
from lakehouse.quality.thresholds import QualityThresholds, ThresholdViolation


logger = get_default_logger()


def calculate_duration_statistics(
    segments: pd.DataFrame,
    segment_type: str = "segment",
) -> Dict[str, Any]:
    """
    Calculate duration statistics for segments (FR-10).
    
    Computes:
    - Minimum duration
    - Maximum duration
    - Mean duration
    - Median duration
    - 5th percentile (p5)
    - 95th percentile (p95)
    - Standard deviation
    
    Args:
        segments: DataFrame with segment data (must have duration or start_time/end_time)
        segment_type: Type of segment for logging (e.g., "span", "beat")
    
    Returns:
        Dictionary containing duration statistics in seconds
    
    Example:
        >>> segments = pd.DataFrame({
        ...     'start_time': [0, 50, 100],
        ...     'end_time': [30, 90, 180]
        ... })
        >>> stats = calculate_duration_statistics(segments, "span")
        >>> stats['mean']
        53.33
    """
    if len(segments) == 0:
        logger.warning(f"No {segment_type}s provided for duration statistics")
        return {
            'count': 0,
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'p5': None,
            'p95': None,
            'std': None,
        }
    
    # Calculate durations
    if 'duration' in segments.columns:
        durations = segments['duration'].copy()
    elif 'start_time' in segments.columns and 'end_time' in segments.columns:
        durations = segments['end_time'] - segments['start_time']
    else:
        raise ValueError(
            f"Segments must have either 'duration' column or 'start_time'/'end_time' columns"
        )
    
    # Remove any NaN or negative values
    durations = durations[durations.notna() & (durations >= 0)]
    
    if len(durations) == 0:
        logger.warning(f"No valid durations found for {segment_type}s")
        return {
            'count': 0,
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'p5': None,
            'p95': None,
            'std': None,
        }
    
    # Calculate statistics
    stats = {
        'count': len(durations),
        'min': round(float(durations.min()), 2),
        'max': round(float(durations.max()), 2),
        'mean': round(float(durations.mean()), 2),
        'median': round(float(durations.median()), 2),
        'p5': round(float(durations.quantile(0.05)), 2),
        'p95': round(float(durations.quantile(0.95)), 2),
        'std': round(float(durations.std()), 2),
    }
    
    logger.debug(
        f"Duration statistics for {len(durations)} {segment_type}s: "
        f"mean={stats['mean']}s, median={stats['median']}s, "
        f"range=[{stats['min']}-{stats['max']}]s"
    )
    
    return stats


def calculate_length_compliance(
    segments: pd.DataFrame,
    min_duration: float,
    max_duration: float,
    segment_type: str = "segment",
) -> Dict[str, Any]:
    """
    Calculate percentage of segments within target duration bounds (FR-11).
    
    For spans: target range 20-120 seconds
    For beats: target range 60-180 seconds
    
    Args:
        segments: DataFrame with segment data
        min_duration: Minimum target duration in seconds
        max_duration: Maximum target duration in seconds
        segment_type: Type of segment for logging
    
    Returns:
        Dictionary containing:
        - within_bounds_count: Number of segments within bounds
        - within_bounds_percent: Percentage within bounds
        - too_short_count: Number below minimum
        - too_short_percent: Percentage below minimum
        - too_long_count: Number above maximum
        - too_long_percent: Percentage above maximum
    
    Example:
        >>> segments = pd.DataFrame({'duration': [10, 30, 50, 90, 150, 200]})
        >>> result = calculate_length_compliance(segments, 20, 120, "span")
        >>> result['within_bounds_percent']
        66.67
    """
    if len(segments) == 0:
        logger.warning(f"No {segment_type}s provided for length compliance check")
        return {
            'total_count': 0,
            'within_bounds_count': 0,
            'within_bounds_percent': 0.0,
            'too_short_count': 0,
            'too_short_percent': 0.0,
            'too_long_count': 0,
            'too_long_percent': 0.0,
        }
    
    # Calculate durations
    if 'duration' in segments.columns:
        durations = segments['duration'].copy()
    elif 'start_time' in segments.columns and 'end_time' in segments.columns:
        durations = segments['end_time'] - segments['start_time']
    else:
        raise ValueError(
            f"Segments must have either 'duration' column or 'start_time'/'end_time' columns"
        )
    
    # Remove any NaN values but keep negative/zero for integrity checking
    durations = durations[durations.notna()]
    
    if len(durations) == 0:
        logger.warning(f"No valid durations found for {segment_type}s")
        return {
            'total_count': 0,
            'within_bounds_count': 0,
            'within_bounds_percent': 0.0,
            'too_short_count': 0,
            'too_short_percent': 0.0,
            'too_long_count': 0,
            'too_long_percent': 0.0,
        }
    
    total_count = len(durations)
    
    # Classify segments
    within_bounds = (durations >= min_duration) & (durations <= max_duration)
    too_short = durations < min_duration
    too_long = durations > max_duration
    
    within_bounds_count = int(within_bounds.sum())
    too_short_count = int(too_short.sum())
    too_long_count = int(too_long.sum())
    
    # Calculate percentages
    within_bounds_percent = round((within_bounds_count / total_count * 100), 2)
    too_short_percent = round((too_short_count / total_count * 100), 2)
    too_long_percent = round((too_long_count / total_count * 100), 2)
    
    logger.info(
        f"Length compliance for {total_count} {segment_type}s "
        f"(target: {min_duration}-{max_duration}s): "
        f"{within_bounds_percent}% within bounds, "
        f"{too_short_percent}% too short, "
        f"{too_long_percent}% too long"
    )
    
    return {
        'total_count': total_count,
        'within_bounds_count': within_bounds_count,
        'within_bounds_percent': within_bounds_percent,
        'too_short_count': too_short_count,
        'too_short_percent': too_short_percent,
        'too_long_count': too_long_count,
        'too_long_percent': too_long_percent,
    }


def generate_histogram_bins(
    segments: pd.DataFrame,
    segment_type: str = "segment",
) -> Dict[str, Any]:
    """
    Generate histogram bin counts for duration distribution (FR-12).
    
    Uses sensible bins:
    - For spans: 0-20s, 20-40s, 40-60s, 60-90s, 90-120s, 120-180s, 180+
    - For beats: 0-60s, 60-90s, 90-120s, 120-150s, 150-180s, 180-240s, 240+
    
    Args:
        segments: DataFrame with segment data
        segment_type: Type of segment ("span" or "beat")
    
    Returns:
        Dictionary containing:
        - bins: List of bin ranges (tuples)
        - counts: List of counts per bin
        - percents: List of percentages per bin
    
    Example:
        >>> segments = pd.DataFrame({'duration': [15, 35, 65, 95, 125, 195]})
        >>> result = generate_histogram_bins(segments, "span")
        >>> len(result['bins'])
        7
    """
    # Define bin edges based on segment type
    if segment_type == "span":
        bin_edges = [0, 20, 40, 60, 90, 120, 180, float('inf')]
        bin_labels = [
            (0, 20), (20, 40), (40, 60), (60, 90),
            (90, 120), (120, 180), (180, None)  # None represents "180+"
        ]
    elif segment_type == "beat":
        bin_edges = [0, 60, 90, 120, 150, 180, 240, float('inf')]
        bin_labels = [
            (0, 60), (60, 90), (90, 120), (120, 150),
            (150, 180), (180, 240), (240, None)  # None represents "240+"
        ]
    else:
        # Default bins for unknown segment types (use span bins)
        logger.warning(
            f"Unknown segment type '{segment_type}', using default span bins"
        )
        bin_edges = [0, 20, 40, 60, 90, 120, 180, float('inf')]
        bin_labels = [
            (0, 20), (20, 40), (40, 60), (60, 90),
            (90, 120), (120, 180), (180, None)
        ]
    
    if len(segments) == 0:
        logger.warning(f"No {segment_type}s provided for histogram generation")
        return {
            'bins': bin_labels,
            'counts': [0] * len(bin_labels),
            'percents': [0.0] * len(bin_labels),
        }
    
    # Calculate durations
    if 'duration' in segments.columns:
        durations = segments['duration'].copy()
    elif 'start_time' in segments.columns and 'end_time' in segments.columns:
        durations = segments['end_time'] - segments['start_time']
    else:
        raise ValueError(
            f"Segments must have either 'duration' column or 'start_time'/'end_time' columns"
        )
    
    # Remove any NaN or negative values for histogram
    durations = durations[durations.notna() & (durations >= 0)]
    
    if len(durations) == 0:
        logger.warning(f"No valid durations found for {segment_type}s")
        return {
            'bins': bin_labels,
            'counts': [0] * len(bin_labels),
            'percents': [0.0] * len(bin_labels),
        }
    
    total_count = len(durations)
    
    # Use pandas cut to bin the durations
    binned = pd.cut(durations, bins=bin_edges, right=False, include_lowest=True)
    counts = binned.value_counts(sort=False).values
    
    # Convert to list of integers
    counts_list = [int(count) for count in counts]
    
    # Calculate percentages
    percents_list = [round((count / total_count * 100), 2) for count in counts_list]
    
    logger.debug(
        f"Histogram for {total_count} {segment_type}s: "
        f"{len(bin_labels)} bins with counts {counts_list}"
    )
    
    return {
        'bins': bin_labels,
        'counts': counts_list,
        'percents': percents_list,
    }


def validate_length_thresholds(
    compliance_metrics: Dict[str, Any],
    thresholds: QualityThresholds,
    segment_type: str = "segment",
) -> List[ThresholdViolation]:
    """
    Validate length compliance against thresholds (FR-13).
    
    Checks:
    - For spans: ≥ 90% within 20-120s (span_length_compliance_min)
    - For beats: ≥ 90% within 60-180s (beat_length_compliance_min)
    
    Args:
        compliance_metrics: Metrics from calculate_length_compliance
        thresholds: Quality thresholds configuration
        segment_type: Type of segment ("span" or "beat")
    
    Returns:
        List of threshold violations
    
    Example:
        >>> metrics = {'within_bounds_percent': 85.0}
        >>> thresholds = QualityThresholds(span_length_compliance_min=90.0)
        >>> violations = validate_length_thresholds(metrics, thresholds, "span")
        >>> len(violations)
        1
    """
    violations = []
    
    within_bounds_percent = compliance_metrics.get('within_bounds_percent', 0.0)
    total_count = compliance_metrics.get('total_count', 0)
    too_short_count = compliance_metrics.get('too_short_count', 0)
    too_short_percent = compliance_metrics.get('too_short_percent', 0.0)
    too_long_count = compliance_metrics.get('too_long_count', 0)
    too_long_percent = compliance_metrics.get('too_long_percent', 0.0)
    
    # Determine threshold and bounds based on segment type
    if segment_type == "span":
        min_compliance = thresholds.span_length_compliance_min
        min_duration = thresholds.span_length_min
        max_duration = thresholds.span_length_max
        threshold_name = 'span_length_compliance_min'
    elif segment_type == "beat":
        min_compliance = thresholds.beat_length_compliance_min
        min_duration = thresholds.beat_length_min
        max_duration = thresholds.beat_length_max
        threshold_name = 'beat_length_compliance_min'
    else:
        logger.warning(
            f"Unknown segment type '{segment_type}' for threshold validation"
        )
        return violations
    
    # Check if compliance meets minimum threshold
    if within_bounds_percent < min_compliance:
        violations.append(ThresholdViolation(
            threshold_name=threshold_name,
            expected=f'>= {min_compliance}%',
            actual=f'{within_bounds_percent}%',
            severity='error',
            message=(
                f'{segment_type.capitalize()} length compliance ({within_bounds_percent}%) '
                f'is below minimum threshold of {min_compliance}%. '
                f'Only {compliance_metrics.get("within_bounds_count", 0)}/{total_count} '
                f'{segment_type}s are within target bounds ({min_duration}-{max_duration}s). '
                f'{too_short_count} are too short ({too_short_percent}%), '
                f'{too_long_count} are too long ({too_long_percent}%).'
            ),
        ))
        
        logger.warning(
            f'{segment_type.capitalize()} length compliance violation: '
            f'{within_bounds_percent}% < {min_compliance}% '
            f'(target: {min_duration}-{max_duration}s)'
        )
    else:
        logger.debug(
            f'{segment_type.capitalize()} length compliance passed: '
            f'{within_bounds_percent}% >= {min_compliance}%'
        )
    
    return violations

