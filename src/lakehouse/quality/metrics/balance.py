"""
Speaker and series balance metrics for quality assessment (Category D).

Calculates speaker distribution, series balance, and identifies long-tail patterns
per PRD requirements FR-18, FR-19.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from lakehouse.logger import get_default_logger


logger = get_default_logger()


def calculate_speaker_distribution(
    segments: pd.DataFrame,
    top_n: int = 10,
    segment_type: str = "segment",
) -> Dict[str, Any]:
    """
    Calculate speaker distribution statistics (FR-18).
    
    Computes:
    - Segment count per speaker
    - Percentage of total segments per speaker
    - Total duration per speaker
    - Average duration per speaker
    - Top N speakers by segment count
    - Long-tail statistics (speakers beyond top N)
    
    Args:
        segments: DataFrame with segment data (must have speaker, start_time, end_time or duration)
        top_n: Number of top speakers to return detailed stats for
        segment_type: Type of segment for logging
    
    Returns:
        Dictionary containing:
        - total_segments: Total number of segments
        - total_speakers: Number of unique speakers
        - speaker_stats: List of per-speaker statistics (sorted by count, descending)
        - top_speakers: Top N speakers with detailed stats
        - long_tail_stats: Statistics for speakers beyond top N
    
    Example:
        >>> segments = pd.DataFrame({
        ...     'speaker': ['A', 'A', 'B', 'B', 'C'],
        ...     'start_time': [0, 30, 60, 90, 120],
        ...     'end_time': [25, 55, 85, 115, 145]
        ... })
        >>> result = calculate_speaker_distribution(segments, top_n=2)
        >>> result['total_speakers']
        3
    """
    if len(segments) == 0:
        logger.warning(f"No {segment_type}s provided for speaker distribution")
        return {
            'total_segments': 0,
            'total_speakers': 0,
            'speaker_stats': [],
            'top_speakers': [],
            'long_tail_stats': {
                'speaker_count': 0,
                'segment_count': 0,
                'segment_percent': 0.0,
                'total_duration': 0.0,
                'avg_duration': 0.0,
            },
        }
    
    if 'speaker' not in segments.columns:
        logger.error(f"speaker column not found in {segment_type}s")
        return {
            'total_segments': len(segments),
            'total_speakers': 0,
            'speaker_stats': [],
            'top_speakers': [],
            'long_tail_stats': {
                'speaker_count': 0,
                'segment_count': 0,
                'segment_percent': 0.0,
                'total_duration': 0.0,
                'avg_duration': 0.0,
            },
        }
    
    total_segments = len(segments)
    
    # Calculate duration if not present
    if 'duration' not in segments.columns:
        if 'start_time' in segments.columns and 'end_time' in segments.columns:
            segments = segments.copy()
            segments['duration'] = segments['end_time'] - segments['start_time']
        else:
            logger.warning(f"Cannot calculate duration for {segment_type}s (missing time columns)")
            segments = segments.copy()
            segments['duration'] = 0.0
    
    # Filter out segments with missing speakers
    valid_segments = segments[segments['speaker'].notna() & (segments['speaker'] != '')]
    
    if len(valid_segments) == 0:
        logger.warning(f"No valid speakers found in {segment_type}s")
        return {
            'total_segments': total_segments,
            'total_speakers': 0,
            'speaker_stats': [],
            'top_speakers': [],
            'long_tail_stats': {
                'speaker_count': 0,
                'segment_count': 0,
                'segment_percent': 0.0,
                'total_duration': 0.0,
                'avg_duration': 0.0,
            },
        }
    
    # Group by speaker
    speaker_groups = valid_segments.groupby('speaker')
    
    speaker_stats = []
    for speaker, group in speaker_groups:
        segment_count = len(group)
        total_duration = group['duration'].sum()
        avg_duration = group['duration'].mean()
        segment_percent = round((segment_count / total_segments * 100), 2)
        
        speaker_stats.append({
            'speaker': speaker,
            'segment_count': segment_count,
            'segment_percent': segment_percent,
            'total_duration': round(float(total_duration), 2),
            'avg_duration': round(float(avg_duration), 2),
        })
    
    # Sort by segment count (descending)
    speaker_stats.sort(key=lambda x: x['segment_count'], reverse=True)
    
    # Split into top N and long tail
    top_speakers = speaker_stats[:top_n]
    long_tail_speakers = speaker_stats[top_n:]
    
    # Calculate long tail statistics
    if long_tail_speakers:
        long_tail_segment_count = sum(s['segment_count'] for s in long_tail_speakers)
        long_tail_total_duration = sum(s['total_duration'] for s in long_tail_speakers)
        long_tail_avg_duration = long_tail_total_duration / long_tail_segment_count if long_tail_segment_count > 0 else 0.0
        
        long_tail_stats = {
            'speaker_count': len(long_tail_speakers),
            'segment_count': long_tail_segment_count,
            'segment_percent': round((long_tail_segment_count / total_segments * 100), 2),
            'total_duration': round(long_tail_total_duration, 2),
            'avg_duration': round(long_tail_avg_duration, 2),
        }
    else:
        long_tail_stats = {
            'speaker_count': 0,
            'segment_count': 0,
            'segment_percent': 0.0,
            'total_duration': 0.0,
            'avg_duration': 0.0,
        }
    
    total_speakers = len(speaker_stats)
    
    logger.info(
        f"Speaker distribution for {total_segments} {segment_type}s: "
        f"{total_speakers} unique speakers, "
        f"top {len(top_speakers)} account for "
        f"{sum(s['segment_percent'] for s in top_speakers):.1f}% of segments"
    )
    
    return {
        'total_segments': total_segments,
        'total_speakers': total_speakers,
        'speaker_stats': speaker_stats,
        'top_speakers': top_speakers,
        'long_tail_stats': long_tail_stats,
    }


def calculate_series_balance(
    segments: pd.DataFrame,
    episodes: Optional[pd.DataFrame] = None,
    segment_type: str = "segment",
) -> Dict[str, Any]:
    """
    Calculate per-series balance statistics (FR-19).
    
    Analyzes distribution across different series (e.g., LOS vs SW).
    Requires series metadata in episodes DataFrame.
    
    Args:
        segments: DataFrame with segment data (must have episode_id)
        episodes: Optional DataFrame with episode metadata (should have episode_id and series columns)
        segment_type: Type of segment for logging
    
    Returns:
        Dictionary containing:
        - total_series: Number of unique series
        - series_stats: List of per-series statistics
        - series_balance_ratio: Ratio between largest and smallest series
        - series_available: Whether series metadata was available
    
    Example:
        >>> segments = pd.DataFrame({
        ...     'episode_id': ['EP1', 'EP1', 'EP2', 'EP2'],
        ...     'speaker': ['A', 'B', 'A', 'B'],
        ...     'duration': [25, 25, 25, 25]
        ... })
        >>> episodes = pd.DataFrame({
        ...     'episode_id': ['EP1', 'EP2'],
        ...     'series': ['LOS', 'SW']
        ... })
        >>> result = calculate_series_balance(segments, episodes)
        >>> result['total_series']
        2
    """
    if len(segments) == 0:
        logger.warning(f"No {segment_type}s provided for series balance")
        return {
            'series_available': False,
            'total_series': 0,
            'series_stats': [],
            'series_balance_ratio': None,
        }
    
    if 'episode_id' not in segments.columns:
        logger.error(f"episode_id column not found in {segment_type}s")
        return {
            'series_available': False,
            'total_series': 0,
            'series_stats': [],
            'series_balance_ratio': None,
        }
    
    # Check if series metadata is available
    if episodes is None or len(episodes) == 0:
        logger.info(f"No episode metadata provided for series balance calculation")
        return {
            'series_available': False,
            'total_series': 0,
            'series_stats': [],
            'series_balance_ratio': None,
        }
    
    if 'series' not in episodes.columns:
        logger.info(f"series column not found in episode metadata")
        return {
            'series_available': False,
            'total_series': 0,
            'series_stats': [],
            'series_balance_ratio': None,
        }
    
    # Calculate duration if not present
    segments_with_duration = segments.copy()
    if 'duration' not in segments_with_duration.columns:
        if 'start_time' in segments_with_duration.columns and 'end_time' in segments_with_duration.columns:
            segments_with_duration['duration'] = segments_with_duration['end_time'] - segments_with_duration['start_time']
        else:
            logger.warning(f"Cannot calculate duration for {segment_type}s (missing time columns)")
            segments_with_duration['duration'] = 0.0
    
    # Merge segments with episode series information
    segments_with_series = segments_with_duration.merge(
        episodes[['episode_id', 'series']], 
        on='episode_id', 
        how='left'
    )
    
    # Filter out segments without valid series
    valid_segments = segments_with_series[segments_with_series['series'].notna() & (segments_with_series['series'] != '')]
    
    if len(valid_segments) == 0:
        logger.warning(f"No segments with valid series metadata found")
        return {
            'series_available': True,
            'total_series': 0,
            'series_stats': [],
            'series_balance_ratio': None,
        }
    
    total_segments = len(segments)
    
    # Group by series
    series_groups = valid_segments.groupby('series')
    
    series_stats = []
    for series, group in series_groups:
        # Count unique episodes in this series
        episode_count = group['episode_id'].nunique()
        segment_count = len(group)
        total_duration = group['duration'].sum()
        avg_duration = group['duration'].mean()
        segment_percent = round((segment_count / total_segments * 100), 2)
        
        # Count unique speakers in this series
        if 'speaker' in group.columns:
            speaker_count = group['speaker'].nunique()
        else:
            speaker_count = 0
        
        series_stats.append({
            'series': series,
            'episode_count': episode_count,
            'segment_count': segment_count,
            'segment_percent': segment_percent,
            'speaker_count': speaker_count,
            'total_duration': round(float(total_duration), 2),
            'avg_duration': round(float(avg_duration), 2),
        })
    
    # Sort by segment count (descending)
    series_stats.sort(key=lambda x: x['segment_count'], reverse=True)
    
    # Calculate balance ratio (largest / smallest)
    if len(series_stats) > 1:
        largest_count = series_stats[0]['segment_count']
        smallest_count = series_stats[-1]['segment_count']
        balance_ratio = round(largest_count / smallest_count, 2) if smallest_count > 0 else None
    else:
        balance_ratio = 1.0 if len(series_stats) == 1 else None
    
    total_series = len(series_stats)
    
    logger.info(
        f"Series balance for {total_segments} {segment_type}s: "
        f"{total_series} unique series, "
        f"balance ratio={balance_ratio}"
    )
    
    return {
        'series_available': True,
        'total_series': total_series,
        'series_stats': series_stats,
        'series_balance_ratio': balance_ratio,
    }

