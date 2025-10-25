"""
Coverage and count metrics for quality assessment (Category A).

Calculates episode coverage, gaps, overlaps, and segment counts
per PRD requirements FR-7, FR-8, FR-9.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from lakehouse.logger import get_default_logger
from lakehouse.quality.thresholds import QualityThresholds, ThresholdViolation


logger = get_default_logger()


def calculate_episode_coverage(
    episodes: pd.DataFrame,
    spans: Optional[pd.DataFrame] = None,
    beats: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Calculate coverage metrics for episodes (FR-7).
    
    For each episode, calculates:
    - Total episode duration (from metadata)
    - Sum of span durations and sum of beat durations
    - Coverage percentage: (sum_durations / episode_duration) * 100
    - Total span count and total beat count
    
    Args:
        episodes: DataFrame with episode metadata (episode_id, duration, etc.)
        spans: DataFrame with span data (optional)
        beats: DataFrame with beat data (optional)
    
    Returns:
        Dictionary containing:
        - per_episode: List of dicts with per-episode metrics
        - global: Dict with global aggregate metrics
    
    Example:
        >>> episodes = pd.DataFrame({
        ...     'episode_id': ['EP001', 'EP002'],
        ...     'duration_minutes': [45.0, 60.0]
        ... })
        >>> spans = pd.DataFrame({
        ...     'episode_id': ['EP001', 'EP001', 'EP002'],
        ...     'start_time': [0.0, 100.0, 0.0],
        ...     'end_time': [50.0, 150.0, 100.0],
        ... })
        >>> metrics = calculate_episode_coverage(episodes, spans)
        >>> metrics['global']['total_episodes']
        2
    """
    logger.info("Calculating episode coverage metrics...")
    
    # Ensure episode duration is in seconds
    if 'duration_seconds' not in episodes.columns:
        if 'duration_minutes' in episodes.columns:
            episodes = episodes.copy()
            episodes['duration_seconds'] = episodes['duration_minutes'] * 60
        elif 'duration' in episodes.columns:
            episodes = episodes.copy()
            episodes['duration_seconds'] = episodes['duration']
        else:
            raise ValueError("Episodes DataFrame must have 'duration_seconds', 'duration_minutes', or 'duration' column")
    
    per_episode_metrics = []
    
    for _, episode in episodes.iterrows():
        episode_id = episode.get('episode_id')
        episode_duration = episode['duration_seconds']
        
        metrics = {
            'episode_id': episode_id,
            'episode_duration_seconds': round(episode_duration, 2),
        }
        
        # Calculate span metrics
        if spans is not None and len(spans) > 0:
            episode_spans = spans[spans['episode_id'] == episode_id]
            
            if len(episode_spans) > 0:
                # Calculate durations
                if 'duration' in episode_spans.columns:
                    span_durations = episode_spans['duration'].sum()
                else:
                    span_durations = (episode_spans['end_time'] - episode_spans['start_time']).sum()
                
                span_count = len(episode_spans)
                span_coverage_pct = (span_durations / episode_duration * 100) if episode_duration > 0 else 0.0
                
                metrics.update({
                    'span_count': span_count,
                    'span_total_duration_seconds': round(span_durations, 2),
                    'span_coverage_percent': round(span_coverage_pct, 2),
                })
            else:
                metrics.update({
                    'span_count': 0,
                    'span_total_duration_seconds': 0.0,
                    'span_coverage_percent': 0.0,
                })
        else:
            metrics.update({
                'span_count': None,
                'span_total_duration_seconds': None,
                'span_coverage_percent': None,
            })
        
        # Calculate beat metrics
        if beats is not None and len(beats) > 0:
            episode_beats = beats[beats['episode_id'] == episode_id]
            
            if len(episode_beats) > 0:
                # Calculate durations
                if 'duration' in episode_beats.columns:
                    beat_durations = episode_beats['duration'].sum()
                else:
                    beat_durations = (episode_beats['end_time'] - episode_beats['start_time']).sum()
                
                beat_count = len(episode_beats)
                beat_coverage_pct = (beat_durations / episode_duration * 100) if episode_duration > 0 else 0.0
                
                metrics.update({
                    'beat_count': beat_count,
                    'beat_total_duration_seconds': round(beat_durations, 2),
                    'beat_coverage_percent': round(beat_coverage_pct, 2),
                })
            else:
                metrics.update({
                    'beat_count': 0,
                    'beat_total_duration_seconds': 0.0,
                    'beat_coverage_percent': 0.0,
                })
        else:
            metrics.update({
                'beat_count': None,
                'beat_total_duration_seconds': None,
                'beat_coverage_percent': None,
            })
        
        per_episode_metrics.append(metrics)
    
    # Calculate global aggregates
    global_metrics = {
        'total_episodes': len(episodes),
        'total_episode_duration_seconds': round(episodes['duration_seconds'].sum(), 2),
    }
    
    # Global span metrics
    if spans is not None and len(spans) > 0:
        if 'duration' in spans.columns:
            total_span_duration = spans['duration'].sum()
        else:
            total_span_duration = (spans['end_time'] - spans['start_time']).sum()
        
        global_span_coverage = (
            (total_span_duration / global_metrics['total_episode_duration_seconds'] * 100)
            if global_metrics['total_episode_duration_seconds'] > 0 else 0.0
        )
        
        global_metrics.update({
            'total_spans': len(spans),
            'total_span_duration_seconds': round(total_span_duration, 2),
            'global_span_coverage_percent': round(global_span_coverage, 2),
        })
    else:
        global_metrics.update({
            'total_spans': 0,
            'total_span_duration_seconds': 0.0,
            'global_span_coverage_percent': 0.0,
        })
    
    # Global beat metrics
    if beats is not None and len(beats) > 0:
        if 'duration' in beats.columns:
            total_beat_duration = beats['duration'].sum()
        else:
            total_beat_duration = (beats['end_time'] - beats['start_time']).sum()
        
        global_beat_coverage = (
            (total_beat_duration / global_metrics['total_episode_duration_seconds'] * 100)
            if global_metrics['total_episode_duration_seconds'] > 0 else 0.0
        )
        
        global_metrics.update({
            'total_beats': len(beats),
            'total_beat_duration_seconds': round(total_beat_duration, 2),
            'global_beat_coverage_percent': round(global_beat_coverage, 2),
        })
    else:
        global_metrics.update({
            'total_beats': 0,
            'total_beat_duration_seconds': 0.0,
            'global_beat_coverage_percent': 0.0,
        })
    
    logger.info(
        f"Coverage metrics calculated: {global_metrics['total_episodes']} episodes, "
        f"{global_metrics.get('total_spans', 0)} spans, {global_metrics.get('total_beats', 0)} beats"
    )
    
    return {
        'per_episode': per_episode_metrics,
        'global': global_metrics,
    }


def detect_gaps_and_overlaps(
    segments: pd.DataFrame,
    episode_id: str,
    episode_duration: float,
) -> Dict[str, Any]:
    """
    Detect gaps and overlaps in segment timeline (FR-8).
    
    Identifies time ranges where:
    - Gaps: No segments cover the time range
    - Overlaps: Multiple segments cover the same time range
    
    Args:
        segments: DataFrame with segment data (must have start_time, end_time)
        episode_id: Episode ID being analyzed
        episode_duration: Total episode duration in seconds
    
    Returns:
        Dictionary containing:
        - gap_count: Number of gaps
        - gap_total_duration: Total duration of gaps in seconds
        - gap_percent: Percentage of episode with gaps
        - overlap_count: Number of overlaps
        - overlap_total_duration: Total duration of overlaps in seconds
        - overlap_percent: Percentage of episode with overlaps
        - gaps: List of gap intervals (start, end, duration)
        - overlaps: List of overlap intervals (start, end, duration, segment_count)
    
    Example:
        >>> segments = pd.DataFrame({
        ...     'start_time': [0.0, 100.0, 150.0],
        ...     'end_time': [50.0, 200.0, 250.0]
        ... })
        >>> result = detect_gaps_and_overlaps(segments, 'EP001', 300.0)
        >>> result['gap_count']
        2
    """
    if len(segments) == 0:
        # No segments = entire episode is a gap
        return {
            'gap_count': 1,
            'gap_total_duration': round(episode_duration, 2),
            'gap_percent': 100.0,
            'overlap_count': 0,
            'overlap_total_duration': 0.0,
            'overlap_percent': 0.0,
            'gaps': [{'start': 0.0, 'end': episode_duration, 'duration': episode_duration}],
            'overlaps': [],
        }
    
    # Sort segments by start time
    segments_sorted = segments.sort_values('start_time').reset_index(drop=True)
    
    # Detect gaps
    gaps = []
    
    # Check for gap at the beginning
    first_start = segments_sorted.iloc[0]['start_time']
    if first_start > 0:
        gap_duration = first_start
        gaps.append({
            'start': 0.0,
            'end': round(first_start, 2),
            'duration': round(gap_duration, 2),
        })
    
    # Check for gaps between segments
    for i in range(len(segments_sorted) - 1):
        current_end = segments_sorted.iloc[i]['end_time']
        next_start = segments_sorted.iloc[i + 1]['start_time']
        
        if next_start > current_end:
            gap_duration = next_start - current_end
            gaps.append({
                'start': round(current_end, 2),
                'end': round(next_start, 2),
                'duration': round(gap_duration, 2),
            })
    
    # Check for gap at the end
    last_end = segments_sorted.iloc[-1]['end_time']
    if last_end < episode_duration:
        gap_duration = episode_duration - last_end
        gaps.append({
            'start': round(last_end, 2),
            'end': round(episode_duration, 2),
            'duration': round(gap_duration, 2),
        })
    
    gap_total_duration = sum(g['duration'] for g in gaps)
    gap_percent = (gap_total_duration / episode_duration * 100) if episode_duration > 0 else 0.0
    
    # Detect overlaps using interval merging approach
    overlaps = []
    
    # Create events for segment starts and ends
    events = []
    for idx, row in segments_sorted.iterrows():
        events.append((row['start_time'], 'start', idx))
        events.append((row['end_time'], 'end', idx))
    
    # Sort events by time, with 'start' events before 'end' events at same time
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'start' else 1))
    
    # Track active segments and detect overlaps
    active_segments = set()
    overlap_start = None
    overlap_segment_count = 0
    
    for time, event_type, seg_idx in events:
        if event_type == 'start':
            active_segments.add(seg_idx)
            
            # If we now have 2+ active segments, we're in an overlap
            if len(active_segments) >= 2:
                if overlap_start is None:
                    overlap_start = time
                    overlap_segment_count = len(active_segments)
                else:
                    # Update segment count if it increased
                    overlap_segment_count = max(overlap_segment_count, len(active_segments))
        
        else:  # event_type == 'end'
            # Before removing, check if we need to close an overlap
            if len(active_segments) >= 2 and overlap_start is not None:
                # End current overlap region
                overlap_duration = time - overlap_start
                if overlap_duration > 0:
                    overlaps.append({
                        'start': round(overlap_start, 2),
                        'end': round(time, 2),
                        'duration': round(overlap_duration, 2),
                        'segment_count': overlap_segment_count,
                    })
                overlap_start = None
                overlap_segment_count = 0
            
            active_segments.remove(seg_idx)
            
            # If still in overlap after removal, start new overlap region
            if len(active_segments) >= 2:
                overlap_start = time
                overlap_segment_count = len(active_segments)
    
    overlap_total_duration = sum(o['duration'] for o in overlaps)
    overlap_percent = (overlap_total_duration / episode_duration * 100) if episode_duration > 0 else 0.0
    
    return {
        'gap_count': len(gaps),
        'gap_total_duration': round(gap_total_duration, 2),
        'gap_percent': round(gap_percent, 2),
        'overlap_count': len(overlaps),
        'overlap_total_duration': round(overlap_total_duration, 2),
        'overlap_percent': round(overlap_percent, 2),
        'gaps': gaps,
        'overlaps': overlaps,
    }


def validate_coverage_thresholds(
    coverage_metrics: Dict[str, Any],
    thresholds: QualityThresholds,
) -> List[ThresholdViolation]:
    """
    Validate coverage metrics against thresholds (FR-9).
    
    Checks:
    - Coverage ≥ coverage_min (default 95%)
    - Gaps ≤ gap_max_percent (default 2%)
    - Overlaps ≤ overlap_max_percent (default 2%)
    
    Args:
        coverage_metrics: Coverage metrics from calculate_episode_coverage
        thresholds: Quality thresholds configuration
    
    Returns:
        List of threshold violations
    
    Example:
        >>> metrics = {
        ...     'per_episode': [
        ...         {'episode_id': 'EP001', 'span_coverage_percent': 92.0,
        ...          'gap_percent': 5.0, 'overlap_percent': 3.0}
        ...     ]
        ... }
        >>> thresholds = QualityThresholds(coverage_min=95.0, gap_max_percent=2.0)
        >>> violations = validate_coverage_thresholds(metrics, thresholds)
        >>> len(violations)
        3
    """
    violations = []
    
    per_episode = coverage_metrics.get('per_episode', [])
    global_metrics = coverage_metrics.get('global', {})
    
    # Check global span coverage
    if 'global_span_coverage_percent' in global_metrics:
        span_coverage = global_metrics['global_span_coverage_percent']
        if span_coverage is not None and span_coverage < thresholds.coverage_min:
            violations.append(ThresholdViolation(
                threshold_name='global_span_coverage_min',
                expected=f'>= {thresholds.coverage_min}%',
                actual=f'{span_coverage}%',
                severity='error',
                message=f'Global span coverage {span_coverage}% is below minimum threshold of {thresholds.coverage_min}%',
            ))
    
    # Check global beat coverage
    if 'global_beat_coverage_percent' in global_metrics:
        beat_coverage = global_metrics['global_beat_coverage_percent']
        if beat_coverage is not None and beat_coverage < thresholds.coverage_min:
            violations.append(ThresholdViolation(
                threshold_name='global_beat_coverage_min',
                expected=f'>= {thresholds.coverage_min}%',
                actual=f'{beat_coverage}%',
                severity='error',
                message=f'Global beat coverage {beat_coverage}% is below minimum threshold of {thresholds.coverage_min}%',
            ))
    
    # Check per-episode coverage, gaps, and overlaps
    episodes_with_low_coverage = []
    episodes_with_high_gaps = []
    episodes_with_high_overlaps = []
    
    for episode in per_episode:
        episode_id = episode.get('episode_id', 'unknown')
        
        # Check span coverage for this episode
        span_coverage = episode.get('span_coverage_percent')
        if span_coverage is not None and span_coverage < thresholds.coverage_min:
            episodes_with_low_coverage.append(f"{episode_id} ({span_coverage}%)")
        
        # Check beat coverage for this episode
        beat_coverage = episode.get('beat_coverage_percent')
        if beat_coverage is not None and beat_coverage < thresholds.coverage_min:
            episodes_with_low_coverage.append(f"{episode_id} beats ({beat_coverage}%)")
        
        # Check gap percentage
        gap_percent = episode.get('gap_percent')
        if gap_percent is not None and gap_percent > thresholds.gap_max_percent:
            episodes_with_high_gaps.append(f"{episode_id} ({gap_percent}%)")
        
        # Check overlap percentage
        overlap_percent = episode.get('overlap_percent')
        if overlap_percent is not None and overlap_percent > thresholds.overlap_max_percent:
            episodes_with_high_overlaps.append(f"{episode_id} ({overlap_percent}%)")
    
    # Add per-episode violations
    if episodes_with_low_coverage:
        # Limit to first 10 episodes in message for readability
        episode_list = ', '.join(episodes_with_low_coverage[:10])
        if len(episodes_with_low_coverage) > 10:
            episode_list += f' ... and {len(episodes_with_low_coverage) - 10} more'
        
        violations.append(ThresholdViolation(
            threshold_name='episode_coverage_min',
            expected=f'>= {thresholds.coverage_min}%',
            actual=f'{len(episodes_with_low_coverage)} episodes below threshold',
            severity='warning',
            message=f'{len(episodes_with_low_coverage)} episode(s) have coverage below {thresholds.coverage_min}%: {episode_list}',
        ))
    
    if episodes_with_high_gaps:
        episode_list = ', '.join(episodes_with_high_gaps[:10])
        if len(episodes_with_high_gaps) > 10:
            episode_list += f' ... and {len(episodes_with_high_gaps) - 10} more'
        
        violations.append(ThresholdViolation(
            threshold_name='episode_gaps_max',
            expected=f'<= {thresholds.gap_max_percent}%',
            actual=f'{len(episodes_with_high_gaps)} episodes above threshold',
            severity='warning',
            message=f'{len(episodes_with_high_gaps)} episode(s) have gaps exceeding {thresholds.gap_max_percent}%: {episode_list}',
        ))
    
    if episodes_with_high_overlaps:
        episode_list = ', '.join(episodes_with_high_overlaps[:10])
        if len(episodes_with_high_overlaps) > 10:
            episode_list += f' ... and {len(episodes_with_high_overlaps) - 10} more'
        
        violations.append(ThresholdViolation(
            threshold_name='episode_overlaps_max',
            expected=f'<= {thresholds.overlap_max_percent}%',
            actual=f'{len(episodes_with_high_overlaps)} episodes above threshold',
            severity='warning',
            message=f'{len(episodes_with_high_overlaps)} episode(s) have overlaps exceeding {thresholds.overlap_max_percent}%: {episode_list}',
        ))
    
    return violations

