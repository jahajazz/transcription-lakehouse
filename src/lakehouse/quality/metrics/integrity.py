"""
Ordering and integrity metrics for quality assessment (Category C).

Checks timestamp monotonicity, integrity violations, and duplicate detection
per PRD requirements FR-14, FR-15, FR-16, FR-17.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import re

from lakehouse.logger import get_default_logger
from lakehouse.quality.thresholds import QualityThresholds, ThresholdViolation

logger = get_default_logger()

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logger.warning("rapidfuzz not available. Near-duplicate detection will use fallback method.")


def check_timestamp_monotonicity(
    segments: pd.DataFrame,
    segment_type: str = "segment",
) -> Dict[str, Any]:
    """
    Check timestamp monotonicity within episodes and speaker streams (FR-14).
    
    Validates that:
    1. Timestamps are monotonically increasing within each episode
    2. end_time[i] <= start_time[i+1] (no overlaps/regressions)
    3. Speaker-specific streams are also monotonic
    
    Args:
        segments: DataFrame with segment data (must have episode_id, speaker, start_time, end_time)
        segment_type: Type of segment for logging
    
    Returns:
        Dictionary containing:
        - total_segments: Total number of segments checked
        - episode_regression_count: Number of timestamp regressions at episode level
        - speaker_regression_count: Number of regressions within speaker streams
        - episode_regressions: List of regression details (episode_id, index, prev_end, curr_start)
        - speaker_regressions: List of regression details (episode_id, speaker, index, prev_end, curr_start)
    
    Example:
        >>> segments = pd.DataFrame({
        ...     'episode_id': ['EP1', 'EP1', 'EP1'],
        ...     'speaker': ['A', 'A', 'A'],
        ...     'start_time': [0, 50, 40],  # Regression at index 2
        ...     'end_time': [30, 70, 60]
        ... })
        >>> result = check_timestamp_monotonicity(segments)
        >>> result['episode_regression_count']
        1
    """
    if len(segments) == 0:
        logger.warning(f"No {segment_type}s provided for monotonicity check")
        return {
            'total_segments': 0,
            'episode_regression_count': 0,
            'speaker_regression_count': 0,
            'episode_regressions': [],
            'speaker_regressions': [],
        }
    
    # Validate required columns
    # Episode-level checks require: episode_id, start_time, end_time
    # Speaker-level checks also require: speaker (optional for beats)
    required_cols = ['episode_id', 'start_time', 'end_time']
    missing_cols = [col for col in required_cols if col not in segments.columns]
    if missing_cols:
        raise ValueError(
            f"Segments missing required columns for monotonicity check: {missing_cols}"
        )
    
    has_speaker_column = 'speaker' in segments.columns
    if not has_speaker_column:
        logger.debug(
            f"Speaker column not found in {segment_type}s, skipping speaker-level monotonicity checks"
        )
    
    episode_regressions = []
    speaker_regressions = []
    
    # Check episode-level monotonicity
    # Group by episode and check that segments are temporally ordered
    for episode_id, episode_segments in segments.groupby('episode_id'):
        # Sort by start_time to check ordering
        sorted_segments = episode_segments.sort_values('start_time').reset_index(drop=True)
        
        for i in range(len(sorted_segments) - 1):
            curr_end = sorted_segments.iloc[i]['end_time']
            next_start = sorted_segments.iloc[i + 1]['start_time']
            
            # Check if current segment's end is after next segment's start (regression)
            if curr_end > next_start:
                episode_regressions.append({
                    'episode_id': episode_id,
                    'segment_index': i,
                    'prev_end': round(float(curr_end), 2),
                    'curr_start': round(float(next_start), 2),
                    'gap': round(float(next_start - curr_end), 2),  # Negative for overlap
                })
    
    # Check speaker-level monotonicity (only if speaker column exists)
    # Group by episode and speaker to check speaker-specific streams
    if has_speaker_column:
        for (episode_id, speaker), speaker_segments in segments.groupby(['episode_id', 'speaker']):
            # Sort by start_time
            sorted_segments = speaker_segments.sort_values('start_time').reset_index(drop=True)
            
            for i in range(len(sorted_segments) - 1):
                curr_end = sorted_segments.iloc[i]['end_time']
                next_start = sorted_segments.iloc[i + 1]['start_time']
                
                # Check if current segment's end is after next segment's start (regression)
                if curr_end > next_start:
                    speaker_regressions.append({
                        'episode_id': episode_id,
                        'speaker': speaker,
                        'segment_index': i,
                        'prev_end': round(float(curr_end), 2),
                        'curr_start': round(float(next_start), 2),
                        'gap': round(float(next_start - curr_end), 2),  # Negative for overlap
                    })
    
    result = {
        'total_segments': len(segments),
        'episode_regression_count': len(episode_regressions),
        'speaker_regression_count': len(speaker_regressions),
        'episode_regressions': episode_regressions,
        'speaker_regressions': speaker_regressions,
    }
    
    if episode_regressions or speaker_regressions:
        logger.warning(
            f"Timestamp regressions detected in {segment_type}s: "
            f"{len(episode_regressions)} at episode level, "
            f"{len(speaker_regressions)} at speaker level"
        )
    else:
        logger.debug(
            f"All {len(segments)} {segment_type}s have monotonic timestamps"
        )
    
    return result


def detect_integrity_violations(
    segments: pd.DataFrame,
    segment_type: str = "segment",
) -> Dict[str, Any]:
    """
    Detect integrity violations in segments (FR-15).
    
    Checks for:
    1. Negative or zero duration (end_time <= start_time)
    2. Missing required fields (episode_id, speaker, text, timestamps)
    3. Invalid timestamp values (NaN, negative)
    
    Args:
        segments: DataFrame with segment data
        segment_type: Type of segment for logging
    
    Returns:
        Dictionary containing:
        - total_segments: Total number of segments
        - negative_duration_count: Number of segments with duration <= 0
        - missing_episode_id_count: Number missing episode_id
        - missing_speaker_count: Number missing speaker
        - missing_text_count: Number missing or empty text
        - missing_timestamps_count: Number with invalid timestamps
        - violations: List of violation details
    
    Example:
        >>> segments = pd.DataFrame({
        ...     'episode_id': ['EP1', 'EP1', 'EP1'],
        ...     'speaker': ['A', '', 'B'],
        ...     'text': ['Hello', 'World', ''],
        ...     'start_time': [0, 50, 100],
        ...     'end_time': [30, 40, 100]  # Second has negative duration
        ... })
        >>> result = detect_integrity_violations(segments)
        >>> result['negative_duration_count']
        1
    """
    if len(segments) == 0:
        logger.warning(f"No {segment_type}s provided for integrity check")
        return {
            'total_segments': 0,
            'negative_duration_count': 0,
            'zero_duration_count': 0,
            'missing_episode_id_count': 0,
            'missing_speaker_count': 0,
            'missing_text_count': 0,
            'missing_timestamps_count': 0,
            'invalid_timestamps_count': 0,
            'violations': [],
        }
    
    violations = []
    total_segments = len(segments)
    
    # Check for negative or zero durations
    negative_duration_indices = []
    zero_duration_indices = []
    
    if 'start_time' in segments.columns and 'end_time' in segments.columns:
        for idx, row in segments.iterrows():
            start = row.get('start_time')
            end = row.get('end_time')
            
            if pd.notna(start) and pd.notna(end):
                duration = end - start
                if duration < 0:
                    negative_duration_indices.append(idx)
                    violations.append({
                        'type': 'negative_duration',
                        'index': idx,
                        'segment_id': row.get('segment_id', row.get('span_id', row.get('beat_id', f'idx_{idx}'))),
                        'start_time': round(float(start), 2),
                        'end_time': round(float(end), 2),
                        'duration': round(float(duration), 2),
                    })
                elif duration == 0:
                    zero_duration_indices.append(idx)
                    violations.append({
                        'type': 'zero_duration',
                        'index': idx,
                        'segment_id': row.get('segment_id', row.get('span_id', row.get('beat_id', f'idx_{idx}'))),
                        'start_time': round(float(start), 2),
                        'end_time': round(float(end), 2),
                    })
    
    # Check for missing episode_id
    missing_episode_id_indices = []
    if 'episode_id' in segments.columns:
        missing_episode_id = segments['episode_id'].isna() | (segments['episode_id'] == '')
        missing_episode_id_indices = segments[missing_episode_id].index.tolist()
        for idx in missing_episode_id_indices:
            violations.append({
                'type': 'missing_episode_id',
                'index': idx,
                'segment_id': segments.loc[idx].get('segment_id', segments.loc[idx].get('span_id', segments.loc[idx].get('beat_id', f'idx_{idx}'))),
            })
    else:
        missing_episode_id_indices = list(range(len(segments)))
        logger.warning(f"episode_id column not found in {segment_type}s")
    
    # Check for missing speaker (column name varies by segment type)
    # - Spans have 'speaker' (single string)
    # - Beats have 'speakers_set' (array of strings)
    missing_speaker_indices = []
    
    if 'speaker' in segments.columns:
        # Spans: check for missing/empty speaker
        missing_speaker = segments['speaker'].isna() | (segments['speaker'] == '')
        missing_speaker_indices = segments[missing_speaker].index.tolist()
        for idx in missing_speaker_indices:
            violations.append({
                'type': 'missing_speaker',
                'index': idx,
                'segment_id': segments.loc[idx].get('segment_id', segments.loc[idx].get('span_id', segments.loc[idx].get('beat_id', f'idx_{idx}'))),
            })
    elif 'speakers_set' in segments.columns:
        # Beats: check for empty speakers_set array
        # Beats should have at least one speaker in the array
        for idx, row in segments.iterrows():
            speakers_set = row.get('speakers_set', [])
            if speakers_set is None or (isinstance(speakers_set, list) and len(speakers_set) == 0):
                missing_speaker_indices.append(idx)
                violations.append({
                    'type': 'missing_speaker',
                    'index': idx,
                    'segment_id': row.get('segment_id', row.get('span_id', row.get('beat_id', f'idx_{idx}'))),
                })
    else:
        # Neither column found - this is unexpected for spans, but don't count as violations
        logger.debug(f"Neither 'speaker' nor 'speakers_set' column found in {segment_type}s")
    
    # Check for missing or empty text
    missing_text_indices = []
    if 'text' in segments.columns:
        missing_text = segments['text'].isna() | (segments['text'].str.strip() == '')
        missing_text_indices = segments[missing_text].index.tolist()
        for idx in missing_text_indices:
            violations.append({
                'type': 'missing_text',
                'index': idx,
                'segment_id': segments.loc[idx].get('segment_id', segments.loc[idx].get('span_id', segments.loc[idx].get('beat_id', f'idx_{idx}'))),
            })
    else:
        missing_text_indices = list(range(len(segments)))
        logger.warning(f"text column not found in {segment_type}s")
    
    # Check for missing or invalid timestamps
    missing_timestamps_indices = []
    invalid_timestamps_indices = []
    
    if 'start_time' in segments.columns and 'end_time' in segments.columns:
        missing_start = segments['start_time'].isna()
        missing_end = segments['end_time'].isna()
        missing_timestamps = missing_start | missing_end
        missing_timestamps_indices = segments[missing_timestamps].index.tolist()
        
        for idx in missing_timestamps_indices:
            violations.append({
                'type': 'missing_timestamps',
                'index': idx,
                'segment_id': segments.loc[idx].get('segment_id', segments.loc[idx].get('span_id', segments.loc[idx].get('beat_id', f'idx_{idx}'))),
                'missing_start': bool(segments.loc[idx]['start_time'] is pd.NA or pd.isna(segments.loc[idx]['start_time'])),
                'missing_end': bool(segments.loc[idx]['end_time'] is pd.NA or pd.isna(segments.loc[idx]['end_time'])),
            })
        
        # Check for negative timestamps (among non-missing)
        valid_timestamps = ~missing_timestamps
        if valid_timestamps.any():
            negative_start = (segments.loc[valid_timestamps, 'start_time'] < 0)
            negative_end = (segments.loc[valid_timestamps, 'end_time'] < 0)
            invalid_timestamps = negative_start | negative_end
            invalid_timestamps_indices = segments[valid_timestamps][invalid_timestamps].index.tolist()
            
            for idx in invalid_timestamps_indices:
                violations.append({
                    'type': 'invalid_timestamps',
                    'index': idx,
                    'segment_id': segments.loc[idx].get('segment_id', segments.loc[idx].get('span_id', segments.loc[idx].get('beat_id', f'idx_{idx}'))),
                    'start_time': float(segments.loc[idx]['start_time']) if pd.notna(segments.loc[idx]['start_time']) else None,
                    'end_time': float(segments.loc[idx]['end_time']) if pd.notna(segments.loc[idx]['end_time']) else None,
                })
    else:
        missing_timestamps_indices = list(range(len(segments)))
        logger.warning(f"start_time/end_time columns not found in {segment_type}s")
    
    result = {
        'total_segments': total_segments,
        'negative_duration_count': len(negative_duration_indices),
        'zero_duration_count': len(zero_duration_indices),
        'missing_episode_id_count': len(missing_episode_id_indices),
        'missing_speaker_count': len(missing_speaker_indices),
        'missing_text_count': len(missing_text_indices),
        'missing_timestamps_count': len(missing_timestamps_indices),
        'invalid_timestamps_count': len(invalid_timestamps_indices),
        'violations': violations,
    }
    
    total_violations = sum([
        result['negative_duration_count'],
        result['zero_duration_count'],
        result['missing_episode_id_count'],
        result['missing_speaker_count'],
        result['missing_text_count'],
        result['missing_timestamps_count'],
        result['invalid_timestamps_count'],
    ])
    
    if total_violations > 0:
        logger.warning(
            f"Integrity violations detected in {segment_type}s: "
            f"{result['negative_duration_count']} negative durations, "
            f"{result['zero_duration_count']} zero durations, "
            f"{result['missing_episode_id_count']} missing episode IDs, "
            f"{result['missing_speaker_count']} missing speakers, "
            f"{result['missing_text_count']} missing text, "
            f"{result['missing_timestamps_count']} missing timestamps, "
            f"{result['invalid_timestamps_count']} invalid timestamps"
        )
    else:
        logger.debug(f"No integrity violations found in {total_segments} {segment_type}s")
    
    return result


def detect_duplicates(
    segments: pd.DataFrame,
    fuzzy_threshold: float = 0.95,
    segment_type: str = "segment",
    force_near_duplicate_check: bool = False,
    min_text_length: int = 10,
) -> Dict[str, Any]:
    """
    Detect exact and near-duplicate segments (FR-16, Task 4.6).
    
    Uses:
    1. Exact duplicates: Normalized text match (lowercase, stripped, no extra whitespace)
    2. Near-duplicates: Fuzzy string matching (e.g., rapidfuzz) with similarity >= threshold
    3. Minimum text length floor: Only check texts >= min_text_length chars (Task 4.6)
       to avoid flagging naturally common short texts like "yes", "no", "mm-hmm" as duplicates
    
    Args:
        segments: DataFrame with segment data (must have text column)
        fuzzy_threshold: Similarity threshold for near-duplicates (0.0-1.0)
        segment_type: Type of segment for logging
        force_near_duplicate_check: Force near-duplicate detection even for large datasets (>10k segments)
        min_text_length: Minimum text length to consider for duplicate detection (default: 10 chars)
    
    Returns:
        Dictionary containing:
        - total_segments: Total number of segments
        - exact_duplicate_count: Number of exact duplicates
        - exact_duplicate_percent: Percentage of exact duplicates
        - near_duplicate_count: Number of near-duplicates (excluding exacts)
        - near_duplicate_percent: Percentage of near-duplicates
        - duplicate_groups: List of duplicate groups with segment IDs
    
    Example:
        >>> segments = pd.DataFrame({
        ...     'segment_id': ['S1', 'S2', 'S3', 'S4'],
        ...     'text': ['Hello world', 'Hello world', 'Hello World!', 'Different']
        ... })
        >>> result = detect_duplicates(segments, fuzzy_threshold=0.95)
        >>> result['exact_duplicate_count']
        2
    """
    if len(segments) == 0:
        logger.warning(f"No {segment_type}s provided for duplicate detection")
        return {
            'total_segments': 0,
            'exact_duplicate_count': 0,
            'exact_duplicate_percent': 0.0,
            'near_duplicate_count': 0,
            'near_duplicate_percent': 0.0,
            'exact_duplicate_groups': [],
            'near_duplicate_groups': [],
        }
    
    if 'text' not in segments.columns:
        logger.error(f"text column not found in {segment_type}s for duplicate detection")
        return {
            'total_segments': len(segments),
            'exact_duplicate_count': 0,
            'exact_duplicate_percent': 0.0,
            'near_duplicate_count': 0,
            'near_duplicate_percent': 0.0,
            'exact_duplicate_groups': [],
            'near_duplicate_groups': [],
        }
    
    total_segments = len(segments)
    
    # Determine segment ID column
    id_col = None
    for col in ['segment_id', 'span_id', 'beat_id', 'section_id']:
        if col in segments.columns:
            id_col = col
            break
    
    if id_col is None:
        # Use index as ID
        segments = segments.copy()
        segments['_temp_id'] = segments.index
        id_col = '_temp_id'
    
    # Normalize text for exact duplicate detection
    def normalize_text(text: Any) -> str:
        """
        Normalize text: lowercase, strip, collapse whitespace.
        
        Args:
            text: Input text (can be string or any type)
        
        Returns:
            Normalized text string
        """
        if pd.isna(text):
            return ""
        # Convert to string, lowercase, strip
        normalized = str(text).lower().strip()
        # Collapse multiple whitespace to single space
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    segments_with_normalized = segments.copy()
    segments_with_normalized['_normalized_text'] = segments_with_normalized['text'].apply(normalize_text)
    
    # Filter by minimum text length (Task 4.6)
    # Only check texts >= min_text_length to avoid flagging naturally common short texts
    segments_with_normalized['_text_length'] = segments_with_normalized['_normalized_text'].apply(len)
    segments_to_check = segments_with_normalized[segments_with_normalized['_text_length'] >= min_text_length].copy()
    
    if len(segments_to_check) < len(segments_with_normalized):
        excluded_count = len(segments_with_normalized) - len(segments_to_check)
        logger.info(
            f"Excluded {excluded_count} {segment_type}s with text < {min_text_length} chars from duplicate detection"
        )
    
    # Detect exact duplicates
    # Use composite key to avoid flagging legitimate cross-episode repetition:
    # - Same text in different episodes (intros, outros, sponsor reads) = NOT duplicate
    # - Same text by different speakers = NOT duplicate
    # - Same text in same episode/speaker/time window = IS duplicate (needs dedup)
    exact_duplicate_groups = []
    exact_duplicate_indices = set()
    
    # Create composite key: (normalized_text, episode_id, speaker, time_bin)
    # Time bin: group by 5-minute windows to catch repeated segments within episode
    TIME_BIN_SECONDS = 300  # 5 minutes
    
    # Add composite key columns
    segments_to_check['_episode_id'] = segments_with_normalized.loc[segments_to_check.index, 'episode_id']
    
    # Speaker handling: use speaker if available, otherwise "NO_SPEAKER"
    if 'speaker' in segments_with_normalized.columns:
        segments_to_check['_speaker'] = segments_with_normalized.loc[segments_to_check.index, 'speaker'].fillna('NO_SPEAKER')
    else:
        segments_to_check['_speaker'] = 'NO_SPEAKER'
    
    # Time bin: divide start_time into 5-minute bins
    if 'start_time' in segments_with_normalized.columns:
        segments_to_check['_time_bin'] = (
            segments_with_normalized.loc[segments_to_check.index, 'start_time'] // TIME_BIN_SECONDS
        ).astype(int)
    else:
        segments_to_check['_time_bin'] = 0
    
    # Group by composite key (normalized_text, episode_id, speaker, time_bin)
    composite_groups = segments_to_check.groupby([
        '_normalized_text', 
        '_episode_id', 
        '_speaker', 
        '_time_bin'
    ])
    
    for (normalized_text, episode_id, speaker, time_bin), group in composite_groups:
        if len(group) > 1 and normalized_text != "":  # Skip empty text
            group_ids = group[id_col].tolist()
            exact_duplicate_groups.append({
                'type': 'exact',
                'count': len(group),
                'segment_ids': group_ids,
                'text_preview': segments_with_normalized.loc[group.index[0], 'text'][:100],  # First 100 chars
                'episode_id': episode_id,
                'speaker': speaker,
            })
            # Mark all but first as duplicates
            exact_duplicate_indices.update(group.index[1:])
    
    exact_duplicate_count = len(exact_duplicate_indices)
    exact_duplicate_percent = round((exact_duplicate_count / total_segments * 100), 2)
    
    logger.debug(
        f"Duplicate detection using composite key (text + episode_id + speaker + time_bin): "
        f"found {exact_duplicate_count} duplicates in {len(exact_duplicate_groups)} groups"
    )
    
    # Detect near-duplicates (excluding exact duplicates)
    near_duplicate_groups = []
    near_duplicate_indices = set()
    
    if not RAPIDFUZZ_AVAILABLE:
        logger.info(f"rapidfuzz not available, skipping near-duplicate detection for {segment_type}s")
    else:
        # Skip near-duplicate detection for large datasets unless forced
        if len(segments_to_check) > 10000 and not force_near_duplicate_check:
            logger.warning(
                f"Skipping near-duplicate detection for {len(segments_to_check)} {segment_type}s "
                f"(too slow for O(n²) comparison). Use --force-duplicate-check to override."
            )
        else:
            if len(segments_to_check) > 10000:
                logger.warning(
                    f"Forcing near-duplicate detection for {len(segments_to_check)} {segment_type}s. "
                    f"This may take a while (O(n²) complexity)..."
                )
            # Only check segments that are not exact duplicates (and meet min length)
            non_exact_duplicate_mask = ~segments_to_check.index.isin(exact_duplicate_indices)
            unique_segments = segments_to_check[non_exact_duplicate_mask].copy()
            
            if len(unique_segments) > 1:
                # For efficiency, only compare unique normalized texts
                unique_texts = unique_segments['_normalized_text'].unique()
                
                # Build groups of near-duplicates
                processed_texts = set()
                
                for i, text1 in enumerate(unique_texts):
                    if text1 in processed_texts or text1 == "":
                        continue
                    
                    similar_group = [text1]
                    processed_texts.add(text1)
                    
                    # Compare with remaining texts
                    for text2 in unique_texts[i+1:]:
                        if text2 in processed_texts or text2 == "":
                            continue
                        
                        # Calculate similarity ratio
                        similarity = fuzz.ratio(text1, text2) / 100.0
                        
                        if similarity >= fuzzy_threshold:
                            similar_group.append(text2)
                            processed_texts.add(text2)
                    
                    # If group has more than one text, record it
                    if len(similar_group) > 1:
                        # Get all segment IDs for these texts
                        group_mask = unique_segments['_normalized_text'].isin(similar_group)
                        group_segments = unique_segments[group_mask]
                        group_ids = group_segments[id_col].tolist()
                        
                        near_duplicate_groups.append({
                            'type': 'near',
                            'count': len(group_segments),
                            'segment_ids': group_ids,
                            'text_preview': group_segments.iloc[0]['text'][:100],
                            'similarity_threshold': fuzzy_threshold,
                        })
                        
                        # Mark all but first as near-duplicates
                        near_duplicate_indices.update(group_segments.index[1:])
    
    near_duplicate_count = len(near_duplicate_indices)
    near_duplicate_percent = round((near_duplicate_count / total_segments * 100), 2)
    
    result = {
        'total_segments': total_segments,
        'segments_checked': len(segments_to_check),
        'min_text_length': min_text_length,
        'exact_duplicate_count': exact_duplicate_count,
        'exact_duplicate_percent': exact_duplicate_percent,
        'near_duplicate_count': near_duplicate_count,
        'near_duplicate_percent': near_duplicate_percent,
        'exact_duplicate_groups': exact_duplicate_groups,
        'near_duplicate_groups': near_duplicate_groups,
    }
    
    if exact_duplicate_count > 0 or near_duplicate_count > 0:
        logger.warning(
            f"Duplicates detected in {segment_type}s: "
            f"{exact_duplicate_count} exact ({exact_duplicate_percent}%), "
            f"{near_duplicate_count} near ({near_duplicate_percent}%)"
        )
    else:
        logger.debug(f"No duplicates found in {total_segments} {segment_type}s")
    
    return result


def validate_integrity_thresholds(
    integrity_metrics: Dict[str, Any],
    thresholds: QualityThresholds,
    segment_type: str = "segment",
) -> List[ThresholdViolation]:
    """
    Validate integrity metrics against thresholds (FR-17).
    
    Checks:
    - Timestamp regressions: 0 allowed (timestamp_regressions_max)
    - Negative durations: 0 allowed (negative_duration_max)
    - Exact duplicates: <= 1% (exact_duplicate_max_percent)
    - Near duplicates: <= 3% (near_duplicate_max_percent)
    
    Args:
        integrity_metrics: Metrics from monotonicity, violations, and duplicate checks
        thresholds: Quality thresholds configuration
        segment_type: Type of segment for logging
    
    Returns:
        List of threshold violations
    
    Example:
        >>> metrics = {
        ...     'episode_regression_count': 5,
        ...     'negative_duration_count': 2,
        ...     'exact_duplicate_percent': 2.5
        ... }
        >>> thresholds = QualityThresholds(timestamp_regressions_max=0)
        >>> violations = validate_integrity_thresholds(metrics, thresholds)
        >>> len(violations)
        3
    """
    violations = []
    
    # Check timestamp regressions (episode level)
    episode_regression_count = integrity_metrics.get('episode_regression_count', 0)
    if episode_regression_count > thresholds.timestamp_regressions_max:
        violations.append(ThresholdViolation(
            threshold_name='timestamp_regressions_max',
            expected=f'<= {thresholds.timestamp_regressions_max}',
            actual=str(episode_regression_count),
            severity='error',
            message=(
                f'{segment_type.capitalize()} timestamp regressions ({episode_regression_count}) '
                f'exceed maximum allowed ({thresholds.timestamp_regressions_max}). '
                f'Segments have overlapping or out-of-order timestamps at episode level.'
            ),
        ))
        logger.warning(
            f'{segment_type.capitalize()} timestamp regression violation: '
            f'{episode_regression_count} > {thresholds.timestamp_regressions_max}'
        )
    
    # Check speaker-level timestamp regressions
    speaker_regression_count = integrity_metrics.get('speaker_regression_count', 0)
    if speaker_regression_count > thresholds.timestamp_regressions_max:
        violations.append(ThresholdViolation(
            threshold_name='speaker_timestamp_regressions_max',
            expected=f'<= {thresholds.timestamp_regressions_max}',
            actual=str(speaker_regression_count),
            severity='error',
            message=(
                f'{segment_type.capitalize()} speaker-level timestamp regressions '
                f'({speaker_regression_count}) exceed maximum allowed '
                f'({thresholds.timestamp_regressions_max}). '
                f'Speaker streams have overlapping or out-of-order timestamps.'
            ),
        ))
        logger.warning(
            f'{segment_type.capitalize()} speaker timestamp regression violation: '
            f'{speaker_regression_count} > {thresholds.timestamp_regressions_max}'
        )
    
    # Check negative durations
    negative_duration_count = integrity_metrics.get('negative_duration_count', 0)
    if negative_duration_count > thresholds.negative_duration_max:
        violations.append(ThresholdViolation(
            threshold_name='negative_duration_max',
            expected=f'<= {thresholds.negative_duration_max}',
            actual=str(negative_duration_count),
            severity='error',
            message=(
                f'{segment_type.capitalize()} negative durations ({negative_duration_count}) '
                f'exceed maximum allowed ({thresholds.negative_duration_max}). '
                f'Segments have end_time < start_time.'
            ),
        ))
        logger.warning(
            f'{segment_type.capitalize()} negative duration violation: '
            f'{negative_duration_count} > {thresholds.negative_duration_max}'
        )
    
    # Check zero durations (use same threshold as negative)
    zero_duration_count = integrity_metrics.get('zero_duration_count', 0)
    if zero_duration_count > thresholds.negative_duration_max:
        violations.append(ThresholdViolation(
            threshold_name='zero_duration_max',
            expected=f'<= {thresholds.negative_duration_max}',
            actual=str(zero_duration_count),
            severity='warning',  # Zero durations are less critical than negative
            message=(
                f'{segment_type.capitalize()} zero durations ({zero_duration_count}) '
                f'exceed maximum allowed ({thresholds.negative_duration_max}). '
                f'Segments have end_time == start_time.'
            ),
        ))
        logger.warning(
            f'{segment_type.capitalize()} zero duration violation: '
            f'{zero_duration_count} > {thresholds.negative_duration_max}'
        )
    
    # Check exact duplicates
    exact_duplicate_percent = integrity_metrics.get('exact_duplicate_percent', 0.0)
    if exact_duplicate_percent > thresholds.exact_duplicate_max_percent:
        exact_duplicate_count = integrity_metrics.get('exact_duplicate_count', 0)
        total_segments = integrity_metrics.get('total_segments', 0)
        
        violations.append(ThresholdViolation(
            threshold_name='exact_duplicate_max_percent',
            expected=f'<= {thresholds.exact_duplicate_max_percent}%',
            actual=f'{exact_duplicate_percent}%',
            severity='error',
            message=(
                f'{segment_type.capitalize()} exact duplicates ({exact_duplicate_percent}%) '
                f'exceed maximum allowed ({thresholds.exact_duplicate_max_percent}%). '
                f'{exact_duplicate_count}/{total_segments} segments have identical normalized text.'
            ),
        ))
        logger.warning(
            f'{segment_type.capitalize()} exact duplicate violation: '
            f'{exact_duplicate_percent}% > {thresholds.exact_duplicate_max_percent}%'
        )
    
    # Check near duplicates
    near_duplicate_percent = integrity_metrics.get('near_duplicate_percent', 0.0)
    if near_duplicate_percent > thresholds.near_duplicate_max_percent:
        near_duplicate_count = integrity_metrics.get('near_duplicate_count', 0)
        total_segments = integrity_metrics.get('total_segments', 0)
        fuzzy_threshold = integrity_metrics.get('fuzzy_threshold', 0.95)
        
        violations.append(ThresholdViolation(
            threshold_name='near_duplicate_max_percent',
            expected=f'<= {thresholds.near_duplicate_max_percent}%',
            actual=f'{near_duplicate_percent}%',
            severity='warning',  # Near-duplicates are less critical than exact
            message=(
                f'{segment_type.capitalize()} near-duplicates ({near_duplicate_percent}%) '
                f'exceed maximum allowed ({thresholds.near_duplicate_max_percent}%). '
                f'{near_duplicate_count}/{total_segments} segments have similar text '
                f'(similarity >= {fuzzy_threshold}).'
            ),
        ))
        logger.warning(
            f'{segment_type.capitalize()} near-duplicate violation: '
            f'{near_duplicate_percent}% > {thresholds.near_duplicate_max_percent}%'
        )
    
    # Check for missing fields (optional - report if significant)
    missing_episode_id_count = integrity_metrics.get('missing_episode_id_count', 0)
    missing_speaker_count = integrity_metrics.get('missing_speaker_count', 0)
    missing_text_count = integrity_metrics.get('missing_text_count', 0)
    
    if missing_episode_id_count > 0:
        violations.append(ThresholdViolation(
            threshold_name='missing_episode_id',
            expected='0',
            actual=str(missing_episode_id_count),
            severity='error',
            message=(
                f'{missing_episode_id_count} {segment_type}(s) are missing episode_id. '
                f'All segments must have a valid episode_id.'
            ),
        ))
    
    if missing_speaker_count > 0:
        violations.append(ThresholdViolation(
            threshold_name='missing_speaker',
            expected='0',
            actual=str(missing_speaker_count),
            severity='warning',
            message=(
                f'{missing_speaker_count} {segment_type}(s) are missing speaker. '
                f'All segments should have a valid speaker identifier.'
            ),
        ))
    
    if missing_text_count > 0:
        violations.append(ThresholdViolation(
            threshold_name='missing_text',
            expected='0',
            actual=str(missing_text_count),
            severity='error',
            message=(
                f'{missing_text_count} {segment_type}(s) are missing or have empty text. '
                f'All segments must have valid text content.'
            ),
        ))
    
    if violations:
        logger.info(
            f'Integrity threshold validation complete for {segment_type}s: '
            f'{len(violations)} violations found'
        )
    else:
        logger.debug(
            f'All integrity thresholds passed for {segment_type}s'
        )
    
    return violations

