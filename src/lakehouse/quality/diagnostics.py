"""
Diagnostic utilities and outlier detection for quality assessment (Category G).

Identifies interesting segments for manual review and exports diagnostic CSV files
per PRD requirements FR-32, FR-33, FR-34.
"""

import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

from lakehouse.logger import get_default_logger


logger = get_default_logger()


def identify_outliers(
    segments_df: pd.DataFrame,
    embeddings: Optional[np.ndarray] = None,
    neighbor_indices: Optional[np.ndarray] = None,
    neighbor_similarities: Optional[np.ndarray] = None,
    outlier_count: int = 20,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Identify outlier segments for manual review (FR-32, subtask 4.1.1).
    
    Finds segments that are statistically interesting:
    - Longest: Top N segments by duration
    - Shortest: Bottom N segments by duration
    - Most Isolated: Segments with lowest mean neighbor similarity (if embeddings available)
    - Most Hubby: Segments with highest mean neighbor similarity (if embeddings available)
    
    Args:
        segments_df: DataFrame with segment data (must have duration column)
        embeddings: Optional embedding matrix aligned with segments_df
        neighbor_indices: Optional precomputed neighbor indices from find_top_k_neighbors()
        neighbor_similarities: Optional precomputed neighbor similarities
        outlier_count: Number of outliers to return per category (default: 20)
    
    Returns:
        Dictionary with keys 'longest', 'shortest', 'most_isolated', 'most_hubby',
        each containing a list of segment dictionaries with metadata
    
    Example:
        >>> outliers = identify_outliers(segments_df, embeddings, outlier_count=10)
        >>> len(outliers['longest'])
        10
    """
    logger.info(f"Identifying outliers: finding top {outlier_count} per category...")
    
    outliers = {
        'longest': [],
        'shortest': [],
        'most_isolated': [],
        'most_hubby': [],
    }
    
    # Check for duration column
    duration_col = None
    for col_name in ['duration', 'duration_seconds', 'length']:
        if col_name in segments_df.columns:
            duration_col = col_name
            break
    
    if duration_col is None:
        logger.warning("No duration column found. Cannot identify length outliers.")
        return outliers
    
    # Identify longest segments
    longest_segments = segments_df.nlargest(outlier_count, duration_col)
    for idx, row in longest_segments.iterrows():
        outliers['longest'].append({
            'segment_idx': int(idx),
            'segment_id': row.get('segment_id', f'idx_{idx}'),
            'episode_id': row.get('episode_id', 'unknown'),
            'speaker_id': row.get('speaker_id', 'unknown'),
            'duration': float(row[duration_col]),
            'start_time': float(row.get('start_time', 0)),
            'end_time': float(row.get('end_time', 0)),
            'text': str(row.get('text', row.get('normalized_text', ''))),
            'metric_value': float(row[duration_col]),
            'metric_name': 'duration_seconds',
        })
    
    # Identify shortest segments
    shortest_segments = segments_df.nsmallest(outlier_count, duration_col)
    for idx, row in shortest_segments.iterrows():
        outliers['shortest'].append({
            'segment_idx': int(idx),
            'segment_id': row.get('segment_id', f'idx_{idx}'),
            'episode_id': row.get('episode_id', 'unknown'),
            'speaker_id': row.get('speaker_id', 'unknown'),
            'duration': float(row[duration_col]),
            'start_time': float(row.get('start_time', 0)),
            'end_time': float(row.get('end_time', 0)),
            'text': str(row.get('text', row.get('normalized_text', ''))),
            'metric_value': float(row[duration_col]),
            'metric_name': 'duration_seconds',
        })
    
    # Identify embedding outliers if embeddings available
    if embeddings is not None and neighbor_similarities is not None:
        # Calculate mean similarity for each segment
        mean_similarities = neighbor_similarities.mean(axis=1)
        
        # Most isolated: lowest mean similarity
        isolated_indices = np.argsort(mean_similarities)[:outlier_count]
        for idx in isolated_indices:
            row = segments_df.iloc[idx]
            outliers['most_isolated'].append({
                'segment_idx': int(idx),
                'segment_id': row.get('segment_id', f'idx_{idx}'),
                'episode_id': row.get('episode_id', 'unknown'),
                'speaker_id': row.get('speaker_id', 'unknown'),
                'duration': float(row.get(duration_col, 0)),
                'start_time': float(row.get('start_time', 0)),
                'end_time': float(row.get('end_time', 0)),
                'text': str(row.get('text', row.get('normalized_text', ''))),
                'metric_value': float(mean_similarities[idx]),
                'metric_name': 'mean_neighbor_similarity',
            })
        
        # Most hubby: highest mean similarity
        hubby_indices = np.argsort(mean_similarities)[-outlier_count:][::-1]
        for idx in hubby_indices:
            row = segments_df.iloc[idx]
            outliers['most_hubby'].append({
                'segment_idx': int(idx),
                'segment_id': row.get('segment_id', f'idx_{idx}'),
                'episode_id': row.get('episode_id', 'unknown'),
                'speaker_id': row.get('speaker_id', 'unknown'),
                'duration': float(row.get(duration_col, 0)),
                'start_time': float(row.get('start_time', 0)),
                'end_time': float(row.get('end_time', 0)),
                'text': str(row.get('text', row.get('normalized_text', ''))),
                'metric_value': float(mean_similarities[idx]),
                'metric_name': 'mean_neighbor_similarity',
            })
        
        logger.info(
            f"Identified outliers: {len(outliers['longest'])} longest, "
            f"{len(outliers['shortest'])} shortest, "
            f"{len(outliers['most_isolated'])} isolated, "
            f"{len(outliers['most_hubby'])} hubby"
        )
    else:
        logger.info(
            f"Identified outliers: {len(outliers['longest'])} longest, "
            f"{len(outliers['shortest'])} shortest "
            f"(embeddings not available for isolation analysis)"
        )
    
    return outliers


def sample_neighbor_lists(
    segments_df: pd.DataFrame,
    embeddings: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_similarities: np.ndarray,
    sample_size: int = 30,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Sample neighbor lists for manual review (FR-33, subtask 4.1.2).
    
    Randomly samples query segments along with their nearest neighbors
    for human inspection of retrieval quality.
    
    Args:
        segments_df: DataFrame with segment data
        embeddings: Embedding matrix aligned with segments_df
        neighbor_indices: Neighbor indices from find_top_k_neighbors()
        neighbor_similarities: Neighbor similarities from find_top_k_neighbors()
        sample_size: Number of query segments to sample (default: 30)
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        List of dictionaries, each containing:
        - query: Query segment info (id, episode, speaker, text, etc.)
        - neighbors: List of neighbor segment info with similarities
    
    Example:
        >>> samples = sample_neighbor_lists(segments_df, embeddings, neighbor_idx, neighbor_sim, sample_size=10)
        >>> len(samples)
        10
    """
    logger.info(f"Sampling {sample_size} neighbor lists for manual review...")
    
    np.random.seed(random_seed)
    
    # Sample random query indices
    n_queries = len(neighbor_indices)
    sample_size = min(sample_size, n_queries)
    sampled_query_indices = np.random.choice(n_queries, size=sample_size, replace=False)
    
    neighbor_samples = []
    
    for i, query_idx in enumerate(sampled_query_indices):
        # Get query segment info
        query_row = segments_df.iloc[query_idx]
        
        query_info = {
            'segment_idx': int(query_idx),
            'segment_id': query_row.get('segment_id', f'idx_{query_idx}'),
            'episode_id': query_row.get('episode_id', 'unknown'),
            'speaker_id': query_row.get('speaker_id', 'unknown'),
            'duration': float(query_row.get('duration', query_row.get('duration_seconds', 0))),
            'start_time': float(query_row.get('start_time', 0)),
            'text': str(query_row.get('text', query_row.get('normalized_text', ''))),
        }
        
        # Get neighbor info
        neighbor_list = []
        for j, neighbor_idx in enumerate(neighbor_indices[query_idx]):
            neighbor_row = segments_df.iloc[neighbor_idx]
            similarity = float(neighbor_similarities[query_idx][j])
            
            neighbor_info = {
                'rank': j + 1,
                'segment_idx': int(neighbor_idx),
                'segment_id': neighbor_row.get('segment_id', f'idx_{neighbor_idx}'),
                'episode_id': neighbor_row.get('episode_id', 'unknown'),
                'speaker_id': neighbor_row.get('speaker_id', 'unknown'),
                'duration': float(neighbor_row.get('duration', neighbor_row.get('duration_seconds', 0))),
                'start_time': float(neighbor_row.get('start_time', 0)),
                'similarity': round(similarity, 4),
                'text': str(neighbor_row.get('text', neighbor_row.get('normalized_text', ''))),
                'same_speaker': neighbor_row.get('speaker_id') == query_row.get('speaker_id'),
                'same_episode': neighbor_row.get('episode_id') == query_row.get('episode_id'),
            }
            neighbor_list.append(neighbor_info)
        
        neighbor_samples.append({
            'sample_id': i + 1,
            'query': query_info,
            'neighbors': neighbor_list,
        })
    
    logger.info(f"Sampled {len(neighbor_samples)} neighbor lists")
    return neighbor_samples


def format_text_excerpt(
    text: str,
    max_length: int = 100,
    add_ellipsis: bool = True,
) -> str:
    """
    Format text excerpt for CSV export (FR-34, subtask 4.1.3).
    
    Truncates text to maximum length and properly escapes for CSV format.
    Adds ellipsis ("...") if text was truncated.
    
    Args:
        text: Input text string
        max_length: Maximum character length (default: 100)
        add_ellipsis: Whether to add "..." when truncating (default: True)
    
    Returns:
        Formatted and escaped text suitable for CSV
    
    Example:
        >>> format_text_excerpt("This is a very long text that needs truncation", max_length=20)
        'This is a very lo...'
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove any existing line breaks and excessive whitespace
    text = ' '.join(text.split())
    
    # Truncate if needed
    if len(text) > max_length:
        if add_ellipsis:
            truncate_length = max_length - 3  # Reserve space for "..."
            text = text[:truncate_length] + "..."
        else:
            text = text[:max_length]
    
    # CSV escaping is handled by csv.writer, but we'll normalize quotes
    text = text.replace('"', '""')  # Escape double quotes
    
    return text


def export_outliers_csv(
    outliers: Dict[str, List[Dict[str, Any]]],
    output_path: Union[str, Path],
) -> None:
    """
    Export outliers to CSV file (FR-34, subtask 4.1.4).
    
    Creates a CSV file with columns:
    - category: Outlier category (longest, shortest, most_isolated, most_hubby)
    - segment_id: Segment identifier
    - episode_id: Episode identifier
    - speaker_id: Speaker identifier
    - duration: Segment duration in seconds
    - start_time: Start timestamp
    - metric_name: Name of the metric that qualified this as an outlier
    - metric_value: Value of that metric
    - text_excerpt: Truncated text preview (100 chars)
    
    Args:
        outliers: Outliers dictionary from identify_outliers()
        output_path: Path to output CSV file
    
    Example:
        >>> outliers = identify_outliers(segments_df, embeddings)
        >>> export_outliers_csv(outliers, "output/diagnostics/outliers.csv")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting outliers to {output_path}...")
    
    # Flatten outliers into rows
    rows = []
    for category, outlier_list in outliers.items():
        for outlier in outlier_list:
            rows.append({
                'category': category,
                'segment_id': outlier['segment_id'],
                'episode_id': outlier['episode_id'],
                'speaker_id': outlier['speaker_id'],
                'duration': round(outlier['duration'], 2),
                'start_time': round(outlier['start_time'], 2),
                'metric_name': outlier['metric_name'],
                'metric_value': round(outlier['metric_value'], 4),
                'text_excerpt': format_text_excerpt(outlier['text'], max_length=100),
            })
    
    # Write CSV
    if rows:
        fieldnames = [
            'category', 'segment_id', 'episode_id', 'speaker_id',
            'duration', 'start_time', 'metric_name', 'metric_value', 'text_excerpt'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Exported {len(rows)} outliers to {output_path}")
    else:
        logger.warning("No outliers to export")


def export_neighbors_csv(
    neighbor_samples: List[Dict[str, Any]],
    output_path: Union[str, Path],
) -> None:
    """
    Export neighbor samples to CSV file (FR-34, subtask 4.1.5).
    
    Creates a CSV file with columns:
    - sample_id: Sample number
    - query_segment_id: Query segment identifier
    - query_episode_id: Query episode identifier
    - query_speaker_id: Query speaker identifier
    - query_text_excerpt: Query text preview (100 chars)
    - neighbor_rank: Neighbor rank (1-k)
    - neighbor_segment_id: Neighbor segment identifier
    - neighbor_episode_id: Neighbor episode identifier
    - neighbor_speaker_id: Neighbor speaker identifier
    - similarity: Cosine similarity score
    - same_speaker: Boolean flag if neighbor has same speaker
    - same_episode: Boolean flag if neighbor from same episode
    - neighbor_text_excerpt: Neighbor text preview (100 chars)
    
    Args:
        neighbor_samples: Neighbor samples from sample_neighbor_lists()
        output_path: Path to output CSV file
    
    Example:
        >>> samples = sample_neighbor_lists(segments_df, embeddings, neighbor_idx, neighbor_sim)
        >>> export_neighbors_csv(samples, "output/diagnostics/neighbors_sample.csv")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting neighbor samples to {output_path}...")
    
    # Flatten samples into rows
    rows = []
    for sample in neighbor_samples:
        sample_id = sample['sample_id']
        query = sample['query']
        
        for neighbor in sample['neighbors']:
            rows.append({
                'sample_id': sample_id,
                'query_segment_id': query['segment_id'],
                'query_episode_id': query['episode_id'],
                'query_speaker_id': query['speaker_id'],
                'query_duration': round(query['duration'], 2),
                'query_text_excerpt': format_text_excerpt(query['text'], max_length=100),
                'neighbor_rank': neighbor['rank'],
                'neighbor_segment_id': neighbor['segment_id'],
                'neighbor_episode_id': neighbor['episode_id'],
                'neighbor_speaker_id': neighbor['speaker_id'],
                'neighbor_duration': round(neighbor['duration'], 2),
                'similarity': neighbor['similarity'],
                'same_speaker': neighbor['same_speaker'],
                'same_episode': neighbor['same_episode'],
                'neighbor_text_excerpt': format_text_excerpt(neighbor['text'], max_length=100),
            })
    
    # Write CSV
    if rows:
        fieldnames = [
            'sample_id', 'query_segment_id', 'query_episode_id', 'query_speaker_id',
            'query_duration', 'query_text_excerpt',
            'neighbor_rank', 'neighbor_segment_id', 'neighbor_episode_id', 'neighbor_speaker_id',
            'neighbor_duration', 'similarity', 'same_speaker', 'same_episode', 'neighbor_text_excerpt'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Exported {len(rows)} neighbor pairs from {len(neighbor_samples)} samples to {output_path}")
    else:
        logger.warning("No neighbor samples to export")
