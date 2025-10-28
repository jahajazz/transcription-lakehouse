"""
Test fixtures for quality assessment tests.

Provides sample data with controlled properties for testing quality metrics:
- Episode metadata with known durations
- Spans and beats with various edge cases
- Embeddings with controlled patterns
- Helper functions to generate custom test scenarios
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


# ============================================================================
# Task 6.1.1: Sample Episode Metadata DataFrame
# ============================================================================

def create_sample_episodes(
    num_episodes: int = 3,
    duration_minutes: Optional[List[float]] = None,
    series: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create sample episode metadata DataFrame.
    
    Args:
        num_episodes: Number of episodes to generate
        duration_minutes: List of durations in minutes (auto-generated if None)
        series: List of series names (cycles through ['LOS', 'SW'] if None)
    
    Returns:
        DataFrame with columns: episode_id, title, duration_minutes, 
        duration_seconds, series, date
    
    Example:
        >>> episodes = create_sample_episodes(num_episodes=2)
        >>> len(episodes)
        2
        >>> 'duration_seconds' in episodes.columns
        True
    """
    if duration_minutes is None:
        # Default to 45, 60, 30 minute episodes
        duration_minutes = [45.0, 60.0, 30.0][:num_episodes]
    
    if series is None:
        # Alternate between LOS and SW
        series = ['LOS' if i % 2 == 0 else 'SW' for i in range(num_episodes)]
    
    # Ensure lists are the right length
    duration_minutes = duration_minutes[:num_episodes] + [45.0] * max(0, num_episodes - len(duration_minutes))
    series = series[:num_episodes] + ['LOS'] * max(0, num_episodes - len(series))
    
    episodes_data = []
    for i in range(num_episodes):
        episode_id = f"TEST-EP{i+1:03d}"
        episodes_data.append({
            'episode_id': episode_id,
            'title': f"Test Episode {i+1}",
            'duration_minutes': duration_minutes[i],
            'duration_seconds': duration_minutes[i] * 60,
            'series': series[i],
            'date': f"2024-01-{i+1:02d}",
        })
    
    return pd.DataFrame(episodes_data)


def get_default_episodes() -> pd.DataFrame:
    """
    Get default test episode metadata (3 episodes).
    
    Returns:
        DataFrame with 3 test episodes:
        - TEST-EP001: 45 minutes, LOS
        - TEST-EP002: 60 minutes, SW
        - TEST-EP003: 30 minutes, LOS
    """
    return create_sample_episodes(num_episodes=3)


# ============================================================================
# Task 6.1.2: Sample Spans DataFrame with Edge Cases
# ============================================================================

def create_sample_spans(
    episode_id: str = "TEST-EP001",
    episode_duration_seconds: float = 2700.0,
    include_edge_cases: bool = True,
) -> pd.DataFrame:
    """
    Create sample spans DataFrame with various edge cases.
    
    Edge cases include:
    - Normal spans (20-120s range)
    - Too short spans (<20s)
    - Too long spans (>120s)
    - Gaps between spans
    - Overlapping spans
    - Negative duration (if include_edge_cases=True)
    - Zero duration (if include_edge_cases=True)
    - Duplicate text
    
    Args:
        episode_id: Episode ID for all spans
        episode_duration_seconds: Total episode duration
        include_edge_cases: Whether to include edge cases for testing
    
    Returns:
        DataFrame with columns: span_id, episode_id, start_time, end_time,
        duration, speaker, text, segment_type
    
    Example:
        >>> spans = create_sample_spans(include_edge_cases=True)
        >>> (spans['duration'] < 0).any()
        True
    """
    spans_data = []
    current_time = 0.0
    span_counter = 1
    
    # Helper to add a span
    def add_span(start, end, speaker, text, segment_type="span"):
        spans_data.append({
            'span_id': f"{episode_id}_S{span_counter:03d}",
            'episode_id': episode_id,
            'start_time': start,
            'end_time': end,
            'duration': end - start,
            'speaker': speaker,
            'speaker_canonical': speaker,  # Task 6.2: New field
            'speaker_role': 'other',  # Task 6.2: New field
            'is_expert': False,  # Task 6.2: New field
            'text': text,
            'segment_type': segment_type,
        })
        return span_counter + 1
    
    # Normal span (good length: 45s)
    span_counter = add_span(
        current_time, current_time + 45.0,
        "Alice",
        "Welcome to our test episode. Today we're going to explore some fascinating topics in artificial intelligence."
    )
    current_time += 45.0
    
    # Short gap (5 seconds)
    current_time += 5.0
    
    # Normal span (good length: 30s)
    span_counter = add_span(
        current_time, current_time + 30.0,
        "Bob",
        "Thanks for having me, Alice. I'm excited to discuss the intersection of technology and philosophy."
    )
    current_time += 30.0
    
    if include_edge_cases:
        # Too short span (15s - below 20s threshold)
        span_counter = add_span(
            current_time, current_time + 15.0,
            "Alice",
            "Great question!"
        )
        current_time += 15.0
    
    # Normal span (60s)
    span_counter = add_span(
        current_time, current_time + 60.0,
        "Bob",
        "One of the most interesting aspects is how machine learning models process information. They don't think like humans do, but they can recognize patterns in ways that sometimes surpass human capabilities."
    )
    current_time += 60.0
    
    if include_edge_cases:
        # Too long span (150s - above 120s threshold)
        span_counter = add_span(
            current_time, current_time + 150.0,
            "Charlie",
            "Let me give you a detailed explanation of the entire history of artificial intelligence, starting from the Dartmouth Conference in 1956, moving through the various AI winters, the resurgence with expert systems, the neural network renaissance, and finally the deep learning revolution. This is a very long explanation that goes on and on to create a span that exceeds the maximum recommended duration."
        )
        current_time += 150.0
    
    # Normal span with overlap (starts before previous ends)
    overlap_start = current_time - 10.0  # Overlap by 10 seconds
    span_counter = add_span(
        overlap_start, overlap_start + 40.0,
        "Alice",
        "If I may interject here, I think we need to distinguish between intelligence and consciousness."
    )
    current_time = overlap_start + 40.0
    
    # Large gap (50 seconds)
    current_time += 50.0
    
    # Normal span (35s)
    span_counter = add_span(
        current_time, current_time + 35.0,
        "Bob",
        "That's an excellent point. Can you elaborate on that distinction?"
    )
    current_time += 35.0
    
    # Duplicate text (exact)
    span_counter = add_span(
        current_time, current_time + 35.0,
        "Charlie",
        "That's an excellent point. Can you elaborate on that distinction?"
    )
    current_time += 35.0
    
    # Near-duplicate text
    span_counter = add_span(
        current_time, current_time + 35.0,
        "Alice",
        "That's an excellent point! Can you elaborate on that distinction?"
    )
    current_time += 35.0
    
    if include_edge_cases:
        # Zero duration span
        span_counter = add_span(
            current_time, current_time,
            "Bob",
            "Zero duration marker"
        )
        
        # Negative duration span (end < start)
        span_counter = add_span(
            current_time + 10.0, current_time,
            "Alice",
            "Negative duration error"
        )
        current_time += 10.0
    
    # Normal span to finish
    span_counter = add_span(
        current_time, current_time + 50.0,
        "Charlie",
        "This has been a fascinating discussion. We'll continue this in our next episode."
    )
    current_time += 50.0
    
    # Leave a gap at the end (don't cover full episode)
    # current_time should be less than episode_duration_seconds
    
    df = pd.DataFrame(spans_data)
    
    # Add some missing fields for testing validation
    if include_edge_cases and len(df) > 0:
        # Make one span have missing speaker
        df.loc[len(df) - 2, 'speaker'] = None
    
    return df


def create_balanced_spans(
    episode_ids: List[str],
    spans_per_episode: int = 10,
    speakers: List[str] = None,
    avg_duration: float = 60.0,
) -> pd.DataFrame:
    """
    Create balanced spans without edge cases (for positive testing).
    
    Args:
        episode_ids: List of episode IDs
        spans_per_episode: Number of spans per episode
        speakers: List of speaker names (defaults to Alice, Bob, Charlie)
        avg_duration: Average span duration in seconds
    
    Returns:
        DataFrame with well-formed spans
    """
    if speakers is None:
        speakers = ["Alice", "Bob", "Charlie"]
    
    spans_data = []
    
    for episode_id in episode_ids:
        current_time = 0.0
        for i in range(spans_per_episode):
            speaker = speakers[i % len(speakers)]
            # Vary duration around average (±20%)
            duration = avg_duration * (0.8 + 0.4 * np.random.random())
            duration = np.clip(duration, 20.0, 120.0)  # Keep within bounds
            
            spans_data.append({
                'span_id': f"{episode_id}_S{i+1:03d}",
                'episode_id': episode_id,
                'start_time': current_time,
                'end_time': current_time + duration,
                'duration': duration,
                'speaker': speaker,
                'speaker_canonical': speaker,  # Task 6.2: New field
                'speaker_role': 'other',  # Task 6.2: New field
                'is_expert': False,  # Task 6.2: New field
                'text': f"This is span {i+1} by {speaker}. Some sample text content.",
                'segment_type': 'span',
            })
            
            current_time += duration + 1.0  # Small gap between spans
    
    return pd.DataFrame(spans_data)


# ============================================================================
# Task 6.1.3: Sample Beats DataFrame
# ============================================================================

def create_sample_beats(
    episode_id: str = "TEST-EP001",
    episode_duration_seconds: float = 2700.0,
    include_edge_cases: bool = True,
) -> pd.DataFrame:
    """
    Create sample beats DataFrame.
    
    Beats are longer segments (60-180s) that may contain multiple speakers.
    Similar edge cases as spans but with different thresholds.
    
    Args:
        episode_id: Episode ID for all beats
        episode_duration_seconds: Total episode duration
        include_edge_cases: Whether to include edge cases for testing
    
    Returns:
        DataFrame with columns: beat_id, episode_id, start_time, end_time,
        duration, speakers, text, segment_type
    
    Example:
        >>> beats = create_sample_beats(include_edge_cases=True)
        >>> (beats['duration'] >= 60).sum() > 0
        True
    """
    beats_data = []
    current_time = 0.0
    beat_counter = 1
    
    # Helper to add a beat
    def add_beat(start, end, speakers, text):
        beats_data.append({
            'beat_id': f"{episode_id}_B{beat_counter:03d}",
            'episode_id': episode_id,
            'start_time': start,
            'end_time': end,
            'duration': end - start,
            'speakers': speakers,
            'speakers_set': list(set(speakers)) if isinstance(speakers, list) else [],  # Task 6.2: New field
            'expert_span_ids': [],  # Task 6.2: New field
            'expert_coverage_pct': 0.0,  # Task 6.2: New field
            'text': text,
            'segment_type': 'beat',
        })
        return beat_counter + 1
    
    # Normal beat (good length: 90s)
    beat_counter = add_beat(
        current_time, current_time + 90.0,
        ["Alice", "Bob"],
        "Welcome to our test episode. Today we're going to explore some fascinating topics in artificial intelligence. Thanks for having me, Alice. I'm excited to discuss the intersection of technology and philosophy."
    )
    current_time += 90.0
    
    # Normal beat (120s)
    beat_counter = add_beat(
        current_time, current_time + 120.0,
        ["Bob", "Charlie"],
        "One of the most interesting aspects is how machine learning models process information. They don't think like humans do, but they can recognize patterns in ways that sometimes surpass human capabilities. If I may interject here, I think we need to distinguish between intelligence and consciousness."
    )
    current_time += 120.0
    
    if include_edge_cases:
        # Too short beat (45s - below 60s threshold)
        beat_counter = add_beat(
            current_time, current_time + 45.0,
            ["Alice"],
            "This is a very short beat that's below the threshold."
        )
        current_time += 45.0
    
    # Normal beat (150s)
    beat_counter = add_beat(
        current_time, current_time + 150.0,
        ["Charlie", "Alice", "Bob"],
        "Intelligence is about problem-solving and pattern recognition. It's computational in nature. Consciousness, on the other hand, involves subjective experience - the feeling of what it's like to be something. A chess computer is intelligent but probably not conscious in any meaningful sense."
    )
    current_time += 150.0
    
    if include_edge_cases:
        # Too long beat (200s - above 180s threshold)
        beat_counter = add_beat(
            current_time, current_time + 200.0,
            ["Bob", "Alice"],
            "This is an extremely long beat that goes on and on with lots of discussion about various topics including philosophy, artificial intelligence, consciousness, the nature of reality, quantum mechanics, and many other fascinating subjects that require extended exploration and detailed explanation that causes this beat to exceed the maximum recommended duration for a beat segment."
        )
        current_time += 200.0
    
    # Gap (no beat for 30 seconds)
    current_time += 30.0
    
    # Normal beat (75s)
    beat_counter = add_beat(
        current_time, current_time + 75.0,
        ["Charlie"],
        "This has been a fascinating discussion. We'll continue this in our next episode."
    )
    current_time += 75.0
    
    if include_edge_cases:
        # Zero duration beat
        beat_counter = add_beat(
            current_time, current_time,
            ["Bob"],
            "Zero duration marker"
        )
    
    # Final beat
    beat_counter = add_beat(
        current_time, current_time + 100.0,
        ["Alice", "Bob", "Charlie"],
        "Thanks everyone for joining us. See you next time!"
    )
    
    return pd.DataFrame(beats_data)


def create_balanced_beats(
    episode_ids: List[str],
    beats_per_episode: int = 5,
    avg_duration: float = 120.0,
) -> pd.DataFrame:
    """
    Create balanced beats without edge cases (for positive testing).
    
    Args:
        episode_ids: List of episode IDs
        beats_per_episode: Number of beats per episode
        avg_duration: Average beat duration in seconds
    
    Returns:
        DataFrame with well-formed beats
    """
    beats_data = []
    
    for episode_id in episode_ids:
        current_time = 0.0
        for i in range(beats_per_episode):
            # Vary duration around average (±20%)
            duration = avg_duration * (0.8 + 0.4 * np.random.random())
            duration = np.clip(duration, 60.0, 180.0)  # Keep within bounds
            
            beats_data.append({
                'beat_id': f"{episode_id}_B{i+1:03d}",
                'episode_id': episode_id,
                'start_time': current_time,
                'end_time': current_time + duration,
                'duration': duration,
                'speakers': ["Alice", "Bob"],
                'speakers_set': ["Alice", "Bob"],  # Task 6.2: New field
                'expert_span_ids': [],  # Task 6.2: New field
                'expert_coverage_pct': 0.0,  # Task 6.2: New field
                'text': f"This is beat {i+1}. Some sample text content with multiple speakers.",
                'segment_type': 'beat',
            })
            
            current_time += duration + 2.0  # Small gap between beats
    
    return pd.DataFrame(beats_data)


# ============================================================================
# Task 6.1.4: Sample Embeddings Arrays
# ============================================================================

def create_sample_embeddings(
    num_segments: int,
    embedding_dim: int = 384,
    pattern: str = "random",
    random_seed: int = 42,
) -> np.ndarray:
    """
    Create sample embedding matrix.
    
    Args:
        num_segments: Number of segments (rows in embedding matrix)
        embedding_dim: Embedding dimensionality (default: 384 like sentence-transformers)
        pattern: Type of embeddings to generate:
            - "random": Random L2-normalized vectors
            - "clustered": Create 3 clusters for testing leakage
            - "length_biased": Correlate norm with segment index (for testing length bias)
            - "adjacent": Make adjacent segments similar (for testing adjacency bias)
        random_seed: Random seed for reproducibility
    
    Returns:
        Array of shape (num_segments, embedding_dim)
    
    Example:
        >>> emb = create_sample_embeddings(100, embedding_dim=384, pattern="random")
        >>> emb.shape
        (100, 384)
    """
    np.random.seed(random_seed)
    
    if pattern == "random":
        # Generate random normalized embeddings
        embeddings = np.random.randn(num_segments, embedding_dim).astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
    
    elif pattern == "clustered":
        # Create 3 clusters to simulate speaker grouping
        cluster_centers = np.random.randn(3, embedding_dim).astype(np.float32)
        cluster_centers = cluster_centers / (np.linalg.norm(cluster_centers, axis=1, keepdims=True) + 1e-8)
        
        # Assign segments to clusters
        cluster_assignments = np.random.randint(0, 3, size=num_segments)
        
        # Generate embeddings around cluster centers
        embeddings = np.zeros((num_segments, embedding_dim), dtype=np.float32)
        for i in range(num_segments):
            center = cluster_centers[cluster_assignments[i]]
            noise = np.random.randn(embedding_dim).astype(np.float32) * 0.1
            embeddings[i] = center + noise
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
    
    elif pattern == "length_biased":
        # Create embeddings where norm correlates with position (simulating length bias)
        base_embeddings = np.random.randn(num_segments, embedding_dim).astype(np.float32)
        # Create varying norms (0.5 to 1.5)
        norms = 0.5 + np.linspace(0, 1.0, num_segments).astype(np.float32)
        
        # Normalize to unit sphere first, then scale by desired norm
        base_norms = np.linalg.norm(base_embeddings, axis=1, keepdims=True)
        embeddings = base_embeddings / (base_norms + 1e-8) * norms[:, np.newaxis]
    
    elif pattern == "adjacent":
        # Make adjacent segments similar
        embeddings = np.zeros((num_segments, embedding_dim), dtype=np.float32)
        
        # Generate base vectors
        base_vectors = np.random.randn(num_segments // 3 + 1, embedding_dim).astype(np.float32)
        base_vectors = base_vectors / (np.linalg.norm(base_vectors, axis=1, keepdims=True) + 1e-8)
        
        # Each segment is similar to nearby segments
        for i in range(num_segments):
            base_idx = i // 3
            noise = np.random.randn(embedding_dim).astype(np.float32) * 0.05
            embeddings[i] = base_vectors[base_idx] + noise
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return embeddings


def create_embeddings_with_speaker_leakage(
    segments_df: pd.DataFrame,
    embedding_dim: int = 384,
    leakage_strength: float = 0.7,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Create embeddings that exhibit speaker leakage.
    
    Segments with the same speaker will have more similar embeddings,
    simulating a model that's learning speaker identity instead of content.
    
    Args:
        segments_df: DataFrame with 'speaker' column
        embedding_dim: Embedding dimensionality
        leakage_strength: How strong the speaker signal is (0-1)
        random_seed: Random seed
    
    Returns:
        Array of shape (len(segments_df), embedding_dim)
    """
    np.random.seed(random_seed)
    
    # Get unique speakers
    speakers = segments_df['speaker'].unique()
    speaker_to_idx = {speaker: i for i, speaker in enumerate(speakers)}
    
    # Create speaker-specific embeddings
    speaker_embeddings = np.random.randn(len(speakers), embedding_dim).astype(np.float32)
    speaker_embeddings = speaker_embeddings / (np.linalg.norm(speaker_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Create content embeddings
    content_embeddings = np.random.randn(len(segments_df), embedding_dim).astype(np.float32)
    content_embeddings = content_embeddings / (np.linalg.norm(content_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Mix speaker and content
    embeddings = np.zeros((len(segments_df), embedding_dim), dtype=np.float32)
    for i, row in segments_df.iterrows():
        speaker_idx = speaker_to_idx.get(row['speaker'], 0)
        embeddings[i] = (
            leakage_strength * speaker_embeddings[speaker_idx] +
            (1 - leakage_strength) * content_embeddings[i]
        )
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    return embeddings


# ============================================================================
# Task 6.1.5: Helper Functions for Controlled Test Data
# ============================================================================

def create_complete_test_dataset(
    num_episodes: int = 3,
    include_edge_cases: bool = True,
    include_embeddings: bool = True,
    embedding_pattern: str = "random",
) -> Dict[str, Any]:
    """
    Create a complete test dataset with episodes, spans, beats, and embeddings.
    
    Args:
        num_episodes: Number of episodes to generate
        include_edge_cases: Whether to include problematic data for testing
        include_embeddings: Whether to generate embeddings
        embedding_pattern: Pattern for embeddings ("random", "clustered", etc.)
    
    Returns:
        Dictionary containing:
        - episodes: DataFrame with episode metadata
        - spans: DataFrame with span data
        - beats: DataFrame with beat data
        - span_embeddings: Array of span embeddings (if include_embeddings=True)
        - beat_embeddings: Array of beat embeddings (if include_embeddings=True)
    
    Example:
        >>> dataset = create_complete_test_dataset(num_episodes=2)
        >>> 'episodes' in dataset
        True
        >>> 'spans' in dataset
        True
    """
    # Create episodes
    episodes = create_sample_episodes(num_episodes=num_episodes)
    
    # Create spans and beats for each episode
    all_spans = []
    all_beats = []
    
    for _, episode in episodes.iterrows():
        episode_id = episode['episode_id']
        duration_seconds = episode['duration_seconds']
        
        # Create spans for this episode
        episode_spans = create_sample_spans(
            episode_id=episode_id,
            episode_duration_seconds=duration_seconds,
            include_edge_cases=include_edge_cases
        )
        all_spans.append(episode_spans)
        
        # Create beats for this episode
        episode_beats = create_sample_beats(
            episode_id=episode_id,
            episode_duration_seconds=duration_seconds,
            include_edge_cases=include_edge_cases
        )
        all_beats.append(episode_beats)
    
    spans_df = pd.concat(all_spans, ignore_index=True) if all_spans else pd.DataFrame()
    beats_df = pd.concat(all_beats, ignore_index=True) if all_beats else pd.DataFrame()
    
    result = {
        'episodes': episodes,
        'spans': spans_df,
        'beats': beats_df,
    }
    
    # Create embeddings if requested
    if include_embeddings:
        if len(spans_df) > 0:
            result['span_embeddings'] = create_sample_embeddings(
                num_segments=len(spans_df),
                pattern=embedding_pattern
            )
        
        if len(beats_df) > 0:
            result['beat_embeddings'] = create_sample_embeddings(
                num_segments=len(beats_df),
                pattern=embedding_pattern
            )
    
    return result


def create_coverage_test_data(
    episode_duration: float = 1000.0,
    coverage_percent: float = 95.0,
    gap_percent: float = 3.0,
    overlap_percent: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create test data with specific coverage, gap, and overlap percentages.
    
    Useful for testing coverage threshold validation.
    
    Args:
        episode_duration: Total episode duration in seconds
        coverage_percent: Target coverage percentage
        gap_percent: Target gap percentage
        overlap_percent: Target overlap percentage
    
    Returns:
        Tuple of (episodes DataFrame, spans DataFrame)
    
    Example:
        >>> episodes, spans = create_coverage_test_data(coverage_percent=90.0)
        >>> # Calculate actual coverage to verify
    """
    episode_id = "TEST-COVERAGE"
    
    # Create episode
    episodes = pd.DataFrame([{
        'episode_id': episode_id,
        'duration_seconds': episode_duration,
        'duration_minutes': episode_duration / 60,
        'series': 'TEST',
        'title': 'Coverage Test Episode',
        'date': '2024-01-01',
    }])
    
    # Calculate target durations
    coverage_duration = episode_duration * (coverage_percent / 100.0)
    gap_duration = episode_duration * (gap_percent / 100.0)
    overlap_duration = episode_duration * (overlap_percent / 100.0)
    
    # Create spans to achieve target coverage
    spans_data = []
    current_time = 0.0
    span_counter = 1
    total_span_duration = 0.0
    
    # Add spans with gaps
    while total_span_duration < coverage_duration and current_time < episode_duration:
        # Span duration (30-60s)
        span_duration = min(45.0, coverage_duration - total_span_duration)
        if span_duration < 10.0:
            break
        
        speaker = 'Alice' if span_counter % 2 == 0 else 'Bob'
        spans_data.append({
            'span_id': f"{episode_id}_S{span_counter:03d}",
            'episode_id': episode_id,
            'start_time': current_time,
            'end_time': current_time + span_duration,
            'duration': span_duration,
            'speaker': speaker,
            'speaker_canonical': speaker,  # Task 6.2: New field
            'speaker_role': 'other',  # Task 6.2: New field
            'is_expert': False,  # Task 6.2: New field
            'text': f'Span {span_counter} content',
            'segment_type': 'span',
        })
        
        total_span_duration += span_duration
        current_time += span_duration
        
        # Add a small gap between spans
        if total_span_duration < coverage_duration:
            gap = min(5.0, gap_duration / 10)  # Distribute gap across multiple segments
            current_time += gap
        
        span_counter += 1
    
    # Add overlapping spans if needed
    if overlap_percent > 0 and len(spans_data) > 2:
        # Create an overlap by making a new span that overlaps with an existing one
        overlap_target_span = spans_data[len(spans_data) // 2]
        overlap_span_duration = min(overlap_duration, overlap_target_span['duration'] / 2)
        
        spans_data.append({
            'span_id': f"{episode_id}_S{span_counter:03d}",
            'episode_id': episode_id,
            'start_time': overlap_target_span['start_time'] + 5.0,
            'end_time': overlap_target_span['start_time'] + 5.0 + overlap_span_duration,
            'duration': overlap_span_duration,
            'speaker': 'Charlie',
            'speaker_canonical': 'Charlie',  # Task 6.2: New field
            'speaker_role': 'other',  # Task 6.2: New field
            'is_expert': False,  # Task 6.2: New field
            'text': 'Overlapping span content',
            'segment_type': 'span',
        })
    
    spans_df = pd.DataFrame(spans_data)
    
    return episodes, spans_df


def create_distribution_test_data(
    num_spans: int = 100,
    target_distribution: str = "good",
) -> pd.DataFrame:
    """
    Create spans with specific duration distributions for testing.
    
    Args:
        num_spans: Number of spans to generate
        target_distribution: Distribution type:
            - "good": 95% within 20-120s range
            - "bad": Only 70% within range
            - "too_short": Many spans < 20s
            - "too_long": Many spans > 120s
    
    Returns:
        DataFrame with spans
    """
    episode_id = "TEST-DIST"
    spans_data = []
    
    np.random.seed(42)
    
    for i in range(num_spans):
        if target_distribution == "good":
            # 95% within range
            if i < 95:
                duration = np.random.uniform(20.0, 120.0)
            else:
                duration = np.random.uniform(5.0, 20.0)
        
        elif target_distribution == "bad":
            # 70% within range
            if i < 70:
                duration = np.random.uniform(20.0, 120.0)
            elif i < 85:
                duration = np.random.uniform(5.0, 20.0)
            else:
                duration = np.random.uniform(120.0, 200.0)
        
        elif target_distribution == "too_short":
            # 50% too short
            if i < 50:
                duration = np.random.uniform(5.0, 20.0)
            else:
                duration = np.random.uniform(20.0, 120.0)
        
        elif target_distribution == "too_long":
            # 50% too long
            if i < 50:
                duration = np.random.uniform(120.0, 200.0)
            else:
                duration = np.random.uniform(20.0, 120.0)
        
        else:
            raise ValueError(f"Unknown distribution: {target_distribution}")
        
        current_time = i * 150.0  # Space them out
        speaker = ['Alice', 'Bob', 'Charlie'][i % 3]
        
        spans_data.append({
            'span_id': f"{episode_id}_S{i+1:03d}",
            'episode_id': episode_id,
            'start_time': current_time,
            'end_time': current_time + duration,
            'duration': duration,
            'speaker': speaker,
            'speaker_canonical': speaker,  # Task 6.2: New field
            'speaker_role': 'other',  # Task 6.2: New field
            'is_expert': False,  # Task 6.2: New field
            'text': f'Span {i+1} content',
            'segment_type': 'span',
        })
    
    return pd.DataFrame(spans_data)


def create_integrity_test_data() -> pd.DataFrame:
    """
    Create spans with specific integrity issues for testing.
    
    Returns:
        DataFrame with spans containing:
        - Timestamp regressions (out of order)
        - Negative durations
        - Zero durations
        - Exact duplicates
        - Near-duplicates
    """
    episode_id = "TEST-INTEGRITY"
    
    spans_data = [
        # Normal span
        {
            'span_id': f"{episode_id}_S001",
            'episode_id': episode_id,
            'start_time': 0.0,
            'end_time': 30.0,
            'duration': 30.0,
            'speaker': 'Alice',
            'text': 'This is the first span.',
            'segment_type': 'span',
        },
        # Normal span
        {
            'span_id': f"{episode_id}_S002",
            'episode_id': episode_id,
            'start_time': 30.0,
            'end_time': 60.0,
            'duration': 30.0,
            'speaker': 'Bob',
            'text': 'This is the second span.',
            'segment_type': 'span',
        },
        # Timestamp regression (starts before previous)
        {
            'span_id': f"{episode_id}_S003",
            'episode_id': episode_id,
            'start_time': 20.0,  # Regression!
            'end_time': 50.0,
            'duration': 30.0,
            'speaker': 'Alice',
            'text': 'This span is out of order.',
            'segment_type': 'span',
        },
        # Normal span
        {
            'span_id': f"{episode_id}_S004",
            'episode_id': episode_id,
            'start_time': 60.0,
            'end_time': 90.0,
            'duration': 30.0,
            'speaker': 'Charlie',
            'text': 'Back to normal order.',
            'segment_type': 'span',
        },
        # Zero duration
        {
            'span_id': f"{episode_id}_S005",
            'episode_id': episode_id,
            'start_time': 90.0,
            'end_time': 90.0,
            'duration': 0.0,
            'speaker': 'Alice',
            'text': 'Zero duration span.',
            'segment_type': 'span',
        },
        # Negative duration
        {
            'span_id': f"{episode_id}_S006",
            'episode_id': episode_id,
            'start_time': 120.0,
            'end_time': 100.0,  # End before start!
            'duration': -20.0,
            'speaker': 'Bob',
            'text': 'Negative duration span.',
            'segment_type': 'span',
        },
        # Exact duplicate (same text as next)
        {
            'span_id': f"{episode_id}_S007",
            'episode_id': episode_id,
            'start_time': 130.0,
            'end_time': 160.0,
            'duration': 30.0,
            'speaker': 'Alice',
            'text': 'This is a duplicate span.',
            'segment_type': 'span',
        },
        # Exact duplicate (same text as previous)
        {
            'span_id': f"{episode_id}_S008",
            'episode_id': episode_id,
            'start_time': 160.0,
            'end_time': 190.0,
            'duration': 30.0,
            'speaker': 'Bob',
            'text': 'This is a duplicate span.',
            'segment_type': 'span',
        },
        # Near-duplicate (very similar text)
        {
            'span_id': f"{episode_id}_S009",
            'episode_id': episode_id,
            'start_time': 190.0,
            'end_time': 220.0,
            'duration': 30.0,
            'speaker': 'Charlie',
            'text': 'This is a duplicate span!',  # Only punctuation differs
            'segment_type': 'span',
        },
    ]
    
    df = pd.DataFrame(spans_data)
    # Task 6.2: Add new speaker metadata fields
    df['speaker_canonical'] = df['speaker']
    df['speaker_role'] = 'other'
    df['is_expert'] = False
    return df


# ============================================================================
# Convenience Functions
# ============================================================================

def get_default_test_data() -> Dict[str, Any]:
    """
    Get default test dataset for quick testing.
    
    Returns:
        Dictionary with episodes, spans, beats, and embeddings
    """
    return create_complete_test_dataset(
        num_episodes=3,
        include_edge_cases=True,
        include_embeddings=True,
        embedding_pattern="random"
    )
