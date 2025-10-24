"""
Utterance normalization with deterministic ID generation.

Converts raw utterance data into normalized records with stable,
content-derived identifiers.
"""

from typing import Dict, List, Any

from lakehouse.ids import generate_utterance_id
from lakehouse.logger import get_default_logger


logger = get_default_logger()


def normalize_utterance(
    utterance: Dict[str, Any],
    position: int,
) -> Dict[str, Any]:
    """
    Normalize a single utterance record with deterministic ID.
    
    Args:
        utterance: Raw utterance dictionary
        position: Position/index of utterance within episode (0-based)
    
    Returns:
        Normalized utterance dictionary with utterance_id and computed duration
    
    Example:
        >>> raw = {"episode_id": "EP1", "start": 0.0, "end": 1.5, "speaker": "A", "text": "Hi"}
        >>> normalized = normalize_utterance(raw, position=0)
        >>> "utterance_id" in normalized
        True
        >>> normalized["duration"]
        1.5
    """
    # Extract required fields
    episode_id = utterance["episode_id"]
    start = utterance["start"]
    end = utterance["end"]
    speaker = utterance["speaker"]
    text = utterance["text"]
    
    # Generate deterministic utterance ID
    utterance_id = generate_utterance_id(
        episode_id=episode_id,
        position=position,
        text=text,
        speaker=speaker,
        start=start,
        end=end,
    )
    
    # Compute duration
    duration = end - start
    
    # Build normalized record
    normalized = {
        "utterance_id": utterance_id,
        "episode_id": episode_id,
        "start": float(start),
        "end": float(end),
        "speaker": speaker,
        "text": text,
        "duration": duration,
    }
    
    # Preserve any additional metadata fields from original
    for key, value in utterance.items():
        if key not in normalized:
            normalized[key] = value
    
    return normalized


def normalize_utterances(
    utterances: List[Dict[str, Any]],
    episode_id: str = None,
) -> List[Dict[str, Any]]:
    """
    Normalize a list of utterances with deterministic IDs.
    
    Utterances are assigned position indices based on their order in the list.
    All utterances must belong to the same episode.
    
    Args:
        utterances: List of raw utterance dictionaries
        episode_id: Episode ID (optional, will use from first utterance if not provided)
    
    Returns:
        List of normalized utterance dictionaries
    
    Raises:
        ValueError: If utterances belong to different episodes
    
    Example:
        >>> raw_utterances = [
        ...     {"episode_id": "EP1", "start": 0.0, "end": 1.5, "speaker": "A", "text": "Hi"},
        ...     {"episode_id": "EP1", "start": 1.5, "end": 3.0, "speaker": "B", "text": "Hello"},
        ... ]
        >>> normalized = normalize_utterances(raw_utterances)
        >>> len(normalized)
        2
    """
    if not utterances:
        logger.warning("No utterances to normalize")
        return []
    
    # Determine episode ID
    if episode_id is None:
        episode_id = utterances[0].get("episode_id")
    
    if not episode_id:
        raise ValueError("episode_id must be provided or present in utterances")
    
    # Verify all utterances belong to same episode
    for idx, utterance in enumerate(utterances):
        utt_episode_id = utterance.get("episode_id")
        if utt_episode_id != episode_id:
            raise ValueError(
                f"Utterance at index {idx} has episode_id '{utt_episode_id}' "
                f"but expected '{episode_id}'"
            )
    
    # Normalize each utterance with its position
    normalized = []
    for position, utterance in enumerate(utterances):
        try:
            normalized_utterance = normalize_utterance(utterance, position)
            normalized.append(normalized_utterance)
        except Exception as e:
            logger.error(f"Failed to normalize utterance at position {position}: {e}")
            raise
    
    logger.info(f"Normalized {len(normalized)} utterances for episode {episode_id}")
    return normalized


def sort_utterances_by_time(utterances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort utterances by start time.
    
    This ensures deterministic ordering for position-based ID generation.
    
    Args:
        utterances: List of utterance dictionaries
    
    Returns:
        Sorted list of utterances
    
    Example:
        >>> utterances = [
        ...     {"start": 10.0, "end": 15.0, ...},
        ...     {"start": 0.0, "end": 5.0, ...},
        ... ]
        >>> sorted_utts = sort_utterances_by_time(utterances)
        >>> sorted_utts[0]["start"]
        0.0
    """
    return sorted(utterances, key=lambda u: u.get("start", 0.0))


def compute_utterance_statistics(utterances: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics about a list of normalized utterances.
    
    Args:
        utterances: List of normalized utterance dictionaries
    
    Returns:
        Dictionary with statistics
    
    Example:
        >>> stats = compute_utterance_statistics(normalized_utterances)
        >>> stats["total_count"]
        150
    """
    if not utterances:
        return {
            "total_count": 0,
            "total_duration": 0.0,
            "speaker_count": 0,
            "speakers": [],
        }
    
    # Get unique speakers
    speakers = set()
    total_duration = 0.0
    
    for utterance in utterances:
        speaker = utterance.get("speaker")
        if speaker:
            speakers.add(speaker)
        
        duration = utterance.get("duration", 0.0)
        total_duration += duration
    
    # Get episode time range
    start_times = [u.get("start", 0.0) for u in utterances]
    end_times = [u.get("end", 0.0) for u in utterances]
    
    episode_start = min(start_times) if start_times else 0.0
    episode_end = max(end_times) if end_times else 0.0
    episode_duration = episode_end - episode_start
    
    return {
        "total_count": len(utterances),
        "total_duration": total_duration,
        "episode_duration": episode_duration,
        "episode_start": episode_start,
        "episode_end": episode_end,
        "speaker_count": len(speakers),
        "speakers": sorted(list(speakers)),
    }


def group_utterances_by_episode(
    utterances: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group utterances by episode ID.
    
    Args:
        utterances: List of utterance dictionaries (possibly from multiple episodes)
    
    Returns:
        Dictionary mapping episode_id to list of utterances
    
    Example:
        >>> all_utterances = [ep1_utt1, ep1_utt2, ep2_utt1]
        >>> grouped = group_utterances_by_episode(all_utterances)
        >>> len(grouped)
        2
    """
    episodes = {}
    
    for utterance in utterances:
        episode_id = utterance.get("episode_id")
        if not episode_id:
            logger.warning("Utterance missing episode_id, skipping")
            continue
        
        if episode_id not in episodes:
            episodes[episode_id] = []
        
        episodes[episode_id].append(utterance)
    
    logger.info(f"Grouped {len(utterances)} utterances into {len(episodes)} episodes")
    return episodes


def normalize_episode(
    utterances: List[Dict[str, Any]],
    episode_id: str = None,
    sort_by_time: bool = True,
) -> List[Dict[str, Any]]:
    """
    Normalize all utterances for a single episode.
    
    This is a convenience function that:
    1. Optionally sorts utterances by time
    2. Normalizes with deterministic IDs
    3. Returns statistics
    
    Args:
        utterances: List of raw utterance dictionaries for one episode
        episode_id: Episode ID (optional)
        sort_by_time: Whether to sort utterances by start time first
    
    Returns:
        List of normalized utterances
    
    Example:
        >>> normalized = normalize_episode(raw_utterances, episode_id="EP1")
        >>> all(u["episode_id"] == "EP1" for u in normalized)
        True
    """
    if not utterances:
        return []
    
    # Sort by time if requested (ensures deterministic ordering)
    if sort_by_time:
        utterances = sort_utterances_by_time(utterances)
    
    # Normalize
    normalized = normalize_utterances(utterances, episode_id=episode_id)
    
    # Log statistics
    stats = compute_utterance_statistics(normalized)
    logger.info(
        f"Episode {normalized[0]['episode_id']}: "
        f"{stats['total_count']} utterances, "
        f"{stats['speaker_count']} speakers, "
        f"{stats['episode_duration']:.1f}s duration"
    )
    
    return normalized

