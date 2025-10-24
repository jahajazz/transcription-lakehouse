"""
Deterministic ID generation utilities for lakehouse artifacts.

All ID generation functions are deterministic: given the same inputs,
they will always produce the same output. This is critical for reproducibility
and idempotent processing.
"""

import hashlib
import json
from typing import Any, Dict, List, Union


def compute_content_hash(content: str, algorithm: str = "sha256") -> str:
    """
    Compute a deterministic hash of content.

    Args:
        content: String content to hash
        algorithm: Hash algorithm to use (default: "sha256")

    Returns:
        Hexadecimal hash string

    Example:
        >>> compute_content_hash("Hello world")
        '64ec88ca00b268e5ba1a35678a1b5316d212f4f366b2477232534a8aeca37f3c'
    """
    hasher = hashlib.new(algorithm)
    hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()


def compute_dict_hash(data: Dict[str, Any], algorithm: str = "sha256") -> str:
    """
    Compute a deterministic hash of a dictionary.

    The dictionary is serialized to JSON with sorted keys to ensure determinism.

    Args:
        data: Dictionary to hash
        algorithm: Hash algorithm to use (default: "sha256")

    Returns:
        Hexadecimal hash string

    Example:
        >>> compute_dict_hash({"a": 1, "b": 2})
        '608de49a4600dbb5b173492759792e4a'
    """
    # Sort keys to ensure deterministic serialization
    json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return compute_content_hash(json_str, algorithm=algorithm)


def generate_utterance_id(
    episode_id: str,
    position: int,
    text: str,
    speaker: str,
    start: float,
    end: float,
) -> str:
    """
    Generate a deterministic utterance ID.

    The ID is based on episode_id, position within episode, and content hash.
    Format: utt_{episode_hash[:12]}_{position:06d}_{content_hash[:8]}

    Args:
        episode_id: Episode identifier
        position: Position/index of utterance within episode (0-based)
        text: Utterance text
        speaker: Speaker name
        start: Start time in seconds
        end: End time in seconds

    Returns:
        Deterministic utterance ID

    Example:
        >>> generate_utterance_id("EP001", 0, "Hello", "Speaker1", 0.0, 1.5)
        'utt_e1b4f4e5c2d9_000000_a1b2c3d4'
    """
    # Hash the episode_id for a stable prefix
    episode_hash = compute_content_hash(episode_id)[:12]
    
    # Create content signature from all fields
    content_data = {
        "text": text,
        "speaker": speaker,
        "start": round(start, 2),  # Round to avoid floating point differences
        "end": round(end, 2),
    }
    content_hash = compute_dict_hash(content_data)[:8]
    
    # Format: utt_{episode_hash}_{position}_{content_hash}
    return f"utt_{episode_hash}_{position:06d}_{content_hash}"


def generate_span_id(
    episode_id: str,
    position: int,
    speaker: str,
    utterance_ids: List[str],
    text: str,
) -> str:
    """
    Generate a deterministic span ID.

    Format: spn_{episode_hash[:12]}_{position:06d}_{content_hash[:8]}

    Args:
        episode_id: Episode identifier
        position: Position/index of span within episode (0-based)
        speaker: Speaker name
        utterance_ids: List of constituent utterance IDs
        text: Concatenated span text

    Returns:
        Deterministic span ID

    Example:
        >>> generate_span_id("EP001", 0, "Speaker1", ["utt_001", "utt_002"], "Hello world")
        'spn_e1b4f4e5c2d9_000000_f3e2d1c0'
    """
    episode_hash = compute_content_hash(episode_id)[:12]
    
    # Content signature includes speaker, utterance IDs, and text
    content_data = {
        "speaker": speaker,
        "utterance_ids": sorted(utterance_ids),  # Sort for determinism
        "text_hash": compute_content_hash(text)[:16],  # Use hash of text to keep ID manageable
    }
    content_hash = compute_dict_hash(content_data)[:8]
    
    return f"spn_{episode_hash}_{position:06d}_{content_hash}"


def generate_beat_id(
    episode_id: str,
    position: int,
    span_ids: List[str],
    text: str,
) -> str:
    """
    Generate a deterministic beat ID.

    Format: bet_{episode_hash[:12]}_{position:06d}_{content_hash[:8]}

    Args:
        episode_id: Episode identifier
        position: Position/index of beat within episode (0-based)
        span_ids: List of constituent span IDs
        text: Concatenated beat text

    Returns:
        Deterministic beat ID

    Example:
        >>> generate_beat_id("EP001", 0, ["spn_001", "spn_002"], "Topic content")
        'bet_e1b4f4e5c2d9_000000_a7b6c5d4'
    """
    episode_hash = compute_content_hash(episode_id)[:12]
    
    content_data = {
        "span_ids": sorted(span_ids),
        "text_hash": compute_content_hash(text)[:16],
    }
    content_hash = compute_dict_hash(content_data)[:8]
    
    return f"bet_{episode_hash}_{position:06d}_{content_hash}"


def generate_section_id(
    episode_id: str,
    position: int,
    beat_ids: List[str],
    text: str,
) -> str:
    """
    Generate a deterministic section ID.

    Format: sec_{episode_hash[:12]}_{position:06d}_{content_hash[:8]}

    Args:
        episode_id: Episode identifier
        position: Position/index of section within episode (0-based)
        beat_ids: List of constituent beat IDs
        text: Concatenated section text

    Returns:
        Deterministic section ID

    Example:
        >>> generate_section_id("EP001", 0, ["bet_001", "bet_002"], "Section content")
        'sec_e1b4f4e5c2d9_000000_d4c5b6a7'
    """
    episode_hash = compute_content_hash(episode_id)[:12]
    
    content_data = {
        "beat_ids": sorted(beat_ids),
        "text_hash": compute_content_hash(text)[:16],
    }
    content_hash = compute_dict_hash(content_data)[:8]
    
    return f"sec_{episode_hash}_{position:06d}_{content_hash}"


def validate_id_format(id_string: str, id_type: str) -> bool:
    """
    Validate that an ID string matches the expected format for its type.

    Args:
        id_string: ID string to validate
        id_type: Type of ID ("utterance", "span", "beat", "section")

    Returns:
        True if valid, False otherwise

    Example:
        >>> validate_id_format("utt_e1b4f4e5c2d9_000000_a1b2c3d4", "utterance")
        True
        >>> validate_id_format("invalid", "utterance")
        False
    """
    if not id_string:
        return False
    
    prefix_map = {
        "utterance": "utt_",
        "span": "spn_",
        "beat": "bet_",
        "section": "sec_",
    }
    
    expected_prefix = prefix_map.get(id_type)
    if not expected_prefix:
        return False
    
    if not id_string.startswith(expected_prefix):
        return False
    
    # Basic format validation: prefix_hash_position_hash
    parts = id_string.split("_")
    if len(parts) != 4:
        return False
    
    # Validate hash and position components
    _, episode_hash, position, content_hash = parts
    
    if len(episode_hash) != 12 or len(content_hash) != 8:
        return False
    
    if len(position) != 6 or not position.isdigit():
        return False
    
    return True

