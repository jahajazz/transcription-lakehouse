"""
Speaker role configuration and enrichment for transcription lakehouse.

Loads speaker role configuration from YAML and provides functions to:
- Determine speaker roles based on configuration
- Enrich spans with speaker metadata (canonical name, role, expert flag)
- Enrich beats with speaker-derived fields (speakers set, expert spans, coverage)
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import yaml

from lakehouse.logger import get_default_logger


logger = get_default_logger()


# Valid speaker role values
VALID_ROLES = {"expert", "host", "guest", "caller", "other"}


class SpeakerRoleConfig:
    """
    Configuration for speaker roles and expert classification.
    
    Loads and validates speaker_roles.yaml configuration file.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize speaker role configuration.
        
        Args:
            config_path: Path to speaker_roles.yaml (default: config/speaker_roles.yaml)
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid or missing required keys
        """
        if config_path is None:
            config_path = Path("config") / "speaker_roles.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Speaker roles configuration file not found: {config_path}\n"
                f"Please create {config_path} with required structure."
            )
        
        logger.info(f"Loading speaker roles configuration from {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError(f"Speaker roles configuration file is empty: {config_path}")
        
        # Validate required keys
        self._validate_config(config, config_path)
        
        # Store configuration
        self.experts: List[str] = config.get("experts", [])
        self.roles: Dict[str, str] = config.get("roles", {})
        self.default_role: str = config.get("default_role", "other")
        
        # Create expert set for fast lookup
        self._expert_set: Set[str] = set(self.experts)
        
        logger.info(
            f"Loaded speaker roles config: {len(self.experts)} experts, "
            f"{len(self.roles)} role mappings, default_role='{self.default_role}'"
        )
    
    def _validate_config(self, config: Dict[str, Any], config_path: Path) -> None:
        """
        Validate configuration structure and values.
        
        Args:
            config: Loaded configuration dictionary
            config_path: Path to config file (for error messages)
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required keys
        required_keys = {"experts", "default_role"}
        missing_keys = required_keys - set(config.keys())
        
        if missing_keys:
            raise ValueError(
                f"Speaker roles configuration missing required keys: {missing_keys}\n"
                f"File: {config_path}\n"
                f"Required keys: {required_keys}"
            )
        
        # Validate experts is a list
        if not isinstance(config["experts"], list):
            raise ValueError(
                f"'experts' must be a list of speaker names\n"
                f"File: {config_path}"
            )
        
        # Validate default_role is valid
        default_role = config.get("default_role")
        if default_role not in VALID_ROLES:
            raise ValueError(
                f"'default_role' must be one of {VALID_ROLES}, got '{default_role}'\n"
                f"File: {config_path}"
            )
        
        # Validate roles map if present
        roles = config.get("roles", {})
        if not isinstance(roles, dict):
            raise ValueError(
                f"'roles' must be a dictionary mapping speaker names to roles\n"
                f"File: {config_path}"
            )
        
        # Validate all role values
        for speaker, role in roles.items():
            if role not in VALID_ROLES:
                raise ValueError(
                    f"Invalid role '{role}' for speaker '{speaker}'\n"
                    f"Valid roles: {VALID_ROLES}\n"
                    f"File: {config_path}"
                )
    
    def is_expert(self, speaker_name: str) -> bool:
        """
        Check if a speaker is classified as an expert.
        
        Args:
            speaker_name: Canonical speaker name
        
        Returns:
            True if speaker is in experts list, False otherwise
        """
        return speaker_name in self._expert_set
    
    def get_role(self, speaker_name: str) -> str:
        """
        Get the role for a speaker.
        
        Logic:
        1. If speaker in roles map, return that role
        2. If speaker in experts list (but not in roles map), return "expert"
        3. Otherwise, return default_role
        
        Args:
            speaker_name: Canonical speaker name
        
        Returns:
            Role string (one of VALID_ROLES)
        """
        # Check explicit role mapping first
        if speaker_name in self.roles:
            return self.roles[speaker_name]
        
        # If in experts list but no explicit role, default to "expert"
        if speaker_name in self._expert_set:
            return "expert"
        
        # Otherwise use default role
        return self.default_role
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "experts": self.experts,
            "roles": self.roles,
            "default_role": self.default_role,
        }


def determine_speaker_role(
    speaker_name: str,
    config: SpeakerRoleConfig,
) -> tuple[str, str, bool]:
    """
    Determine speaker metadata fields based on configuration.
    
    Args:
        speaker_name: Canonical speaker name from transcript
        config: SpeakerRoleConfig instance
    
    Returns:
        Tuple of (speaker_canonical, speaker_role, is_expert)
        - speaker_canonical: Same as input (canonical name)
        - speaker_role: Role string from config
        - is_expert: Boolean flag if speaker is an expert
    
    Example:
        >>> config = SpeakerRoleConfig()
        >>> determine_speaker_role("Fr Stephen De Young", config)
        ('Fr Stephen De Young', 'expert', True)
    """
    speaker_canonical = speaker_name  # Already canonical from input
    speaker_role = config.get_role(speaker_name)
    is_expert = config.is_expert(speaker_name)
    
    return speaker_canonical, speaker_role, is_expert


def enrich_spans_with_speaker_metadata(
    spans: List[Dict[str, Any]],
    config: SpeakerRoleConfig,
) -> List[Dict[str, Any]]:
    """
    Enrich span records with speaker metadata fields.
    
    Adds three new fields to each span:
    - speaker_canonical: Canonical speaker name (same as 'speaker' field)
    - speaker_role: Role classification from config
    - is_expert: Boolean flag if speaker is an expert
    
    Args:
        spans: List of span dictionaries
        config: SpeakerRoleConfig instance
    
    Returns:
        List of enriched span dictionaries (modifies in place and returns)
    
    Example:
        >>> config = SpeakerRoleConfig()
        >>> spans = [{"speaker": "Fr Stephen De Young", "text": "..."}]
        >>> enriched = enrich_spans_with_speaker_metadata(spans, config)
        >>> enriched[0]["is_expert"]
        True
    """
    for span in spans:
        speaker_name = span.get("speaker", "")
        
        if not speaker_name:
            logger.warning(f"Span {span.get('span_id', 'unknown')} has no speaker field")
            # Use default values for missing speaker
            span["speaker_canonical"] = ""
            span["speaker_role"] = config.default_role
            span["is_expert"] = False
        else:
            speaker_canonical, speaker_role, is_expert = determine_speaker_role(
                speaker_name, config
            )
            span["speaker_canonical"] = speaker_canonical
            span["speaker_role"] = speaker_role
            span["is_expert"] = is_expert
    
    return spans


def enrich_beats_with_speaker_metadata(
    beats: List[Dict[str, Any]],
    spans: List[Dict[str, Any]],
    config: SpeakerRoleConfig,
) -> List[Dict[str, Any]]:
    """
    Enrich beat records with speaker-derived metadata fields.
    
    Adds three new fields to each beat:
    - speakers_set: List of unique canonical speaker names from member spans
    - expert_span_ids: List of span IDs where is_expert=True
    - expert_coverage_pct: Percentage of beat content spoken by experts (0-100)
    
    Args:
        beats: List of beat dictionaries
        spans: List of span dictionaries (must include speaker metadata)
        config: SpeakerRoleConfig instance
    
    Returns:
        List of enriched beat dictionaries (modifies in place and returns)
    
    Example:
        >>> beats = [{"beat_id": "b1", "span_ids": ["s1", "s2"]}]
        >>> spans = [
        ...     {"span_id": "s1", "is_expert": True, "speaker_canonical": "Expert"},
        ...     {"span_id": "s2", "is_expert": False, "speaker_canonical": "Other"}
        ... ]
        >>> enriched = enrich_beats_with_speaker_metadata(beats, spans, config)
        >>> enriched[0]["expert_span_ids"]
        ['s1']
    """
    # Create span lookup dictionary for fast access
    span_lookup = {span["span_id"]: span for span in spans if "span_id" in span}
    
    for beat in beats:
        span_ids = beat.get("span_ids", [])
        
        if not span_ids:
            logger.warning(f"Beat {beat.get('beat_id', 'unknown')} has no span_ids")
            beat["speakers_set"] = []
            beat["expert_span_ids"] = []
            beat["expert_coverage_pct"] = 0.0
            continue
        
        # Collect speaker names and expert span IDs
        speakers = []
        expert_span_ids = []
        
        for span_id in span_ids:
            span = span_lookup.get(span_id)
            if span is None:
                logger.warning(
                    f"Span {span_id} referenced in beat {beat.get('beat_id', 'unknown')} not found"
                )
                continue
            
            # Collect speaker
            speaker_canonical = span.get("speaker_canonical")
            if speaker_canonical:
                speakers.append(speaker_canonical)
            
            # Collect expert spans
            if span.get("is_expert", False):
                expert_span_ids.append(span_id)
        
        # Deduplicate speakers while preserving order
        speakers_set = []
        seen = set()
        for speaker in speakers:
            if speaker not in seen:
                speakers_set.append(speaker)
                seen.add(speaker)
        
        # Calculate expert coverage percentage
        expert_coverage_pct = calculate_expert_coverage_pct(
            span_ids, span_lookup, expert_span_ids
        )
        
        # Add enriched fields
        beat["speakers_set"] = speakers_set
        beat["expert_span_ids"] = expert_span_ids
        beat["expert_coverage_pct"] = expert_coverage_pct
    
    return beats


def calculate_expert_coverage_pct(
    span_ids: List[str],
    span_lookup: Dict[str, Dict[str, Any]],
    expert_span_ids: List[str],
) -> float:
    """
    Calculate percentage of beat content spoken by experts.
    
    Uses token-weighted calculation if token_count available,
    otherwise falls back to character-weighted (text length).
    
    Args:
        span_ids: All span IDs in the beat
        span_lookup: Dictionary mapping span_id to span record
        expert_span_ids: List of expert span IDs
    
    Returns:
        Percentage (0.0 to 100.0) of expert content
    
    Example:
        >>> span_lookup = {
        ...     "s1": {"token_count": 100, "text": "a" * 500},
        ...     "s2": {"token_count": 50, "text": "b" * 250}
        ... }
        >>> calculate_expert_coverage_pct(["s1", "s2"], span_lookup, ["s1"])
        66.66666666666666
    """
    if not span_ids:
        return 0.0
    
    expert_span_ids_set = set(expert_span_ids)
    
    # Try token-weighted first
    total_tokens = 0
    expert_tokens = 0
    has_token_counts = False
    
    for span_id in span_ids:
        span = span_lookup.get(span_id)
        if span is None:
            continue
        
        token_count = span.get("token_count")
        if token_count is not None and token_count > 0:
            has_token_counts = True
            total_tokens += token_count
            if span_id in expert_span_ids_set:
                expert_tokens += token_count
    
    if has_token_counts and total_tokens > 0:
        return (expert_tokens / total_tokens) * 100.0
    
    # Fallback to character-weighted
    total_chars = 0
    expert_chars = 0
    
    for span_id in span_ids:
        span = span_lookup.get(span_id)
        if span is None:
            continue
        
        text = span.get("text", "")
        char_count = len(text)
        total_chars += char_count
        
        if span_id in expert_span_ids_set:
            expert_chars += char_count
    
    if total_chars > 0:
        return (expert_chars / total_chars) * 100.0
    
    # Edge case: no tokens or text
    logger.warning(
        f"Unable to calculate expert coverage for beat with span_ids {span_ids}: "
        f"no token counts or text found"
    )
    return 0.0

