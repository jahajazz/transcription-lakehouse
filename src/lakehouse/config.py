"""
Configuration loading and validation for the lakehouse package.

Loads YAML configuration files with sensible defaults and supports
environment variable overrides.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# Default configuration values (fallback if files not found)
DEFAULT_AGGREGATION_CONFIG = {
    "spans": {
        "min_duration": 1.0,
        "max_silence_gap": 0.5,
        "break_on_speaker_change": True,
    },
    "beats": {
        "similarity_threshold": 0.7,
        "min_spans_per_beat": 1,
        "max_spans_per_beat": None,
        "window_size": 3,
        "use_embeddings": True,
        "fallback_method": "heuristic",
    },
    "sections": {
        "target_duration_minutes": 8.0,
        "min_duration_minutes": 5.0,
        "max_duration_minutes": 12.0,
        "allow_semantic_overflow": True,
        "boundary_similarity_threshold": 0.5,
        "prefer_time_boundaries": False,
    },
    "general": {
        "preserve_metadata": True,
        "validate_references": True,
        "max_text_length": None,
    },
}

DEFAULT_EMBEDDING_CONFIG = {
    "model": {
        "provider": "local",
        "name": "all-MiniLM-L6-v2",
        "device": "cpu",
        "cache_dir": None,
    },
    "fallback": {
        "enabled": True,
        "provider": "openai",
        "model_name": "text-embedding-3-small",
    },
    "openai": {
        "api_key": None,
        "organization": None,
        "api_base": None,
        "timeout": 30,
        "max_retries": 3,
    },
    "generation": {
        "batch_size": 32,
        "normalize_embeddings": True,
        "max_text_length": 8192,
        "show_progress": True,
        "num_workers": 1,
    },
    "artifacts": {
        "embed_spans": True,
        "embed_beats": True,
        "embed_sections": False,
        "embed_utterances": False,
    },
    "cache": {
        "enabled": True,
        "cache_dir": "embeddings/cache",
        "invalidation": "content_hash",
    },
    "quality": {
        "check_validity": True,
        "min_dimension": 128,
        "max_dimension": 4096,
    },
}

DEFAULT_VALIDATION_CONFIG = {
    "schemas": {
        "enforce_schemas": True,
        "required_fields": {
            "utterance": ["utterance_id", "episode_id", "start", "end", "speaker", "text"],
            "span": ["span_id", "episode_id", "speaker", "start_time", "end_time", "text", "utterance_ids"],
            "beat": ["beat_id", "episode_id", "start_time", "end_time", "text", "span_ids"],
            "section": ["section_id", "episode_id", "start_time", "end_time", "duration_minutes", "text", "beat_ids"],
        },
    },
    "quality_checks": {
        "timestamps": {
            "check_monotonic": True,
            "check_end_after_start": True,
            "check_non_negative": True,
            "max_gap": None,
        },
        "text": {
            "check_non_empty": True,
            "min_length": 1,
            "max_length": None,
            "check_null": True,
        },
        "ids": {
            "check_format": True,
            "check_duplicates": True,
            "check_references": True,
        },
        "numeric": {
            "check_nan": True,
            "check_inf": True,
            "check_non_negative_duration": True,
        },
    },
    "coverage": {
        "min_utterance_to_span_coverage": 95.0,
        "min_span_to_beat_coverage": 90.0,
        "min_section_duration_coverage": 90.0,
        "fail_on_low_coverage": False,
    },
    "sanity_checks": {
        "check_non_empty_tables": True,
        "max_speakers_per_episode": 10,
        "min_utterances_per_episode": 1,
        "max_utterances_per_episode": None,
        "min_episode_duration_minutes": 1.0,
        "max_episode_duration_minutes": 300.0,
        "max_span_to_utterance_ratio": 1.0,
        "max_beat_to_span_ratio": 1.0,
    },
    "error_handling": {
        "fail_fast": False,
        "max_errors_to_report": 100,
        "include_row_details": True,
        "severity_levels": {
            "empty_text": "error",
            "missing_id": "error",
            "duplicate_id": "error",
            "timestamp_order": "warning",
            "low_coverage": "warning",
            "long_text": "info",
        },
    },
    "reporting": {
        "generate_report": True,
        "report_format": "both",
        "include_statistics": True,
        "include_sample_errors": True,
        "sample_errors_per_type": 5,
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary
        override: Override dictionary (takes precedence)

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        file_path: Path to YAML file

    Returns:
        Dictionary with YAML contents

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is not valid YAML
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config if config is not None else {}


def load_config(
    config_dir: Optional[Path] = None,
    config_type: str = "aggregation",
) -> Dict[str, Any]:
    """
    Load configuration from YAML file with fallback to defaults.

    Args:
        config_dir: Directory containing config files (default: ./config)
        config_type: Type of config to load ("aggregation", "embedding", "validation")

    Returns:
        Configuration dictionary

    Example:
        >>> config = load_config(Path("config"), "aggregation")
        >>> threshold = config["beats"]["similarity_threshold"]
    """
    # Determine config directory
    if config_dir is None:
        config_dir = Path("config")
    else:
        config_dir = Path(config_dir)
    
    # Map config type to filename and default
    config_map = {
        "aggregation": ("aggregation_config.yaml", DEFAULT_AGGREGATION_CONFIG),
        "embedding": ("embedding_config.yaml", DEFAULT_EMBEDDING_CONFIG),
        "validation": ("validation_rules.yaml", DEFAULT_VALIDATION_CONFIG),
    }
    
    if config_type not in config_map:
        raise ValueError(f"Invalid config_type: {config_type}. Must be one of {list(config_map.keys())}")
    
    filename, default_config = config_map[config_type]
    config_path = config_dir / filename
    
    # Try to load from file, fall back to defaults
    try:
        user_config = load_yaml_file(config_path)
        # Merge user config with defaults (user config takes precedence)
        config = deep_merge(default_config, user_config)
    except FileNotFoundError:
        # Use defaults if file not found
        config = default_config.copy()
    
    # Apply environment variable overrides for sensitive data
    if config_type == "embedding":
        # Override OpenAI API key from environment variable
        env_api_key = os.environ.get("OPENAI_API_KEY")
        if env_api_key:
            config["openai"]["api_key"] = env_api_key
    
    return config


def load_speaker_roles(config_path: Optional[Path] = None):
    """
    Load speaker roles configuration.
    
    Args:
        config_path: Path to speaker_roles.yaml (default: config/speaker_roles.yaml)
    
    Returns:
        SpeakerRoleConfig instance
    
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config is invalid
    
    Example:
        >>> from lakehouse.config import load_speaker_roles
        >>> config = load_speaker_roles()
        >>> config.is_expert("Fr Stephen De Young")
        True
    """
    from lakehouse.speaker_roles import SpeakerRoleConfig
    
    return SpeakerRoleConfig(config_path)


class Config:
    """
    Configuration manager for lakehouse.

    Provides convenient access to all configuration types.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing config files (default: ./config)
        """
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self._aggregation_config = None
        self._embedding_config = None
        self._validation_config = None

    @property
    def aggregation(self) -> Dict[str, Any]:
        """Get aggregation configuration (lazy load)."""
        if self._aggregation_config is None:
            self._aggregation_config = load_config(self.config_dir, "aggregation")
        return self._aggregation_config

    @property
    def embedding(self) -> Dict[str, Any]:
        """Get embedding configuration (lazy load)."""
        if self._embedding_config is None:
            self._embedding_config = load_config(self.config_dir, "embedding")
        return self._embedding_config

    @property
    def validation(self) -> Dict[str, Any]:
        """Get validation configuration (lazy load)."""
        if self._validation_config is None:
            self._validation_config = load_config(self.config_dir, "validation")
        return self._validation_config

    def reload(self) -> None:
        """Reload all configurations from disk."""
        self._aggregation_config = None
        self._embedding_config = None
        self._validation_config = None

    def get(self, config_type: str, *keys: str, default: Any = None) -> Any:
        """
        Get a nested configuration value by key path.

        Args:
            config_type: Type of config ("aggregation", "embedding", "validation")
            *keys: Keys to traverse (e.g., "beats", "similarity_threshold")
            default: Default value if key path not found

        Returns:
            Configuration value or default

        Example:
            >>> config = Config()
            >>> threshold = config.get("aggregation", "beats", "similarity_threshold")
            0.7
        """
        config_map = {
            "aggregation": self.aggregation,
            "embedding": self.embedding,
            "validation": self.validation,
        }
        
        if config_type not in config_map:
            return default
        
        value = config_map[config_type]
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value

