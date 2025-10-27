"""
Configuration management for lakehouse snapshots.

Handles snapshot configuration loading, version parsing, and snapshot root resolution.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from lakehouse.logger import get_default_logger


logger = get_default_logger()


# Default configuration values
DEFAULT_SNAPSHOT_CONFIG = {
    "version": {
        "major": 0,
        "minor": 1,
        "patch": 0,
    },
    "snapshot_root": "${SNAPSHOT_ROOT:-./snapshots}",
    "lake_root": "${LAKE_ROOT}",
    "auto_increment": True,
    "provisional_suffix": "-provisional",
}


# Semantic version regex pattern
SEMVER_PATTERN = re.compile(
    r"^v?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?P<suffix>[-+].*)?$"
)


class SnapshotConfig:
    """
    Configuration manager for snapshot operations.
    
    Loads configuration from YAML file or uses defaults. Handles version
    management, snapshot root resolution, and configuration validation.
    
    Attributes:
        major: Major version number
        minor: Minor version number
        patch: Patch version number
        snapshot_root_template: Template for base directory where snapshots are created
        lake_root_template: Template for directory where snapshots are consumed from
        auto_increment: Whether to auto-increment version on collision
        provisional_suffix: Suffix to append to provisional snapshots
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize snapshot configuration.
        
        Args:
            config_path: Path to snapshot_config.yaml file (optional)
            overrides: Dictionary of configuration overrides (optional)
        
        Example:
            >>> config = SnapshotConfig()
            >>> config.get_version_string()
            '0.1.0-provisional'
        """
        # Load base configuration
        if config_path and config_path.exists():
            logger.info(f"Loading snapshot config from {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
            config = self._merge_config(DEFAULT_SNAPSHOT_CONFIG, user_config)
        else:
            logger.debug("Using default snapshot configuration")
            config = DEFAULT_SNAPSHOT_CONFIG.copy()
        
        # Apply overrides
        if overrides:
            config = self._merge_config(config, overrides)
        
        # Extract configuration values
        version_config = config.get("version", {})
        self.major: int = version_config.get("major", 0)
        self.minor: int = version_config.get("minor", 1)
        self.patch: int = version_config.get("patch", 0)
        
        self.snapshot_root_template: str = config.get("snapshot_root", "./snapshots")
        self.lake_root_template: str = config.get("lake_root", "${LAKE_ROOT}")
        self.auto_increment: bool = config.get("auto_increment", True)
        self.provisional_suffix: str = config.get("provisional_suffix", "-provisional")
        
        logger.debug(
            f"Snapshot config initialized: v{self.major}.{self.minor}.{self.patch}, "
            f"auto_increment={self.auto_increment}"
        )
    
    def get_version_tuple(self) -> Tuple[int, int, int]:
        """
        Get version as tuple.
        
        Returns:
            Tuple of (major, minor, patch)
        """
        return (self.major, self.minor, self.patch)
    
    def get_version_string(self, include_suffix: bool = True) -> str:
        """
        Get version as formatted string.
        
        Args:
            include_suffix: Whether to include provisional suffix
        
        Returns:
            Version string (e.g., "0.1.0-provisional")
        """
        suffix = self.provisional_suffix if include_suffix else ""
        return format_version(self.major, self.minor, self.patch, suffix)
    
    def _merge_config(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
        
        Returns:
            Merged configuration dictionary
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result


def parse_version(version_string: str) -> Tuple[int, int, int, str]:
    """
    Parse semantic version string into components.
    
    Supports both "v1.2.3" and "1.2.3" formats, with optional suffix
    like "-provisional", "-rc1", "+metadata".
    
    Args:
        version_string: Version string to parse (e.g., "1.0.0-provisional")
    
    Returns:
        Tuple of (major, minor, patch, suffix)
    
    Raises:
        ValueError: If version string is not valid semver format
    
    Example:
        >>> parse_version("1.0.0-provisional")
        (1, 0, 0, '-provisional')
        >>> parse_version("v2.3.4")
        (2, 3, 4, '')
    """
    match = SEMVER_PATTERN.match(version_string)
    
    if not match:
        raise ValueError(
            f"Invalid semantic version string: '{version_string}'. "
            f"Expected format: 'MAJOR.MINOR.PATCH[-suffix]'"
        )
    
    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch"))
    suffix = match.group("suffix") or ""
    
    return (major, minor, patch, suffix)


def format_version(major: int, minor: int, patch: int, suffix: str = "") -> str:
    """
    Format version components into semantic version string.
    
    Args:
        major: Major version number
        minor: Minor version number
        patch: Patch version number
        suffix: Optional suffix (e.g., "-provisional", "-rc1")
    
    Returns:
        Formatted version string (e.g., "1.0.0-provisional")
    
    Example:
        >>> format_version(1, 0, 0, "-provisional")
        '1.0.0-provisional'
        >>> format_version(2, 3, 4)
        '2.3.4'
    """
    version = f"{major}.{minor}.{patch}"
    if suffix:
        version += suffix
    return version


def find_existing_snapshots(snapshot_root: Path) -> List[str]:
    """
    Find all existing snapshot versions in snapshot root directory.
    
    Scans the snapshot_root directory for subdirectories matching the
    version pattern (vMAJOR.MINOR.PATCH[-suffix]).
    
    Args:
        snapshot_root: Root directory containing snapshots
    
    Returns:
        List of version strings found (e.g., ["1.0.0-provisional", "1.0.1-provisional"])
    
    Example:
        >>> find_existing_snapshots(Path("snapshots"))
        ['0.1.0-provisional', '0.1.1-provisional']
    """
    if not snapshot_root.exists():
        logger.debug(f"Snapshot root does not exist: {snapshot_root}")
        return []
    
    versions = []
    
    for entry in snapshot_root.iterdir():
        if not entry.is_dir():
            continue
        
        # Try to parse directory name as version
        dir_name = entry.name
        # Strip leading 'v' if present
        version_str = dir_name[1:] if dir_name.startswith("v") else dir_name
        
        try:
            # Validate it's a proper version
            parse_version(version_str)
            versions.append(version_str)
            logger.debug(f"Found existing snapshot: {version_str}")
        except ValueError:
            # Not a valid version directory, skip it
            logger.debug(f"Skipping non-version directory: {dir_name}")
            continue
    
    return sorted(versions, key=lambda v: parse_version(v)[:3])


def get_next_version(
    current_version: Tuple[int, int, int],
    existing_versions: List[str],
    suffix: str = "",
) -> str:
    """
    Get next available version, incrementing patch if collision exists.
    
    If the current version exists in existing_versions, increments the patch
    number until finding a non-colliding version.
    
    Args:
        current_version: Tuple of (major, minor, patch)
        existing_versions: List of existing version strings
        suffix: Optional suffix to append (e.g., "-provisional")
    
    Returns:
        Next available version string
    
    Example:
        >>> get_next_version((1, 0, 0), ["1.0.0-provisional"], "-provisional")
        '1.0.1-provisional'
        >>> get_next_version((1, 0, 0), [], "-provisional")
        '1.0.0-provisional'
    """
    major, minor, patch = current_version
    
    # Parse existing versions to extract just the numeric parts
    existing_tuples = []
    for ver_str in existing_versions:
        try:
            maj, min_, pat, _ = parse_version(ver_str)
            existing_tuples.append((maj, min_, pat))
        except ValueError:
            continue
    
    # Keep incrementing patch until we find a non-colliding version
    while (major, minor, patch) in existing_tuples:
        logger.debug(
            f"Version collision detected: {major}.{minor}.{patch}, incrementing patch"
        )
        patch += 1
    
    version_string = format_version(major, minor, patch, suffix)
    logger.info(f"Next available version: {version_string}")
    
    return version_string


def resolve_snapshot_root(config: SnapshotConfig) -> Path:
    """
    Resolve snapshot root path from configuration.
    
    Supports environment variable expansion in the form ${VAR_NAME:-default}.
    Falls back to ./snapshots if not configured.
    
    Args:
        config: SnapshotConfig instance
    
    Returns:
        Resolved Path to snapshot root directory
    
    Example:
        >>> config = SnapshotConfig()
        >>> resolve_snapshot_root(config)
        PosixPath('snapshots')
    """
    template = config.snapshot_root_template
    
    # Handle environment variable expansion: ${VAR_NAME:-default}
    env_pattern = re.compile(r"\$\{([^:}]+)(?::-)?(.*?)\}")
    
    def replace_env(match):
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) else "./snapshots"
        value = os.environ.get(var_name, default_value)
        logger.debug(f"Resolved ${{{var_name}}} to: {value}")
        return value
    
    resolved = env_pattern.sub(replace_env, template)
    snapshot_root = Path(resolved).expanduser().resolve()
    
    logger.info(f"Resolved snapshot root: {snapshot_root}")
    
    return snapshot_root


def resolve_lake_root(config: SnapshotConfig) -> Optional[Path]:
    """
    Resolve lake root path from configuration.
    
    Supports environment variable expansion in the form ${VAR_NAME:-default}.
    Returns None if LAKE_ROOT is not set (optional for producer-only workflows).
    
    Args:
        config: SnapshotConfig instance
    
    Returns:
        Resolved Path to lake root directory, or None if not configured
    
    Example:
        >>> config = SnapshotConfig()
        >>> lake_root = resolve_lake_root(config)
        >>> print(lake_root)  # None if LAKE_ROOT not set
    """
    template = config.lake_root_template
    
    # Handle environment variable expansion: ${VAR_NAME:-default}
    env_pattern = re.compile(r"\$\{([^:}]+)(?::-)?(.*?)\}")
    
    def replace_env(match):
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) else None
        value = os.environ.get(var_name, default_value)
        if value is None:
            logger.debug(f"${{{var_name}}} is not set, lake_root will be None")
        else:
            logger.debug(f"Resolved ${{{var_name}}} to: {value}")
        return value or ""
    
    resolved = env_pattern.sub(replace_env, template)
    
    # If still contains unresolved variables or is empty, return None
    if not resolved or resolved == "None" or "${" in resolved:
        logger.debug("Lake root not configured (LAKE_ROOT not set)")
        return None
    
    lake_root = Path(resolved).expanduser().resolve()
    logger.info(f"Resolved lake root: {lake_root}")
    
    return lake_root


def ensure_snapshot_root_writable(snapshot_root: Path) -> None:
    """
    Ensure snapshot root directory exists and is writable.
    
    Creates the directory if it doesn't exist. Validates write permissions
    by attempting to create the directory.
    
    Args:
        snapshot_root: Path to snapshot root directory
    
    Raises:
        PermissionError: If directory cannot be created or is not writable
        OSError: If directory creation fails for other reasons
    
    Example:
        >>> ensure_snapshot_root_writable(Path("snapshots"))
        # Creates ./snapshots/ if it doesn't exist
    """
    try:
        # Create directory if it doesn't exist
        if not snapshot_root.exists():
            logger.info(f"Creating snapshot root directory: {snapshot_root}")
            snapshot_root.mkdir(parents=True, exist_ok=True)
        
        # Verify directory is writable by attempting to create a test file
        test_file = snapshot_root / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
            logger.debug(f"Verified snapshot root is writable: {snapshot_root}")
        except (PermissionError, OSError) as e:
            raise PermissionError(
                f"Snapshot root directory is not writable: {snapshot_root}. "
                f"Please check permissions."
            ) from e
    
    except PermissionError as e:
        logger.error(f"Permission denied creating snapshot root: {snapshot_root}")
        raise PermissionError(
            f"Cannot create or write to snapshot root directory: {snapshot_root}. "
            f"Please check permissions or specify a different location."
        ) from e
    
    except OSError as e:
        logger.error(f"Failed to create snapshot root: {snapshot_root}: {e}")
        raise OSError(
            f"Failed to create snapshot root directory: {snapshot_root}. "
            f"Error: {e}"
        ) from e

