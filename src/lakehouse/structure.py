"""
Lakehouse directory structure management.

Creates and manages the versioned directory layout for the lakehouse,
including raw data, normalized layers, aggregations, embeddings, and catalogs.
"""

import json
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime

from lakehouse.logger import get_default_logger


logger = get_default_logger()


# Standard lakehouse directory structure
LAKEHOUSE_DIRECTORIES = [
    "raw",
    "normalized",
    "spans",
    "beats",
    "sections",
    "embeddings",
    "ann_index",
    "catalogs",
    "catalogs/validation_reports",
]

# Versioned directories (support multiple processing versions)
VERSIONED_DIRECTORIES = [
    "normalized",
    "spans",
    "beats",
    "sections",
    "embeddings",
    "ann_index",
]


class LakehouseStructure:
    """
    Manages the lakehouse directory structure.
    """
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize lakehouse structure manager.
        
        Args:
            base_path: Base directory for the lakehouse
        """
        self.base_path = Path(base_path)
        logger.info(f"Initialized lakehouse structure at {self.base_path}")
    
    def initialize(self, version: str = "v1") -> None:
        """
        Initialize the complete lakehouse directory structure.
        
        Creates all necessary directories including versioned subdirectories.
        
        Args:
            version: Initial version string (default: "v1")
        
        Example:
            >>> structure = LakehouseStructure("/data/lakehouse")
            >>> structure.initialize()
        """
        logger.info(f"Initializing lakehouse structure with version {version}")
        
        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create all standard directories
        for directory in LAKEHOUSE_DIRECTORIES:
            dir_path = self.base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
        
        # Create versioned subdirectories
        for directory in VERSIONED_DIRECTORIES:
            versioned_path = self.base_path / directory / version
            versioned_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created versioned directory: {versioned_path}")
        
        # Create metadata file
        self._write_metadata(version)
        
        logger.info(f"Lakehouse structure initialized successfully at {self.base_path}")
    
    def get_raw_path(self) -> Path:
        """Get path to raw data directory."""
        return self.base_path / "raw"
    
    def get_normalized_path(self, version: str = "v1") -> Path:
        """Get path to normalized data directory."""
        return self.base_path / "normalized" / version
    
    def get_spans_path(self, version: str = "v1") -> Path:
        """Get path to spans directory."""
        return self.base_path / "spans" / version
    
    def get_beats_path(self, version: str = "v1") -> Path:
        """Get path to beats directory."""
        return self.base_path / "beats" / version
    
    def get_sections_path(self, version: str = "v1") -> Path:
        """Get path to sections directory."""
        return self.base_path / "sections" / version
    
    def get_embeddings_path(self, version: str = "v1") -> Path:
        """Get path to embeddings directory."""
        return self.base_path / "embeddings" / version
    
    def get_ann_index_path(self, version: str = "v1") -> Path:
        """Get path to ANN index directory."""
        return self.base_path / "ann_index" / version
    
    def get_catalogs_path(self) -> Path:
        """Get path to catalogs directory."""
        return self.base_path / "catalogs"
    
    def get_validation_reports_path(self) -> Path:
        """Get path to validation reports directory."""
        return self.base_path / "catalogs" / "validation_reports"
    
    def create_new_version(self, version: str) -> None:
        """
        Create a new version of all versioned directories.
        
        Args:
            version: Version string (e.g., "v2", "v3")
        
        Example:
            >>> structure.create_new_version("v2")
        """
        logger.info(f"Creating new version: {version}")
        
        for directory in VERSIONED_DIRECTORIES:
            versioned_path = self.base_path / directory / version
            versioned_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created versioned directory: {versioned_path}")
        
        logger.info(f"Version {version} created successfully")
    
    def list_versions(self, artifact_type: str = "normalized") -> list:
        """
        List all available versions for an artifact type.
        
        Args:
            artifact_type: Type of artifact ("normalized", "spans", "beats", etc.)
        
        Returns:
            Sorted list of version strings
        
        Example:
            >>> structure.list_versions("normalized")
            ['v1', 'v2', 'v3']
        """
        artifact_dir = self.base_path / artifact_type
        
        if not artifact_dir.exists():
            return []
        
        versions = [
            d.name for d in artifact_dir.iterdir()
            if d.is_dir() and d.name.startswith("v")
        ]
        
        return sorted(versions)
    
    def get_latest_version(self, artifact_type: str = "normalized") -> Optional[str]:
        """
        Get the latest version for an artifact type.
        
        Args:
            artifact_type: Type of artifact
        
        Returns:
            Latest version string or None if no versions exist
        
        Example:
            >>> structure.get_latest_version("normalized")
            'v3'
        """
        versions = self.list_versions(artifact_type)
        return versions[-1] if versions else None
    
    def exists(self) -> bool:
        """Check if lakehouse structure exists."""
        return self.base_path.exists() and (self.base_path / "raw").exists()
    
    def _write_metadata(self, version: str) -> None:
        """
        Write metadata file about the lakehouse structure.
        
        Args:
            version: Current version string
        """
        metadata = {
            "created_at": datetime.now().isoformat(),
            "lakehouse_path": str(self.base_path),
            "initial_version": version,
            "structure_version": "1.0",
            "directories": LAKEHOUSE_DIRECTORIES,
            "versioned_directories": VERSIONED_DIRECTORIES,
        }
        
        metadata_path = self.base_path / "lakehouse_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"Written metadata to {metadata_path}")
    
    def get_structure_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the lakehouse structure.
        
        Returns:
            Dictionary with structure information
        
        Example:
            >>> summary = structure.get_structure_summary()
            >>> summary["exists"]
            True
        """
        summary = {
            "base_path": str(self.base_path),
            "exists": self.exists(),
            "directories": {},
        }
        
        if self.exists():
            for directory in LAKEHOUSE_DIRECTORIES:
                dir_path = self.base_path / directory
                summary["directories"][directory] = {
                    "exists": dir_path.exists(),
                    "path": str(dir_path),
                }
                
                # Add version info for versioned directories
                if directory in VERSIONED_DIRECTORIES:
                    summary["directories"][directory]["versions"] = self.list_versions(directory)
        
        return summary


def initialize_lakehouse(
    base_path: Union[str, Path],
    version: str = "v1",
) -> LakehouseStructure:
    """
    Initialize a new lakehouse directory structure.
    
    Convenience function that creates the structure and returns the manager.
    
    Args:
        base_path: Base directory for the lakehouse
        version: Initial version string (default: "v1")
    
    Returns:
        LakehouseStructure instance
    
    Example:
        >>> structure = initialize_lakehouse("/data/transcript-lakehouse")
        >>> structure.get_normalized_path()
        PosixPath('/data/transcript-lakehouse/normalized/v1')
    """
    structure = LakehouseStructure(base_path)
    structure.initialize(version=version)
    return structure


def get_or_create_lakehouse(
    base_path: Union[str, Path],
    version: str = "v1",
) -> LakehouseStructure:
    """
    Get existing lakehouse structure or create if it doesn't exist.
    
    Args:
        base_path: Base directory for the lakehouse
        version: Version string (default: "v1")
    
    Returns:
        LakehouseStructure instance
    
    Example:
        >>> structure = get_or_create_lakehouse("/data/lakehouse")
    """
    structure = LakehouseStructure(base_path)
    
    if not structure.exists():
        logger.info(f"Lakehouse not found at {base_path}, initializing...")
        structure.initialize(version=version)
    else:
        logger.info(f"Using existing lakehouse at {base_path}")
    
    return structure

