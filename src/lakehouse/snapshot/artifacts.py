"""
Artifact discovery and file collection for lakehouse snapshots.

Handles discovery of all consumer-facing artifacts, file copying with checksums,
and snapshot directory structure creation.
"""

import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq

from lakehouse.logger import get_default_logger


logger = get_default_logger()


class ArtifactDiscovery:
    """
    Discovers and catalogs lakehouse artifacts for snapshot creation.
    
    Scans the lakehouse directory structure to find all consumer-facing
    artifacts including spans, beats, sections, embeddings, indexes, and catalogs.
    """
    
    def __init__(self, lakehouse_path: Path):
        """
        Initialize artifact discovery.
        
        Args:
            lakehouse_path: Path to lakehouse base directory
        """
        self.lakehouse_path = Path(lakehouse_path)
        logger.info(f"Initialized artifact discovery for: {self.lakehouse_path}")
    
    def discover_all(self, version: str = "v1") -> Dict[str, List[Path]]:
        """
        Discover all consumer-facing artifacts.
        
        Args:
            version: Version of artifacts to discover (default: "v1")
        
        Returns:
            Dictionary mapping artifact type to list of file paths
        
        Example:
            >>> discovery = ArtifactDiscovery(Path("lakehouse"))
            >>> artifacts = discovery.discover_all("v1")
            >>> artifacts.keys()
            dict_keys(['spans', 'beats', 'sections', 'embeddings', 'indexes', 'catalogs'])
        """
        artifacts = {
            "spans": [],
            "beats": [],
            "sections": [],
            "embeddings": [],
            "indexes": [],
            "catalogs": [],
        }
        
        logger.info(f"Discovering artifacts for version: {version}")
        
        # Discover each artifact type
        artifacts["spans"] = self._discover_spans(version)
        artifacts["beats"] = self._discover_beats(version)
        artifacts["sections"] = self._discover_sections(version)
        artifacts["embeddings"] = self._discover_embeddings(version)
        artifacts["indexes"] = self._discover_indexes(version)
        artifacts["catalogs"] = self._discover_catalogs()
        
        # Log discovery summary
        total_files = sum(len(files) for files in artifacts.values())
        logger.info(f"Discovered {total_files} artifacts across {len(artifacts)} categories")
        
        for artifact_type, files in artifacts.items():
            if files:
                logger.debug(f"  {artifact_type}: {len(files)} files")
        
        return artifacts
    
    def _discover_spans(self, version: str) -> List[Path]:
        """Discover span artifacts."""
        spans_dir = self.lakehouse_path / "spans" / version
        return self._find_files(spans_dir, "*.parquet")
    
    def _discover_beats(self, version: str) -> List[Path]:
        """Discover beat artifacts."""
        beats_dir = self.lakehouse_path / "beats" / version
        return self._find_files(beats_dir, "*.parquet")
    
    def _discover_sections(self, version: str) -> List[Path]:
        """Discover section artifacts."""
        sections_dir = self.lakehouse_path / "sections" / version
        return self._find_files(sections_dir, "*.parquet")
    
    def _discover_embeddings(self, version: str) -> List[Path]:
        """Discover embedding artifacts."""
        embeddings_dir = self.lakehouse_path / "embeddings" / version
        return self._find_files(embeddings_dir, "*.parquet")
    
    def _discover_indexes(self, version: str) -> List[Path]:
        """Discover ANN index artifacts."""
        indexes_dir = self.lakehouse_path / "ann_index" / version
        files = []
        
        # Find FAISS index files and their metadata
        files.extend(self._find_files(indexes_dir, "*.faiss"))
        files.extend(self._find_files(indexes_dir, "*.json"))
        
        return files
    
    def _discover_catalogs(self) -> List[Path]:
        """Discover catalog artifacts."""
        catalogs_dir = self.lakehouse_path / "catalogs"
        files = []
        
        # Find Parquet and JSON catalog files
        files.extend(self._find_files(catalogs_dir, "*.parquet"))
        files.extend(self._find_files(catalogs_dir, "*.json"))
        
        return files
    
    def _find_files(self, directory: Path, pattern: str) -> List[Path]:
        """
        Find files matching pattern in directory.
        
        Args:
            directory: Directory to search
            pattern: Glob pattern to match
        
        Returns:
            List of matching file paths
        """
        if not directory.exists():
            logger.debug(f"Directory does not exist: {directory}")
            return []
        
        files = list(directory.glob(pattern))
        logger.debug(f"Found {len(files)} files matching {pattern} in {directory}")
        
        return files


def discover_artifacts(lakehouse_path: Path, version: str = "v1") -> Dict[str, List[Path]]:
    """
    Discover all consumer-facing artifacts in lakehouse.
    
    Convenience function that creates an ArtifactDiscovery instance and
    discovers all artifacts. Returns categorized dictionary of artifact paths.
    
    Args:
        lakehouse_path: Path to lakehouse base directory
        version: Version of artifacts to discover (default: "v1")
    
    Returns:
        Dictionary mapping artifact type to list of file paths:
        - 'spans': List of span Parquet files
        - 'beats': List of beat Parquet files
        - 'sections': List of section Parquet files
        - 'embeddings': List of embedding Parquet files
        - 'indexes': List of ANN index files (.faiss, .json)
        - 'catalogs': List of catalog files (.parquet, .json)
    
    Example:
        >>> artifacts = discover_artifacts(Path("lakehouse"), "v1")
        >>> len(artifacts['spans'])
        5
    """
    discovery = ArtifactDiscovery(lakehouse_path)
    return discovery.discover_all(version)


def find_latest_qa_report(base_path: Path) -> Optional[Path]:
    """
    Find the most recent QA report in quality_reports directory.
    
    Scans the quality_reports/ directory for timestamped subdirectories
    (format: YYYYMMDD_HHMMSS) and returns the path to quality_assessment.md
    in the most recent one.
    
    Args:
        base_path: Base path containing quality_reports/ directory
    
    Returns:
        Path to quality_assessment.md in most recent report, or None if not found
    
    Example:
        >>> qa_report = find_latest_qa_report(Path("."))
        >>> qa_report
        PosixPath('quality_reports/20251026_205140/report/quality_assessment.md')
    """
    qa_reports_dir = base_path / "quality_reports"
    
    if not qa_reports_dir.exists():
        logger.debug(f"QA reports directory does not exist: {qa_reports_dir}")
        return None
    
    # Find all timestamped directories
    timestamped_dirs = []
    for entry in qa_reports_dir.iterdir():
        if entry.is_dir() and len(entry.name) == 15:  # YYYYMMDD_HHMMSS format
            try:
                # Validate format by checking if name matches pattern
                parts = entry.name.split('_')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    timestamped_dirs.append(entry)
            except (ValueError, IndexError):
                continue
    
    if not timestamped_dirs:
        logger.debug(f"No timestamped QA report directories found in {qa_reports_dir}")
        return None
    
    # Sort by directory name (timestamp) to get most recent
    latest_dir = sorted(timestamped_dirs, key=lambda d: d.name, reverse=True)[0]
    
    # Look for quality_assessment.md in report subdirectory
    qa_report_path = latest_dir / "report" / "quality_assessment.md"
    
    if qa_report_path.exists():
        logger.info(f"Found latest QA report: {qa_report_path}")
        return qa_report_path
    else:
        logger.warning(f"QA report directory exists but quality_assessment.md not found: {latest_dir}")
        return None


def calculate_sha256(file_path: Path) -> str:
    """
    Calculate SHA-256 checksum of a file.
    
    Reads file in chunks to handle large files efficiently.
    
    Args:
        file_path: Path to file to checksum
    
    Returns:
        Hexadecimal SHA-256 checksum string
    
    Raises:
        FileNotFoundError: If file does not exist
        IOError: If file cannot be read
    
    Example:
        >>> checksum = calculate_sha256(Path("data.parquet"))
        >>> len(checksum)
        64
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    sha256_hash = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            # Read in 64KB chunks for efficiency
            for chunk in iter(lambda: f.read(65536), b""):
                sha256_hash.update(chunk)
        
        checksum = sha256_hash.hexdigest()
        logger.debug(f"Calculated SHA-256 for {file_path.name}: {checksum[:16]}...")
        return checksum
    
    except IOError as e:
        logger.error(f"Failed to read file for checksum: {file_path}: {e}")
        raise IOError(f"Cannot read file for checksum calculation: {file_path}") from e


def get_parquet_row_count(file_path: Path) -> Optional[int]:
    """
    Get row count from Parquet file using PyArrow.
    
    Efficiently reads metadata without loading full file into memory.
    
    Args:
        file_path: Path to Parquet file
    
    Returns:
        Number of rows in Parquet file, or None if file is not Parquet or cannot be read
    
    Example:
        >>> row_count = get_parquet_row_count(Path("spans.parquet"))
        >>> row_count
        50000
    """
    if not file_path.exists():
        logger.warning(f"Parquet file not found: {file_path}")
        return None
    
    # Only process .parquet files
    if file_path.suffix.lower() != ".parquet":
        logger.debug(f"File is not Parquet: {file_path}")
        return None
    
    try:
        # Read Parquet metadata without loading data
        parquet_file = pq.ParquetFile(file_path)
        row_count = parquet_file.metadata.num_rows
        logger.debug(f"Parquet file {file_path.name} has {row_count:,} rows")
        return row_count
    
    except Exception as e:
        logger.warning(f"Failed to read Parquet metadata from {file_path}: {e}")
        return None


def copy_artifact_with_metadata(src: Path, dest: Path) -> Dict[str, Any]:
    """
    Copy artifact file and collect metadata.
    
    Copies the source file to destination and calculates:
    - File size (bytes)
    - SHA-256 checksum
    - Row count (for Parquet files)
    
    Args:
        src: Source file path
        dest: Destination file path
    
    Returns:
        Dictionary with metadata:
        - 'path': Relative path (just filename)
        - 'bytes': File size in bytes
        - 'sha256': SHA-256 checksum
        - 'rows': Row count for Parquet files, None otherwise
    
    Raises:
        FileNotFoundError: If source file does not exist
        IOError: If copy operation fails
    
    Example:
        >>> metadata = copy_artifact_with_metadata(
        ...     Path("lakehouse/spans/v1/spans.parquet"),
        ...     Path("snapshot/spans/spans.parquet")
        ... )
        >>> metadata['bytes']
        12345678
        >>> metadata['sha256']
        'abc123...'
        >>> metadata['rows']
        50000
    """
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    try:
        # Ensure destination directory exists
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(src, dest)
        logger.debug(f"Copied {src.name} to {dest}")
        
        # Get file size
        file_size = dest.stat().st_size
        
        # Calculate checksum
        checksum = calculate_sha256(dest)
        
        # Get row count for Parquet files
        row_count = get_parquet_row_count(dest)
        
        metadata = {
            "path": dest.name,  # Just the filename for now
            "bytes": file_size,
            "sha256": checksum,
            "rows": row_count,
        }
        
        logger.info(
            f"Copied {src.name}: {file_size:,} bytes, "
            f"checksum {checksum[:16]}..., "
            f"rows={row_count if row_count else 'N/A'}"
        )
        
        return metadata
    
    except (IOError, OSError) as e:
        logger.error(f"Failed to copy file from {src} to {dest}: {e}")
        raise IOError(f"Failed to copy artifact: {src} -> {dest}") from e


def create_snapshot_structure(snapshot_root: Path, version: str) -> Dict[str, Path]:
    """
    Create snapshot directory structure.
    
    Creates the versioned snapshot directory and all required subdirectories
    for organizing different artifact types.
    
    Args:
        snapshot_root: Base path for snapshots
        version: Version string for this snapshot (e.g., "1.0.0-provisional")
    
    Returns:
        Dictionary mapping directory names to their paths:
        - 'root': Snapshot root directory
        - 'spans': Spans subdirectory
        - 'beats': Beats subdirectory
        - 'sections': Sections subdirectory
        - 'embeddings': Embeddings subdirectory
        - 'indexes': ANN indexes subdirectory
        - 'catalogs': Catalogs subdirectory
        - 'quality_report': QA report subdirectory
    
    Raises:
        OSError: If directory creation fails
    
    Example:
        >>> dirs = create_snapshot_structure(Path("snapshots"), "1.0.0-provisional")
        >>> dirs['root']
        PosixPath('snapshots/v1.0.0-provisional')
        >>> dirs['spans']
        PosixPath('snapshots/v1.0.0-provisional/spans')
    """
    # Create versioned snapshot directory (with 'v' prefix)
    snapshot_dir = snapshot_root / f"v{version}"
    
    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created snapshot directory: {snapshot_dir}")
        
        # Define subdirectories
        subdirs = {
            "root": snapshot_dir,
            "spans": snapshot_dir / "spans",
            "beats": snapshot_dir / "beats",
            "sections": snapshot_dir / "sections",
            "embeddings": snapshot_dir / "embeddings",
            "indexes": snapshot_dir / "indexes",
            "catalogs": snapshot_dir / "catalogs",
            "quality_report": snapshot_dir / "quality_report",
        }
        
        # Create all subdirectories
        for name, path in subdirs.items():
            if name != "root":  # root already created
                path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created subdirectory: {path.name}/")
        
        logger.info(f"Created snapshot structure with {len(subdirs) - 1} subdirectories")
        
        return subdirs
    
    except OSError as e:
        logger.error(f"Failed to create snapshot structure at {snapshot_dir}: {e}")
        raise OSError(f"Failed to create snapshot directory structure: {e}") from e


def copy_all_artifacts(
    artifacts: Dict[str, List[Path]],
    dest_dirs: Dict[str, Path],
    qa_report_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Copy all discovered artifacts to snapshot with metadata collection.
    
    Copies artifacts from each category to their respective destination
    directories. Calculates checksums and collects metadata for each file.
    Fails immediately if any artifact is missing or unreadable (FR-10).
    
    Args:
        artifacts: Dictionary mapping artifact types to lists of source paths
        dest_dirs: Dictionary mapping artifact types to destination directories
        qa_report_path: Optional path to QA report to include in snapshot
    
    Returns:
        List of file metadata dictionaries, one per copied file
    
    Raises:
        FileNotFoundError: If any required artifact is missing
        IOError: If any artifact cannot be read or copied
    
    Example:
        >>> artifacts = {
        ...     'spans': [Path("lakehouse/spans/v1/spans.parquet")],
        ...     'beats': [Path("lakehouse/beats/v1/beats.parquet")],
        ... }
        >>> dest_dirs = {
        ...     'spans': Path("snapshot/spans"),
        ...     'beats': Path("snapshot/beats"),
        ... }
        >>> metadata_list = copy_all_artifacts(artifacts, dest_dirs)
        >>> len(metadata_list)
        2
    """
    all_metadata = []
    total_files = sum(len(files) for files in artifacts.values())
    
    # Add QA report if present
    if qa_report_path:
        total_files += 1
    
    logger.info(f"Copying {total_files} artifacts to snapshot...")
    
    # Map artifact types to destination directory keys
    type_to_dir_key = {
        "spans": "spans",
        "beats": "beats",
        "sections": "sections",
        "embeddings": "embeddings",
        "indexes": "indexes",
        "catalogs": "catalogs",
    }
    
    # Copy each artifact type
    for artifact_type, source_files in artifacts.items():
        if not source_files:
            logger.debug(f"No {artifact_type} artifacts to copy")
            continue
        
        # Get destination directory
        dir_key = type_to_dir_key.get(artifact_type)
        if not dir_key or dir_key not in dest_dirs:
            logger.warning(f"No destination directory for artifact type: {artifact_type}")
            continue
        
        dest_dir = dest_dirs[dir_key]
        
        logger.info(f"Copying {len(source_files)} {artifact_type} artifacts...")
        
        # Copy each file
        for source_file in source_files:
            # Validate source file exists (FR-10: fail immediately if missing)
            if not source_file.exists():
                error_msg = f"Required artifact missing: {source_file}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # Validate source file is readable (FR-10)
            if not source_file.is_file():
                error_msg = f"Required artifact is not a file: {source_file}"
                logger.error(error_msg)
                raise IOError(error_msg)
            
            try:
                # Determine destination path
                dest_file = dest_dir / source_file.name
                
                # Copy with metadata collection
                metadata = copy_artifact_with_metadata(source_file, dest_file)
                
                # Update metadata with relative path from snapshot root
                metadata["path"] = f"{dir_key}/{source_file.name}"
                
                all_metadata.append(metadata)
            
            except (FileNotFoundError, IOError) as e:
                # Re-raise with context (FR-10: fail immediately)
                error_msg = f"Failed to copy required artifact {source_file}: {e}"
                logger.error(error_msg)
                raise IOError(error_msg) from e
    
    # Copy QA report if provided
    if qa_report_path:
        logger.info("Copying QA report...")
        
        if not qa_report_path.exists():
            error_msg = f"QA report not found: {qa_report_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            dest_file = dest_dirs["quality_report"] / "quality_assessment.md"
            metadata = copy_artifact_with_metadata(qa_report_path, dest_file)
            metadata["path"] = "quality_report/quality_assessment.md"
            all_metadata.append(metadata)
        
        except (FileNotFoundError, IOError) as e:
            error_msg = f"Failed to copy QA report: {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e
    
    logger.info(f"Successfully copied {len(all_metadata)} files to snapshot")
    
    return all_metadata

