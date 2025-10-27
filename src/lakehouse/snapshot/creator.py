"""
Snapshot creator for orchestrating lakehouse snapshot creation.

Coordinates all components: configuration, artifact discovery, file copying,
manifest generation, validation, and snapshot note creation.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from lakehouse.snapshot.config import (
    SnapshotConfig,
    resolve_snapshot_root,
    ensure_snapshot_root_writable,
    find_existing_snapshots,
    get_next_version,
)
from lakehouse.snapshot.artifacts import (
    discover_artifacts,
    find_latest_qa_report,
    create_snapshot_structure,
    copy_all_artifacts,
)
from lakehouse.snapshot.manifest import (
    ManifestGenerator,
    write_manifest,
)
from lakehouse.snapshot.validator import validate_snapshot, generate_validation_report
from lakehouse.logger import get_default_logger


logger = get_default_logger()


class SnapshotCreator:
    """
    Orchestrates complete snapshot creation workflow.
    
    Coordinates configuration, artifact discovery, directory creation,
    file copying, manifest generation, validation, and snapshot note creation.
    """
    
    def __init__(self, lakehouse_path: Path, config: Optional[SnapshotConfig] = None):
        """
        Initialize snapshot creator.
        
        Args:
            lakehouse_path: Path to lakehouse base directory
            config: Optional SnapshotConfig instance (creates default if None)
        """
        self.lakehouse_path = Path(lakehouse_path)
        self.config = config if config else SnapshotConfig()
        
        logger.info(f"Initialized snapshot creator for: {self.lakehouse_path}")
    
    def create(
        self,
        version_override: Optional[str] = None,
        lakehouse_version: str = "v1",
    ) -> Dict[str, Any]:
        """
        Create a complete snapshot.
        
        Orchestrates the full workflow:
        1. Resolve configuration and version
        2. Discover artifacts
        3. Create snapshot structure
        4. Copy artifacts with checksums
        5. Generate manifest
        6. Write snapshot note
        7. Validate snapshot
        
        Args:
            version_override: Optional version string to override config
            lakehouse_version: Lakehouse artifact version to snapshot (default: "v1")
        
        Returns:
            Dictionary with snapshot creation results:
            - snapshot_path: Path to created snapshot
            - version: Snapshot version string
            - manifest: Generated manifest dictionary
            - validation_result: Validation result dictionary
            - files_copied: Number of files copied
        
        Example:
            >>> creator = SnapshotCreator(Path("lakehouse"))
            >>> result = creator.create()
            >>> result['validation_result']['status']
            'PASS'
        """
        return create_snapshot(
            lakehouse_path=self.lakehouse_path,
            config=self.config,
            version_override=version_override,
            lakehouse_version=lakehouse_version,
        )


def create_snapshot(
    lakehouse_path: Path,
    config: SnapshotConfig,
    version_override: Optional[str] = None,
    lakehouse_version: str = "v1",
) -> Dict[str, Any]:
    """
    Create a complete lakehouse snapshot.
    
    Orchestrates the entire snapshot creation workflow with all components.
    
    Args:
        lakehouse_path: Path to lakehouse base directory
        config: SnapshotConfig instance
        version_override: Optional version override
        lakehouse_version: Lakehouse version to snapshot
    
    Returns:
        Dictionary with complete snapshot information
    
    Raises:
        FileNotFoundError: If required artifacts are missing
        IOError: If snapshot creation fails
    """
    logger.info("=" * 60)
    logger.info("Starting snapshot creation")
    logger.info("=" * 60)
    
    # Step 1: Resolve snapshot root and ensure writable
    logger.info("Step 1: Resolving snapshot root")
    snapshot_root = resolve_snapshot_root(config)
    ensure_snapshot_root_writable(snapshot_root)
    
    # Step 2: Determine version
    logger.info("Step 2: Determining snapshot version")
    if version_override:
        version = version_override
        logger.info(f"Using version override: {version}")
    else:
        # Get base version from config
        base_version = config.get_version_tuple()
        
        # Check for existing versions
        existing_versions = find_existing_snapshots(snapshot_root)
        
        # Get next available version with auto-increment
        version = get_next_version(
            base_version,
            existing_versions,
            config.provisional_suffix if config.auto_increment else "",
        )
        
        logger.info(f"Determined version: {version}")
    
    # Step 3: Discover artifacts
    logger.info(f"Step 3: Discovering artifacts (version: {lakehouse_version})")
    artifacts = discover_artifacts(lakehouse_path, lakehouse_version)
    
    total_artifacts = sum(len(files) for files in artifacts.values())
    logger.info(f"Discovered {total_artifacts} artifacts")
    
    # Step 4: Find QA report
    logger.info("Step 4: Finding latest QA report")
    qa_report_path = find_latest_qa_report(Path.cwd())  # Search from current directory
    if qa_report_path:
        logger.info(f"Found QA report: {qa_report_path}")
    else:
        logger.info("No QA report found")
    
    # Step 5: Create snapshot structure
    logger.info("Step 5: Creating snapshot directory structure")
    dest_dirs = create_snapshot_structure(snapshot_root, version)
    snapshot_path = dest_dirs["root"]
    logger.info(f"Created snapshot at: {snapshot_path}")
    
    # Step 6: Copy all artifacts
    logger.info("Step 6: Copying artifacts to snapshot")
    files_metadata = copy_all_artifacts(artifacts, dest_dirs, qa_report_path)
    logger.info(f"Copied {len(files_metadata)} files")
    
    # Step 7: Generate manifest
    logger.info("Step 7: Generating manifest")
    manifest_generator = ManifestGenerator(lakehouse_path)
    manifest = manifest_generator.generate(
        version=version,
        files_metadata=files_metadata,
        qa_report_path=qa_report_path,
    )
    
    # Step 8: Write manifest
    logger.info("Step 8: Writing manifest")
    write_manifest(manifest, snapshot_path)
    
    # Step 9: Generate and write snapshot note
    logger.info("Step 9: Generating snapshot note")
    snapshot_note = generate_snapshot_note(
        snapshot_info={
            "version": version,
            "snapshot_path": snapshot_path,
            "files_copied": len(files_metadata),
        },
        manifest=manifest,
        snapshot_path=snapshot_path,
    )
    write_snapshot_note(snapshot_note, snapshot_path)
    
    # Step 10: Validate snapshot
    logger.info("Step 10: Validating snapshot")
    validation_result = validate_snapshot(snapshot_path)
    
    logger.info("=" * 60)
    logger.info(f"Snapshot creation complete: {validation_result['status']}")
    logger.info("=" * 60)
    
    return {
        "snapshot_path": snapshot_path,
        "version": version,
        "manifest": manifest,
        "validation_result": validation_result,
        "files_copied": len(files_metadata),
    }


def generate_snapshot_note(
    snapshot_info: Dict[str, Any],
    manifest: Dict[str, Any],
    snapshot_path: Path,
) -> str:
    """
    Generate snapshot note content.
    
    Creates comprehensive snapshot_note.txt with version, timestamp,
    QA status, usage instructions, and LAKE_ROOT export command.
    
    Args:
        snapshot_info: Dictionary with snapshot information
        manifest: Generated manifest dictionary
        snapshot_path: Path to snapshot root
    
    Returns:
        Snapshot note content as string
    
    Example:
        >>> note = generate_snapshot_note(info, manifest, Path("snapshot"))
        >>> 'export LAKE_ROOT=' in note
        True
    """
    version = snapshot_info.get("version", "unknown")
    files_copied = snapshot_info.get("files_copied", 0)
    
    # Extract from manifest
    created_at = manifest.get("created_at", "unknown")
    producer = manifest.get("producer", {})
    repo = producer.get("repo", "unknown")
    commit = producer.get("commit", "unknown")
    
    qa_status = manifest.get("qa_status", {})
    qa_state = qa_status.get("state", "UNKNOWN")
    qa_summary = qa_status.get("summary", "No information available")
    
    # Build note content
    lines = []
    
    lines.append("PROVISIONAL LAKEHOUSE SNAPSHOT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Version: v{version}")
    lines.append(f"Created: {created_at}")
    lines.append(f"Producer: {repo} (commit: {commit[:12]}...)")
    lines.append("")
    
    lines.append(f"QUALITY STATUS: {qa_state}")
    lines.append(f"Summary: {qa_summary}")
    lines.append("")
    
    if qa_state != "PASS":
        lines.append("⚠️  WARNING: This is a PROVISIONAL snapshot.")
        lines.append("Quality assessment indicates known issues or is not available.")
        lines.append("Consumers should validate results and be prepared for data")
        lines.append("quality limitations.")
    else:
        lines.append("✓ This snapshot has passed quality assessment.")
        lines.append("However, it is still marked as PROVISIONAL for migration purposes.")
    
    lines.append("")
    lines.append("Intended Use: Migration to desktop environment for local development")
    lines.append("")
    
    lines.append("USAGE")
    lines.append("-" * 60)
    lines.append("To use this snapshot, set the LAKE_ROOT environment variable:")
    lines.append("")
    
    # Generate OS-specific export commands
    snapshot_path_str = str(snapshot_path.resolve())
    
    # Windows (PowerShell)
    lines.append("  # Windows (PowerShell)")
    lines.append(f'  $env:LAKE_ROOT = "{snapshot_path_str}"')
    lines.append("")
    
    # Windows (CMD)
    lines.append("  # Windows (Command Prompt)")
    lines.append(f'  set LAKE_ROOT={snapshot_path_str}')
    lines.append("")
    
    # Unix/Linux/macOS
    lines.append("  # Linux/macOS (bash/zsh)")
    lines.append(f'  export LAKE_ROOT="{snapshot_path_str}"')
    lines.append("")
    
    lines.append("Then access artifacts through the standard lakehouse API.")
    lines.append("")
    
    lines.append("CONTENTS")
    lines.append("-" * 60)
    lines.append(f"Total Files: {files_copied}")
    lines.append("")
    
    # List artifact categories with file counts
    file_counts = {}
    for file_entry in manifest.get("files", []):
        path = file_entry.get("path", "")
        category = path.split("/")[0] if "/" in path else "other"
        file_counts[category] = file_counts.get(category, 0) + 1
    
    for category, count in sorted(file_counts.items()):
        lines.append(f"  - {category.title()}: {count} files")
    
    lines.append("")
    lines.append("For detailed inventory, checksums, and metadata:")
    lines.append("  See lake_manifest.json in this directory")
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def write_snapshot_note(note_content: str, snapshot_path: Path) -> None:
    """
    Write snapshot note to file.
    
    Writes snapshot_note.txt to snapshot root directory.
    
    Args:
        note_content: Snapshot note content string
        snapshot_path: Path to snapshot root directory
    
    Raises:
        IOError: If note cannot be written
    
    Example:
        >>> write_snapshot_note("Note content", Path("snapshot"))
        # Creates snapshot/snapshot_note.txt
    """
    note_path = snapshot_path / "snapshot_note.txt"
    
    try:
        note_path.write_text(note_content, encoding="utf-8")
        logger.info(f"Wrote snapshot note to {note_path}")
    
    except (IOError, OSError) as e:
        logger.error(f"Failed to write snapshot note: {e}")
        raise IOError(f"Failed to write snapshot note: {e}") from e

