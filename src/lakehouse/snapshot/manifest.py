"""
Manifest generation for lakehouse snapshots.

Creates comprehensive manifest files with checksums, schemas, QA status,
and provenance information.
"""

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from lakehouse.logger import get_default_logger


logger = get_default_logger()


class ManifestGenerator:
    """
    Generates manifest files for lakehouse snapshots.
    
    Creates comprehensive lake_manifest.json files with all required metadata
    including version, provenance, file inventory, checksums, and QA status.
    """
    
    def __init__(self, lakehouse_path: Path):
        """
        Initialize manifest generator.
        
        Args:
            lakehouse_path: Path to lakehouse base directory
        """
        self.lakehouse_path = Path(lakehouse_path)
        logger.info(f"Initialized manifest generator for: {self.lakehouse_path}")
    
    def generate(
        self,
        version: str,
        files_metadata: List[Dict[str, Any]],
        qa_report_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate complete manifest dictionary.
        
        Args:
            version: Snapshot version string
            files_metadata: List of file metadata dictionaries
            qa_report_path: Optional path to QA report
        
        Returns:
            Complete manifest dictionary
        """
        logger.info(f"Generating manifest for version {version}")
        
        # Get Git commit hash
        git_commit = get_git_commit_hash()
        
        # Determine QA status
        qa_status = determine_qa_status(qa_report_path)
        
        # Get schema versions
        schema_versions = get_schema_versions(self.lakehouse_path)
        
        # Build files array with proper structure
        files_array = build_files_array(files_metadata)
        
        # Determine QA report relative path
        qa_report_rel_path = None
        if qa_report_path:
            # Find the report path in files_metadata
            for file_meta in files_metadata:
                if "quality_report" in file_meta.get("path", ""):
                    qa_report_rel_path = file_meta["path"]
                    break
        
        # Create complete manifest
        manifest = create_manifest(
            version=version,
            files=files_array,
            qa_status=qa_status,
            schema_versions=schema_versions,
            git_commit=git_commit,
            qa_report_rel_path=qa_report_rel_path,
        )
        
        logger.info(f"Generated manifest with {len(files_array)} files")
        
        return manifest


def get_git_commit_hash() -> str:
    """
    Get current Git commit hash using subprocess.
    
    Runs `git rev-parse HEAD` to get the current commit hash.
    Returns "unknown" if not in a Git repository or Git is not available.
    
    Returns:
        Git commit hash (SHA-1) or "unknown"
    
    Example:
        >>> commit = get_git_commit_hash()
        >>> len(commit) == 40 or commit == "unknown"
        True
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        
        commit_hash = result.stdout.strip()
        logger.info(f"Git commit: {commit_hash[:12]}...")
        return commit_hash
    
    except subprocess.CalledProcessError:
        logger.warning("Not in a Git repository, using 'unknown' for commit hash")
        return "unknown"
    
    except FileNotFoundError:
        logger.warning("Git not found in PATH, using 'unknown' for commit hash")
        return "unknown"
    
    except subprocess.TimeoutExpired:
        logger.warning("Git command timed out, using 'unknown' for commit hash")
        return "unknown"
    
    except Exception as e:
        logger.warning(f"Failed to get Git commit: {e}, using 'unknown'")
        return "unknown"


def parse_qa_report(qa_report_path: Path) -> Dict[str, Any]:
    """
    Parse QA report to extract status, summary, and metrics.
    
    Reads quality_assessment.md and extracts:
    - Overall status (PASS/FAIL)
    - Summary text
    - Key metrics for invariants field
    
    Args:
        qa_report_path: Path to quality_assessment.md file
    
    Returns:
        Dictionary with 'state', 'summary', and 'invariants'
    
    Example:
        >>> report_data = parse_qa_report(Path("quality_assessment.md"))
        >>> report_data['state'] in ['PASS', 'FAIL', 'UNKNOWN']
        True
    """
    if not qa_report_path or not qa_report_path.exists():
        logger.debug("QA report not found or not provided")
        return {
            "state": "UNKNOWN",
            "summary": "No QA report available",
            "invariants": {},
        }
    
    try:
        content = qa_report_path.read_text(encoding="utf-8")
        
        # Extract status from report
        # Look for patterns like "Overall Status: GREEN" or "Status: PASS"
        state = "UNKNOWN"
        
        # Check for RAG status (GREEN/AMBER/RED)
        if re.search(r"Overall Status:?\s*🟢|GREEN", content, re.IGNORECASE):
            state = "PASS"
        elif re.search(r"Overall Status:?\s*(🔴|RED)", content, re.IGNORECASE):
            state = "FAIL"
        elif re.search(r"Overall Status:?\s*(🟠|AMBER)", content, re.IGNORECASE):
            state = "FAIL"  # Treat AMBER as FAIL
        
        # Extract summary (first few lines of assessment section)
        summary_match = re.search(
            r"## (?:Summary|Assessment Summary|Overview)(.*?)(?=##|\Z)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        
        if summary_match:
            summary_text = summary_match.group(1).strip()
            # Get first 200 characters as summary
            summary = " ".join(summary_text.split()[:30])
            if len(summary_text) > 200:
                summary = summary[:200] + "..."
        else:
            # Fallback: create summary from status
            summary = f"QA assessment completed with status: {state}"
        
        # Extract key metrics for invariants
        invariants = {}
        
        # Look for common metrics patterns
        metric_patterns = [
            (r"Total Episodes:?\s*(\d+)", "episode_count"),
            (r"Total Spans:?\s*([\d,]+)", "span_count"),
            (r"Total Beats:?\s*([\d,]+)", "beat_count"),
            (r"Duplicate Rate:?\s*([\d.]+)%?", "duplicate_rate"),
            (r"Coverage:?\s*([\d.]+)%", "coverage_percentage"),
        ]
        
        for pattern, key in metric_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = match.group(1).replace(",", "")
                try:
                    # Try to convert to number
                    if "." in value:
                        invariants[key] = float(value)
                    else:
                        invariants[key] = int(value)
                except ValueError:
                    pass
        
        logger.info(f"Parsed QA report: state={state}, {len(invariants)} metrics")
        
        return {
            "state": state,
            "summary": summary,
            "invariants": invariants,
        }
    
    except Exception as e:
        logger.error(f"Failed to parse QA report {qa_report_path}: {e}")
        return {
            "state": "UNKNOWN",
            "summary": f"Failed to parse QA report: {e}",
            "invariants": {},
        }


def determine_qa_status(qa_report_path: Optional[Path]) -> Dict[str, Any]:
    """
    Determine QA status for manifest.
    
    Returns complete qa_status object with state, summary, invariants,
    and provisional flag.
    
    Args:
        qa_report_path: Optional path to QA report
    
    Returns:
        Dictionary with qa_status fields:
        - state: "PASS", "FAIL", or "UNKNOWN"
        - summary: Human-readable summary
        - invariants: Dictionary of QA metrics
        - provisional: Always True for provisional snapshots
    
    Example:
        >>> qa_status = determine_qa_status(None)
        >>> qa_status['provisional']
        True
        >>> qa_status['state']
        'UNKNOWN'
    """
    # Parse QA report if available
    qa_data = parse_qa_report(qa_report_path) if qa_report_path else {
        "state": "UNKNOWN",
        "summary": "No QA report available",
        "invariants": {},
    }
    
    # Build complete qa_status object
    qa_status = {
        "state": qa_data["state"],
        "summary": qa_data["summary"],
        "invariants": qa_data["invariants"],
        "provisional": True,  # Always true for provisional snapshots
    }
    
    logger.debug(f"QA status determined: {qa_status['state']}")
    
    return qa_status


def get_schema_versions(lakehouse_path: Path) -> Dict[str, str]:
    """
    Get schema versions for artifacts from lakehouse metadata.
    
    Reads lakehouse_metadata.json if available, otherwise returns
    default schema versions (all "1.0").
    
    Args:
        lakehouse_path: Path to lakehouse base directory
    
    Returns:
        Dictionary mapping artifact names to schema versions
    
    Example:
        >>> schemas = get_schema_versions(Path("lakehouse"))
        >>> schemas['spans']
        '1.0'
    """
    metadata_path = lakehouse_path / "lakehouse_metadata.json"
    
    # Default schema versions
    default_schemas = {
        "spans": "1.0",
        "beats": "1.0",
        "sections": "1.0",
        "embeddings": "1.0",
    }
    
    if not metadata_path.exists():
        logger.debug("Lakehouse metadata not found, using default schema versions")
        return default_schemas
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Extract schema versions if present
        schemas = metadata.get("schema_versions", default_schemas)
        
        logger.info(f"Loaded schema versions from lakehouse metadata")
        return schemas
    
    except Exception as e:
        logger.warning(f"Failed to read lakehouse metadata: {e}, using defaults")
        return default_schemas


def build_files_array(file_metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build files array for manifest with proper media types.
    
    Converts file metadata to manifest format with appropriate media_type
    categorization based on file extension and path.
    
    Args:
        file_metadata_list: List of file metadata dictionaries
    
    Returns:
        List of formatted file entries for manifest
    
    Example:
        >>> files = build_files_array([
        ...     {"path": "spans/spans.parquet", "bytes": 1000, "sha256": "abc", "rows": 100}
        ... ])
        >>> files[0]['media_type']
        'parquet'
    """
    files_array = []
    
    for file_meta in file_metadata_list:
        path = file_meta.get("path", "")
        
        # Determine media type from file path/extension
        if path.endswith(".parquet"):
            media_type = "parquet"
        elif path.endswith(".faiss"):
            media_type = "ann-index"
        elif "catalog" in path.lower() and path.endswith(".json"):
            media_type = "catalog-db"
        elif "quality_report" in path or "quality_assessment" in path:
            media_type = "report"
        elif path.endswith(".json"):
            media_type = "ann-index"  # Index metadata
        else:
            media_type = "other"
        
        # Build file entry
        file_entry = {
            "path": path,
            "media_type": media_type,
            "bytes": file_meta.get("bytes", 0),
            "sha256": file_meta.get("sha256", ""),
            "rows": file_meta.get("rows"),  # May be None
            "notes": file_meta.get("notes", ""),
        }
        
        files_array.append(file_entry)
    
    logger.debug(f"Built files array with {len(files_array)} entries")
    
    return files_array


def create_manifest(
    version: str,
    files: List[Dict[str, Any]],
    qa_status: Dict[str, Any],
    schema_versions: Dict[str, str],
    git_commit: str,
    qa_report_rel_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create complete manifest dictionary with all required fields.
    
    Builds the full lake_manifest.json structure with version, timestamps,
    provenance, contracts, files inventory, and QA status.
    
    Args:
        version: Snapshot version string
        files: List of file entries
        qa_status: QA status dictionary
        schema_versions: Schema version mappings
        git_commit: Git commit hash
        qa_report_rel_path: Relative path to QA report in snapshot
    
    Returns:
        Complete manifest dictionary
    
    Example:
        >>> manifest = create_manifest("1.0.0", [], {}, {}, "abc123")
        >>> manifest['lake_version']
        '1.0.0'
    """
    # Generate ISO-8601 UTC timestamp
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    manifest = {
        "lake_version": version,
        "created_at": created_at,
        "producer": {
            "repo": "transcription-lakehouse",
            "commit": git_commit,
            "task": "SNAPSHOT",
            "report_path": qa_report_rel_path,
        },
        "contracts": {
            "schemas": schema_versions,
            "compatibility": {},  # Optional, can be omitted initially
        },
        "files": files,
        "qa_status": qa_status,
    }
    
    logger.info(f"Created manifest for version {version} at {created_at}")
    
    return manifest


def write_manifest(manifest: Dict[str, Any], snapshot_root: Path) -> None:
    """
    Write manifest as pretty-printed JSON to snapshot root.
    
    Writes lake_manifest.json with proper formatting (2-space indent).
    
    Args:
        manifest: Complete manifest dictionary
        snapshot_root: Snapshot root directory
    
    Raises:
        IOError: If manifest cannot be written
    
    Example:
        >>> write_manifest({"lake_version": "1.0.0"}, Path("snapshot"))
        # Creates snapshot/lake_manifest.json
    """
    manifest_path = snapshot_root / "lake_manifest.json"
    
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Wrote manifest to {manifest_path}")
    
    except (IOError, OSError) as e:
        logger.error(f"Failed to write manifest to {manifest_path}: {e}")
        raise IOError(f"Failed to write manifest: {e}") from e

