"""
Tests for snapshot artifact discovery and file collection.

Tests artifact discovery, SHA-256 calculation, Parquet row counting, and file copying.
"""

import tempfile
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from lakehouse.snapshot.artifacts import (
    ArtifactDiscovery,
    discover_artifacts,
    find_latest_qa_report,
    calculate_sha256,
    get_parquet_row_count,
    copy_artifact_with_metadata,
    create_snapshot_structure,
    copy_all_artifacts,
)


@pytest.fixture
def mock_lakehouse(tmp_path):
    """Create a mock lakehouse structure for testing."""
    lakehouse = tmp_path / "lakehouse"
    lakehouse.mkdir()
    
    # Create versioned directories
    for artifact_type in ["spans", "beats", "sections", "embeddings", "ann_index"]:
        (lakehouse / artifact_type / "v1").mkdir(parents=True)
    
    (lakehouse / "catalogs").mkdir()
    
    # Create some mock Parquet files
    schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    table = pa.table({"id": ["1", "2", "3"], "text": ["a", "b", "c"]})
    
    pq.write_table(table, lakehouse / "spans" / "v1" / "spans.parquet")
    pq.write_table(table, lakehouse / "beats" / "v1" / "beats.parquet")
    pq.write_table(table, lakehouse / "sections" / "v1" / "sections.parquet")
    pq.write_table(table, lakehouse / "embeddings" / "v1" / "span_embeddings.parquet")
    
    # Create mock index files
    (lakehouse / "ann_index" / "v1" / "span_index.faiss").write_text("mock faiss index")
    (lakehouse / "ann_index" / "v1" / "span_index.ids.json").write_text('{"ids": [1, 2, 3]}')
    
    # Create mock catalog files
    pq.write_table(table, lakehouse / "catalogs" / "episodes.parquet")
    (lakehouse / "catalogs" / "speakers.json").write_text('{"speakers": []}')
    
    return lakehouse


@pytest.fixture
def mock_qa_reports(tmp_path):
    """Create mock QA report structure."""
    qa_dir = tmp_path / "quality_reports"
    
    # Create multiple timestamped reports
    report1 = qa_dir / "20251025_120000" / "report"
    report1.mkdir(parents=True)
    (report1 / "quality_assessment.md").write_text("# QA Report 1\n\nStatus: PASS")
    
    report2 = qa_dir / "20251026_143000" / "report"
    report2.mkdir(parents=True)
    (report2 / "quality_assessment.md").write_text("# QA Report 2\n\nStatus: FAIL")
    
    return tmp_path


class TestArtifactDiscovery:
    """Tests for ArtifactDiscovery class."""
    
    def test_discovery_initialization(self, mock_lakehouse):
        """Test ArtifactDiscovery initialization."""
        discovery = ArtifactDiscovery(mock_lakehouse)
        assert discovery.lakehouse_path == mock_lakehouse
    
    def test_discover_all_artifacts(self, mock_lakehouse):
        """Test discovering all artifacts."""
        discovery = ArtifactDiscovery(mock_lakehouse)
        artifacts = discovery.discover_all("v1")
        
        assert "spans" in artifacts
        assert "beats" in artifacts
        assert "sections" in artifacts
        assert "embeddings" in artifacts
        assert "indexes" in artifacts
        assert "catalogs" in artifacts
        
        # Verify some files were found
        assert len(artifacts["spans"]) == 1
        assert len(artifacts["beats"]) == 1
        assert len(artifacts["embeddings"]) == 1
        assert len(artifacts["indexes"]) == 2  # .faiss + .json
        assert len(artifacts["catalogs"]) == 2  # .parquet + .json
    
    def test_discover_artifacts_function(self, mock_lakehouse):
        """Test discover_artifacts convenience function."""
        artifacts = discover_artifacts(mock_lakehouse, "v1")
        
        assert isinstance(artifacts, dict)
        assert len(artifacts["spans"]) > 0
    
    def test_discover_nonexistent_version(self, mock_lakehouse):
        """Test discovering artifacts for non-existent version."""
        discovery = ArtifactDiscovery(mock_lakehouse)
        artifacts = discovery.discover_all("v99")
        
        # Versioned artifacts should be empty
        assert len(artifacts["spans"]) == 0
        assert len(artifacts["beats"]) == 0
        assert len(artifacts["sections"]) == 0
        assert len(artifacts["embeddings"]) == 0
        assert len(artifacts["indexes"]) == 0
        
        # Catalogs are not versioned, so they may still be found
        # (This is expected behavior)


class TestQAReportDiscovery:
    """Tests for QA report discovery."""
    
    def test_find_latest_qa_report(self, mock_qa_reports):
        """Test finding the latest QA report."""
        qa_report = find_latest_qa_report(mock_qa_reports)
        
        assert qa_report is not None
        assert qa_report.name == "quality_assessment.md"
        assert "20251026_143000" in str(qa_report)  # Latest timestamp
    
    def test_find_qa_report_no_directory(self, tmp_path):
        """Test finding QA report when directory doesn't exist."""
        qa_report = find_latest_qa_report(tmp_path / "nonexistent")
        assert qa_report is None
    
    def test_find_qa_report_empty_directory(self, tmp_path):
        """Test finding QA report in empty directory."""
        qa_dir = tmp_path / "quality_reports"
        qa_dir.mkdir()
        
        qa_report = find_latest_qa_report(tmp_path)
        assert qa_report is None


class TestChecksumAndRowCount:
    """Tests for checksum calculation and Parquet row counting."""
    
    def test_calculate_sha256(self, tmp_path):
        """Test SHA-256 checksum calculation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")
        
        checksum = calculate_sha256(test_file)
        
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 is 64 hex characters
        
        # Verify checksum is consistent
        checksum2 = calculate_sha256(test_file)
        assert checksum == checksum2
    
    def test_calculate_sha256_nonexistent_file(self, tmp_path):
        """Test checksum calculation for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            calculate_sha256(tmp_path / "nonexistent.txt")
    
    def test_get_parquet_row_count(self, mock_lakehouse):
        """Test getting row count from Parquet file."""
        parquet_file = mock_lakehouse / "spans" / "v1" / "spans.parquet"
        row_count = get_parquet_row_count(parquet_file)
        
        assert row_count == 3  # We created a table with 3 rows
    
    def test_get_parquet_row_count_nonparquet_file(self, tmp_path):
        """Test getting row count from non-Parquet file."""
        text_file = tmp_path / "test.txt"
        text_file.write_text("not a parquet file")
        
        row_count = get_parquet_row_count(text_file)
        assert row_count is None
    
    def test_get_parquet_row_count_nonexistent_file(self, tmp_path):
        """Test getting row count from nonexistent file."""
        row_count = get_parquet_row_count(tmp_path / "nonexistent.parquet")
        assert row_count is None


class TestFileCopying:
    """Tests for file copying with metadata."""
    
    def test_copy_artifact_with_metadata(self, tmp_path):
        """Test copying artifact and collecting metadata."""
        # Create source file
        source = tmp_path / "source.txt"
        source.write_text("Test content")
        
        # Copy to destination
        dest = tmp_path / "dest" / "source.txt"
        metadata = copy_artifact_with_metadata(source, dest)
        
        assert dest.exists()
        assert metadata["path"] == "source.txt"
        assert metadata["bytes"] == len("Test content")
        assert len(metadata["sha256"]) == 64
        assert metadata["rows"] is None  # Not a Parquet file
    
    def test_copy_parquet_with_metadata(self, mock_lakehouse, tmp_path):
        """Test copying Parquet file with row count."""
        source = mock_lakehouse / "spans" / "v1" / "spans.parquet"
        dest = tmp_path / "spans.parquet"
        
        metadata = copy_artifact_with_metadata(source, dest)
        
        assert dest.exists()
        assert metadata["rows"] == 3
        assert metadata["bytes"] > 0
    
    def test_copy_artifact_nonexistent_source(self, tmp_path):
        """Test copying nonexistent source file."""
        source = tmp_path / "nonexistent.txt"
        dest = tmp_path / "dest.txt"
        
        with pytest.raises(FileNotFoundError):
            copy_artifact_with_metadata(source, dest)


class TestSnapshotStructure:
    """Tests for snapshot directory structure creation."""
    
    def test_create_snapshot_structure(self, tmp_path):
        """Test creating snapshot directory structure."""
        snapshot_root = tmp_path / "snapshots"
        version = "1.0.0-provisional"
        
        dirs = create_snapshot_structure(snapshot_root, version)
        
        assert dirs["root"].exists()
        assert dirs["root"].name == "v1.0.0-provisional"
        assert dirs["spans"].exists()
        assert dirs["beats"].exists()
        assert dirs["sections"].exists()
        assert dirs["embeddings"].exists()
        assert dirs["indexes"].exists()
        assert dirs["catalogs"].exists()
        assert dirs["quality_report"].exists()
    
    def test_create_snapshot_structure_idempotent(self, tmp_path):
        """Test creating snapshot structure multiple times."""
        snapshot_root = tmp_path / "snapshots"
        version = "1.0.0-provisional"
        
        dirs1 = create_snapshot_structure(snapshot_root, version)
        dirs2 = create_snapshot_structure(snapshot_root, version)
        
        assert dirs1["root"] == dirs2["root"]


class TestCopyAllArtifacts:
    """Tests for copying all artifacts."""
    
    def test_copy_all_artifacts_success(self, mock_lakehouse, tmp_path):
        """Test copying all discovered artifacts."""
        # Discover artifacts
        artifacts = discover_artifacts(mock_lakehouse, "v1")
        
        # Create destination structure
        dest_dirs = create_snapshot_structure(tmp_path / "snapshots", "1.0.0-provisional")
        
        # Copy all artifacts
        metadata_list = copy_all_artifacts(artifacts, dest_dirs)
        
        assert len(metadata_list) > 0
        assert all("path" in m for m in metadata_list)
        assert all("bytes" in m for m in metadata_list)
        assert all("sha256" in m for m in metadata_list)
        
        # Verify files were actually copied
        assert (dest_dirs["spans"] / "spans.parquet").exists()
        assert (dest_dirs["beats"] / "beats.parquet").exists()
    
    def test_copy_all_artifacts_with_qa_report(self, mock_lakehouse, mock_qa_reports, tmp_path):
        """Test copying artifacts including QA report."""
        artifacts = discover_artifacts(mock_lakehouse, "v1")
        dest_dirs = create_snapshot_structure(tmp_path / "snapshots", "1.0.0-provisional")
        qa_report = find_latest_qa_report(mock_qa_reports)
        
        metadata_list = copy_all_artifacts(artifacts, dest_dirs, qa_report)
        
        # Should include QA report in metadata
        qa_metadata = [m for m in metadata_list if "quality_report" in m["path"]]
        assert len(qa_metadata) == 1
        
        # Verify QA report was copied
        assert (dest_dirs["quality_report"] / "quality_assessment.md").exists()
    
    def test_copy_all_artifacts_missing_source(self, tmp_path):
        """Test copying artifacts with missing source file."""
        # Create artifacts dict with nonexistent file
        artifacts = {
            "spans": [tmp_path / "nonexistent.parquet"],
            "beats": [],
            "sections": [],
            "embeddings": [],
            "indexes": [],
            "catalogs": [],
        }
        
        dest_dirs = create_snapshot_structure(tmp_path / "snapshots", "1.0.0-provisional")
        
        # Should raise FileNotFoundError (FR-10)
        with pytest.raises(FileNotFoundError, match="Required artifact missing"):
            copy_all_artifacts(artifacts, dest_dirs)

