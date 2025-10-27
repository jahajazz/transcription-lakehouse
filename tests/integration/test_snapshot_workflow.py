"""
Integration tests for complete snapshot workflow.

Tests end-to-end snapshot creation, validation, reproducibility,
version collision handling, error handling, and QA status detection.
"""

import json
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from lakehouse.snapshot.config import SnapshotConfig
from lakehouse.snapshot.creator import SnapshotCreator
from lakehouse.snapshot.validator import validate_snapshot
from lakehouse.snapshot.artifacts import calculate_sha256


@pytest.fixture
def complete_lakehouse(tmp_path):
    """Create a complete lakehouse with all artifact types."""
    lakehouse = tmp_path / "lakehouse"
    lakehouse.mkdir()
    
    # Create versioned artifact directories
    schema = pa.schema([
        ("id", pa.string()),
        ("text", pa.string()),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
    ])
    
    # Create spans
    spans_data = {
        "id": ["span_1", "span_2", "span_3"],
        "text": ["Hello world", "This is a test", "Final span"],
        "start_time": [0.0, 5.0, 10.0],
        "end_time": [5.0, 10.0, 15.0],
    }
    (lakehouse / "spans" / "v1").mkdir(parents=True)
    pq.write_table(pa.table(spans_data, schema=schema), lakehouse / "spans" / "v1" / "spans.parquet")
    
    # Create beats
    beats_data = {
        "id": ["beat_1", "beat_2"],
        "text": ["First beat content", "Second beat content"],
        "start_time": [0.0, 7.5],
        "end_time": [7.5, 15.0],
    }
    (lakehouse / "beats" / "v1").mkdir(parents=True)
    pq.write_table(pa.table(beats_data, schema=schema), lakehouse / "beats" / "v1" / "beats.parquet")
    
    # Create sections
    sections_data = {
        "id": ["section_1"],
        "text": ["Complete section"],
        "start_time": [0.0],
        "end_time": [15.0],
    }
    (lakehouse / "sections" / "v1").mkdir(parents=True)
    pq.write_table(pa.table(sections_data, schema=schema), lakehouse / "sections" / "v1" / "sections.parquet")
    
    # Create embeddings
    embedding_schema = pa.schema([("id", pa.string()), ("embedding", pa.list_(pa.float32()))])
    embedding_data = {
        "id": ["span_1", "span_2", "span_3"],
        "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
    }
    (lakehouse / "embeddings" / "v1").mkdir(parents=True)
    pq.write_table(
        pa.table(embedding_data, schema=embedding_schema),
        lakehouse / "embeddings" / "v1" / "span_embeddings.parquet"
    )
    
    # Create ANN indexes
    (lakehouse / "ann_index" / "v1").mkdir(parents=True)
    (lakehouse / "ann_index" / "v1" / "span_index.faiss").write_bytes(b"mock_faiss_index_data")
    (lakehouse / "ann_index" / "v1" / "span_index.ids.json").write_text('{"ids": ["span_1", "span_2", "span_3"]}')
    
    # Create catalogs
    (lakehouse / "catalogs").mkdir()
    catalog_schema = pa.schema([("episode_id", pa.string()), ("title", pa.string())])
    catalog_data = {"episode_id": ["ep1", "ep2"], "title": ["Episode 1", "Episode 2"]}
    pq.write_table(pa.table(catalog_data, schema=catalog_schema), lakehouse / "catalogs" / "episodes.parquet")
    
    # Create lakehouse metadata
    metadata = {
        "created_at": "2025-10-27T00:00:00Z",
        "lakehouse_path": str(lakehouse),
        "initial_version": "v1",
        "structure_version": "1.0",
        "schema_versions": {
            "spans": "1.0",
            "beats": "1.0",
            "sections": "1.0",
            "embeddings": "1.0",
        },
    }
    with open(lakehouse / "lakehouse_metadata.json", "w") as f:
        json.dump(metadata, f)
    
    return lakehouse


@pytest.fixture
def qa_report_pass(tmp_path):
    """Create a QA report with PASS status."""
    base = tmp_path / "lakehouse_pass"
    base.mkdir()
    qa_dir = base / "quality_reports" / "20251027_120000" / "report"
    qa_dir.mkdir(parents=True)
    
    report_content = """
# Quality Assessment Report

## Summary
Overall Status: 🟢 GREEN

All quality checks passed successfully.

## Dataset Metrics
- Total Episodes: 2
- Total Spans: 3
- Total Beats: 2
- Duplicate Rate: 0.0%
- Coverage: 100.0%

All validation checks completed successfully.
"""
    (qa_dir / "quality_assessment.md").write_text(report_content, encoding='utf-8')
    return base


@pytest.fixture
def qa_report_fail(tmp_path):
    """Create a QA report with FAIL status."""
    base = tmp_path / "lakehouse_fail"
    base.mkdir()
    qa_dir = base / "quality_reports" / "20251027_120000" / "report"
    qa_dir.mkdir(parents=True)
    
    report_content = """
# Quality Assessment Report

## Summary
Overall Status: 🔴 RED

Critical issues detected in the dataset.

## Dataset Metrics
- Total Episodes: 2
- Total Spans: 3
- Total Beats: 2
- Duplicate Rate: 15.5%
- Coverage: 85.0%

Quality issues require attention.
"""
    (qa_dir / "quality_assessment.md").write_text(report_content, encoding='utf-8')
    return base


class TestEndToEndWorkflow:
    """Test complete end-to-end snapshot workflow (Task 6.6)."""
    
    def test_complete_snapshot_creation_and_validation(self, complete_lakehouse, tmp_path):
        """Test creating a complete snapshot and validating it."""
        snapshot_root = tmp_path / "snapshots"
        
        config = SnapshotConfig(overrides={"snapshot_root": str(snapshot_root)})
        creator = SnapshotCreator(complete_lakehouse, config)
        
        # Create snapshot
        result = creator.create(lakehouse_version="v1")
        
        # Verify result structure
        assert 'snapshot_path' in result
        assert 'version' in result
        assert 'manifest' in result
        assert 'validation_result' in result
        assert 'files_copied' in result
        
        # Verify snapshot exists
        snapshot_path = result['snapshot_path']
        assert snapshot_path.exists()
        assert snapshot_path.is_dir()
        
        # Verify all required files
        assert (snapshot_path / "lake_manifest.json").exists()
        assert (snapshot_path / "snapshot_note.txt").exists()
        
        # Verify subdirectories
        assert (snapshot_path / "spans").exists()
        assert (snapshot_path / "beats").exists()
        assert (snapshot_path / "sections").exists()
        assert (snapshot_path / "embeddings").exists()
        assert (snapshot_path / "indexes").exists()
        assert (snapshot_path / "catalogs").exists()
        
        # Verify artifacts were copied
        assert (snapshot_path / "spans" / "spans.parquet").exists()
        assert (snapshot_path / "beats" / "beats.parquet").exists()
        assert (snapshot_path / "sections" / "sections.parquet").exists()
        assert (snapshot_path / "embeddings" / "span_embeddings.parquet").exists()
        assert (snapshot_path / "indexes" / "span_index.faiss").exists()
        
        # Verify validation passed
        assert result['validation_result']['status'] == 'PASS'
        
        # Verify manifest structure
        manifest = result['manifest']
        assert 'lake_version' in manifest
        assert 'created_at' in manifest
        assert 'producer' in manifest
        assert 'contracts' in manifest
        assert 'files' in manifest
        assert 'qa_status' in manifest
        
        # Verify files were cataloged
        assert len(manifest['files']) > 0
        
        # All files should have checksums
        for file_entry in manifest['files']:
            assert 'sha256' in file_entry
            assert 'bytes' in file_entry
            assert len(file_entry['sha256']) == 64  # SHA-256 hex length


class TestImmutability:
    """Test snapshot immutability and reproducibility (Task 6.7 - FR-27)."""
    
    def test_identical_checksums_from_same_inputs(self, complete_lakehouse, tmp_path):
        """Test that creating snapshots from same inputs yields identical checksums."""
        snapshot_root = tmp_path / "snapshots"
        config = SnapshotConfig(overrides={"snapshot_root": str(snapshot_root)})
        
        # Create first snapshot
        creator1 = SnapshotCreator(complete_lakehouse, config)
        result1 = creator1.create(version_override="test-1.0.0")
        
        # Create second snapshot with different version but same inputs
        creator2 = SnapshotCreator(complete_lakehouse, config)
        result2 = creator2.create(version_override="test-2.0.0")
        
        # Get file lists from both manifests
        files1 = {f['path']: f['sha256'] for f in result1['manifest']['files']}
        files2 = {f['path']: f['sha256'] for f in result2['manifest']['files']}
        
        # Should have same files
        assert set(files1.keys()) == set(files2.keys())
        
        # Checksums should be identical for same artifact types
        # (excluding qa_report which may have timestamps)
        for path in files1:
            if 'quality_report' not in path:
                assert files1[path] == files2[path], f"Checksum mismatch for {path}"


class TestVersionCollision:
    """Test version collision handling (Task 6.8 - FR-3)."""
    
    def test_auto_increment_on_collision(self, complete_lakehouse, tmp_path):
        """Test that version auto-increments when collision detected."""
        snapshot_root = tmp_path / "snapshots"
        config = SnapshotConfig(overrides={
            "snapshot_root": str(snapshot_root),
            "version": {"major": 1, "minor": 0, "patch": 0},
        })
        
        creator = SnapshotCreator(complete_lakehouse, config)
        
        # Create first snapshot
        result1 = creator.create()
        version1 = result1['version']
        
        # Create second snapshot - should auto-increment
        result2 = creator.create()
        version2 = result2['version']
        
        # Parse versions
        from lakehouse.snapshot.config import parse_version
        v1_major, v1_minor, v1_patch, _ = parse_version(version1)
        v2_major, v2_minor, v2_patch, _ = parse_version(version2)
        
        # Should have incremented patch version
        assert v1_major == v2_major
        assert v1_minor == v2_minor
        assert v2_patch == v1_patch + 1
        
        # Both snapshots should exist
        assert result1['snapshot_path'].exists()
        assert result2['snapshot_path'].exists()


class TestErrorHandling:
    """Test error handling for missing artifacts (Task 6.9 - FR-10)."""
    
    def test_graceful_handling_of_missing_artifacts(self, tmp_path):
        """Test that snapshot creation continues gracefully if some artifacts are missing.
        
        This tests the provisional snapshot behavior: we don't fail the entire
        snapshot if individual files are missing. Instead, we log and continue.
        """
        # Create incomplete lakehouse
        lakehouse = tmp_path / "lakehouse"
        lakehouse.mkdir()
        
        # Create structure but add a reference to non-existent file
        # by manipulating the discovery process
        (lakehouse / "spans" / "v1").mkdir(parents=True)
        
        # Create a parquet file that we'll delete to simulate missing artifact
        schema = pa.schema([("id", pa.string()), ("text", pa.string())])
        data = {"id": ["1"], "text": ["test"]}
        test_file = lakehouse / "spans" / "v1" / "spans.parquet"
        pq.write_table(pa.table(data, schema=schema), test_file)
        
        snapshot_root = tmp_path / "snapshots"
        config = SnapshotConfig(overrides={"snapshot_root": str(snapshot_root)})
        
        # Delete the file after discovery would find it
        # This simulates a file being removed during snapshot creation
        creator = SnapshotCreator(lakehouse, config)
        
        # First verify it works with file present
        result1 = creator.create(version_override="working-1.0.0")
        assert result1['validation_result']['status'] == 'PASS'
        files_in_first = result1['files_copied']
        assert files_in_first >= 1  # Should have copied at least the spans file
        
        # Now delete the source file and try again
        test_file.unlink()
        
        # Should succeed but with fewer files (provisional snapshot behavior)
        result2 = creator.create(version_override="working-2.0.0")
        assert result2['validation_result']['status'] == 'PASS'
        files_in_second = result2['files_copied']
        # Should have copied fewer files since the spans file is missing
        assert files_in_second < files_in_first


class TestQAStatusDetection:
    """Test QA status detection (Task 6.10)."""
    
    def test_qa_status_pass(self, complete_lakehouse, qa_report_pass, tmp_path):
        """Test snapshot with PASS QA status."""
        snapshot_root = tmp_path / "snapshots"
        config = SnapshotConfig(overrides={"snapshot_root": str(snapshot_root)})
        
        # Change to QA report directory so it can be found
        import os
        original_cwd = os.getcwd()
        os.chdir(qa_report_pass)
        
        try:
            creator = SnapshotCreator(complete_lakehouse, config)
            result = creator.create()
            
            # Check QA status in manifest
            qa_status = result['manifest']['qa_status']
            assert qa_status['state'] == 'PASS'
            assert qa_status['provisional'] is True
            
            # Validation should pass without warnings
            validation = result['validation_result']
            assert validation['status'] == 'PASS'
        finally:
            os.chdir(original_cwd)
    
    def test_qa_status_fail(self, complete_lakehouse, qa_report_fail, tmp_path):
        """Test snapshot with FAIL QA status."""
        snapshot_root = tmp_path / "snapshots"
        config = SnapshotConfig(overrides={"snapshot_root": str(snapshot_root)})
        
        import os
        original_cwd = os.getcwd()
        os.chdir(qa_report_fail)
        
        try:
            creator = SnapshotCreator(complete_lakehouse, config)
            result = creator.create()
            
            # Check QA status
            qa_status = result['manifest']['qa_status']
            assert qa_status['state'] == 'FAIL'
            assert qa_status['provisional'] is True
            
            # Validation should still PASS but with warnings (FR-25)
            validation = result['validation_result']
            assert validation['status'] == 'PASS'
            assert len(validation['warnings']) > 0
            assert any('FAIL' in w for w in validation['warnings'])
        finally:
            os.chdir(original_cwd)
    
    def test_qa_status_unknown(self, complete_lakehouse, tmp_path):
        """Test snapshot with no QA report (UNKNOWN status)."""
        snapshot_root = tmp_path / "snapshots"
        config = SnapshotConfig(overrides={"snapshot_root": str(snapshot_root)})
        
        creator = SnapshotCreator(complete_lakehouse, config)
        result = creator.create()
        
        # Check QA status
        qa_status = result['manifest']['qa_status']
        assert qa_status['state'] == 'UNKNOWN'
        assert qa_status['provisional'] is True
        
        # Validation should pass with warning
        validation = result['validation_result']
        assert validation['status'] == 'PASS'
        assert len(validation['warnings']) > 0
        assert any('UNKNOWN' in w for w in validation['warnings'])


class TestTestFixtures:
    """Test that test fixtures work correctly (Task 6.11)."""
    
    def test_complete_lakehouse_fixture(self, complete_lakehouse):
        """Test that complete lakehouse fixture is properly structured."""
        assert complete_lakehouse.exists()
        assert (complete_lakehouse / "spans" / "v1" / "spans.parquet").exists()
        assert (complete_lakehouse / "beats" / "v1" / "beats.parquet").exists()
        assert (complete_lakehouse / "lakehouse_metadata.json").exists()
    
    def test_qa_report_fixtures(self, qa_report_pass, qa_report_fail):
        """Test that QA report fixtures are created correctly."""
        assert (qa_report_pass / "quality_reports").exists()
        assert (qa_report_fail / "quality_reports").exists()

