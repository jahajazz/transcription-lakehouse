"""
Tests for snapshot creator and orchestration.

Tests the complete snapshot creation workflow including orchestration,
note generation, and CLI integration.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from lakehouse.snapshot.creator import (
    SnapshotCreator,
    create_snapshot,
    generate_snapshot_note,
    write_snapshot_note,
)
from lakehouse.snapshot.config import SnapshotConfig


@pytest.fixture
def mock_full_lakehouse(tmp_path):
    """Create a complete mock lakehouse for end-to-end testing."""
    lakehouse = tmp_path / "lakehouse"
    lakehouse.mkdir()
    
    # Create versioned directories with Parquet files
    schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    table = pa.table({"id": ["1", "2"], "text": ["a", "b"]})
    
    for artifact_type in ["spans", "beats", "sections", "embeddings"]:
        (lakehouse / artifact_type / "v1").mkdir(parents=True)
        pq.write_table(table, lakehouse / artifact_type / "v1" / f"{artifact_type}.parquet")
    
    # Create ANN index
    (lakehouse / "ann_index" / "v1").mkdir(parents=True)
    (lakehouse / "ann_index" / "v1" / "index.faiss").write_text("mock")
    (lakehouse / "ann_index" / "v1" / "index.json").write_text("{}")
    
    # Create catalogs
    (lakehouse / "catalogs").mkdir()
    pq.write_table(table, lakehouse / "catalogs" / "episodes.parquet")
    
    # Create lakehouse metadata
    metadata = {
        "schema_versions": {
            "spans": "1.0",
            "beats": "1.0",
            "sections": "1.0",
            "embeddings": "1.0",
        }
    }
    with open(lakehouse / "lakehouse_metadata.json", "w") as f:
        json.dump(metadata, f)
    
    return lakehouse


class TestSnapshotNoteGeneration:
    """Tests for snapshot note generation."""
    
    def test_generate_snapshot_note(self, tmp_path):
        """Test generating snapshot note."""
        snapshot_info = {
            "version": "1.0.0-provisional",
            "snapshot_path": tmp_path / "snapshot",
            "files_copied": 10,
        }
        
        manifest = {
            "created_at": "2025-10-27T12:00:00Z",
            "producer": {
                "repo": "test-repo",
                "commit": "abc123def456",
            },
            "qa_status": {
                "state": "PASS",
                "summary": "All checks passed",
            },
            "files": [
                {"path": "spans/spans.parquet", "bytes": 100},
                {"path": "beats/beats.parquet", "bytes": 200},
            ],
        }
        
        note = generate_snapshot_note(snapshot_info, manifest, tmp_path / "snapshot")
        
        assert "PROVISIONAL LAKEHOUSE SNAPSHOT" in note
        assert "v1.0.0-provisional" in note
        assert "2025-10-27T12:00:00Z" in note
        assert "PASS" in note
        assert "export LAKE_ROOT=" in note
        assert "$env:LAKE_ROOT" in note  # Windows version
    
    def test_generate_snapshot_note_with_failures(self, tmp_path):
        """Test generating snapshot note with QA failures."""
        snapshot_info = {
            "version": "1.0.0-provisional",
            "files_copied": 5,
        }
        
        manifest = {
            "created_at": "2025-10-27T12:00:00Z",
            "producer": {"repo": "test", "commit": "abc123"},
            "qa_status": {
                "state": "FAIL",
                "summary": "Issues detected",
            },
            "files": [],
        }
        
        note = generate_snapshot_note(snapshot_info, manifest, tmp_path)
        
        assert "FAIL" in note
        assert "WARNING" in note or "⚠" in note
    
    def test_write_snapshot_note(self, tmp_path):
        """Test writing snapshot note to file."""
        snapshot_path = tmp_path / "snapshot"
        snapshot_path.mkdir()
        
        note_content = "Test snapshot note content"
        
        write_snapshot_note(note_content, snapshot_path)
        
        note_file = snapshot_path / "snapshot_note.txt"
        assert note_file.exists()
        assert note_file.read_text(encoding="utf-8") == note_content


class TestSnapshotCreator:
    """Tests for SnapshotCreator class."""
    
    def test_snapshot_creator_initialization(self, tmp_path):
        """Test SnapshotCreator initialization."""
        lakehouse = tmp_path / "lakehouse"
        lakehouse.mkdir()
        
        creator = SnapshotCreator(lakehouse)
        
        assert creator.lakehouse_path == lakehouse
        assert isinstance(creator.config, SnapshotConfig)
    
    def test_snapshot_creator_with_config(self, tmp_path):
        """Test SnapshotCreator with custom config."""
        lakehouse = tmp_path / "lakehouse"
        lakehouse.mkdir()
        
        config = SnapshotConfig()
        creator = SnapshotCreator(lakehouse, config)
        
        assert creator.config is config


class TestCreateSnapshotWorkflow:
    """Tests for create_snapshot orchestration."""
    
    def test_create_snapshot_basic_workflow(self, mock_full_lakehouse, tmp_path):
        """Test complete snapshot creation workflow."""
        snapshot_root = tmp_path / "snapshots"
        
        config = SnapshotConfig(overrides={"snapshot_root": str(snapshot_root)})
        
        # Mock Git to avoid dependency
        with patch('lakehouse.snapshot.manifest.get_git_commit_hash', return_value="test123"):
            result = create_snapshot(
                lakehouse_path=mock_full_lakehouse,
                config=config,
                lakehouse_version="v1",
            )
        
        # Check result structure
        assert 'snapshot_path' in result
        assert 'version' in result
        assert 'manifest' in result
        assert 'validation_result' in result
        assert 'files_copied' in result
        
        # Check snapshot was created
        snapshot_path = result['snapshot_path']
        assert snapshot_path.exists()
        assert (snapshot_path / "lake_manifest.json").exists()
        assert (snapshot_path / "snapshot_note.txt").exists()
        
        # Check validation
        assert result['validation_result']['status'] in ['PASS', 'FAIL']
        
        # Check files were copied
        assert result['files_copied'] > 0
    
    def test_create_snapshot_with_version_override(self, mock_full_lakehouse, tmp_path):
        """Test snapshot creation with version override."""
        snapshot_root = tmp_path / "snapshots"
        config = SnapshotConfig(overrides={"snapshot_root": str(snapshot_root)})
        
        with patch('lakehouse.snapshot.manifest.get_git_commit_hash', return_value="test123"):
            result = create_snapshot(
                lakehouse_path=mock_full_lakehouse,
                config=config,
                version_override="2.0.0-custom",
            )
        
        assert result['version'] == "2.0.0-custom"
        assert result['snapshot_path'].name == "v2.0.0-custom"
    
    def test_create_snapshot_missing_artifacts(self, tmp_path):
        """Test snapshot creation with missing artifacts."""
        lakehouse = tmp_path / "lakehouse"
        lakehouse.mkdir()
        
        # Create minimal structure but no actual files
        (lakehouse / "spans" / "v1").mkdir(parents=True)
        
        snapshot_root = tmp_path / "snapshots"
        config = SnapshotConfig(overrides={"snapshot_root": str(snapshot_root)})
        
        # Should complete even with no artifacts (empty snapshot)
        with patch('lakehouse.snapshot.manifest.get_git_commit_hash', return_value="test123"):
            result = create_snapshot(
                lakehouse_path=lakehouse,
                config=config,
            )
        
        # Should succeed but with few/no files
        assert result['files_copied'] >= 0


class TestSnapshotCreatorIntegration:
    """Integration tests for SnapshotCreator."""
    
    def test_creator_create_method(self, mock_full_lakehouse, tmp_path):
        """Test SnapshotCreator.create() method."""
        snapshot_root = tmp_path / "snapshots"
        config = SnapshotConfig(overrides={"snapshot_root": str(snapshot_root)})
        
        creator = SnapshotCreator(mock_full_lakehouse, config)
        
        with patch('lakehouse.snapshot.manifest.get_git_commit_hash', return_value="test123"):
            result = creator.create(lakehouse_version="v1")
        
        assert 'snapshot_path' in result
        assert result['snapshot_path'].exists()
    
    def test_multiple_snapshots_auto_increment(self, mock_full_lakehouse, tmp_path):
        """Test creating multiple snapshots with auto-increment."""
        snapshot_root = tmp_path / "snapshots"
        config = SnapshotConfig(overrides={"snapshot_root": str(snapshot_root)})
        
        creator = SnapshotCreator(mock_full_lakehouse, config)
        
        with patch('lakehouse.snapshot.manifest.get_git_commit_hash', return_value="test123"):
            # Create first snapshot
            result1 = creator.create()
            version1 = result1['version']
            
            # Create second snapshot - should auto-increment
            result2 = creator.create()
            version2 = result2['version']
        
        # Versions should be different (auto-incremented)
        assert version1 != version2
        
        # Both snapshots should exist
        assert result1['snapshot_path'].exists()
        assert result2['snapshot_path'].exists()

