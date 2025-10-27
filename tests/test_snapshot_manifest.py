"""
Tests for snapshot manifest generation.

Tests Git integration, QA report parsing, manifest generation, and JSON writing.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

import pytest

from lakehouse.snapshot.manifest import (
    ManifestGenerator,
    get_git_commit_hash,
    parse_qa_report,
    determine_qa_status,
    get_schema_versions,
    build_files_array,
    create_manifest,
    write_manifest,
)


class TestGitIntegration:
    """Tests for Git commit hash retrieval."""
    
    def test_get_git_commit_hash_success(self):
        """Test getting Git commit hash successfully."""
        # Mock subprocess to return a commit hash
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "abc123def456789012345678901234567890abcd\n"
            mock_result.returncode = 0
            mock_run.return_value = mock_result
            
            commit = get_git_commit_hash()
            
            assert commit == "abc123def456789012345678901234567890abcd"
            mock_run.assert_called_once()
    
    def test_get_git_commit_hash_not_git_repo(self):
        """Test getting commit hash when not in Git repo."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(128, 'git')
            
            commit = get_git_commit_hash()
            
            assert commit == "unknown"
    
    def test_get_git_commit_hash_git_not_found(self):
        """Test getting commit hash when Git is not installed."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()
            
            commit = get_git_commit_hash()
            
            assert commit == "unknown"
    
    def test_get_git_commit_hash_timeout(self):
        """Test getting commit hash with timeout."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired('git', 5)
            
            commit = get_git_commit_hash()
            
            assert commit == "unknown"


class TestQAReportParsing:
    """Tests for QA report parsing."""
    
    def test_parse_qa_report_pass_status(self, tmp_path):
        """Test parsing QA report with PASS status."""
        report = tmp_path / "quality_assessment.md"
        report.write_text("""
# Quality Assessment Report

## Summary
Overall Status: 🟢 GREEN

All quality checks passed successfully.

## Dataset Metrics
- Total Episodes: 100
- Total Spans: 50000
- Total Beats: 12000
- Duplicate Rate: 0.5%
- Coverage: 98.5%
""", encoding='utf-8')
        
        result = parse_qa_report(report)
        
        assert result['state'] == 'PASS'
        assert 'passed' in result['summary'].lower()
        assert result['invariants']['episode_count'] == 100
        assert result['invariants']['span_count'] == 50000
        assert result['invariants']['beat_count'] == 12000
        assert result['invariants']['duplicate_rate'] == 0.5
        assert result['invariants']['coverage_percentage'] == 98.5
    
    def test_parse_qa_report_fail_status(self, tmp_path):
        """Test parsing QA report with FAIL status."""
        report = tmp_path / "quality_assessment.md"
        report.write_text("""
# Quality Assessment Report

Overall Status: 🔴 RED

Critical issues detected.
""", encoding='utf-8')
        
        result = parse_qa_report(report)
        
        assert result['state'] == 'FAIL'
        assert 'critical' in result['summary'].lower() or 'fail' in result['summary'].lower()
    
    def test_parse_qa_report_amber_status(self, tmp_path):
        """Test parsing QA report with AMBER status (treated as FAIL)."""
        report = tmp_path / "quality_assessment.md"
        report.write_text("""
# Quality Assessment Report

Overall Status: 🟠 AMBER

Some warnings detected.
""", encoding='utf-8')
        
        result = parse_qa_report(report)
        
        assert result['state'] == 'FAIL'  # AMBER treated as FAIL
    
    def test_parse_qa_report_no_file(self):
        """Test parsing non-existent QA report."""
        result = parse_qa_report(Path("/nonexistent/report.md"))
        
        assert result['state'] == 'UNKNOWN'
        assert 'no qa report' in result['summary'].lower()
        assert result['invariants'] == {}
    
    def test_parse_qa_report_malformed(self, tmp_path):
        """Test parsing malformed QA report."""
        report = tmp_path / "quality_assessment.md"
        report.write_text("This is not a properly formatted QA report")
        
        result = parse_qa_report(report)
        
        # Should handle gracefully
        assert result['state'] in ['PASS', 'FAIL', 'UNKNOWN']
    
    def test_determine_qa_status_with_report(self, tmp_path):
        """Test determining QA status with report."""
        report = tmp_path / "quality_assessment.md"
        report.write_text("Overall Status: GREEN")
        
        qa_status = determine_qa_status(report)
        
        assert qa_status['state'] == 'PASS'
        assert qa_status['provisional'] is True
        assert 'summary' in qa_status
        assert 'invariants' in qa_status
    
    def test_determine_qa_status_no_report(self):
        """Test determining QA status without report."""
        qa_status = determine_qa_status(None)
        
        assert qa_status['state'] == 'UNKNOWN'
        assert qa_status['provisional'] is True
        assert qa_status['summary'] == 'No QA report available'


class TestSchemaVersions:
    """Tests for schema version retrieval."""
    
    def test_get_schema_versions_defaults(self, tmp_path):
        """Test getting default schema versions."""
        lakehouse = tmp_path / "lakehouse"
        lakehouse.mkdir()
        
        schemas = get_schema_versions(lakehouse)
        
        assert schemas['spans'] == '1.0'
        assert schemas['beats'] == '1.0'
        assert schemas['sections'] == '1.0'
        assert schemas['embeddings'] == '1.0'
    
    def test_get_schema_versions_from_metadata(self, tmp_path):
        """Test getting schema versions from metadata file."""
        lakehouse = tmp_path / "lakehouse"
        lakehouse.mkdir()
        
        metadata = {
            "schema_versions": {
                "spans": "2.0",
                "beats": "1.5",
                "sections": "1.0",
                "embeddings": "1.0",
            }
        }
        
        with open(lakehouse / "lakehouse_metadata.json", "w") as f:
            json.dump(metadata, f)
        
        schemas = get_schema_versions(lakehouse)
        
        assert schemas['spans'] == '2.0'
        assert schemas['beats'] == '1.5'
    
    def test_get_schema_versions_malformed_metadata(self, tmp_path):
        """Test getting schema versions with malformed metadata."""
        lakehouse = tmp_path / "lakehouse"
        lakehouse.mkdir()
        
        # Write invalid JSON
        (lakehouse / "lakehouse_metadata.json").write_text("not valid json")
        
        schemas = get_schema_versions(lakehouse)
        
        # Should fall back to defaults
        assert schemas['spans'] == '1.0'


class TestFilesArray:
    """Tests for building files array."""
    
    def test_build_files_array_parquet(self):
        """Test building files array with Parquet files."""
        file_metadata = [
            {
                "path": "spans/spans.parquet",
                "bytes": 12345,
                "sha256": "abc123",
                "rows": 1000,
                "notes": "",
            }
        ]
        
        files = build_files_array(file_metadata)
        
        assert len(files) == 1
        assert files[0]['media_type'] == 'parquet'
        assert files[0]['path'] == 'spans/spans.parquet'
        assert files[0]['bytes'] == 12345
        assert files[0]['rows'] == 1000
    
    def test_build_files_array_ann_index(self):
        """Test building files array with ANN index files."""
        file_metadata = [
            {
                "path": "indexes/span_index.faiss",
                "bytes": 5000,
                "sha256": "def456",
                "rows": None,
                "notes": "",
            },
            {
                "path": "indexes/span_index.ids.json",
                "bytes": 100,
                "sha256": "ghi789",
                "rows": None,
                "notes": "",
            }
        ]
        
        files = build_files_array(file_metadata)
        
        assert len(files) == 2
        assert files[0]['media_type'] == 'ann-index'
        assert files[1]['media_type'] == 'ann-index'
    
    def test_build_files_array_catalog(self):
        """Test building files array with catalog files."""
        file_metadata = [
            {
                "path": "catalogs/episodes.json",
                "bytes": 200,
                "sha256": "jkl012",
                "rows": None,
                "notes": "",
            }
        ]
        
        files = build_files_array(file_metadata)
        
        assert len(files) == 1
        assert files[0]['media_type'] == 'catalog-db'
    
    def test_build_files_array_report(self):
        """Test building files array with QA report."""
        file_metadata = [
            {
                "path": "quality_report/quality_assessment.md",
                "bytes": 5000,
                "sha256": "mno345",
                "rows": None,
                "notes": "",
            }
        ]
        
        files = build_files_array(file_metadata)
        
        assert len(files) == 1
        assert files[0]['media_type'] == 'report'
    
    def test_build_files_array_mixed(self):
        """Test building files array with mixed file types."""
        file_metadata = [
            {"path": "spans/spans.parquet", "bytes": 100, "sha256": "a", "rows": 10, "notes": ""},
            {"path": "indexes/index.faiss", "bytes": 200, "sha256": "b", "rows": None, "notes": ""},
            {"path": "catalogs/catalog.json", "bytes": 50, "sha256": "c", "rows": None, "notes": ""},
        ]
        
        files = build_files_array(file_metadata)
        
        assert len(files) == 3
        assert files[0]['media_type'] == 'parquet'
        assert files[1]['media_type'] == 'ann-index'
        assert files[2]['media_type'] == 'catalog-db'


class TestManifestCreation:
    """Tests for manifest creation."""
    
    def test_create_manifest_basic(self):
        """Test creating basic manifest."""
        files = [
            {"path": "spans/spans.parquet", "media_type": "parquet", 
             "bytes": 100, "sha256": "abc", "rows": 10, "notes": ""}
        ]
        qa_status = {"state": "PASS", "summary": "OK", "invariants": {}, "provisional": True}
        schemas = {"spans": "1.0"}
        
        manifest = create_manifest(
            version="1.0.0-provisional",
            files=files,
            qa_status=qa_status,
            schema_versions=schemas,
            git_commit="abc123",
            qa_report_rel_path=None,
        )
        
        assert manifest['lake_version'] == '1.0.0-provisional'
        assert 'created_at' in manifest
        assert manifest['producer']['repo'] == 'transcription-lakehouse'
        assert manifest['producer']['commit'] == 'abc123'
        assert manifest['producer']['task'] == 'SNAPSHOT'
        assert manifest['contracts']['schemas'] == schemas
        assert manifest['files'] == files
        assert manifest['qa_status'] == qa_status
    
    def test_create_manifest_with_qa_report(self):
        """Test creating manifest with QA report path."""
        manifest = create_manifest(
            version="1.0.0",
            files=[],
            qa_status={"state": "PASS", "summary": "", "invariants": {}, "provisional": True},
            schema_versions={},
            git_commit="abc",
            qa_report_rel_path="quality_report/quality_assessment.md",
        )
        
        assert manifest['producer']['report_path'] == 'quality_report/quality_assessment.md'
    
    def test_create_manifest_timestamp_format(self):
        """Test manifest timestamp is in ISO-8601 UTC format."""
        manifest = create_manifest(
            version="1.0.0",
            files=[],
            qa_status={"state": "PASS", "summary": "", "invariants": {}, "provisional": True},
            schema_versions={},
            git_commit="abc",
        )
        
        # Check timestamp format: YYYY-MM-DDTHH:MM:SSZ
        assert 'T' in manifest['created_at']
        assert manifest['created_at'].endswith('Z')
        assert len(manifest['created_at']) == 20  # ISO-8601 UTC format


class TestManifestWriting:
    """Tests for manifest writing."""
    
    def test_write_manifest_success(self, tmp_path):
        """Test writing manifest to file."""
        manifest = {
            "lake_version": "1.0.0-provisional",
            "created_at": "2025-10-27T12:00:00Z",
            "files": [],
        }
        
        snapshot_root = tmp_path / "snapshot"
        snapshot_root.mkdir()
        
        write_manifest(manifest, snapshot_root)
        
        manifest_file = snapshot_root / "lake_manifest.json"
        assert manifest_file.exists()
        
        # Verify content
        with open(manifest_file, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['lake_version'] == '1.0.0-provisional'
    
    def test_write_manifest_pretty_printed(self, tmp_path):
        """Test manifest is pretty-printed."""
        manifest = {"lake_version": "1.0.0", "files": [{"path": "test"}]}
        
        snapshot_root = tmp_path / "snapshot"
        snapshot_root.mkdir()
        
        write_manifest(manifest, snapshot_root)
        
        content = (snapshot_root / "lake_manifest.json").read_text()
        
        # Should have indentation (not single line)
        assert '\n' in content
        assert '  ' in content  # 2-space indent
    
    def test_write_manifest_unicode(self, tmp_path):
        """Test manifest handles Unicode properly."""
        manifest = {"lake_version": "1.0.0", "summary": "Test with émoji 🎉"}
        
        snapshot_root = tmp_path / "snapshot"
        snapshot_root.mkdir()
        
        write_manifest(manifest, snapshot_root)
        
        with open(snapshot_root / "lake_manifest.json", 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        assert loaded['summary'] == "Test with émoji 🎉"


class TestManifestGenerator:
    """Tests for ManifestGenerator class."""
    
    def test_manifest_generator_initialization(self, tmp_path):
        """Test ManifestGenerator initialization."""
        lakehouse = tmp_path / "lakehouse"
        lakehouse.mkdir()
        
        generator = ManifestGenerator(lakehouse)
        
        assert generator.lakehouse_path == lakehouse
    
    def test_manifest_generator_generate(self, tmp_path):
        """Test full manifest generation."""
        lakehouse = tmp_path / "lakehouse"
        lakehouse.mkdir()
        
        generator = ManifestGenerator(lakehouse)
        
        files_metadata = [
            {"path": "spans/spans.parquet", "bytes": 100, "sha256": "abc", "rows": 10, "notes": ""}
        ]
        
        manifest = generator.generate(
            version="1.0.0-provisional",
            files_metadata=files_metadata,
            qa_report_path=None,
        )
        
        assert manifest['lake_version'] == '1.0.0-provisional'
        assert 'created_at' in manifest
        assert 'producer' in manifest
        assert 'files' in manifest
        assert 'qa_status' in manifest
        assert len(manifest['files']) == 1

