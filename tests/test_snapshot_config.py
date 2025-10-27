"""
Tests for snapshot configuration management.

Tests version parsing, formatting, collision detection, and configuration loading.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from lakehouse.snapshot.config import (
    SnapshotConfig,
    parse_version,
    format_version,
    find_existing_snapshots,
    get_next_version,
    resolve_snapshot_root,
    ensure_snapshot_root_writable,
)


class TestVersionParsing:
    """Tests for version parsing and formatting."""
    
    def test_parse_version_basic(self):
        """Test parsing basic semantic version."""
        major, minor, patch, suffix = parse_version("1.2.3")
        assert major == 1
        assert minor == 2
        assert patch == 3
        assert suffix == ""
    
    def test_parse_version_with_v_prefix(self):
        """Test parsing version with 'v' prefix."""
        major, minor, patch, suffix = parse_version("v2.3.4")
        assert major == 2
        assert minor == 3
        assert patch == 4
        assert suffix == ""
    
    def test_parse_version_with_suffix(self):
        """Test parsing version with suffix."""
        major, minor, patch, suffix = parse_version("1.0.0-provisional")
        assert major == 1
        assert minor == 0
        assert patch == 0
        assert suffix == "-provisional"
    
    def test_parse_version_with_rc_suffix(self):
        """Test parsing version with release candidate suffix."""
        major, minor, patch, suffix = parse_version("2.1.0-rc1")
        assert major == 2
        assert minor == 1
        assert patch == 0
        assert suffix == "-rc1"
    
    def test_parse_version_with_metadata(self):
        """Test parsing version with build metadata."""
        major, minor, patch, suffix = parse_version("1.0.0+20240101")
        assert major == 1
        assert minor == 0
        assert patch == 0
        assert suffix == "+20240101"
    
    def test_parse_version_invalid_format(self):
        """Test parsing invalid version format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid semantic version"):
            parse_version("1.2")
        
        with pytest.raises(ValueError, match="Invalid semantic version"):
            parse_version("abc")
        
        with pytest.raises(ValueError, match="Invalid semantic version"):
            parse_version("1.2.3.4")
    
    def test_format_version_basic(self):
        """Test formatting basic version."""
        version = format_version(1, 2, 3)
        assert version == "1.2.3"
    
    def test_format_version_with_suffix(self):
        """Test formatting version with suffix."""
        version = format_version(1, 0, 0, "-provisional")
        assert version == "1.0.0-provisional"
    
    def test_format_version_zero_values(self):
        """Test formatting version with zero values."""
        version = format_version(0, 1, 0)
        assert version == "0.1.0"


class TestSnapshotDiscovery:
    """Tests for snapshot discovery and collision handling."""
    
    def test_find_existing_snapshots_empty_directory(self, tmp_path):
        """Test finding snapshots in empty directory."""
        snapshots = find_existing_snapshots(tmp_path)
        assert snapshots == []
    
    def test_find_existing_snapshots_nonexistent_directory(self):
        """Test finding snapshots in non-existent directory."""
        non_existent = Path("/nonexistent/path")
        snapshots = find_existing_snapshots(non_existent)
        assert snapshots == []
    
    def test_find_existing_snapshots_with_versions(self, tmp_path):
        """Test finding multiple version directories."""
        # Create version directories
        (tmp_path / "v1.0.0-provisional").mkdir()
        (tmp_path / "v1.0.1-provisional").mkdir()
        (tmp_path / "v2.0.0").mkdir()
        
        # Create non-version directories (should be ignored)
        (tmp_path / "other_dir").mkdir()
        (tmp_path / "README.md").touch()
        
        snapshots = find_existing_snapshots(tmp_path)
        
        assert len(snapshots) == 3
        assert "1.0.0-provisional" in snapshots
        assert "1.0.1-provisional" in snapshots
        assert "2.0.0" in snapshots
    
    def test_find_existing_snapshots_sorted(self, tmp_path):
        """Test snapshots are returned in sorted order."""
        # Create in non-sorted order
        (tmp_path / "v2.0.0").mkdir()
        (tmp_path / "v1.0.0").mkdir()
        (tmp_path / "v1.5.0").mkdir()
        
        snapshots = find_existing_snapshots(tmp_path)
        
        assert snapshots == ["1.0.0", "1.5.0", "2.0.0"]
    
    def test_get_next_version_no_collision(self):
        """Test getting next version with no collision."""
        version = get_next_version((1, 0, 0), [], "-provisional")
        assert version == "1.0.0-provisional"
    
    def test_get_next_version_with_collision(self):
        """Test getting next version with collision."""
        existing = ["1.0.0-provisional"]
        version = get_next_version((1, 0, 0), existing, "-provisional")
        assert version == "1.0.1-provisional"
    
    def test_get_next_version_multiple_collisions(self):
        """Test getting next version with multiple collisions."""
        existing = ["1.0.0-provisional", "1.0.1-provisional", "1.0.2-provisional"]
        version = get_next_version((1, 0, 0), existing, "-provisional")
        assert version == "1.0.3-provisional"
    
    def test_get_next_version_different_major_minor(self):
        """Test getting next version doesn't collide with different major/minor."""
        existing = ["1.0.0-provisional", "2.0.0-provisional"]
        version = get_next_version((1, 0, 0), existing, "-provisional")
        assert version == "1.0.1-provisional"


class TestSnapshotConfig:
    """Tests for SnapshotConfig class."""
    
    def test_snapshot_config_defaults(self):
        """Test SnapshotConfig with default values."""
        config = SnapshotConfig()
        
        assert config.major == 0
        assert config.minor == 1
        assert config.patch == 0
        assert config.auto_increment is True
        assert config.provisional_suffix == "-provisional"
    
    def test_snapshot_config_get_version_tuple(self):
        """Test getting version as tuple."""
        config = SnapshotConfig()
        version_tuple = config.get_version_tuple()
        
        assert version_tuple == (0, 1, 0)
    
    def test_snapshot_config_get_version_string(self):
        """Test getting version as string."""
        config = SnapshotConfig()
        version_string = config.get_version_string()
        
        assert version_string == "0.1.0-provisional"
    
    def test_snapshot_config_get_version_string_no_suffix(self):
        """Test getting version string without suffix."""
        config = SnapshotConfig()
        version_string = config.get_version_string(include_suffix=False)
        
        assert version_string == "0.1.0"
    
    def test_snapshot_config_from_yaml(self, tmp_path):
        """Test loading SnapshotConfig from YAML file."""
        config_file = tmp_path / "snapshot_config.yaml"
        config_data = {
            "version": {"major": 2, "minor": 3, "patch": 4},
            "snapshot_root": "./custom_snapshots",
            "auto_increment": False,
            "provisional_suffix": "-rc",
        }
        
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        config = SnapshotConfig(config_path=config_file)
        
        assert config.major == 2
        assert config.minor == 3
        assert config.patch == 4
        assert config.snapshot_root_template == "./custom_snapshots"
        assert config.auto_increment is False
        assert config.provisional_suffix == "-rc"
    
    def test_snapshot_config_with_overrides(self):
        """Test SnapshotConfig with overrides."""
        overrides = {
            "version": {"major": 5, "minor": 6, "patch": 7},
            "auto_increment": False,
        }
        
        config = SnapshotConfig(overrides=overrides)
        
        assert config.major == 5
        assert config.minor == 6
        assert config.patch == 7
        assert config.auto_increment is False


class TestSnapshotRootResolution:
    """Tests for snapshot root resolution."""
    
    def test_resolve_snapshot_root_default(self):
        """Test resolving snapshot root with default."""
        config = SnapshotConfig()
        snapshot_root = resolve_snapshot_root(config)
        
        # Should resolve to ./snapshots relative to current directory
        assert snapshot_root.name == "snapshots"
    
    def test_resolve_snapshot_root_with_env_var(self, monkeypatch, tmp_path):
        """Test resolving snapshot root with environment variable."""
        custom_path = tmp_path / "custom_snapshots"
        monkeypatch.setenv("SNAPSHOT_ROOT", str(custom_path))
        
        config = SnapshotConfig()
        snapshot_root = resolve_snapshot_root(config)
        
        assert snapshot_root == custom_path
    
    def test_ensure_snapshot_root_writable_creates_directory(self, tmp_path):
        """Test ensure_snapshot_root_writable creates directory."""
        snapshot_root = tmp_path / "new_snapshots"
        
        assert not snapshot_root.exists()
        
        ensure_snapshot_root_writable(snapshot_root)
        
        assert snapshot_root.exists()
        assert snapshot_root.is_dir()
    
    def test_ensure_snapshot_root_writable_existing_directory(self, tmp_path):
        """Test ensure_snapshot_root_writable with existing directory."""
        snapshot_root = tmp_path / "existing_snapshots"
        snapshot_root.mkdir()
        
        # Should not raise error
        ensure_snapshot_root_writable(snapshot_root)
        
        assert snapshot_root.exists()
    
    def test_ensure_snapshot_root_writable_permission_check(self, tmp_path):
        """Test ensure_snapshot_root_writable verifies write permissions."""
        snapshot_root = tmp_path / "writable_snapshots"
        
        ensure_snapshot_root_writable(snapshot_root)
        
        # Verify we can write to the directory
        test_file = snapshot_root / "test.txt"
        test_file.write_text("test")
        assert test_file.exists()
        test_file.unlink()

