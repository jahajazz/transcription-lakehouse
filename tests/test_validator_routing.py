"""
Unit tests for validator routing system (Task 3.11).

Tests that quality checks are correctly routed to appropriate tables
based on their role (base vs embedding) and configuration.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from lakehouse.quality.validator_router import ValidatorRouter


# Fixtures


@pytest.fixture
def valid_routing_config():
    """Valid routing configuration for testing."""
    return {
        "tables": {
            "spans": {
                "role": "base",
                "description": "Single-speaker segments",
                "checks": ["coverage", "length_buckets", "duplicates", "text_quality"]
            },
            "beats": {
                "role": "base",
                "description": "Semantic meaning units",
                "checks": ["length_buckets", "order_no_overlap", "text_quality"]
            },
            "span_embeddings": {
                "role": "embedding",
                "description": "Vector embeddings for spans",
                "checks": ["dim_consistency", "id_join_back", "nn_leakage"]
            },
            "beat_embeddings": {
                "role": "embedding",
                "description": "Vector embeddings for beats",
                "checks": ["dim_consistency", "id_join_back", "nn_leakage"]
            }
        },
        "check_requirements": {
            "coverage": {
                "required_columns": ["start_time", "end_time", "episode_id"],
                "table_roles": ["base"]
            },
            "dim_consistency": {
                "required_columns": ["embedding"],
                "table_roles": ["embedding"]
            }
        },
        "error_handling": {
            "misconfigured_check": "fail",
            "missing_columns": "skip",
            "log_skipped_checks": True
        }
    }


@pytest.fixture
def temp_routing_config(valid_routing_config):
    """Create a temporary routing config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        yaml.dump(valid_routing_config, f)
        config_path = Path(f.name)
    
    yield config_path
    
    # Cleanup
    if config_path.exists():
        config_path.unlink()


@pytest.fixture
def router(temp_routing_config):
    """Create a ValidatorRouter with test config."""
    return ValidatorRouter(temp_routing_config)


# Test: Configuration Loading


def test_router_loads_valid_config(temp_routing_config):
    """Test that ValidatorRouter loads a valid config file."""
    router = ValidatorRouter(temp_routing_config)
    
    assert router.tables is not None
    assert len(router.tables) == 4
    assert "spans" in router.tables
    assert "span_embeddings" in router.tables


def test_router_fails_on_missing_config():
    """Test that ValidatorRouter raises FileNotFoundError for missing config."""
    with pytest.raises(FileNotFoundError):
        ValidatorRouter(Path("nonexistent_config.yaml"))


def test_router_fails_on_empty_config():
    """Test that ValidatorRouter raises ValueError for empty config."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        f.write("")  # Empty file
        config_path = Path(f.name)
    
    try:
        with pytest.raises(ValueError, match="empty"):
            ValidatorRouter(config_path)
    finally:
        config_path.unlink()


def test_router_fails_on_invalid_role():
    """Test that ValidatorRouter raises ValueError for invalid table role."""
    invalid_config = {
        "tables": {
            "test_table": {
                "role": "invalid_role",  # Should be 'base' or 'embedding'
                "checks": []
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        yaml.dump(invalid_config, f)
        config_path = Path(f.name)
    
    try:
        with pytest.raises(ValueError, match="invalid role"):
            ValidatorRouter(config_path)
    finally:
        config_path.unlink()


# Test: should_run_check()


def test_should_run_check_returns_true_for_configured_check(router):
    """Test that should_run_check returns True for configured checks."""
    assert router.should_run_check("spans", "coverage") is True
    assert router.should_run_check("span_embeddings", "dim_consistency") is True


def test_should_run_check_returns_false_for_unconfigured_check(router):
    """Test that should_run_check returns False for unconfigured checks."""
    # Coverage is not configured for embedding tables
    assert router.should_run_check("span_embeddings", "coverage") is False
    
    # dim_consistency is not configured for base tables
    assert router.should_run_check("spans", "dim_consistency") is False


def test_should_run_check_allows_unknown_table(router):
    """Test that should_run_check allows checks on unknown tables."""
    # Unknown tables should allow checks (permissive fallback)
    assert router.should_run_check("unknown_table", "any_check") is True


# Test: get_table_role()


def test_get_table_role_returns_correct_role(router):
    """Test that get_table_role returns the correct role."""
    assert router.get_table_role("spans") == "base"
    assert router.get_table_role("beats") == "base"
    assert router.get_table_role("span_embeddings") == "embedding"
    assert router.get_table_role("beat_embeddings") == "embedding"


def test_get_table_role_returns_unknown_for_missing_table(router):
    """Test that get_table_role returns 'unknown' for missing tables."""
    assert router.get_table_role("nonexistent_table") == "unknown"


# Test: get_checks_for_table()


def test_get_checks_for_table_returns_check_list(router):
    """Test that get_checks_for_table returns the configured checks."""
    span_checks = router.get_checks_for_table("spans")
    assert "coverage" in span_checks
    assert "length_buckets" in span_checks
    assert "duplicates" in span_checks
    
    embedding_checks = router.get_checks_for_table("span_embeddings")
    assert "dim_consistency" in embedding_checks
    assert "nn_leakage" in embedding_checks


def test_get_checks_for_table_returns_empty_for_unknown_table(router):
    """Test that get_checks_for_table returns empty list for unknown tables."""
    assert router.get_checks_for_table("unknown_table") == []


# Test: get_tables_by_role()


def test_get_tables_by_role_returns_base_tables(router):
    """Test that get_tables_by_role returns all base tables."""
    base_tables = router.get_tables_by_role("base")
    assert "spans" in base_tables
    assert "beats" in base_tables
    assert "span_embeddings" not in base_tables


def test_get_tables_by_role_returns_embedding_tables(router):
    """Test that get_tables_by_role returns all embedding tables."""
    embedding_tables = router.get_tables_by_role("embedding")
    assert "span_embeddings" in embedding_tables
    assert "beat_embeddings" in embedding_tables
    assert "spans" not in embedding_tables


# Test: get_check_requirements()


def test_get_check_requirements_returns_requirements(router):
    """Test that get_check_requirements returns check requirements."""
    coverage_reqs = router.get_check_requirements("coverage")
    assert coverage_reqs is not None
    assert "required_columns" in coverage_reqs
    assert "episode_id" in coverage_reqs["required_columns"]


def test_get_check_requirements_returns_none_for_unknown_check(router):
    """Test that get_check_requirements returns None for unknown checks."""
    assert router.get_check_requirements("unknown_check") is None


# Test: validate_check_for_table()


def test_validate_check_for_table_passes_with_required_columns(router):
    """Test that validate_check_for_table passes when columns are present."""
    can_run, reason = router.validate_check_for_table(
        "spans",
        "coverage",
        {"start_time", "end_time", "episode_id", "text"}
    )
    assert can_run is True
    assert reason is None


def test_validate_check_for_table_fails_with_missing_columns(router):
    """Test that validate_check_for_table fails when columns are missing."""
    can_run, reason = router.validate_check_for_table(
        "spans",
        "coverage",
        {"start_time", "text"}  # Missing end_time, episode_id
    )
    assert can_run is False
    assert "Missing required columns" in reason


def test_validate_check_for_table_fails_for_unconfigured_check(router):
    """Test that validate_check_for_table fails for unconfigured checks."""
    can_run, reason = router.validate_check_for_table(
        "spans",
        "dim_consistency",  # Not configured for base tables
        {"embedding"}
    )
    assert can_run is False
    assert "not configured" in reason


# Test: generate_routing_summary()


def test_generate_routing_summary_returns_complete_summary(router):
    """Test that generate_routing_summary returns all expected fields."""
    summary = router.generate_routing_summary()
    
    assert "total_tables" in summary
    assert summary["total_tables"] == 4
    
    assert "base_tables" in summary
    assert len(summary["base_tables"]) == 2
    
    assert "embedding_tables" in summary
    assert len(summary["embedding_tables"]) == 2
    
    assert "checks_per_table" in summary
    assert "spans" in summary["checks_per_table"]


# Test: Integration with QualityAssessor


def test_router_prevents_text_checks_on_embeddings(router):
    """Test that text checks are not routed to embedding tables (Task 3.12 preview)."""
    # Text quality check should not run on embedding tables
    assert router.should_run_check("span_embeddings", "text_quality") is False
    assert router.should_run_check("beat_embeddings", "text_quality") is False
    
    # But should run on base tables
    assert router.should_run_check("spans", "text_quality") is True


def test_router_prevents_timestamp_checks_on_embeddings(router):
    """Test that timestamp checks are not routed to embedding tables (Task 3.12 preview)."""
    # Coverage (which requires timestamps) should not run on embedding tables
    assert router.should_run_check("span_embeddings", "coverage") is False
    
    # But should run on base tables
    assert router.should_run_check("spans", "coverage") is True


def test_router_prevents_vector_checks_on_base_tables(router):
    """Test that vector checks are not routed to base tables."""
    # Vector checks should not run on base tables
    assert router.should_run_check("spans", "dim_consistency") is False
    assert router.should_run_check("beats", "nn_leakage") is False
    
    # But should run on embedding tables
    assert router.should_run_check("span_embeddings", "dim_consistency") is True


# Test: Edge Cases


def test_router_handles_empty_checks_list():
    """Test that router handles tables with no checks configured."""
    config = {
        "tables": {
            "empty_table": {
                "role": "base",
                "checks": []  # No checks
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        yaml.dump(config, f)
        config_path = Path(f.name)
    
    try:
        router = ValidatorRouter(config_path)
        assert router.get_checks_for_table("empty_table") == []
        assert router.should_run_check("empty_table", "any_check") is False
    finally:
        config_path.unlink()


def test_to_dict_returns_complete_config(router):
    """Test that to_dict returns a complete serializable config."""
    config_dict = router.to_dict()
    
    assert "tables" in config_dict
    assert "check_requirements" in config_dict
    assert "error_handling" in config_dict
    assert "summary" in config_dict
    
    # Should be JSON-serializable
    import json
    json_str = json.dumps(config_dict)
    assert json_str is not None

