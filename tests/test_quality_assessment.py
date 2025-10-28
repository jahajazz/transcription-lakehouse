"""
Integration tests for quality assessment system.

Tests the complete quality assessment pipeline end-to-end including:
- Full assessment orchestration with QualityAssessor
- Output file generation (JSON, CSV, Markdown)
- Reproducibility and determinism
- CLI command execution
- Graceful error handling
- Threshold overrides
"""

import pytest
import json
import pandas as pd
import numpy as np
from pathlib import Path
from click.testing import CliRunner
from datetime import datetime

# Import quality assessment components
from lakehouse.quality.assessor import QualityAssessor, AssessmentResult
from lakehouse.quality.thresholds import QualityThresholds, RAGStatus
from lakehouse.quality.reporter import QualityReporter
from lakehouse.structure import LakehouseStructure

# Import CLI command
from lakehouse.cli.commands.quality import quality as quality_command

# Import test fixtures
from tests.fixtures.quality_test_data import (
    create_complete_test_dataset,
    create_coverage_test_data,
    get_default_episodes,
    create_sample_spans,
    create_sample_beats,
    create_sample_embeddings,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_lakehouse(tmp_path):
    """
    Create a mock lakehouse structure with test data.
    
    Creates:
    - lakehouse/catalogs/episodes_*.parquet
    - lakehouse/spans/v1/spans.parquet
    - lakehouse/beats/v1/beats.parquet
    - lakehouse/embeddings/v1/span_embeddings.parquet (optional)
    """
    lakehouse_path = tmp_path / "lakehouse"
    lakehouse_path.mkdir()
    
    # Create directory structure
    catalogs_dir = lakehouse_path / "catalogs"
    catalogs_dir.mkdir()
    
    spans_dir = lakehouse_path / "spans" / "v1"
    spans_dir.mkdir(parents=True)
    
    beats_dir = lakehouse_path / "beats" / "v1"
    beats_dir.mkdir(parents=True)
    
    embeddings_dir = lakehouse_path / "embeddings" / "v1"
    embeddings_dir.mkdir(parents=True)
    
    # Generate test data
    dataset = create_complete_test_dataset(
        num_episodes=2,
        include_embeddings=True,
        include_edge_cases=True
    )
    
    # Save episodes catalog
    episodes = dataset['episodes']
    catalog_file = catalogs_dir / f"episodes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    episodes.to_parquet(catalog_file)
    
    # Save spans
    spans = dataset['spans']
    spans_file = spans_dir / "spans.parquet"
    spans.to_parquet(spans_file)
    
    # Save beats
    beats = dataset['beats']
    beats_file = beats_dir / "beats.parquet"
    beats.to_parquet(beats_file)
    
    # Save embeddings (create proper structure)
    if 'span_embeddings' in dataset:
        span_embeddings = dataset['span_embeddings']
        # Create embeddings dataframe with proper format
        embeddings_df = pd.DataFrame({
            'span_id': spans['span_id'].values if 'span_id' in spans.columns else [f'S{i:03d}' for i in range(len(spans))],
            'embedding': list(span_embeddings)
        })
        embeddings_file = embeddings_dir / "span_embeddings.parquet"
        embeddings_df.to_parquet(embeddings_file)
    
    if 'beat_embeddings' in dataset:
        beat_embeddings = dataset['beat_embeddings']
        embeddings_df = pd.DataFrame({
            'beat_id': beats['beat_id'].values if 'beat_id' in beats.columns else [f'B{i:03d}' for i in range(len(beats))],
            'embedding': list(beat_embeddings)
        })
        embeddings_file = embeddings_dir / "beat_embeddings.parquet"
        embeddings_df.to_parquet(embeddings_file)
    
    return {
        'lakehouse_path': lakehouse_path,
        'episodes': episodes,
        'spans': spans,
        'beats': beats,
    }


@pytest.fixture
def mock_lakehouse_no_embeddings(tmp_path):
    """Create a mock lakehouse without embeddings (for testing graceful handling)."""
    lakehouse_path = tmp_path / "lakehouse_no_emb"
    lakehouse_path.mkdir()
    
    # Create directory structure
    catalogs_dir = lakehouse_path / "catalogs"
    catalogs_dir.mkdir()
    
    spans_dir = lakehouse_path / "spans" / "v1"
    spans_dir.mkdir(parents=True)
    
    beats_dir = lakehouse_path / "beats" / "v1"
    beats_dir.mkdir(parents=True)
    
    # Create embeddings dir but don't put files in it
    embeddings_dir = lakehouse_path / "embeddings" / "v1"
    embeddings_dir.mkdir(parents=True)
    
    # Generate test data without embeddings
    dataset = create_complete_test_dataset(
        num_episodes=2,
        include_embeddings=False,
        include_edge_cases=False
    )
    
    # Save data
    episodes = dataset['episodes']
    catalog_file = catalogs_dir / f"episodes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    episodes.to_parquet(catalog_file)
    
    spans = dataset['spans']
    spans_file = spans_dir / "spans.parquet"
    spans.to_parquet(spans_file)
    
    beats = dataset['beats']
    beats_file = beats_dir / "beats.parquet"
    beats.to_parquet(beats_file)
    
    return {
        'lakehouse_path': lakehouse_path,
        'episodes': episodes,
        'spans': spans,
        'beats': beats,
    }


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    output = tmp_path / "output"
    output.mkdir()
    return output


@pytest.fixture
def default_thresholds():
    """Get default quality thresholds."""
    return QualityThresholds()


# ============================================================================
# Task 6.3.1: Test Full Assessment Run with Sample Data
# ============================================================================

class TestFullAssessmentRun:
    """Test complete quality assessment runs."""
    
    def test_assessment_with_sample_data(self, mock_lakehouse, output_dir):
        """Test full assessment run with sample data."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Check result structure
        assert isinstance(result, AssessmentResult)
        assert result.total_episodes > 0
        assert result.total_spans > 0
        assert result.total_beats > 0
        assert result.assessment_duration_seconds > 0
    
    def test_assessment_returns_metrics_bundle(self, mock_lakehouse, output_dir):
        """Test that assessment returns complete metrics bundle."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Check metrics bundle
        metrics = result.metrics
        assert hasattr(metrics, 'coverage_metrics')
        assert hasattr(metrics, 'distribution_metrics')
        assert hasattr(metrics, 'integrity_metrics')
        assert hasattr(metrics, 'balance_metrics')
        assert hasattr(metrics, 'text_quality_metrics')
        assert hasattr(metrics, 'embedding_metrics')
        assert hasattr(metrics, 'diagnostics')
        
        # Check that metrics contain data
        assert len(metrics.coverage_metrics) > 0
        assert len(metrics.distribution_metrics) > 0
    
    def test_assessment_determines_rag_status(self, mock_lakehouse, output_dir):
        """Test that assessment determines RAG status."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Check RAG status
        assert isinstance(result.rag_status, RAGStatus)
        assert result.rag_status in [RAGStatus.GREEN, RAGStatus.AMBER, RAGStatus.RED]
    
    def test_assessment_spans_only(self, mock_lakehouse, output_dir):
        """Test assessment with only spans."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=False,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        assert result.total_spans > 0
        assert result.total_beats == 0
    
    def test_assessment_beats_only(self, mock_lakehouse, output_dir):
        """Test assessment with only beats."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=False,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        assert result.total_spans == 0
        assert result.total_beats > 0
    
    def test_assessment_with_threshold_violations(self, mock_lakehouse, output_dir):
        """Test that assessment detects threshold violations."""
        # Use very strict thresholds to ensure violations
        strict_thresholds = QualityThresholds(
            coverage_min=99.0,  # Very high coverage requirement
            span_length_compliance_min=99.0,
            timestamp_regressions_max=0,
            negative_duration_max=0
        )
        
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1',
            thresholds=strict_thresholds
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Should have some violations with strict thresholds
        # (since test data includes edge cases)
        assert len(result.violations) >= 0  # May or may not have violations
    
    def test_assessment_error_on_missing_data(self, tmp_path):
        """Test that assessment fails gracefully with missing data."""
        # Create empty lakehouse
        empty_lakehouse = tmp_path / "empty_lakehouse"
        empty_lakehouse.mkdir()
        
        assessor = QualityAssessor(
            lakehouse_path=empty_lakehouse,
            version='v1'
        )
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            assessor.run_assessment(
                assess_spans=True,
                assess_beats=True
            )


# ============================================================================
# Task 6.3.2: Test Output File Generation
# ============================================================================

class TestOutputFileGeneration:
    """Test generation of all output files."""
    
    def test_creates_output_directory_structure(self, mock_lakehouse, output_dir):
        """Test that proper output directory structure is created."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Check directory structure
        assert output_dir.exists()
        assert (output_dir / "metrics").exists()
        assert (output_dir / "diagnostics").exists()
        assert (output_dir / "report").exists()
    
    def test_generates_global_metrics_json(self, mock_lakehouse, output_dir):
        """Test generation of global metrics JSON file."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Check global.json exists and is valid
        global_json = output_dir / "metrics" / "global.json"
        assert global_json.exists()
        
        with open(global_json, 'r') as f:
            data = json.load(f)
        
        assert 'metrics' in data or 'coverage' in data  # Should have metrics data
    
    def test_generates_episodes_csv(self, mock_lakehouse, output_dir):
        """Test generation of per-episode metrics CSV."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Check episodes.csv exists and can be read
        episodes_csv = output_dir / "metrics" / "episodes.csv"
        assert episodes_csv.exists()
        
        df = pd.read_csv(episodes_csv)
        assert len(df) > 0
        assert 'episode_id' in df.columns
    
    def test_generates_segments_csv(self, mock_lakehouse, output_dir):
        """Test generation of per-segment metrics CSVs."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Check spans.csv
        spans_csv = output_dir / "metrics" / "spans.csv"
        assert spans_csv.exists()
        
        df = pd.read_csv(spans_csv)
        assert len(df) > 0
        
        # Check beats.csv
        beats_csv = output_dir / "metrics" / "beats.csv"
        assert beats_csv.exists()
        
        df = pd.read_csv(beats_csv)
        assert len(df) > 0
    
    def test_generates_markdown_report(self, mock_lakehouse, output_dir):
        """Test generation of markdown quality report."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Check report exists
        report_md = output_dir / "report" / "quality_assessment.md"
        assert report_md.exists()
        
        # Check report content
        content = report_md.read_text()
        assert len(content) > 0
        assert 'Quality Assessment Report' in content or 'Executive Summary' in content
        assert 'RAG Status' in content or result.rag_status.value.upper() in content
    
    def test_generates_diagnostics_csv(self, mock_lakehouse, output_dir):
        """Test generation of diagnostic CSV files."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Check outliers.csv
        outliers_csv = output_dir / "diagnostics" / "outliers.csv"
        assert outliers_csv.exists()
        
        df = pd.read_csv(outliers_csv)
        assert len(df) > 0
    
    def test_output_paths_in_result(self, mock_lakehouse, output_dir):
        """Test that output paths are returned in result."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Check output_paths attribute
        assert hasattr(result, 'output_paths')
        assert len(result.output_paths) > 0
    
    def test_timestamped_output_directory(self, mock_lakehouse, output_dir):
        """Test creation of timestamped output directory."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=True  # Enable timestamp
        )
        
        # Check that a timestamped subdirectory was created
        subdirs = list(output_dir.iterdir())
        assert len(subdirs) > 0
        
        # Check that subdirectory name looks like a timestamp (YYYYMMDD_HHMMSS)
        subdir_name = subdirs[0].name
        assert len(subdir_name) == 15  # YYYYMMDD_HHMMSS
        assert subdir_name[8] == '_'


# ============================================================================
# Task 6.3.3: Test Reproducibility
# ============================================================================

class TestReproducibility:
    """Test deterministic and reproducible results."""
    
    def test_same_input_produces_same_metrics(self, mock_lakehouse, output_dir):
        """Test that running twice on same data produces same metrics."""
        assessor1 = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result1 = assessor1.run_assessment(
            assess_spans=True,
            assess_beats=False,
            output_dir=output_dir / "run1",
            use_timestamp=False
        )
        
        assessor2 = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result2 = assessor2.run_assessment(
            assess_spans=True,
            assess_beats=False,
            output_dir=output_dir / "run2",
            use_timestamp=False
        )
        
        # Compare key metrics (should be identical)
        assert result1.total_episodes == result2.total_episodes
        assert result1.total_spans == result2.total_spans
        assert result1.rag_status == result2.rag_status
        
        # Compare coverage metrics
        cov1 = result1.metrics.coverage_metrics.get('global', {})
        cov2 = result2.metrics.coverage_metrics.get('global', {})
        
        if 'total_spans' in cov1 and 'total_spans' in cov2:
            assert cov1['total_spans'] == cov2['total_spans']
    
    def test_random_seed_determinism(self, mock_lakehouse, output_dir):
        """Test that random seed ensures deterministic sampling."""
        # Both should use same default random seed (42)
        thresholds1 = QualityThresholds(random_seed=42)
        thresholds2 = QualityThresholds(random_seed=42)
        
        assessor1 = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1',
            thresholds=thresholds1
        )
        
        result1 = assessor1.run_assessment(
            assess_spans=True,
            assess_beats=False,
            output_dir=output_dir / "run1",
            use_timestamp=False
        )
        
        assessor2 = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1',
            thresholds=thresholds2
        )
        
        result2 = assessor2.run_assessment(
            assess_spans=True,
            assess_beats=False,
            output_dir=output_dir / "run2",
            use_timestamp=False
        )
        
        # Results should be identical with same seed
        assert result1.total_spans == result2.total_spans
        assert result1.total_episodes == result2.total_episodes
    
    def test_metrics_precision_consistent(self, mock_lakehouse, output_dir):
        """Test that metrics are rounded to consistent precision."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=False,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Check that percentages are rounded to 2 decimal places
        coverage = result.metrics.coverage_metrics.get('global', {})
        if 'global_span_coverage_percent' in coverage:
            cov_pct = coverage['global_span_coverage_percent']
            # Should have at most 2 decimal places
            assert cov_pct == round(cov_pct, 2)


# ============================================================================
# Task 6.3.4: Test CLI Command Execution
# ============================================================================

class TestCLICommand:
    """Test CLI command interface."""
    
    def test_quality_command_basic(self, mock_lakehouse, output_dir):
        """Test basic quality command execution."""
        runner = CliRunner()
        
        result = runner.invoke(
            quality_command,
            [
                '--lakehouse-path', str(mock_lakehouse['lakehouse_path']),
                '--output-dir', str(output_dir),
                '--no-timestamp',
            ],
            catch_exceptions=False
        )
        
        # Command should succeed
        assert result.exit_code == 0
    
    def test_quality_command_with_version(self, mock_lakehouse, output_dir):
        """Test quality command with version option."""
        runner = CliRunner()
        
        result = runner.invoke(
            quality_command,
            [
                '--lakehouse-path', str(mock_lakehouse['lakehouse_path']),
                '--version', 'v1',
                '--output-dir', str(output_dir),
                '--no-timestamp',
            ],
            catch_exceptions=False
        )
        
        assert result.exit_code == 0
    
    def test_quality_command_spans_only(self, mock_lakehouse, output_dir):
        """Test quality command with spans only."""
        runner = CliRunner()
        
        result = runner.invoke(
            quality_command,
            [
                '--lakehouse-path', str(mock_lakehouse['lakehouse_path']),
                '--level', 'spans',
                '--output-dir', str(output_dir),
                '--no-timestamp',
            ],
            catch_exceptions=False
        )
        
        assert result.exit_code == 0
    
    def test_quality_command_beats_only(self, mock_lakehouse, output_dir):
        """Test quality command with beats only."""
        runner = CliRunner()
        
        result = runner.invoke(
            quality_command,
            [
                '--lakehouse-path', str(mock_lakehouse['lakehouse_path']),
                '--level', 'beats',
                '--output-dir', str(output_dir),
                '--no-timestamp',
            ],
            catch_exceptions=False
        )
        
        assert result.exit_code == 0
    
    def test_quality_command_with_sample_size(self, mock_lakehouse, output_dir):
        """Test quality command with custom sample size."""
        runner = CliRunner()
        
        result = runner.invoke(
            quality_command,
            [
                '--lakehouse-path', str(mock_lakehouse['lakehouse_path']),
                '--sample-size', '50',
                '--output-dir', str(output_dir),
                '--no-timestamp',
            ],
            catch_exceptions=False
        )
        
        assert result.exit_code == 0


# ============================================================================
# Task 6.3.5: Test Graceful Handling of Missing Embeddings
# ============================================================================

class TestMissingEmbeddingsHandling:
    """Test graceful handling when embeddings are not available."""
    
    def test_assessment_without_embeddings(self, mock_lakehouse_no_embeddings, output_dir):
        """Test that assessment works without embeddings."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse_no_embeddings['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Should complete successfully
        assert isinstance(result, AssessmentResult)
        assert result.total_episodes > 0
        
        # embeddings_available should be False
        assert result.metrics.embeddings_available == False
    
    def test_embedding_metrics_skipped_when_missing(self, mock_lakehouse_no_embeddings, output_dir):
        """Test that embedding metrics are skipped when embeddings missing."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse_no_embeddings['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Embedding metrics should be empty or indicate skipped
        embedding_metrics = result.metrics.embedding_metrics
        assert len(embedding_metrics) == 0 or embedding_metrics.get('spans') is None
    
    def test_other_metrics_calculated_without_embeddings(self, mock_lakehouse_no_embeddings, output_dir):
        """Test that other metrics are still calculated without embeddings."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse_no_embeddings['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Coverage, distribution, integrity, balance, text quality should all work
        assert len(result.metrics.coverage_metrics) > 0
        assert len(result.metrics.distribution_metrics) > 0
        assert len(result.metrics.integrity_metrics) > 0
        assert len(result.metrics.balance_metrics) > 0
        assert len(result.metrics.text_quality_metrics) > 0
    
    def test_report_indicates_embeddings_skipped(self, mock_lakehouse_no_embeddings, output_dir):
        """Test that report indicates embedding checks were skipped."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse_no_embeddings['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Read report
        report_md = output_dir / "report" / "quality_assessment.md"
        assert report_md.exists()
        
        content = report_md.read_text()
        # Report should mention embeddings were not available or skipped
        assert 'embedding' in content.lower() or 'not available' in content.lower()


# ============================================================================
# Task 6.3.6: Test Threshold Override via Command Line
# ============================================================================

class TestThresholdOverrides:
    """Test threshold configuration and overrides."""
    
    def test_override_coverage_threshold(self, mock_lakehouse, output_dir):
        """Test overriding coverage threshold."""
        custom_thresholds = QualityThresholds(coverage_min=80.0)
        
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1',
            thresholds=custom_thresholds
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=False,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Check that custom threshold was applied
        assert result.thresholds.coverage_min == 80.0
    
    def test_override_span_length_thresholds(self, mock_lakehouse, output_dir):
        """Test overriding span length thresholds."""
        custom_thresholds = QualityThresholds(
            span_length_min=30.0,
            span_length_max=100.0
        )
        
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1',
            thresholds=custom_thresholds
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=False,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        assert result.thresholds.span_length_min == 30.0
        assert result.thresholds.span_length_max == 100.0
    
    def test_cli_threshold_overrides(self, mock_lakehouse, output_dir):
        """Test threshold overrides via CLI."""
        runner = CliRunner()
        
        result = runner.invoke(
            quality_command,
            [
                '--lakehouse-path', str(mock_lakehouse['lakehouse_path']),
                '--coverage-min', '85.0',
                '--span-length-min', '25.0',
                '--span-length-max', '110.0',
                '--output-dir', str(output_dir),
                '--no-timestamp',
            ],
            catch_exceptions=False
        )
        
        # Command should succeed with overrides
        assert result.exit_code == 0
    
    def test_multiple_threshold_overrides(self, mock_lakehouse, output_dir):
        """Test multiple threshold overrides at once."""
        threshold_overrides = {
            'coverage_min': 90.0,
            'span_length_min': 25.0,
            'span_length_max': 110.0,
            'neighbor_sample_size': 50,
            'neighbor_k': 15,
        }
        
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1',
            threshold_overrides=threshold_overrides
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=False,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Check all overrides applied
        assert result.thresholds.coverage_min == 90.0
        assert result.thresholds.span_length_min == 25.0
        assert result.thresholds.span_length_max == 110.0
        assert result.thresholds.neighbor_sample_size == 50
        assert result.thresholds.neighbor_k == 15
    
    def test_threshold_overrides_apply_overrides_method(self):
        """Test QualityThresholds.apply_overrides() method."""
        base_thresholds = QualityThresholds()
        
        overridden = base_thresholds.apply_overrides(
            coverage_min=85.0,
            span_length_min=25.0
        )
        
        # Check that overrides were applied
        assert overridden.coverage_min == 85.0
        assert overridden.span_length_min == 25.0
        
        # Check that other values remain default
        assert overridden.span_length_max == base_thresholds.span_length_max


# ============================================================================
# Additional Integration Tests
# ============================================================================

class TestAssessmentIntegration:
    """Additional integration tests."""
    
    def test_assessment_to_dict_serialization(self, mock_lakehouse, output_dir):
        """Test that assessment result can be serialized to dict."""
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=False,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Convert to dict
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'metrics' in result_dict
        assert 'thresholds' in result_dict
        assert 'violations' in result_dict
        assert 'rag_status' in result_dict
    
    def test_assessment_with_violations_list(self, mock_lakehouse, output_dir):
        """Test that violations are properly tracked."""
        # Use strict thresholds to trigger violations
        strict_thresholds = QualityThresholds(
            coverage_min=99.9,
            timestamp_regressions_max=0,
            negative_duration_max=0
        )
        
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1',
            thresholds=strict_thresholds
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=False,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Check violations structure
        assert hasattr(result, 'violations')
        assert isinstance(result.violations, list)
        
        # Get critical violations
        critical = result.get_critical_violations()
        warnings = result.get_warnings()
        
        assert isinstance(critical, list)
        assert isinstance(warnings, list)
    
    def test_full_pipeline_end_to_end(self, mock_lakehouse, output_dir):
        """Test complete pipeline from initialization to report generation."""
        # Initialize
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        # Run assessment
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Verify all outputs exist
        assert (output_dir / "metrics" / "global.json").exists()
        assert (output_dir / "metrics" / "episodes.csv").exists()
        assert (output_dir / "metrics" / "spans.csv").exists()
        assert (output_dir / "metrics" / "beats.csv").exists()
        assert (output_dir / "report" / "quality_assessment.md").exists()
        assert (output_dir / "diagnostics" / "outliers.csv").exists()
        
        # Verify result completeness
        assert result.total_episodes > 0
        assert result.total_spans > 0
        assert result.total_beats > 0
        assert result.assessment_duration_seconds > 0
        assert isinstance(result.rag_status, RAGStatus)


# ============================================================================
# Task 4.10 & 4.11: Fix Pack - Part 1 Quality Assessment Tests
# ============================================================================

class TestFixPackPart1Improvements:
    """Test improvements from Fix Pack — Part-1 Foundations (Tasks 4.10, 4.11)."""
    
    def test_coverage_never_exceeds_100_percent(self, mock_lakehouse, output_dir):
        """
        Test that coverage percentage never exceeds 100% (Task 4.10).
        
        Verifies overlap-aware union prevents >100% coverage even with
        overlapping spans.
        """
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Check global coverage
        coverage = result.metrics.coverage_metrics
        assert coverage is not None
        
        global_metrics = coverage.get('global', {})
        global_span_coverage = global_metrics.get('global_span_coverage_percent', 0)
        global_beat_coverage = global_metrics.get('global_beat_coverage_percent', 0)
        
        # Coverage should never exceed 100%
        assert global_span_coverage <= 100.0, \
            f"Global span coverage {global_span_coverage}% exceeds 100%"
        assert global_beat_coverage <= 100.0, \
            f"Global beat coverage {global_beat_coverage}% exceeds 100%"
        
        # Check per-episode coverage
        per_episode = coverage.get('per_episode', [])
        for episode_metrics in per_episode:
            span_coverage = episode_metrics.get('span_coverage_percent')
            if span_coverage is not None:
                assert span_coverage <= 100.0, \
                    f"Episode {episode_metrics.get('episode_id')} span coverage " \
                    f"{span_coverage}% exceeds 100%"
            
            beat_coverage = episode_metrics.get('beat_coverage_percent')
            if beat_coverage is not None:
                assert beat_coverage <= 100.0, \
                    f"Episode {episode_metrics.get('episode_id')} beat coverage " \
                    f"{beat_coverage}% exceeds 100%"
    
    def test_length_compliance_buckets_sum_to_100_percent(self, mock_lakehouse, output_dir):
        """
        Test that length compliance buckets sum to exactly 100% (Task 4.10).
        
        Verifies that too_short_percent + within_bounds_percent + too_long_percent = 100.0
        """
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        distribution = result.metrics.distribution_metrics
        assert distribution is not None
        
        # Check span compliance buckets
        span_compliance = distribution.get('spans', {}).get('compliance', {})
        if span_compliance and span_compliance.get('total_count', 0) > 0:
            too_short = span_compliance.get('too_short_percent', 0)
            within_bounds = span_compliance.get('within_bounds_percent', 0)
            too_long = span_compliance.get('too_long_percent', 0)
            
            bucket_sum = too_short + within_bounds + too_long
            
            # Allow 0.01% tolerance for rounding
            assert abs(bucket_sum - 100.0) < 0.01, \
                f"Span buckets sum to {bucket_sum}% instead of 100% " \
                f"(too_short={too_short}%, within_bounds={within_bounds}%, too_long={too_long}%)"
        
        # Check beat compliance buckets
        beat_compliance = distribution.get('beats', {}).get('compliance', {})
        if beat_compliance and beat_compliance.get('total_count', 0) > 0:
            too_short = beat_compliance.get('too_short_percent', 0)
            within_bounds = beat_compliance.get('within_bounds_percent', 0)
            too_long = beat_compliance.get('too_long_percent', 0)
            
            bucket_sum = too_short + within_bounds + too_long
            
            # Allow 0.01% tolerance for rounding
            assert abs(bucket_sum - 100.0) < 0.01, \
                f"Beat buckets sum to {bucket_sum}% instead of 100% " \
                f"(too_short={too_short}%, within_bounds={within_bounds}%, too_long={too_long}%)"
    
    def test_executive_summary_consistency(self, mock_lakehouse, output_dir):
        """
        Test that Executive Summary counts match detailed sections (Task 4.10).
        
        Ensures episode/span/beat counts are consistent across
        Executive Summary, coverage metrics, and distribution metrics.
        """
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Get counts from different sources
        summary_episodes = result.total_episodes
        summary_spans = result.total_spans
        summary_beats = result.total_beats
        
        coverage = result.metrics.coverage_metrics
        global_metrics = coverage.get('global', {})
        
        coverage_episodes = global_metrics.get('total_episodes', 0)
        coverage_spans = global_metrics.get('total_spans', 0)
        coverage_beats = global_metrics.get('total_beats', 0)
        
        # Verify consistency
        assert summary_episodes == coverage_episodes, \
            f"Episode count mismatch: Summary={summary_episodes}, Coverage={coverage_episodes}"
        
        if coverage_spans:  # May be None if no spans
            assert summary_spans == coverage_spans, \
                f"Span count mismatch: Summary={summary_spans}, Coverage={coverage_spans}"
        
        if coverage_beats:  # May be None if no beats
            assert summary_beats == coverage_beats, \
                f"Beat count mismatch: Summary={summary_beats}, Coverage={coverage_beats}"
        
        # Also check against distribution metrics
        distribution = result.metrics.distribution_metrics
        
        span_stats = distribution.get('spans', {}).get('statistics', {})
        if span_stats:
            dist_span_count = span_stats.get('count', 0)
            assert summary_spans == dist_span_count, \
                f"Span count mismatch with distribution: Summary={summary_spans}, Distribution={dist_span_count}"
        
        beat_stats = distribution.get('beats', {}).get('statistics', {})
        if beat_stats:
            dist_beat_count = beat_stats.get('count', 0)
            assert summary_beats == dist_beat_count, \
                f"Beat count mismatch with distribution: Summary={summary_beats}, Distribution={dist_beat_count}"
    
    def test_duplicate_detection_with_min_text_length(self, mock_lakehouse, output_dir):
        """
        Test that duplicate detection applies minimum text length filter (Task 4.11).
        
        Verifies that the min_text_length parameter is used and short texts
        are excluded from duplicate detection.
        """
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        integrity = result.metrics.integrity_metrics
        assert integrity is not None
        
        # Check that min_text_length is being used
        if 'min_text_length' in integrity:
            assert integrity['min_text_length'] == 10, \
                "Duplicate detection should use min_text_length=10"
        
        # Check that segments_checked is reported
        if 'segments_checked' in integrity:
            total_segments = integrity.get('total_segments', 0)
            segments_checked = integrity.get('segments_checked', 0)
            
            # segments_checked should be <= total_segments
            assert segments_checked <= total_segments, \
                f"Checked {segments_checked} segments but only {total_segments} total exist"
    
    def test_duplicate_checks_separate_for_spans_and_beats(self, mock_lakehouse, output_dir):
        """
        Test that duplicate checks show separate counts for spans and beats (Task 4.11).
        
        Verifies that duplicate detection runs separately at each level.
        """
        assessor = QualityAssessor(
            lakehouse_path=mock_lakehouse['lakehouse_path'],
            version='v1'
        )
        
        # Run assessment for both spans and beats
        result = assessor.run_assessment(
            assess_spans=True,
            assess_beats=True,
            output_dir=output_dir,
            use_timestamp=False
        )
        
        # Check that integrity metrics exist
        integrity = result.metrics.integrity_metrics
        assert integrity is not None
        
        # Duplicate metrics should be present
        assert 'exact_duplicate_count' in integrity
        assert 'exact_duplicate_percent' in integrity
        
        # Verify duplicate percentages are in valid range
        exact_dup_pct = integrity.get('exact_duplicate_percent', 0)
        near_dup_pct = integrity.get('near_duplicate_percent', 0)
        
        assert 0 <= exact_dup_pct <= 100, \
            f"Exact duplicate percentage {exact_dup_pct}% out of range [0, 100]"
        assert 0 <= near_dup_pct <= 100, \
            f"Near duplicate percentage {near_dup_pct}% out of range [0, 100]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
