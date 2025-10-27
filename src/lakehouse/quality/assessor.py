"""
Quality assessment orchestration for spans and beats.

Coordinates all quality metric calculations, threshold validation,
and report generation for comprehensive data quality assessment.
"""

import json
import yaml
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime

from lakehouse.logger import get_default_logger
from lakehouse.quality.thresholds import QualityThresholds, ThresholdViolation, RAGStatus
from lakehouse.structure import LakehouseStructure

# Import metric calculators
from lakehouse.quality.metrics import coverage, distribution, integrity, balance, text_quality, embedding
from lakehouse.quality import diagnostics
from lakehouse.quality import reporter


logger = get_default_logger()


@dataclass
class MetricsBundle:
    """
    Container for all quality metrics across categories.
    
    Stores results from all metric calculators for use in reporting
    and threshold validation.
    """
    # Category A: Coverage & Count
    coverage_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Category B: Length & Distribution
    distribution_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Category C: Ordering & Integrity
    integrity_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Category D: Speaker & Series Balance
    balance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Category E: Text Quality
    text_quality_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Category F: Embedding Sanity
    embedding_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Category G: Diagnostics (outliers, samples)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    lakehouse_version: str = "v1"
    embeddings_available: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "coverage": self.coverage_metrics,
            "distribution": self.distribution_metrics,
            "integrity": self.integrity_metrics,
            "balance": self.balance_metrics,
            "text_quality": self.text_quality_metrics,
            "embedding": self.embedding_metrics,
            "diagnostics": self.diagnostics,
            "metadata": {
                "timestamp": self.timestamp.isoformat(),
                "lakehouse_version": self.lakehouse_version,
                "embeddings_available": self.embeddings_available,
            }
        }


@dataclass
class AssessmentResult:
    """
    Complete quality assessment result.
    
    Contains metrics bundle, threshold violations, RAG status,
    and summary information for reporting.
    """
    metrics: MetricsBundle
    thresholds: QualityThresholds
    violations: List[ThresholdViolation] = field(default_factory=list)
    rag_status: RAGStatus = RAGStatus.GREEN
    
    # Summary counts
    total_episodes: int = 0
    total_spans: int = 0
    total_beats: int = 0
    
    # Processing metadata
    assessment_duration_seconds: float = 0.0
    output_paths: Dict[str, Path] = field(default_factory=dict)
    
    def get_critical_violations(self) -> List[ThresholdViolation]:
        """Get all error-level violations."""
        return [v for v in self.violations if v.severity == "error"]
    
    def get_warnings(self) -> List[ThresholdViolation]:
        """Get all warning-level violations."""
        return [v for v in self.violations if v.severity == "warning"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metrics": self.metrics.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "violations": [
                {
                    "threshold": v.threshold_name,
                    "expected": v.expected,
                    "actual": v.actual,
                    "severity": v.severity,
                    "message": v.message,
                }
                for v in self.violations
            ],
            "rag_status": self.rag_status.value,
            "summary": {
                "total_episodes": self.total_episodes,
                "total_spans": self.total_spans,
                "total_beats": self.total_beats,
                "critical_violations": len(self.get_critical_violations()),
                "warnings": len(self.get_warnings()),
                "assessment_duration_seconds": self.assessment_duration_seconds,
            }
        }


class QualityAssessor:
    """
    Main quality assessment orchestrator.
    
    Coordinates data loading, metric calculation, threshold validation,
    and report generation for comprehensive quality assessment.
    """
    
    def __init__(
        self,
        lakehouse_path: Union[str, Path],
        version: str = "v1",
        config_path: Optional[Union[str, Path]] = None,
        threshold_overrides: Optional[Dict[str, Any]] = None,
        thresholds: Optional[QualityThresholds] = None,
    ):
        """
        Initialize quality assessor (Tasks 5.6.2, 5.6.3).
        
        Args:
            lakehouse_path: Path to lakehouse base directory
            version: Lakehouse version to assess (default: v1)
            config_path: Path to quality thresholds YAML config (optional)
            threshold_overrides: Dictionary of threshold overrides from CLI (optional)
            thresholds: Pre-configured QualityThresholds instance (optional)
        """
        self.lakehouse_path = Path(lakehouse_path)
        self.version = version
        self.structure = LakehouseStructure(lakehouse_path)
        
        # Load thresholds (Task 5.6.3)
        if thresholds is not None:
            self.thresholds = thresholds
        else:
            self.thresholds = self._load_thresholds(config_path, threshold_overrides)
        
        # Data containers (loaded on demand)
        self._episodes: Optional[pd.DataFrame] = None
        self._spans: Optional[pd.DataFrame] = None
        self._beats: Optional[pd.DataFrame] = None
        self._span_embeddings: Optional[Tuple[pd.DataFrame, np.ndarray]] = None
        self._beat_embeddings: Optional[Tuple[pd.DataFrame, np.ndarray]] = None
        
        logger.info(f"Initialized QualityAssessor for {lakehouse_path} version {version}")
    
    def _load_thresholds(
        self,
        config_path: Optional[Union[str, Path]],
        overrides: Optional[Dict[str, Any]],
    ) -> QualityThresholds:
        """
        Load thresholds from config with CLI overrides (Task 5.6.3).
        
        Args:
            config_path: Path to YAML config file
            overrides: Dictionary of override values
        
        Returns:
            QualityThresholds instance
        """
        # Start with defaults
        thresholds = QualityThresholds()
        
        # Load from config file if provided
        if config_path is not None:
            config_path = Path(config_path)
            if config_path.exists():
                logger.info(f"Loading thresholds from {config_path}")
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                if config:
                    thresholds = QualityThresholds.from_dict(config)
            else:
                logger.warning(f"Config file not found: {config_path}, using defaults")
        else:
            # Try default config location
            default_config = Path("config") / "quality_thresholds.yaml"
            if default_config.exists():
                logger.info(f"Loading thresholds from {default_config}")
                with open(default_config, 'r') as f:
                    config = yaml.safe_load(f)
                if config:
                    thresholds = QualityThresholds.from_dict(config)
        
        # Apply CLI overrides
        if overrides:
            thresholds = thresholds.apply_overrides(**overrides)
            logger.info(f"Applied {len(overrides)} threshold overrides from CLI")
        
        return thresholds
    
    def load_episodes(self) -> pd.DataFrame:
        """
        Load episode metadata from catalogs.
        
        Returns:
            DataFrame with episode metadata (episode_id, duration, etc.)
        
        Raises:
            FileNotFoundError: If episode catalog not found
        """
        if self._episodes is not None:
            return self._episodes
        
        logger.info("Loading episode metadata...")
        
        # Look for most recent episode catalog
        catalogs_path = self.structure.get_catalogs_path()
        catalog_files = sorted(catalogs_path.glob("episodes_*.parquet"), reverse=True)
        
        if not catalog_files:
            raise FileNotFoundError(
                f"No episode catalog found in {catalogs_path}. "
                "Run 'lakehouse catalog' first."
            )
        
        catalog_file = catalog_files[0]
        logger.info(f"Loading episode catalog from {catalog_file}")
        
        self._episodes = pd.read_parquet(catalog_file)
        logger.info(f"Loaded {len(self._episodes)} episodes")
        
        return self._episodes
    
    def load_spans(self) -> pd.DataFrame:
        """
        Load spans data.
        
        Returns:
            DataFrame with span data
        
        Raises:
            FileNotFoundError: If spans data not found
        """
        if self._spans is not None:
            return self._spans
        
        logger.info("Loading spans data...")
        
        spans_path = self.structure.get_spans_path(self.version)
        spans_file = spans_path / "spans.parquet"
        
        if not spans_file.exists():
            raise FileNotFoundError(
                f"Spans data not found at {spans_file}. "
                "Run 'lakehouse materialize --artifact span' first."
            )
        
        self._spans = pd.read_parquet(spans_file)
        logger.info(f"Loaded {len(self._spans)} spans")
        
        return self._spans
    
    def load_beats(self) -> pd.DataFrame:
        """
        Load beats data.
        
        Returns:
            DataFrame with beat data
        
        Raises:
            FileNotFoundError: If beats data not found
        """
        if self._beats is not None:
            return self._beats
        
        logger.info("Loading beats data...")
        
        beats_path = self.structure.get_beats_path(self.version)
        beats_file = beats_path / "beats.parquet"
        
        if not beats_file.exists():
            raise FileNotFoundError(
                f"Beats data not found at {beats_file}. "
                "Run 'lakehouse materialize --artifact beat' first."
            )
        
        self._beats = pd.read_parquet(beats_file)
        logger.info(f"Loaded {len(self._beats)} beats")
        
        return self._beats
    
    def load_embeddings(
        self, 
        level: str,
        segments_df: Optional[pd.DataFrame] = None
    ) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
        """
        Load embeddings for specified level (gracefully handles missing) (Task 5.6.2, 5.6.7).
        
        Args:
            level: "span" or "beat"
            segments_df: Segments dataframe to align embeddings with
        
        Returns:
            Tuple of (merged_df, embedding_matrix) or None if not available
        """
        if level == "span" and self._span_embeddings is not None:
            return self._span_embeddings
        elif level == "beat" and self._beat_embeddings is not None:
            return self._beat_embeddings
        
        logger.info(f"Loading {level} embeddings...")
        
        embeddings_path = self.structure.get_embeddings_path(self.version)
        embeddings_file = embeddings_path / f"{level}_embeddings.parquet"
        
        # Use the embedding.load_embeddings function which has graceful handling
        result = embedding.load_embeddings(embeddings_file, segments_df)
        
        if result is not None:
            if level == "span":
                self._span_embeddings = result
            else:
                self._beat_embeddings = result
        
        return result
    
    def run_assessment(
        self,
        assess_spans: bool = True,
        assess_beats: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        use_timestamp: bool = True,
        force_near_duplicate_check: bool = False,
    ) -> AssessmentResult:
        """
        Run comprehensive quality assessment (Task 5.6.1).
        
        Coordinates all metric calculations, threshold validation,
        report generation, and file exports.
        
        Args:
            assess_spans: Whether to assess spans (default: True)
            assess_beats: Whether to assess beats (default: True)
            output_dir: Output directory for reports/exports (optional)
            use_timestamp: Whether to create timestamped subdirectory (default: True)
            force_near_duplicate_check: Force near-duplicate detection for large datasets (default: False)
        
        Returns:
            AssessmentResult with metrics, violations, and RAG status
        """
        start_time = time.time()
        logger.info(
            f"Starting quality assessment "
            f"(spans={'yes' if assess_spans else 'no'}, beats={'yes' if assess_beats else 'no'})"
        )
        
        # Set random seed for reproducibility (Task 5.6.6, FR-51)
        np.random.seed(self.thresholds.random_seed)
        logger.info(f"Set random seed to {self.thresholds.random_seed} for reproducibility")
        
        # Initialize results
        metrics = MetricsBundle(
            lakehouse_version=self.version,
            timestamp=datetime.now()
        )
        all_violations: List[ThresholdViolation] = []
        
        # Task 5.6.2: Load data from lakehouse (FR-3)
        try:
            episodes = self.load_episodes()
            spans_df = self.load_spans() if assess_spans else None
            beats_df = self.load_beats() if assess_beats else None
            
        except FileNotFoundError as e:
            logger.error(f"Failed to load required data: {e}")
            raise
        
        # Task 5.6.7: Try to load embeddings (FR-50 - graceful handling)
        span_embeddings_result = None
        beat_embeddings_result = None
        span_embeddings_matrix = None
        beat_embeddings_matrix = None
        
        if assess_spans and spans_df is not None:
            span_embeddings_result = self.load_embeddings("span", spans_df)
            if span_embeddings_result:
                _, span_embeddings_matrix = span_embeddings_result
        
        if assess_beats and beats_df is not None:
            beat_embeddings_result = self.load_embeddings("beat", beats_df)
            if beat_embeddings_result:
                _, beat_embeddings_matrix = beat_embeddings_result
        
        metrics.embeddings_available = (
            span_embeddings_matrix is not None or beat_embeddings_matrix is not None
        )
        
        logger.info(
            f"Data loaded: {len(episodes)} episodes, "
            f"{len(spans_df) if spans_df is not None else 0} spans, "
            f"{len(beats_df) if beats_df is not None else 0} beats, "
            f"embeddings={'available' if metrics.embeddings_available else 'not available'}"
        )
        
        # Task 5.6.4: Call all metric calculators
        logger.info("Calculating quality metrics...")
        
        # Category A: Coverage & Count Metrics
        if assess_spans or assess_beats:
            logger.info("Calculating coverage metrics...")
            coverage_metrics = coverage.calculate_episode_coverage(
                episodes, spans_df, beats_df
            )
            metrics.coverage_metrics = coverage_metrics
            
            # Validate coverage thresholds
            coverage_violations = coverage.validate_coverage_thresholds(
                coverage_metrics, self.thresholds
            )
            all_violations.extend(coverage_violations)
        
        # Category B: Distribution Metrics (for spans)
        if assess_spans and spans_df is not None:
            logger.info("Calculating span distribution metrics...")
            span_distribution = distribution.calculate_duration_statistics(
                spans_df, segment_type="spans"
            )
            span_compliance = distribution.calculate_length_compliance(
                spans_df,
                min_duration=self.thresholds.span_length_min,
                max_duration=self.thresholds.span_length_max,
                segment_type="spans"
            )
            
            metrics.distribution_metrics['spans'] = {
                'statistics': span_distribution,
                'compliance': span_compliance,
            }
            
            # Validate distribution thresholds for spans
            dist_violations = distribution.validate_length_thresholds(
                span_compliance,
                self.thresholds.span_length_compliance_min,
                segment_type="spans"
            )
            all_violations.extend(dist_violations)
        
        # Category B: Distribution Metrics (for beats)
        if assess_beats and beats_df is not None:
            logger.info("Calculating beat distribution metrics...")
            beat_distribution = distribution.calculate_duration_statistics(
                beats_df, segment_type="beats"
            )
            beat_compliance = distribution.calculate_length_compliance(
                beats_df,
                min_duration=self.thresholds.beat_length_min,
                max_duration=self.thresholds.beat_length_max,
                segment_type="beats"
            )
            
            metrics.distribution_metrics['beats'] = {
                'statistics': beat_distribution,
                'compliance': beat_compliance,
            }
            
            # Validate distribution thresholds for beats
            dist_violations = distribution.validate_length_thresholds(
                beat_compliance,
                self.thresholds.beat_length_compliance_min,
                segment_type="beats"
            )
            all_violations.extend(dist_violations)
        
        # Category C: Integrity Metrics (combines spans and beats)
        combined_segments = []
        if spans_df is not None:
            combined_segments.append(spans_df)
        if beats_df is not None:
            combined_segments.append(beats_df)
        
        if combined_segments:
            logger.info("Calculating integrity metrics...")
            all_segments = pd.concat(combined_segments, ignore_index=True)
            
            integrity_metrics = integrity.check_timestamp_monotonicity(all_segments)
            violations_data = integrity.detect_integrity_violations(all_segments)
            duplicates_data = integrity.detect_duplicates(
                all_segments,
                fuzzy_threshold=self.thresholds.near_duplicate_threshold,
                force_near_duplicate_check=force_near_duplicate_check
            )
            
            # Combine integrity metrics
            metrics.integrity_metrics = {
                **integrity_metrics,
                **violations_data,
                **duplicates_data,
            }
            
            # Validate integrity thresholds
            integrity_violations = integrity.validate_integrity_thresholds(
                metrics.integrity_metrics, self.thresholds
            )
            all_violations.extend(integrity_violations)
        
        # Category D: Balance Metrics (combines spans and beats)
        if combined_segments:
            logger.info("Calculating balance metrics...")
            balance_metrics = balance.calculate_speaker_distribution(
                all_segments,
                top_n=self.thresholds.top_speakers_count
            )
            metrics.balance_metrics = balance_metrics
        
        # Category E: Text Quality Metrics (combines spans and beats)
        if combined_segments:
            logger.info("Calculating text quality metrics...")
            text_metrics = text_quality.calculate_text_metrics(all_segments)
            lexical_density = text_quality.calculate_lexical_density(all_segments)
            top_terms = text_quality.extract_top_terms(all_segments, top_n=20)
            
            metrics.text_quality_metrics = {
                'statistics': text_metrics,
                'lexical_density': lexical_density,
                'top_terms': top_terms,
            }
        
        # Category F: Embedding Sanity Checks (Task 5.6.7: FR-50 - graceful handling)
        embedding_metrics_dict = {}
        
        # Process span embeddings if available
        if assess_spans and spans_df is not None and span_embeddings_matrix is not None:
            logger.info("Calculating span embedding metrics...")
            embedding_metrics_dict['spans'] = self._calculate_embedding_metrics(
                spans_df, span_embeddings_matrix, "spans"
            )
            
            # Validate embedding thresholds
            emb_violations = self._validate_embedding_metrics(
                embedding_metrics_dict['spans']
            )
            all_violations.extend(emb_violations)
        
        # Process beat embeddings if available
        if assess_beats and beats_df is not None and beat_embeddings_matrix is not None:
            logger.info("Calculating beat embedding metrics...")
            embedding_metrics_dict['beats'] = self._calculate_embedding_metrics(
                beats_df, beat_embeddings_matrix, "beats"
            )
            
            # Validate embedding thresholds
            emb_violations = self._validate_embedding_metrics(
                embedding_metrics_dict['beats']
            )
            all_violations.extend(emb_violations)
        
        metrics.embedding_metrics = embedding_metrics_dict
        
        # Category G: Diagnostics & Outliers
        if combined_segments:
            logger.info("Identifying outliers and generating diagnostic samples...")
            
            # Use span embeddings for diagnostics if available, otherwise beat embeddings
            diag_embeddings = span_embeddings_matrix if span_embeddings_matrix is not None else beat_embeddings_matrix
            diag_segments = spans_df if spans_df is not None else beats_df
            
            # Get neighbor indices if embeddings available
            neighbor_indices = None
            neighbor_similarities = None
            
            if diag_embeddings is not None and diag_segments is not None:
                # Sample and find neighbors for diagnostics
                sampled = embedding.stratified_sample_segments(
                    diag_segments,
                    sample_size=self.thresholds.neighbor_sample_size,
                    random_seed=self.thresholds.random_seed
                )
                query_indices = sampled.index.values
                
                neighbor_indices, neighbor_similarities = embedding.find_top_k_neighbors(
                    query_indices=query_indices,
                    embeddings=diag_embeddings,
                    k=self.thresholds.neighbor_k,
                    exclude_self=True
                )
            
            # Identify outliers
            outliers = diagnostics.identify_outliers(
                all_segments,
                embeddings=diag_embeddings,
                neighbor_indices=neighbor_indices,
                neighbor_similarities=neighbor_similarities,
                outlier_count=self.thresholds.outlier_count
            )
            
            metrics.diagnostics['outliers'] = outliers
            
            # Sample neighbor lists if embeddings available
            if diag_embeddings is not None and neighbor_indices is not None:
                neighbor_samples = diagnostics.sample_neighbor_lists(
                    diag_segments,
                    diag_embeddings,
                    neighbor_indices,
                    neighbor_similarities,
                    sample_size=self.thresholds.neighbor_list_sample_size,
                    random_seed=self.thresholds.random_seed
                )
                metrics.diagnostics['neighbor_samples'] = neighbor_samples
        
        # Task 5.6.5: Determine RAG status
        rag_status = self._determine_rag_status(all_violations)
        
        # Calculate assessment duration
        duration = time.time() - start_time
        
        # Task 5.6.5: Create result
        result = AssessmentResult(
            metrics=metrics,
            thresholds=self.thresholds,
            violations=all_violations,
            rag_status=rag_status,
            total_episodes=len(episodes),
            total_spans=len(spans_df) if spans_df is not None else 0,
            total_beats=len(beats_df) if beats_df is not None else 0,
            assessment_duration_seconds=round(duration, 2),
        )
        
        logger.info(
            f"Assessment complete in {duration:.2f}s. "
            f"RAG Status: {rag_status.value.upper()}, "
            f"Violations: {len(all_violations)} ({len(result.get_critical_violations())} errors, "
            f"{len(result.get_warnings())} warnings)"
        )
        
        # Generate outputs if output_dir specified
        if output_dir is not None:
            output_paths = self._generate_outputs(
                result, episodes, spans_df, beats_df, output_dir, use_timestamp
            )
            result.output_paths = output_paths
        
        return result
    
    def _calculate_embedding_metrics(
        self,
        segments_df: pd.DataFrame,
        embeddings_matrix: np.ndarray,
        segment_type: str,
    ) -> Dict[str, Any]:
        """
        Calculate all embedding sanity check metrics (Task 5.6.4).
        
        Args:
            segments_df: DataFrame with segment data
            embeddings_matrix: Embedding matrix (N x D)
            segment_type: "spans" or "beats"
        
        Returns:
            Dictionary with all embedding metrics
        """
        metrics_dict = {}
        
        # Sample segments for neighbor analysis
        sampled = embedding.stratified_sample_segments(
            segments_df,
            sample_size=self.thresholds.neighbor_sample_size,
            random_seed=self.thresholds.random_seed
        )
        query_indices = sampled.index.values
        
        # Find neighbors for sampled segments
        neighbor_indices, neighbor_similarities = embedding.find_top_k_neighbors(
            query_indices=query_indices,
            embeddings=embeddings_matrix,
            k=self.thresholds.neighbor_k,
            exclude_self=True
        )
        
        # FR-23: Neighbor coherence
        coherence_result = embedding.assess_neighbor_coherence(
            segments_df=segments_df,
            embeddings=embeddings_matrix,
            sample_size=self.thresholds.neighbor_sample_size,
            k=self.thresholds.neighbor_k,
            random_seed=42
        )
        metrics_dict['neighbor_coherence'] = coherence_result
        
        # FR-24, FR-25: Leakage detection
        speaker_leakage = embedding.calculate_speaker_leakage(
            segments_df=segments_df,
            embeddings=embeddings_matrix,
            sample_size=self.thresholds.neighbor_sample_size,
            k=self.thresholds.neighbor_k,
            random_seed=42
        )
        episode_leakage = embedding.calculate_episode_leakage(
            segments_df=segments_df,
            embeddings=embeddings_matrix,
            sample_size=self.thresholds.neighbor_sample_size,
            k=self.thresholds.neighbor_k,
            random_seed=42
        )
        metrics_dict['speaker_leakage'] = speaker_leakage
        metrics_dict['episode_leakage'] = episode_leakage
        
        # FR-26, FR-27: Length bias
        length_bias_corr = embedding.calculate_length_bias_correlation(
            segments_df=segments_df,
            embeddings=embeddings_matrix,
            sample_size=self.thresholds.neighbor_sample_size,
            k=self.thresholds.neighbor_k,
            random_seed=42
        )
        metrics_dict['length_bias'] = length_bias_corr
        
        # FR-28: Lexical vs embedding similarity alignment
        similarity_corr_result = embedding.calculate_similarity_correlation(
            segments_df=segments_df,
            embeddings=embeddings_matrix,
            n_pairs=min(1000, len(segments_df) // 2),
            lexical_method='jaccard',
            random_seed=42
        )
        metrics_dict['similarity_correlation'] = similarity_corr_result
        
        # FR-29, FR-30, FR-31: Cross-series and adjacency bias
        cross_series_result = embedding.calculate_cross_series_neighbors(
            segments_df=segments_df,
            embeddings=embeddings_matrix,
            sample_size=self.thresholds.neighbor_sample_size,
            k=self.thresholds.neighbor_k,
            random_seed=42
        )
        adjacency_result = embedding.calculate_adjacency_bias(
            segments_df=segments_df,
            embeddings=embeddings_matrix,
            sample_size=self.thresholds.neighbor_sample_size,
            k=self.thresholds.neighbor_k,
            adjacency_tolerance_seconds=5.0,
            random_seed=42
        )
        metrics_dict['cross_series'] = cross_series_result
        metrics_dict['adjacency_bias'] = adjacency_result
        
        return metrics_dict
    
    def _validate_embedding_metrics(
        self,
        embedding_metrics: Dict[str, Any]
    ) -> List[ThresholdViolation]:
        """
        Validate embedding metrics against thresholds (Task 5.6.4).
        
        Args:
            embedding_metrics: Dictionary of embedding metrics
        
        Returns:
            List of threshold violations
        """
        violations = []
        
        # Validate leakage thresholds (FR-24, FR-25)
        speaker_leakage_dict = embedding_metrics.get('speaker_leakage', {})
        episode_leakage_dict = embedding_metrics.get('episode_leakage', {})
        
        leakage_violations = embedding.validate_leakage_thresholds(
            speaker_leakage=speaker_leakage_dict,
            episode_leakage=episode_leakage_dict,
            thresholds=self.thresholds
        )
        violations.extend(leakage_violations)
        
        # Validate length bias (FR-26, FR-27)
        length_bias_dict = embedding_metrics.get('length_bias', {})
        length_bias_violations = embedding.validate_length_bias_threshold(
            length_bias_metrics=length_bias_dict,
            thresholds=self.thresholds
        )
        violations.extend(length_bias_violations)
        
        # Validate adjacency bias (FR-30, FR-31)
        adjacency_dict = embedding_metrics.get('adjacency_bias', {})
        adjacency_violations = embedding.validate_adjacency_threshold(
            adjacency_metrics=adjacency_dict,
            thresholds=self.thresholds
        )
        violations.extend(adjacency_violations)
        
        return violations
    
    def _generate_outputs(
        self,
        result: AssessmentResult,
        episodes: pd.DataFrame,
        spans_df: Optional[pd.DataFrame],
        beats_df: Optional[pd.DataFrame],
        output_dir: Union[str, Path],
        use_timestamp: bool,
    ) -> Dict[str, Path]:
        """
        Generate all output files (reports, metrics, diagnostics) (Task 5.6.4).
        
        Args:
            result: Assessment result
            episodes: Episodes dataframe
            spans_df: Spans dataframe (optional)
            beats_df: Beats dataframe (optional)
            output_dir: Base output directory
            use_timestamp: Whether to create timestamped subdirectory
        
        Returns:
            Dictionary mapping output types to file paths
        """
        logger.info("Generating output files...")
        
        # Create output directory structure
        output_paths = reporter.create_output_structure(
            base_output_dir=output_dir,
            use_timestamp=use_timestamp
        )
        
        # Initialize reporter
        quality_reporter = reporter.QualityReporter(result)
        
        # Generate markdown report
        report_path = output_paths['report'] / "quality_assessment.md"
        quality_reporter.generate_markdown_report(report_path)
        logger.info(f"Generated markdown report: {report_path}")
        
        # TODO: Implement export methods for CSV/JSON
        # These export methods need to be implemented in the reporter module
        logger.info("CSV/JSON exports not yet implemented - skipping")
        
        logger.info(f"All outputs written to: {output_paths['root']}")
        
        return output_paths
    
    def _determine_rag_status(self, violations: List[ThresholdViolation]) -> RAGStatus:
        """
        Determine RAG status based on threshold violations.
        
        Args:
            violations: List of threshold violations
        
        Returns:
            RAGStatus (GREEN, AMBER, or RED)
        """
        errors = [v for v in violations if v.severity == "error"]
        warnings = [v for v in violations if v.severity == "warning"]
        
        if len(errors) >= 3:
            # Multiple critical failures
            return RAGStatus.RED
        elif len(errors) > 0:
            # Any critical failures
            return RAGStatus.RED
        elif len(warnings) > 2:
            # Multiple warnings
            return RAGStatus.AMBER
        elif len(warnings) > 0:
            # Some warnings
            return RAGStatus.AMBER
        else:
            # All checks passed
            return RAGStatus.GREEN
    
    @classmethod
    def from_config(
        cls,
        lakehouse_path: Path,
        version: str = "v1",
        config_path: Optional[Path] = None,
        **threshold_overrides,
    ) -> "QualityAssessor":
        """
        Create QualityAssessor from configuration file with optional overrides.
        
        Args:
            lakehouse_path: Path to lakehouse
            version: Lakehouse version
            config_path: Path to quality_thresholds.yaml (uses default if None)
            **threshold_overrides: CLI overrides for specific thresholds
        
        Returns:
            QualityAssessor instance
        """
        # Load thresholds from config
        if config_path is None:
            config_path = Path("config") / "quality_thresholds.yaml"
        
        thresholds = QualityThresholds()  # Start with defaults
        
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if config:
                thresholds = QualityThresholds.from_dict(config)
                logger.info(f"Loaded thresholds from {config_path}")
        else:
            logger.warning(
                f"Config file not found at {config_path}, using defaults"
            )
        
        # Apply CLI overrides
        if threshold_overrides:
            thresholds = thresholds.apply_overrides(**threshold_overrides)
            logger.info(f"Applied {len(threshold_overrides)} threshold overrides")
        
        return cls(lakehouse_path, version, thresholds)

