"""
Quality assessment orchestration for spans and beats.

Coordinates all quality metric calculations, threshold validation,
and report generation for comprehensive data quality assessment.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from lakehouse.logger import get_default_logger
from lakehouse.quality.thresholds import QualityThresholds, ThresholdViolation, RAGStatus
from lakehouse.structure import LakehouseStructure


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
        lakehouse_path: Path,
        version: str = "v1",
        thresholds: Optional[QualityThresholds] = None,
    ):
        """
        Initialize quality assessor.
        
        Args:
            lakehouse_path: Path to lakehouse base directory
            version: Lakehouse version to assess (default: v1)
            thresholds: Quality thresholds (uses defaults if None)
        """
        self.lakehouse_path = Path(lakehouse_path)
        self.version = version
        self.thresholds = thresholds or QualityThresholds()
        self.structure = LakehouseStructure(lakehouse_path)
        
        # Data containers (loaded on demand)
        self._episodes: Optional[pd.DataFrame] = None
        self._spans: Optional[pd.DataFrame] = None
        self._beats: Optional[pd.DataFrame] = None
        self._span_embeddings: Optional[pd.DataFrame] = None
        self._beat_embeddings: Optional[pd.DataFrame] = None
        
        logger.info(f"Initialized QualityAssessor for {lakehouse_path} version {version}")
    
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
    
    def load_embeddings(self, level: str) -> Optional[pd.DataFrame]:
        """
        Load embeddings for specified level (gracefully handles missing).
        
        Args:
            level: "span" or "beat"
        
        Returns:
            DataFrame with embeddings or None if not available
        """
        if level == "span" and self._span_embeddings is not None:
            return self._span_embeddings
        elif level == "beat" and self._beat_embeddings is not None:
            return self._beat_embeddings
        
        logger.info(f"Loading {level} embeddings...")
        
        embeddings_path = self.structure.get_embeddings_path(self.version)
        embeddings_file = embeddings_path / f"{level}_embeddings.parquet"
        
        if not embeddings_file.exists():
            logger.warning(
                f"{level.capitalize()} embeddings not found at {embeddings_file}. "
                "Embedding sanity checks will be skipped."
            )
            return None
        
        try:
            embeddings = pd.read_parquet(embeddings_file)
            logger.info(f"Loaded embeddings for {len(embeddings)} {level}s")
            
            if level == "span":
                self._span_embeddings = embeddings
            else:
                self._beat_embeddings = embeddings
            
            return embeddings
        except Exception as e:
            logger.warning(f"Failed to load {level} embeddings: {e}")
            return None
    
    def run_assessment(
        self,
        levels: List[str] = None,
    ) -> AssessmentResult:
        """
        Run comprehensive quality assessment.
        
        Args:
            levels: List of levels to assess ("spans", "beats", or both)
                   Defaults to ["spans", "beats"]
        
        Returns:
            AssessmentResult with metrics, violations, and RAG status
        """
        if levels is None:
            levels = ["spans", "beats"]
        
        start_time = datetime.now()
        logger.info(f"Starting quality assessment for levels: {levels}")
        
        # Set random seed for reproducibility
        np.random.seed(self.thresholds.random_seed)
        
        # Initialize results
        metrics = MetricsBundle(lakehouse_version=self.version)
        violations: List[ThresholdViolation] = []
        
        # Load data
        try:
            episodes = self.load_episodes()
            spans = self.load_spans() if "spans" in levels else None
            beats = self.load_beats() if "beats" in levels else None
            
            # Try to load embeddings (optional)
            span_embeddings = None
            beat_embeddings = None
            if "spans" in levels:
                span_embeddings = self.load_embeddings("span")
            if "beats" in levels:
                beat_embeddings = self.load_embeddings("beat")
            
            metrics.embeddings_available = (
                span_embeddings is not None or beat_embeddings is not None
            )
            
        except FileNotFoundError as e:
            logger.error(f"Failed to load required data: {e}")
            raise
        
        # TODO: Call metric calculators here (will be implemented in subsequent tasks)
        # - Category A: Coverage metrics
        # - Category B: Distribution metrics
        # - Category C: Integrity metrics
        # - Category D: Balance metrics
        # - Category E: Text quality metrics
        # - Category F: Embedding metrics (if embeddings available)
        # - Category G: Diagnostics
        
        # Placeholder for now
        logger.info("Metric calculation will be implemented in subsequent tasks")
        
        # Determine RAG status based on violations
        rag_status = self._determine_rag_status(violations)
        
        # Create result
        duration = (datetime.now() - start_time).total_seconds()
        
        result = AssessmentResult(
            metrics=metrics,
            thresholds=self.thresholds,
            violations=violations,
            rag_status=rag_status,
            total_episodes=len(episodes),
            total_spans=len(spans) if spans is not None else 0,
            total_beats=len(beats) if beats is not None else 0,
            assessment_duration_seconds=duration,
        )
        
        logger.info(
            f"Assessment complete in {duration:.2f}s. "
            f"RAG Status: {rag_status.value.upper()}"
        )
        
        return result
    
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

