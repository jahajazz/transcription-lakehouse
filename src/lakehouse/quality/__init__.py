"""
Quality assessment module for transcript data.

Provides comprehensive quality assessment for spans and beats including:
- Coverage and count metrics
- Length distribution analysis
- Ordering and integrity checks
- Speaker and series balance
- Text quality proxies
- Embedding sanity checks
- Outlier detection and diagnostics
"""

from lakehouse.quality.assessor import QualityAssessor
from lakehouse.quality.thresholds import QualityThresholds, RAGStatus

__all__ = [
    "QualityAssessor",
    "QualityThresholds",
    "RAGStatus",
]

