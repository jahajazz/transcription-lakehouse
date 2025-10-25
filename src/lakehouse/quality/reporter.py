"""
Quality assessment report generation.

Generates comprehensive markdown reports with metrics, visualizations,
and recommendations per PRD requirements FR-35 through FR-40.

Also handles metrics export to JSON and CSV formats per FR-4, FR-5, FR-6.
"""

import json
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from datetime import datetime

from lakehouse.logger import get_default_logger
from lakehouse.quality.thresholds import RAGStatus, ThresholdViolation


logger = get_default_logger()


class QualityReporter:
    """
    Generates markdown reports for quality assessment results.
    
    Implements FR-35 through FR-40 with executive summary, detailed metrics,
    ASCII visualizations, and actionable recommendations.
    """
    
    def __init__(self, assessment_result):
        """
        Initialize reporter with assessment results.
        
        Args:
            assessment_result: AssessmentResult object from quality assessment
        """
        self.result = assessment_result
        self.metrics = assessment_result.metrics
        self.thresholds = assessment_result.thresholds
        self.violations = assessment_result.violations
        self.rag_status = assessment_result.rag_status
    
    def generate_markdown_report(self, output_path: Path) -> None:
        """
        Generate complete markdown report (FR-35, subtask 5.1.1).
        
        Creates a comprehensive markdown file with all sections:
        - Executive summary with RAG status
        - Configuration details
        - Detailed metrics for all categories
        - ASCII visualizations
        - Findings and recommendations
        - Go/No-Go recommendation
        
        Args:
            output_path: Path to output markdown file
        
        Example:
            >>> reporter = QualityReporter(assessment_result)
            >>> reporter.generate_markdown_report(Path("output/report/quality_report.md"))
        """
        logger.info(f"Generating quality assessment report: {output_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sections = []
        
        # Generate all report sections
        sections.append(self._generate_header())
        sections.append(self.generate_executive_summary())
        sections.append(self._generate_configuration_section())
        sections.append(self._generate_coverage_section())
        sections.append(self._generate_distribution_section())
        sections.append(self._generate_integrity_section())
        sections.append(self._generate_balance_section())
        sections.append(self._generate_text_quality_section())
        sections.append(self._generate_embedding_section())
        sections.append(self._generate_outliers_section())
        sections.append(self.generate_findings_and_remediation())
        sections.append(self.generate_go_nogo_recommendation())
        sections.append(self._generate_footer())
        
        # Write to file
        report_content = "\n\n".join(sections)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Quality report written to {output_path}")
    
    def generate_executive_summary(self) -> str:
        """
        Generate executive summary section (FR-36, subtask 5.1.2).
        
        Includes:
        - RAG status with visual indicator
        - Assessment timestamp
        - Dataset counts (episodes, segments)
        - Pass/fail summary
        - Critical issues highlighted
        
        Returns:
            Markdown string for executive summary
        """
        rag_emoji = {
            RAGStatus.GREEN: "🟢",
            RAGStatus.AMBER: "🟠",
            RAGStatus.RED: "🔴",
        }
        
        emoji = rag_emoji.get(self.rag_status, "⚪")
        
        # Count violations by severity
        errors = [v for v in self.violations if v.severity == "error"]
        warnings = [v for v in self.violations if v.severity == "warning"]
        
        summary = f"""## Executive Summary

**Overall Status:** {emoji} **{self.rag_status.value.upper()}**

**Assessment Date:** {self.metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

**Dataset Overview:**
- Episodes: {self.result.total_episodes:,}
- Spans: {self.result.total_spans:,}
- Beats: {self.result.total_beats:,}
- Embeddings Available: {'Yes' if self.metrics.embeddings_available else 'No'}

**Quality Check Results:**
- ✅ Passed: {len(self.thresholds.to_dict()) * 3 - len(self.violations)} checks
- ⚠️ Warnings: {len(warnings)} issues
- ❌ Errors: {len(errors)} critical failures

**Processing Time:** {self.result.assessment_duration_seconds:.2f} seconds
"""
        
        if errors:
            summary += "\n**Critical Issues:**\n"
            for error in errors[:5]:  # Show top 5
                summary += f"- {error.threshold_name}: {error.message}\n"
            if len(errors) > 5:
                summary += f"- ... and {len(errors) - 5} more\n"
        
        return summary
    
    def determine_rag_status(
        self,
        violations: List[ThresholdViolation]
    ) -> RAGStatus:
        """
        Determine RAG status from violations (FR-36, subtask 5.1.3).
        
        Logic:
        - GREEN: No violations (all checks pass)
        - AMBER: 1-2 non-critical violations (warnings only)
        - RED: Multiple violations or any critical (error) violations
        
        Args:
            violations: List of threshold violations
        
        Returns:
            RAGStatus enum value
        
        Example:
            >>> violations = [ThresholdViolation(..., severity='warning')]
            >>> status = reporter.determine_rag_status(violations)
            >>> status
            RAGStatus.AMBER
        """
        if not violations:
            return RAGStatus.GREEN
        
        # Count by severity
        errors = [v for v in violations if v.severity == "error"]
        warnings = [v for v in violations if v.severity == "warning"]
        
        # Any error = RED
        if errors:
            return RAGStatus.RED
        
        # Multiple warnings = RED
        if len(warnings) > 2:
            return RAGStatus.RED
        
        # 1-2 warnings = AMBER
        if warnings:
            return RAGStatus.AMBER
        
        return RAGStatus.GREEN
    
    def generate_ascii_histogram(
        self,
        values: List[float],
        bins: int = 20,
        max_width: int = 60,
        title: str = "Distribution"
    ) -> str:
        """
        Generate ASCII histogram visualization (FR-37, subtask 5.1.4).
        
        Creates a text-based histogram using block characters for
        displaying duration distributions in the markdown report.
        
        Args:
            values: List of numeric values to plot
            bins: Number of histogram bins (default: 20)
            max_width: Maximum width of bars in characters (default: 60)
            title: Histogram title
        
        Returns:
            Markdown-formatted ASCII histogram string
        
        Example:
            >>> durations = [30.5, 45.2, 50.1, 55.8, ...]
            >>> histogram = reporter.generate_ascii_histogram(durations, bins=10, title="Span Durations")
        """
        if not values or len(values) == 0:
            return f"**{title}**\n\n(No data available)\n"
        
        values = np.array(values)
        
        # Calculate histogram
        counts, bin_edges = np.histogram(values, bins=bins)
        max_count = counts.max() if len(counts) > 0 and counts.max() > 0 else 1
        
        # Build ASCII visualization
        histogram = f"**{title}**\n\n```\n"
        
        for i, count in enumerate(counts):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            
            # Calculate bar width
            bar_width = int((count / max_count) * max_width) if max_count > 0 else 0
            bar = "█" * bar_width
            
            # Format label
            if i == 0:
                label = f"{bin_start:6.1f}s - {bin_end:6.1f}s"
            else:
                label = f"{bin_start:6.1f}s - {bin_end:6.1f}s"
            
            histogram += f"{label} | {bar} {count:,}\n"
        
        histogram += "```\n"
        
        # Add statistics
        histogram += f"\n**Statistics:**\n"
        histogram += f"- Count: {len(values):,}\n"
        histogram += f"- Mean: {values.mean():.2f}s\n"
        histogram += f"- Median: {np.median(values):.2f}s\n"
        histogram += f"- Std Dev: {values.std():.2f}s\n"
        histogram += f"- Min: {values.min():.2f}s\n"
        histogram += f"- Max: {values.max():.2f}s\n"
        
        return histogram
    
    def generate_findings_and_remediation(self) -> str:
        """
        Generate findings and recommendations section (FR-38, subtask 5.1.6).
        
        Provides specific, actionable recommendations for each failed threshold:
        - What the issue is
        - Why it matters
        - How to remediate
        
        Returns:
            Markdown string with findings and recommendations
        """
        if not self.violations:
            return """## Findings and Recommendations

✅ **All quality checks passed!** No issues detected.

The dataset meets all quality thresholds. No remediation actions required.
"""
        
        findings = "## Findings and Recommendations\n\n"
        
        # Group violations by category
        coverage_violations = [v for v in self.violations if 'coverage' in v.threshold_name.lower() or 'gap' in v.threshold_name.lower() or 'overlap' in v.threshold_name.lower()]
        length_violations = [v for v in self.violations if 'length' in v.threshold_name.lower() or 'duration' in v.threshold_name.lower()]
        integrity_violations = [v for v in self.violations if 'integrity' in v.threshold_name.lower() or 'duplicate' in v.threshold_name.lower() or 'timestamp' in v.threshold_name.lower()]
        leakage_violations = [v for v in self.violations if 'leakage' in v.threshold_name.lower() or 'speaker' in v.threshold_name.lower() or 'episode' in v.threshold_name.lower()]
        bias_violations = [v for v in self.violations if 'bias' in v.threshold_name.lower() or 'adjacency' in v.threshold_name.lower()]
        
        # Generate recommendations for each category
        if coverage_violations:
            findings += self._generate_coverage_recommendations(coverage_violations)
        
        if length_violations:
            findings += self._generate_length_recommendations(length_violations)
        
        if integrity_violations:
            findings += self._generate_integrity_recommendations(integrity_violations)
        
        if leakage_violations:
            findings += self._generate_leakage_recommendations(leakage_violations)
        
        if bias_violations:
            findings += self._generate_bias_recommendations(bias_violations)
        
        return findings
    
    def generate_go_nogo_recommendation(self) -> str:
        """
        Generate go/no-go recommendation section (FR-39, subtask 5.1.7).
        
        Provides clear recommendation on whether the dataset is suitable
        for production use based on overall RAG status.
        
        Returns:
            Markdown string with go/no-go recommendation
        """
        if self.rag_status == RAGStatus.GREEN:
            return """## Go/No-Go Recommendation

### ✅ **GO** - Ready for Production

The dataset has passed all quality checks and meets the required thresholds. 
It is suitable for:
- Production embedding generation
- RAG system deployment
- Semantic search applications
- Model fine-tuning

**Confidence Level:** High

No blocking issues detected. Proceed with confidence.
"""
        
        elif self.rag_status == RAGStatus.AMBER:
            return """## Go/No-Go Recommendation

### ⚠️ **CONDITIONAL GO** - Proceed with Caution

The dataset has minor quality issues that should be addressed but do not block production use.

**Recommendation:**
- Proceed with production deployment
- Monitor the identified warning areas closely
- Plan remediation for the next iteration
- Document known limitations

**Confidence Level:** Moderate

The issues identified are non-critical but should be tracked.
"""
        
        else:  # RED
            return """## Go/No-Go Recommendation

### ❌ **NO-GO** - Not Ready for Production

The dataset has critical quality issues that must be resolved before production use.

**Recommendation:**
- **DO NOT** proceed with production deployment
- Address all critical (error-level) violations
- Re-run quality assessment after fixes
- Review data pipeline for systematic issues

**Confidence Level:** High

Proceeding without fixes could lead to:
- Poor search quality
- Unreliable semantic matching
- System instability
- User experience issues

**Required Actions:** See "Findings and Recommendations" section above for specific remediation steps.
"""
    
    # Internal helper methods for report sections
    
    def _generate_header(self) -> str:
        """Generate report header."""
        return f"""# Quality Assessment Report

**Lakehouse Version:** {self.metrics.lakehouse_version}
**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
"""
    
    def _generate_configuration_section(self) -> str:
        """Generate configuration section showing thresholds used."""
        config = self.thresholds.to_dict()
        
        section = "## Configuration\n\n**Thresholds Used:**\n\n"
        
        for category, thresholds in config.items():
            section += f"### {category.replace('_', ' ').title()}\n\n"
            for key, value in thresholds.items():
                section += f"- `{key}`: {value}\n"
            section += "\n"
        
        return section
    
    def _generate_coverage_section(self) -> str:
        """Generate coverage metrics section (FR-37, subtask 5.1.5)."""
        coverage = self.metrics.coverage_metrics
        
        if not coverage:
            return "## Category A: Coverage & Count Metrics\n\n(No coverage data available)\n"
        
        section = "## Category A: Coverage & Count Metrics\n\n"
        
        global_metrics = coverage.get('global', {})
        
        section += "### Global Coverage\n\n"
        section += f"- Total Episodes: {global_metrics.get('total_episodes', 0):,}\n"
        section += f"- Total Spans: {global_metrics.get('total_spans', 0):,}\n"
        section += f"- Total Beats: {global_metrics.get('total_beats', 0):,}\n"
        section += f"- Span Coverage: {global_metrics.get('global_span_coverage_percent', 0):.2f}%\n"
        section += f"- Beat Coverage: {global_metrics.get('global_beat_coverage_percent', 0):.2f}%\n"
        
        return section
    
    def _generate_distribution_section(self) -> str:
        """Generate distribution metrics section (FR-37, subtask 5.1.5)."""
        distribution = self.metrics.distribution_metrics
        
        if not distribution:
            return "## Category B: Length & Distribution Metrics\n\n(No distribution data available)\n"
        
        section = "## Category B: Length & Distribution Metrics\n\n"
        
        # Span statistics
        span_stats = distribution.get('spans', {}).get('statistics', {})
        if span_stats:
            section += "### Span Duration Statistics\n\n"
            section += f"- Mean: {span_stats.get('mean', 0):.2f}s\n"
            section += f"- Median: {span_stats.get('median', 0):.2f}s\n"
            section += f"- Std Dev: {span_stats.get('std', 0):.2f}s\n"
            section += f"- Min: {span_stats.get('min', 0):.2f}s\n"
            section += f"- Max: {span_stats.get('max', 0):.2f}s\n"
            section += f"- P5: {span_stats.get('p5', 0):.2f}s\n"
            section += f"- P95: {span_stats.get('p95', 0):.2f}s\n"
            
            compliance = distribution.get('spans', {}).get('compliance', {})
            if compliance:
                section += f"\n**Length Compliance:**\n"
                section += f"- Within bounds (20-120s): {compliance.get('within_bounds_percent', 0):.2f}%\n"
                section += f"- Below minimum: {compliance.get('below_min_percent', 0):.2f}%\n"
                section += f"- Above maximum: {compliance.get('above_max_percent', 0):.2f}%\n"
        
        # Beat statistics
        beat_stats = distribution.get('beats', {}).get('statistics', {})
        if beat_stats:
            section += "\n### Beat Duration Statistics\n\n"
            section += f"- Mean: {beat_stats.get('mean', 0):.2f}s\n"
            section += f"- Median: {beat_stats.get('median', 0):.2f}s\n"
            section += f"- Std Dev: {beat_stats.get('std', 0):.2f}s\n"
            section += f"- Min: {beat_stats.get('min', 0):.2f}s\n"
            section += f"- Max: {beat_stats.get('max', 0):.2f}s\n"
            
            compliance = distribution.get('beats', {}).get('compliance', {})
            if compliance:
                section += f"\n**Length Compliance:**\n"
                section += f"- Within bounds (60-180s): {compliance.get('within_bounds_percent', 0):.2f}%\n"
        
        return section
    
    def _generate_integrity_section(self) -> str:
        """Generate integrity metrics section (FR-37, subtask 5.1.5)."""
        integrity = self.metrics.integrity_metrics
        
        if not integrity:
            return "## Category C: Ordering & Integrity Metrics\n\n(No integrity data available)\n"
        
        section = "## Category C: Ordering & Integrity Metrics\n\n"
        
        section += f"- Timestamp Regressions: {integrity.get('timestamp_regressions', 0):,}\n"
        section += f"- Negative Durations: {integrity.get('negative_durations', 0):,}\n"
        section += f"- Exact Duplicates: {integrity.get('exact_duplicate_count', 0):,} ({integrity.get('exact_duplicate_percent', 0):.2f}%)\n"
        section += f"- Near Duplicates: {integrity.get('near_duplicate_count', 0):,} ({integrity.get('near_duplicate_percent', 0):.2f}%)\n"
        
        return section
    
    def _generate_balance_section(self) -> str:
        """Generate balance metrics section (FR-37, subtask 5.1.5)."""
        balance = self.metrics.balance_metrics
        
        if not balance:
            return "## Category D: Speaker & Series Balance\n\n(No balance data available)\n"
        
        section = "## Category D: Speaker & Series Balance\n\n"
        
        speaker_stats = balance.get('speakers', {})
        section += f"- Unique Speakers: {speaker_stats.get('unique_count', 0):,}\n"
        section += f"- Segments per Speaker (avg): {speaker_stats.get('avg_segments_per_speaker', 0):.1f}\n"
        
        top_speakers = speaker_stats.get('top_speakers', [])
        if top_speakers:
            section += f"\n**Top {len(top_speakers)} Speakers:**\n\n"
            for speaker in top_speakers[:5]:
                section += f"- {speaker.get('speaker_id', 'unknown')}: {speaker.get('count', 0):,} segments ({speaker.get('percent', 0):.1f}%)\n"
        
        return section
    
    def _generate_text_quality_section(self) -> str:
        """Generate text quality section (FR-37, subtask 5.1.5)."""
        text_quality = self.metrics.text_quality_metrics
        
        if not text_quality:
            return "## Category E: Text Quality Proxy Metrics\n\n(No text quality data available)\n"
        
        section = "## Category E: Text Quality Proxy Metrics\n\n"
        
        stats = text_quality.get('statistics', {})
        section += f"- Average Token Count: {stats.get('avg_token_count', 0):.1f}\n"
        section += f"- Average Word Count: {stats.get('avg_word_count', 0):.1f}\n"
        section += f"- Average Character Count: {stats.get('avg_char_count', 0):.1f}\n"
        section += f"- Average Lexical Density: {stats.get('avg_lexical_density', 0):.3f}\n"
        
        return section
    
    def _generate_embedding_section(self) -> str:
        """Generate embedding metrics section (FR-37, subtask 5.1.5)."""
        embedding = self.metrics.embedding_metrics
        
        if not embedding or not self.metrics.embeddings_available:
            return "## Category F: Embedding Sanity Checks\n\n(Embeddings not available - checks skipped)\n"
        
        section = "## Category F: Embedding Sanity Checks\n\n"
        
        # Coherence
        coherence = embedding.get('coherence', {})
        if coherence:
            section += f"### Neighbor Coherence\n\n"
            section += f"- Assessment: {coherence.get('assessment', 'unknown')}\n"
            section += f"- Mean Similarity: {coherence.get('mean_neighbor_similarity', 0):.3f}\n"
            section += f"- Mean Coherence Score: {coherence.get('mean_coherence', 0):.3f}\n"
        
        # Leakage
        speaker_leakage = embedding.get('speaker_leakage', {})
        if speaker_leakage:
            section += f"\n### Speaker Leakage\n\n"
            section += f"- Mean Same-Speaker %: {speaker_leakage.get('mean_same_speaker_percent', 0):.1f}%\n"
        
        episode_leakage = embedding.get('episode_leakage', {})
        if episode_leakage:
            section += f"\n### Episode Leakage\n\n"
            section += f"- Mean Same-Episode %: {episode_leakage.get('mean_same_episode_percent', 0):.1f}%\n"
        
        # Length bias
        length_bias = embedding.get('length_bias', {})
        if length_bias:
            section += f"\n### Length Bias\n\n"
            section += f"- Duration-Norm Correlation: {length_bias.get('duration_norm_correlation', 0):.3f}\n"
            section += f"- Duration-Similarity Correlation: {length_bias.get('duration_similarity_correlation', 0):.3f}\n"
        
        return section
    
    def _generate_outliers_section(self) -> str:
        """Generate outliers section (FR-37, subtask 5.1.5)."""
        diagnostics = self.metrics.diagnostics
        
        if not diagnostics:
            return "## Category G: Diagnostics & Outliers\n\n(No diagnostic data available)\n"
        
        section = "## Category G: Diagnostics & Outliers\n\n"
        
        outliers = diagnostics.get('outliers', {})
        
        for category in ['longest', 'shortest', 'most_isolated', 'most_hubby']:
            outlier_list = outliers.get(category, [])
            if outlier_list:
                section += f"### {category.replace('_', ' ').title()}\n\n"
                section += f"Found {len(outlier_list)} outliers. "
                section += f"See `diagnostics/outliers.csv` for details.\n\n"
        
        return section
    
    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f"""---

*Report generated by Transcript Lakehouse Quality Assessment System*

*For questions or issues, refer to the quality assessment documentation.*
"""
    
    # Recommendation helpers
    
    def _generate_coverage_recommendations(self, violations: List[ThresholdViolation]) -> str:
        """Generate recommendations for coverage violations."""
        section = "### Coverage Issues\n\n"
        for v in violations:
            section += f"**Issue:** {v.message}\n\n"
            section += "**Impact:** Incomplete coverage may result in missing context for semantic search.\n\n"
            section += "**Remediation:**\n"
            section += "- Review segmentation logic to ensure all audio is properly covered\n"
            section += "- Check for gaps in the source transcripts\n"
            section += "- Verify episode duration metadata is accurate\n\n"
        return section
    
    def _generate_length_recommendations(self, violations: List[ThresholdViolation]) -> str:
        """Generate recommendations for length violations."""
        section = "### Length Distribution Issues\n\n"
        for v in violations:
            section += f"**Issue:** {v.message}\n\n"
            section += "**Impact:** Segments outside target ranges may affect embedding quality and search relevance.\n\n"
            section += "**Remediation:**\n"
            section += "- Adjust segmentation parameters (min/max duration)\n"
            section += "- Review aggregation logic for beats\n"
            section += "- Consider filtering out extreme outliers\n\n"
        return section
    
    def _generate_integrity_recommendations(self, violations: List[ThresholdViolation]) -> str:
        """Generate recommendations for integrity violations."""
        section = "### Data Integrity Issues\n\n"
        for v in violations:
            section += f"**Issue:** {v.message}\n\n"
            section += "**Impact:** Data integrity issues can cause processing errors and unreliable results.\n\n"
            section += "**Remediation:**\n"
            section += "- Review data pipeline for timestamp handling errors\n"
            section += "- Implement deduplication in ingestion pipeline\n"
            section += "- Validate input data before processing\n\n"
        return section
    
    def _generate_leakage_recommendations(self, violations: List[ThresholdViolation]) -> str:
        """Generate recommendations for leakage violations."""
        section = "### Embedding Leakage Issues\n\n"
        for v in violations:
            section += f"**Issue:** {v.message}\n\n"
            section += "**Impact:** Embeddings may be encoding metadata rather than semantic content.\n\n"
            section += "**Remediation:**\n"
            section += "- Review embedding model for potential biases\n"
            section += "- Consider using a different embedding model\n"
            section += "- Ensure text normalization removes speaker/episode markers\n"
            section += "- Test with cross-validation across episodes/speakers\n\n"
        return section
    
    def _generate_bias_recommendations(self, violations: List[ThresholdViolation]) -> str:
        """Generate recommendations for bias violations."""
        section = "### Embedding Bias Issues\n\n"
        for v in violations:
            section += f"**Issue:** {v.message}\n\n"
            section += "**Impact:** Biased embeddings can produce unreliable search results.\n\n"
            section += "**Remediation:**\n"
            section += "- Investigate embedding model for length/position biases\n"
            section += "- Consider normalization or calibration techniques\n"
            section += "- Test alternative embedding models\n"
            section += "- Increase diversity in training/embedding data\n\n"
        return section


# Metrics Export Functions (FR-4, FR-5, FR-6)


def export_global_metrics_json(
    assessment_result,
    output_path: Union[str, Path],
) -> None:
    """
    Export global metrics to JSON file (FR-4, subtask 5.2.1).
    
    Creates a JSON file with all global-level metrics:
    - Assessment metadata (timestamp, RAG status, duration)
    - Coverage metrics (episode counts, coverage percentages)
    - Distribution statistics (mean, median, std for spans/beats)
    - Integrity counts (regressions, duplicates)
    - Balance metrics (speaker counts, distribution)
    - Text quality (average token counts, lexical density)
    - Embedding metrics (coherence, leakage, bias)
    - Threshold violations summary
    
    Args:
        assessment_result: AssessmentResult object from quality assessment
        output_path: Path to output JSON file
    
    Example:
        >>> export_global_metrics_json(assessment_result, "output/metrics/global.json")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting global metrics to {output_path}...")
    
    metrics = assessment_result.metrics
    
    # Build global metrics dictionary
    global_metrics = {
        "metadata": {
            "assessment_timestamp": metrics.timestamp.isoformat(),
            "rag_status": assessment_result.rag_status.value,
            "lakehouse_version": metrics.lakehouse_version,
            "assessment_duration_seconds": assessment_result.assessment_duration_seconds,
            "embeddings_available": metrics.embeddings_available,
        },
        "dataset_summary": {
            "total_episodes": assessment_result.total_episodes,
            "total_spans": assessment_result.total_spans,
            "total_beats": assessment_result.total_beats,
        },
        "coverage": metrics.coverage_metrics.get('global', {}),
        "distribution": {
            "spans": metrics.distribution_metrics.get('spans', {}),
            "beats": metrics.distribution_metrics.get('beats', {}),
        },
        "integrity": {
            "timestamp_regressions": metrics.integrity_metrics.get('timestamp_regressions', 0),
            "negative_durations": metrics.integrity_metrics.get('negative_durations', 0),
            "exact_duplicates": {
                "count": metrics.integrity_metrics.get('exact_duplicate_count', 0),
                "percent": metrics.integrity_metrics.get('exact_duplicate_percent', 0.0),
            },
            "near_duplicates": {
                "count": metrics.integrity_metrics.get('near_duplicate_count', 0),
                "percent": metrics.integrity_metrics.get('near_duplicate_percent', 0.0),
            },
        },
        "balance": {
            "speakers": metrics.balance_metrics.get('speakers', {}),
        },
        "text_quality": metrics.text_quality_metrics.get('statistics', {}),
        "embedding": metrics.embedding_metrics if metrics.embeddings_available else {},
        "violations": {
            "total_count": len(assessment_result.violations),
            "error_count": len([v for v in assessment_result.violations if v.severity == "error"]),
            "warning_count": len([v for v in assessment_result.violations if v.severity == "warning"]),
            "violations_list": [
                {
                    "threshold_name": v.threshold_name,
                    "expected": str(v.expected),
                    "actual": str(v.actual),
                    "severity": v.severity,
                    "message": v.message,
                }
                for v in assessment_result.violations
            ],
        },
        "thresholds_used": assessment_result.thresholds.to_dict(),
    }
    
    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(global_metrics, f, indent=2, default=str)
    
    logger.info(f"Global metrics exported to {output_path}")


def export_episodes_csv(
    episodes: pd.DataFrame,
    coverage_metrics: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """
    Export per-episode metrics to CSV file (FR-5, subtask 5.2.2).
    
    Creates a CSV file with one row per episode containing:
    - Episode ID
    - Duration
    - Span count and coverage percentage
    - Beat count and coverage percentage
    - Gap and overlap statistics
    
    Args:
        episodes: DataFrame with episode metadata
        coverage_metrics: Coverage metrics from calculate_episode_coverage()
        output_path: Path to output CSV file
    
    Example:
        >>> export_episodes_csv(episodes_df, coverage_metrics, "output/metrics/episodes.csv")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting per-episode metrics to {output_path}...")
    
    # Extract per-episode metrics
    per_episode = coverage_metrics.get('per_episode', [])
    
    if not per_episode:
        logger.warning("No per-episode metrics available for export")
        return
    
    # Convert to DataFrame
    episodes_df = pd.DataFrame(per_episode)
    
    # Ensure consistent column order
    column_order = [
        'episode_id',
        'episode_duration_seconds',
        'span_count',
        'span_total_duration_seconds',
        'span_coverage_percent',
        'beat_count',
        'beat_total_duration_seconds',
        'beat_coverage_percent',
    ]
    
    # Add gap/overlap columns if available
    if 'gap_count' in episodes_df.columns:
        column_order.extend([
            'gap_count',
            'gap_total_duration',
            'gap_percent',
            'overlap_count',
            'overlap_total_duration',
            'overlap_percent',
        ])
    
    # Select available columns
    available_columns = [col for col in column_order if col in episodes_df.columns]
    episodes_df = episodes_df[available_columns]
    
    # Write to CSV
    episodes_df.to_csv(output_path, index=False)
    
    logger.info(f"Exported {len(episodes_df)} episode metrics to {output_path}")


def export_segments_csv(
    segments: pd.DataFrame,
    segment_type: str,
    integrity_metrics: Dict[str, Any],
    distribution_metrics: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """
    Export per-segment metrics to CSV file (FR-6, subtask 5.2.3).
    
    Creates a CSV file with one row per segment containing:
    - Segment ID, episode ID, speaker ID
    - Start time, end time, duration
    - Text excerpt
    - Quality flags:
      - is_duplicate: Boolean for exact duplicates
      - is_near_duplicate: Boolean for near-duplicates
      - is_out_of_bounds: Boolean for length violations
      - is_short: Boolean for below minimum length
      - is_long: Boolean for above maximum length
    
    Args:
        segments: DataFrame with segment data
        segment_type: Type of segments ('spans' or 'beats')
        integrity_metrics: Integrity metrics with duplicate information
        distribution_metrics: Distribution metrics with length compliance
        output_path: Path to output CSV file
    
    Example:
        >>> export_segments_csv(spans_df, 'spans', integrity_metrics, distribution_metrics, 
        ...                     "output/metrics/spans.csv")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting {segment_type} metrics to {output_path}...")
    
    if segments is None or len(segments) == 0:
        logger.warning(f"No {segment_type} data available for export")
        return
    
    # Create export dataframe
    export_df = segments.copy()
    
    # Add quality flags
    # Duplicate flags
    duplicate_indices = set(integrity_metrics.get('exact_duplicate_indices', []))
    near_duplicate_indices = set(integrity_metrics.get('near_duplicate_indices', []))
    
    export_df['is_duplicate'] = export_df.index.isin(duplicate_indices)
    export_df['is_near_duplicate'] = export_df.index.isin(near_duplicate_indices)
    
    # Length compliance flags
    segment_dist = distribution_metrics.get(segment_type, {})
    segment_compliance = segment_dist.get('compliance', {})
    
    if segment_compliance:
        below_min_indices = set(segment_compliance.get('below_min_indices', []))
        above_max_indices = set(segment_compliance.get('above_max_indices', []))
        within_bounds_indices = set(segment_compliance.get('within_bounds_indices', []))
        
        export_df['is_short'] = export_df.index.isin(below_min_indices)
        export_df['is_long'] = export_df.index.isin(above_max_indices)
        export_df['is_out_of_bounds'] = ~export_df.index.isin(within_bounds_indices)
    else:
        export_df['is_short'] = False
        export_df['is_long'] = False
        export_df['is_out_of_bounds'] = False
    
    # Calculate duration if not present
    if 'duration' not in export_df.columns:
        if 'start_time' in export_df.columns and 'end_time' in export_df.columns:
            export_df['duration'] = export_df['end_time'] - export_df['start_time']
    
    # Truncate text for export
    text_col = 'text' if 'text' in export_df.columns else 'normalized_text'
    if text_col in export_df.columns:
        export_df['text_excerpt'] = export_df[text_col].apply(
            lambda t: str(t)[:200] + '...' if isinstance(t, str) and len(str(t)) > 200 else str(t)
        )
    
    # Select columns for export
    base_columns = [
        'segment_id', 'episode_id', 'speaker_id',
        'start_time', 'end_time', 'duration',
    ]
    
    flag_columns = [
        'is_duplicate', 'is_near_duplicate',
        'is_short', 'is_long', 'is_out_of_bounds',
    ]
    
    text_columns = ['text_excerpt'] if 'text_excerpt' in export_df.columns else []
    
    # Build final column list
    export_columns = []
    for col in base_columns:
        if col in export_df.columns:
            export_columns.append(col)
    
    export_columns.extend(flag_columns)
    export_columns.extend(text_columns)
    
    # Export
    export_df[export_columns].to_csv(output_path, index=False)
    
    logger.info(f"Exported {len(export_df)} {segment_type} with quality flags to {output_path}")
