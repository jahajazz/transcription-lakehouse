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
        
        # Validate consistency (Task 4.7)
        self._validate_summary_consistency()
    
    def _validate_summary_consistency(self) -> None:
        """
        Validate that Executive Summary counts match detailed section data (Task 4.7).
        
        Checks:
        - Episode count matches coverage metrics
        - Span/beat counts match distribution metrics
        - Embedding availability flag is consistent
        
        Logs warnings if inconsistencies are detected.
        """
        coverage = self.metrics.coverage_metrics
        distribution = self.metrics.distribution_metrics
        
        if coverage:
            global_metrics = coverage.get('global', {})
            
            # Validate episode count
            coverage_episodes = global_metrics.get('total_episodes', 0)
            if self.result.total_episodes != coverage_episodes:
                logger.warning(
                    f"Inconsistency: Executive Summary shows {self.result.total_episodes} episodes, "
                    f"but coverage metrics show {coverage_episodes}"
                )
            
            # Validate span count
            coverage_spans = global_metrics.get('total_spans', 0)
            if coverage_spans and self.result.total_spans != coverage_spans:
                logger.warning(
                    f"Inconsistency: Executive Summary shows {self.result.total_spans} spans, "
                    f"but coverage metrics show {coverage_spans}"
                )
            
            # Validate beat count
            coverage_beats = global_metrics.get('total_beats', 0)
            if coverage_beats and self.result.total_beats != coverage_beats:
                logger.warning(
                    f"Inconsistency: Executive Summary shows {self.result.total_beats} beats, "
                    f"but coverage metrics show {coverage_beats}"
                )
        
        # Validate span count against distribution metrics
        if distribution:
            span_stats = distribution.get('spans', {}).get('statistics', {})
            if span_stats:
                dist_span_count = span_stats.get('count', 0)
                if dist_span_count and self.result.total_spans != dist_span_count:
                    logger.warning(
                        f"Inconsistency: Executive Summary shows {self.result.total_spans} spans, "
                        f"but distribution metrics show {dist_span_count}"
                    )
            
            beat_stats = distribution.get('beats', {}).get('statistics', {})
            if beat_stats:
                dist_beat_count = beat_stats.get('count', 0)
                if dist_beat_count and self.result.total_beats != dist_beat_count:
                    logger.warning(
                        f"Inconsistency: Executive Summary shows {self.result.total_beats} beats, "
                        f"but distribution metrics show {dist_beat_count}"
                    )
        
        logger.debug("Executive Summary consistency validation complete")
    
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
        sections.append(self._generate_table_validation_map_section())  # Task 3.10
        sections.append(self._generate_coverage_section())
        sections.append(self._generate_distribution_section())
        sections.append(self._generate_integrity_section())
        sections.append(self._generate_balance_section())
        sections.append(self._generate_text_quality_section())
        sections.append(self._generate_embedding_section())
        sections.append(self._generate_outliers_section())
        sections.append(self._generate_changes_in_this_run_section())  # Task 4.8, 4.9
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
    
    def _generate_table_validation_map_section(self) -> str:
        """
        Generate table validation map section (Task 3.10).
        
        Shows which quality checks ran on which tables based on validator routing.
        This helps confirm that text/timestamp checks only ran on base tables
        and vector checks only ran on embedding tables.
        
        Returns:
            Markdown formatted section
        """
        validation_map = self.result.validation_map
        
        if not validation_map or "_summary" not in validation_map:
            return "## Table Validation Map\n\n(No validator routing information available)\n"
        
        section = "## Table Validation Map\n\n"
        section += "_This section shows which quality checks were executed on each table._\n\n"
        
        summary = validation_map.get("_summary", {})
        section += f"**Summary:** Assessed {summary.get('total_tables_assessed', 0)} tables "
        section += f"with validator routing {'enabled' if summary.get('router_loaded', False) else 'disabled'}.\n\n"
        
        # Create table showing checks per table
        tables_assessed = summary.get("tables_assessed", [])
        
        if tables_assessed:
            section += "| Table | Role | Checks | Check Types |\n"
            section += "|-------|------|--------|-------------|\n"
            
            for table_name in tables_assessed:
                table_info = validation_map.get(table_name, {})
                role = table_info.get("role", "unknown")
                check_count = table_info.get("check_count", 0)
                configured_checks = table_info.get("configured_checks", [])
                
                # Abbreviate check list if too long
                if len(configured_checks) > 4:
                    check_list = ", ".join(configured_checks[:4]) + f", ... ({len(configured_checks) - 4} more)"
                else:
                    check_list = ", ".join(configured_checks) if configured_checks else "none"
                
                section += f"| `{table_name}` | {role} | {check_count} | {check_list} |\n"
            
            section += "\n"
        
        # Add notes about routing behavior
        section += "**Routing Behavior:**\n\n"
        section += "- **Base tables** (spans, beats, sections): "
        section += "Text/timestamp checks (coverage, length_buckets, duplicates, etc.)\n"
        section += "- **Embedding tables** (span_embeddings, beat_embeddings): "
        section += "Vector checks (dim_consistency, nn_leakage, adjacency_bias, etc.)\n"
        section += "- This prevents spurious warnings like \"No timestamp columns found\" on embedding tables\n"
        
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
                section += f"- Below minimum: {compliance.get('too_short_percent', 0):.2f}%\n"
                section += f"- Above maximum: {compliance.get('too_long_percent', 0):.2f}%\n"
        
        # Beat statistics (Task 4.5: Show min, median, p95, max)
        beat_stats = distribution.get('beats', {}).get('statistics', {})
        if beat_stats:
            section += "\n### Beat Duration Statistics\n\n"
            section += f"- Mean: {beat_stats.get('mean', 0):.2f}s\n"
            section += f"- Median: {beat_stats.get('median', 0):.2f}s\n"
            section += f"- Std Dev: {beat_stats.get('std', 0):.2f}s\n"
            section += f"- Min: {beat_stats.get('min', 0):.2f}s\n"
            section += f"- Max: {beat_stats.get('max', 0):.2f}s\n"
            section += f"- P5: {beat_stats.get('p5', 0):.2f}s\n"
            section += f"- P95: {beat_stats.get('p95', 0):.2f}s\n"
            
            compliance = distribution.get('beats', {}).get('compliance', {})
            if compliance:
                section += f"\n**Length Compliance:**\n"
                section += f"- Within bounds (60-180s): {compliance.get('within_bounds_percent', 0):.2f}%\n"
                section += f"- Below minimum: {compliance.get('too_short_percent', 0):.2f}%\n"
                section += f"- Above maximum: {compliance.get('too_long_percent', 0):.2f}%\n"
        
        return section
    
    def _generate_integrity_section(self) -> str:
        """Generate integrity metrics section (FR-37, subtask 5.1.5)."""
        integrity = self.metrics.integrity_metrics
        
        if not integrity:
            return "## Category C: Ordering & Integrity Metrics\n\n(No integrity data available)\n"
        
        section = "## Category C: Ordering & Integrity Metrics\n\n"
        
        # CRITICAL FIX: Integrity metrics are now separated by table (spans vs beats)
        # This prevents false positives from hierarchical overlap
        # See CRITICAL_ISSUES_REMEDIATION_PLAN.md for details
        
        span_integrity = integrity.get('spans', {})
        beat_integrity = integrity.get('beats', {})
        
        # Report span integrity
        if span_integrity:
            section += "**Span Integrity:**\n"
            section += f"- Timestamp Regressions: {span_integrity.get('episode_regression_count', 0):,}\n"
            section += f"- Negative Durations: {span_integrity.get('negative_duration_count', 0):,}\n"
            section += f"- Exact Duplicates: {span_integrity.get('exact_duplicate_count', 0):,} ({span_integrity.get('exact_duplicate_percent', 0):.2f}%)\n"
            section += f"- Near Duplicates: {span_integrity.get('near_duplicate_count', 0):,} ({span_integrity.get('near_duplicate_percent', 0):.2f}%)\n"
            section += "\n"
        
        # Report beat integrity
        if beat_integrity:
            section += "**Beat Integrity:**\n"
            section += f"- Timestamp Regressions: {beat_integrity.get('episode_regression_count', 0):,}\n"
            section += f"- Negative Durations: {beat_integrity.get('negative_duration_count', 0):,}\n"
            section += f"- Exact Duplicates: {beat_integrity.get('exact_duplicate_count', 0):,} ({beat_integrity.get('exact_duplicate_percent', 0):.2f}%)\n"
            section += f"- Near Duplicates: {beat_integrity.get('near_duplicate_count', 0):,} ({beat_integrity.get('near_duplicate_percent', 0):.2f}%)\n"
        
        return section
    
    def _generate_balance_section(self) -> str:
        """Generate balance metrics section (FR-37, subtask 5.1.5)."""
        balance = self.metrics.balance_metrics
        
        if not balance:
            return "## Category D: Speaker & Series Balance\n\n(No balance data available)\n"
        
        section = "## Category D: Speaker & Series Balance\n\n"
        
        # FIXED: Access keys directly from balance dict (no 'speakers' wrapper)
        # Calculator returns flat structure, not nested under 'speakers' key
        section += f"- Unique Speakers: {balance.get('total_speakers', 0):,}\n"
        section += f"- Segments per Speaker (avg): {balance.get('avg_segments_per_speaker', 0):.1f}\n"
        
        top_speakers = balance.get('top_speakers', [])
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
        section += f"- Average Token Count: {stats.get('avg_tokens', 0):.1f}\n"
        section += f"- Average Word Count: {stats.get('avg_words', 0):.1f}\n"
        section += f"- Average Character Count: {stats.get('avg_characters', 0):.1f}\n"
        section += f"- Total Segments: {stats.get('total_segments', 0):,}\n"
        
        # Lexical density is in a separate dict
        lexical = text_quality.get('lexical_density', {})
        section += f"\n**Lexical Density (Content vs Stopwords):**\n"
        section += f"- Overall Lexical Density: {lexical.get('lexical_density', 0):.3f}\n"
        section += f"- Average per Segment: {lexical.get('avg_lexical_density', 0):.3f}\n"
        section += f"- Content Words: {lexical.get('content_words', 0):,}\n"
        section += f"- Stopwords: {lexical.get('stopword_count', 0):,}\n"
        
        # Top terms
        top_terms = text_quality.get('top_terms', {})
        top_unigrams = top_terms.get('top_unigrams', [])
        top_bigrams = top_terms.get('top_bigrams', [])
        
        if top_unigrams:
            section += f"\n**Top 10 Terms (unigrams, excluding stopwords):**\n"
            for i, (term, count) in enumerate(top_unigrams[:10], 1):
                section += f"{i}. {term}: {count:,} occurrences\n"
        
        if top_bigrams:
            section += f"\n**Top 10 Bigrams:**\n"
            for i, (bigram, count) in enumerate(top_bigrams[:10], 1):
                section += f"{i}. \"{bigram}\": {count:,} occurrences\n"
        
        return section
    
    def _generate_embedding_section(self) -> str:
        """Generate embedding metrics section (FR-37, subtask 5.1.5)."""
        embedding = self.metrics.embedding_metrics
        
        if not embedding or not self.metrics.embeddings_available:
            return "## Category F: Embedding Sanity Checks\n\n(Embeddings not available - checks skipped)\n"
        
        section = "## Category F: Embedding Sanity Checks\n\n"
        
        # Process spans and beats separately (like integrity metrics)
        for segment_type in ['spans', 'beats']:
            seg_metrics = embedding.get(segment_type, {})
            if not seg_metrics:
                continue
            
            section += f"### {segment_type.capitalize()} Embedding Metrics\n\n"
            
            # Neighbor Coherence
            coherence = seg_metrics.get('neighbor_coherence', {})
            if coherence:
                section += f"**Neighbor Coherence:**\n"
                section += f"- Assessment: {coherence.get('assessment', 'unknown')}\n"
                section += f"- Mean Similarity: {coherence.get('mean_neighbor_similarity', 0):.3f}\n"
                section += f"- Mean Coherence Score: {coherence.get('mean_coherence', 0):.3f}\n\n"
            
            # Speaker Leakage (same-speaker %)
            speaker_leakage = seg_metrics.get('speaker_leakage', {})
            if speaker_leakage:
                same_speaker_pct = speaker_leakage.get('mean_same_speaker_percent', 0)
                section += f"**Speaker Leakage:**\n"
                section += f"- Same-Speaker %: {same_speaker_pct:.1f}%\n"
                section += f"- Assessment: {'✓ Good' if same_speaker_pct < 30 else '✗ High leakage'}\n\n"
            
            # Episode Leakage (same-episode %)
            episode_leakage = seg_metrics.get('episode_leakage', {})
            if episode_leakage:
                same_episode_pct = episode_leakage.get('mean_same_episode_percent', 0)
                section += f"**Episode Leakage:**\n"
                section += f"- Same-Episode %: {same_episode_pct:.1f}%\n"
                section += f"- Assessment: {'✓ Good' if same_episode_pct < 50 else '✗ High leakage'}\n\n"
            
            # Adjacency Bias (adjacency %)
            adjacency = seg_metrics.get('adjacency_bias', {})
            if adjacency:
                adj_pct = adjacency.get('mean_adjacent_percent', 0)
                section += f"**Adjacency Bias:**\n"
                section += f"- Adjacent Neighbors %: {adj_pct:.1f}%\n"
                section += f"- Assessment: {'✓ Good' if adj_pct < 20 else '✗ High bias'}\n\n"
            
            # Length Bias (length-sim correlation)
            length_bias = seg_metrics.get('length_bias', {})
            if length_bias:
                duration_corr = length_bias.get('duration_similarity_correlation', 0)
                section += f"**Length Bias:**\n"
                section += f"- Length-Similarity Correlation: {duration_corr:.3f}\n"
                section += f"- Duration-Norm Correlation: {length_bias.get('duration_norm_correlation', 0):.3f}\n"
                section += f"- Assessment: {'✓ Good' if abs(duration_corr) < 0.3 else '✗ Biased'}\n\n"
            
            # Similarity Correlation
            sim_corr = seg_metrics.get('similarity_correlation', {})
            if sim_corr:
                corr_val = sim_corr.get('correlation', 0)
                section += f"**Lexical-Embedding Alignment:**\n"
                section += f"- Correlation: {corr_val:.3f}\n"
                section += f"- Assessment: {'✓ Good' if corr_val > 0.5 else '⚠ Low alignment'}\n\n"
            
            # Cross-series diversity
            cross_series = seg_metrics.get('cross_series', {})
            if cross_series:
                cross_pct = cross_series.get('mean_cross_series_percent', 0)
                section += f"**Cross-Series Diversity:**\n"
                section += f"- Cross-Series Neighbors %: {cross_pct:.1f}%\n"
                section += f"- Assessment: {'✓ Good diversity' if cross_pct > 20 else '⚠ Low diversity'}\n\n"
        
        # Sample neighbor lists from diagnostics
        neighbor_samples = self.metrics.diagnostics.get('neighbor_samples', [])
        if neighbor_samples:
            section += f"\n### Sample Neighbor Lists (Top {min(5, len(neighbor_samples))} examples)\n\n"
            for i, sample in enumerate(neighbor_samples[:5], 1):
                query_text = sample.get('query_text', '')[:80]
                section += f"**{i}. Query:** \"{query_text}...\"\n"
                section += f"   **Episode:** {sample.get('query_episode', 'unknown')}\n"
                section += f"   **Top 3 Neighbors:**\n"
                for j, neighbor in enumerate(sample.get('neighbors', [])[:3], 1):
                    neighbor_text = neighbor.get('text', '')[:60]
                    similarity = neighbor.get('similarity', 0)
                    episode = neighbor.get('episode_id', 'unknown')
                    section += f"   {j}. [{similarity:.3f}] (ep: {episode}) \"{neighbor_text}...\"\n"
                section += "\n"
        
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
    
    def _generate_changes_in_this_run_section(self) -> str:
        """
        Generate "Changes in This Run" section (Task 4.8, 4.9).
        
        Documents the specific improvements implemented in Fix Pack — Part-1:
        - R1: Speaker mapping & expert flags
        - R2: Semantic sections (topic-based blocks)
        - R3: Validator routing (proper check routing)
        
        Returns:
            Markdown formatted section describing changes
        """
        section = "## Changes in This Run\n\n"
        section += "_This section documents improvements implemented in Fix Pack — Part-1 Foundations._\n\n"
        
        # R1: Speaker mapping & expert flags (Task 4.9)
        section += "### R1: Speaker Metadata Enrichment\n\n"
        section += "**Speaker roles and expert identification are now configured via `config/speaker_roles.yaml`.**\n\n"
        section += "- **Spans** now include:\n"
        section += "  - `speaker_canonical`: Canonical speaker name (consistent across episodes)\n"
        section += "  - `speaker_role`: Role (expert, host, guest, other)\n"
        section += "  - `is_expert`: Boolean flag for expert speakers\n"
        section += "- **Beats** now include:\n"
        section += "  - `speakers_set`: Array of all canonical speakers in the beat\n"
        section += "  - `expert_span_ids`: IDs of spans with expert speakers\n"
        section += "  - `expert_coverage_pct`: Percentage of beat duration with expert speech (0-100)\n\n"
        section += "**Impact:** You can now filter, analyze, and report on expert vs. non-expert content. "
        section += "Expert speakers are configurable without code changes.\n\n"
        
        # R2: Semantic sections (Task 4.9)
        section += "### R2: Semantic Sections (Topic-Based Blocks)\n\n"
        section += "**Sections are now generated as topic-based blocks using beat embeddings and cosine similarity.**\n\n"
        section += "- Previously: 1 section per episode (entire episode as single block)\n"
        section += "- Now: Multiple sections per episode based on semantic boundaries\n"
        section += "- Configuration: `config/aggregation_config.yaml` (section parameters)\n"
        section += "- Algorithm:\n"
        section += "  1. Load beat embeddings (required)\n"
        section += "  2. Detect topic changes using cosine similarity between adjacent beats\n"
        section += "  3. Break sections at strong semantic boundaries (configurable threshold)\n"
        section += "  4. Balance section duration with semantic coherence\n"
        section += "- Each section includes: `section_id`, `episode_id`, `beat_ids`, `title`, `synopsis`\n\n"
        section += "**Impact:** Sections now represent logical topic blocks rather than arbitrary time windows. "
        section += "Useful for content navigation, topic analysis, and structured retrieval.\n\n"
        
        # R3: Validator routing (Task 4.9)
        section += "### R3: Validator Routing System\n\n"
        section += "**Quality checks are now routed to appropriate tables based on their role.**\n\n"
        section += "- **Base tables** (spans, beats, sections): Text/timestamp checks only\n"
        section += "  - Examples: coverage, length_buckets, duplicates, text_quality\n"
        section += "- **Embedding tables** (span_embeddings, beat_embeddings): Vector checks only\n"
        section += "  - Examples: dim_consistency, nn_leakage, adjacency_bias, length_sim_corr\n"
        section += "- Configuration: `config/validator_routing.yaml`\n"
        section += "- See \"Table Validation Map\" section above for which checks ran on which tables\n\n"
        section += "**Impact:** Eliminates spurious warnings like \"No timestamp columns found\" on embedding tables. "
        section += "Reports are cleaner and checks are only run where they make sense.\n\n"
        
        # Additional fixes
        section += "### Additional Quality Improvements\n\n"
        section += "- **Coverage calculation**: Now uses overlap-aware union (no more >100% coverage)\n"
        section += "- **Length compliance buckets**: Now sum to exactly 100%\n"
        section += "- **Duplicate detection**: Minimum text length filter (10 chars) to avoid false positives on short texts\n"
        section += "- **Report consistency**: Statistics now match between Executive Summary and detailed sections\n\n"
        
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
            "spans": {
                "timestamp_regressions": metrics.integrity_metrics.get('spans', {}).get('episode_regression_count', 0),
                "negative_durations": metrics.integrity_metrics.get('spans', {}).get('negative_duration_count', 0),
                "exact_duplicates": {
                    "count": metrics.integrity_metrics.get('spans', {}).get('exact_duplicate_count', 0),
                    "percent": metrics.integrity_metrics.get('spans', {}).get('exact_duplicate_percent', 0.0),
                },
                "near_duplicates": {
                    "count": metrics.integrity_metrics.get('spans', {}).get('near_duplicate_count', 0),
                    "percent": metrics.integrity_metrics.get('spans', {}).get('near_duplicate_percent', 0.0),
                },
            },
            "beats": {
                "timestamp_regressions": metrics.integrity_metrics.get('beats', {}).get('episode_regression_count', 0),
                "negative_durations": metrics.integrity_metrics.get('beats', {}).get('negative_duration_count', 0),
                "exact_duplicates": {
                    "count": metrics.integrity_metrics.get('beats', {}).get('exact_duplicate_count', 0),
                    "percent": metrics.integrity_metrics.get('beats', {}).get('exact_duplicate_percent', 0.0),
                },
                "near_duplicates": {
                    "count": metrics.integrity_metrics.get('beats', {}).get('near_duplicate_count', 0),
                    "percent": metrics.integrity_metrics.get('beats', {}).get('near_duplicate_percent', 0.0),
                },
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


# Output Directory Management (FR-48, FR-49)


def create_output_structure(
    base_output_dir: Union[str, Path],
    use_timestamp: bool = True,
) -> Dict[str, Path]:
    """
    Create output directory structure for quality assessment (FR-48, FR-49, subtask 5.3.1, 5.3.2).
    
    Creates a timestamped directory structure with subdirectories for:
    - metrics/: JSON and CSV metrics exports
    - diagnostics/: CSV files for outliers and neighbor samples
    - report/: Markdown quality report
    
    Args:
        base_output_dir: Base directory for quality assessment outputs
        use_timestamp: Whether to create timestamped subdirectory (default: True)
    
    Returns:
        Dictionary with paths to created directories:
        - 'root': Root output directory
        - 'metrics': Metrics subdirectory
        - 'diagnostics': Diagnostics subdirectory
        - 'report': Report subdirectory
    
    Raises:
        OSError: If directory creation fails (with proper error handling)
    
    Example:
        >>> paths = create_output_structure("output/quality", use_timestamp=True)
        >>> paths['root']
        PosixPath('output/quality/20251025_143022')
        >>> paths['metrics']
        PosixPath('output/quality/20251025_143022/metrics')
    """
    base_output_dir = Path(base_output_dir)
    
    try:
        # Create timestamped directory if requested
        if use_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            root_dir = base_output_dir / timestamp
        else:
            root_dir = base_output_dir
        
        # Create root directory
        root_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {root_dir}")
        
        # Create subdirectories
        subdirs = {
            'root': root_dir,
            'metrics': root_dir / 'metrics',
            'diagnostics': root_dir / 'diagnostics',
            'report': root_dir / 'report',
        }
        
        for name, path in subdirs.items():
            if name != 'root':  # root already created
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Created subdirectory: {path}")
                except OSError as e:
                    logger.error(f"Failed to create {name} directory at {path}: {e}")
                    raise OSError(f"Failed to create {name} directory: {e}") from e
        
        logger.info(
            f"Output structure created: "
            f"metrics={subdirs['metrics']}, "
            f"diagnostics={subdirs['diagnostics']}, "
            f"report={subdirs['report']}"
        )
        
        return subdirs
    
    except PermissionError as e:
        logger.error(f"Permission denied creating output directory {base_output_dir}: {e}")
        raise PermissionError(
            f"Cannot create output directory at {base_output_dir}. "
            f"Check write permissions."
        ) from e
    
    except OSError as e:
        logger.error(f"Failed to create output directory structure: {e}")
        raise OSError(
            f"Failed to create output directory structure at {base_output_dir}: {e}"
        ) from e
    
    except Exception as e:
        logger.error(f"Unexpected error creating output structure: {e}")
        raise RuntimeError(
            f"Unexpected error creating output structure: {e}"
        ) from e
