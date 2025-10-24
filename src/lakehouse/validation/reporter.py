"""
Validation report generation and formatting.

Provides utilities for generating human-readable and machine-readable validation reports.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from lakehouse.logger import get_default_logger
from lakehouse.validation.checks import ValidationReport, ValidationCheck


logger = get_default_logger()
console = Console(legacy_windows=False)


class ValidationReporter:
    """Generates validation reports in multiple formats."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize validation reporter.
        
        Args:
            output_dir: Directory to save reports (optional)
        """
        self.output_dir = output_dir
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_summary_report(
        self,
        reports: Dict[str, ValidationReport],
        format: str = "console"
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate a summary report across all validation results.
        
        Args:
            reports: Dictionary of validation reports
            format: Output format ("console", "json", "text")
        
        Returns:
            Formatted report string or dictionary
        """
        if format == "json":
            return self._generate_json_summary(reports)
        elif format == "text":
            return self._generate_text_summary(reports)
        else:
            return self._generate_console_summary(reports)
    
    def _generate_console_summary(self, reports: Dict[str, ValidationReport]) -> str:
        """Generate console-formatted summary."""
        if not reports:
            return "No validation reports to display"
        
        # Create main summary table
        table = Table(title="Validation Summary", show_header=True, header_style="bold magenta")
        table.add_column("Artifact", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Total Checks", justify="right", style="magenta")
        table.add_column("Passed", justify="right", style="green")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Errors", justify="right", style="red")
        table.add_column("Warnings", justify="right", style="yellow")
        
        total_checks = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        total_warnings = 0
        
        for artifact_name, report in reports.items():
            passed_checks = len(report.get_passed_checks())
            failed_checks = len(report.get_failed_checks())
            errors = len(report.get_errors())
            warnings = len(report.get_warnings())
            
            # Determine status
            if errors > 0:
                status = "[red]FAIL[/red]"
            elif warnings > 0:
                status = "[yellow]WARN[/yellow]"
            else:
                status = "[green]PASS[/green]"
            
            table.add_row(
                artifact_name,
                status,
                str(len(report.checks)),
                str(passed_checks),
                str(failed_checks),
                str(errors),
                str(warnings),
            )
            
            total_checks += len(report.checks)
            total_passed += passed_checks
            total_failed += failed_checks
            total_errors += errors
            total_warnings += warnings
        
        # Add totals row
        overall_status = "[green]PASS[/green]" if total_errors == 0 else "[red]FAIL[/red]"
        table.add_row(
            "[bold]TOTAL[/bold]",
            overall_status,
            str(total_checks),
            str(total_passed),
            str(total_failed),
            str(total_errors),
            str(total_warnings),
        )
        
        return str(table)
    
    def _generate_text_summary(self, reports: Dict[str, ValidationReport]) -> str:
        """Generate plain text summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("VALIDATION SUMMARY REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Artifacts: {len(reports)}")
        lines.append("")
        
        total_checks = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        total_warnings = 0
        
        for artifact_name, report in reports.items():
            lines.append(f"Artifact: {artifact_name}")
            lines.append(f"  Type: {report.artifact_type}")
            lines.append(f"  Version: {report.version}")
            lines.append(f"  Timestamp: {report.timestamp}")
            
            passed_checks = len(report.get_passed_checks())
            failed_checks = len(report.get_failed_checks())
            errors = len(report.get_errors())
            warnings = len(report.get_warnings())
            
            lines.append(f"  Total Checks: {len(report.checks)}")
            lines.append(f"  Passed: {passed_checks}")
            lines.append(f"  Failed: {failed_checks}")
            lines.append(f"  Errors: {errors}")
            lines.append(f"  Warnings: {warnings}")
            
            if report.statistics:
                lines.append("  Statistics:")
                for key, value in report.statistics.items():
                    lines.append(f"    {key}: {value}")
            
            lines.append("")
            
            total_checks += len(report.checks)
            total_passed += passed_checks
            total_failed += failed_checks
            total_errors += errors
            total_warnings += warnings
        
        lines.append("=" * 60)
        lines.append("OVERALL SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Total Checks: {total_checks}")
        lines.append(f"Passed: {total_passed}")
        lines.append(f"Failed: {total_failed}")
        lines.append(f"Errors: {total_errors}")
        lines.append(f"Warnings: {total_warnings}")
        lines.append(f"Overall Status: {'PASS' if total_errors == 0 else 'FAIL'}")
        
        return "\n".join(lines)
    
    def _generate_json_summary(self, reports: Dict[str, ValidationReport]) -> Dict[str, Any]:
        """Generate JSON summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_artifacts": len(reports),
            "artifacts": {},
            "overall": {
                "total_checks": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "warnings": 0,
                "status": "unknown"
            }
        }
        
        total_checks = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        total_warnings = 0
        
        for artifact_name, report in reports.items():
            passed_checks = len(report.get_passed_checks())
            failed_checks = len(report.get_failed_checks())
            errors = len(report.get_errors())
            warnings = len(report.get_warnings())
            
            artifact_summary = {
                "artifact_type": report.artifact_type,
                "version": report.version,
                "timestamp": report.timestamp.isoformat(),
                "total_checks": len(report.checks),
                "passed": passed_checks,
                "failed": failed_checks,
                "errors": errors,
                "warnings": warnings,
                "status": "pass" if errors == 0 else "fail",
                "statistics": report.statistics,
                "checks": [check.to_dict() for check in report.checks]
            }
            
            summary["artifacts"][artifact_name] = artifact_summary
            
            total_checks += len(report.checks)
            total_passed += passed_checks
            total_failed += failed_checks
            total_errors += errors
            total_warnings += warnings
        
        summary["overall"] = {
            "total_checks": total_checks,
            "passed": total_passed,
            "failed": total_failed,
            "errors": total_errors,
            "warnings": total_warnings,
            "status": "pass" if total_errors == 0 else "fail"
        }
        
        return summary
    
    def generate_detailed_report(
        self,
        report: ValidationReport,
        format: str = "console"
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate detailed report for a single artifact.
        
        Args:
            report: Validation report to format
            format: Output format ("console", "json", "text")
        
        Returns:
            Formatted report string or dictionary
        """
        if format == "json":
            return self._generate_json_detailed(report)
        elif format == "text":
            return self._generate_text_detailed(report)
        else:
            return self._generate_console_detailed(report)
    
    def _generate_console_detailed(self, report: ValidationReport) -> str:
        """Generate console-formatted detailed report."""
        # Create header panel
        header_text = Text()
        header_text.append(f"Validation Report: {report.artifact_type} (v{report.version})", style="bold blue")
        header_text.append(f"\nTimestamp: {report.timestamp}")
        header_text.append(f"\nTotal Checks: {len(report.checks)}")
        
        header_panel = Panel(header_text, title="Report Header", border_style="blue")
        
        # Create statistics table
        stats_table = Table(title="Statistics", show_header=True, header_style="bold magenta")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right", style="magenta")
        
        for key, value in report.statistics.items():
            stats_table.add_row(key, str(value))
        
        # Create checks table
        checks_table = Table(title="Validation Checks", show_header=True, header_style="bold magenta")
        checks_table.add_column("Check Name", style="cyan", no_wrap=True)
        checks_table.add_column("Status", justify="center")
        checks_table.add_column("Severity", justify="center")
        checks_table.add_column("Message", style="white")
        
        for check in report.checks:
            status = "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]"
            severity_color = {
                "error": "red",
                "warning": "yellow",
                "info": "blue"
            }.get(check.severity, "white")
            
            checks_table.add_row(
                check.check_name,
                status,
                f"[{severity_color}]{check.severity.upper()}[/{severity_color}]",
                check.message
            )
        
        return f"{header_panel}\n\n{stats_table}\n\n{checks_table}"
    
    def _generate_text_detailed(self, report: ValidationReport) -> str:
        """Generate plain text detailed report."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"VALIDATION REPORT: {report.artifact_type.upper()} (v{report.version})")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {report.timestamp}")
        lines.append(f"Total Checks: {len(report.checks)}")
        lines.append("")
        
        # Statistics
        if report.statistics:
            lines.append("STATISTICS:")
            lines.append("-" * 40)
            for key, value in report.statistics.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        # Checks
        lines.append("VALIDATION CHECKS:")
        lines.append("-" * 40)
        
        for check in report.checks:
            status = "PASS" if check.passed else "FAIL"
            lines.append(f"  {check.check_name}: {status} ({check.severity.upper()})")
            lines.append(f"    Message: {check.message}")
            if check.details:
                lines.append(f"    Details: {check.details}")
            lines.append("")
        
        # Summary
        passed = len(report.get_passed_checks())
        failed = len(report.get_failed_checks())
        errors = len(report.get_errors())
        warnings = len(report.get_warnings())
        
        lines.append("SUMMARY:")
        lines.append("-" * 40)
        lines.append(f"  Passed: {passed}")
        lines.append(f"  Failed: {failed}")
        lines.append(f"  Errors: {errors}")
        lines.append(f"  Warnings: {warnings}")
        lines.append(f"  Overall Status: {'PASS' if errors == 0 else 'FAIL'}")
        
        return "\n".join(lines)
    
    def _generate_json_detailed(self, report: ValidationReport) -> Dict[str, Any]:
        """Generate JSON detailed report."""
        return {
            "artifact_type": report.artifact_type,
            "version": report.version,
            "timestamp": report.timestamp.isoformat(),
            "statistics": report.statistics,
            "checks": [check.to_dict() for check in report.checks],
            "summary": {
                "total_checks": len(report.checks),
                "passed": len(report.get_passed_checks()),
                "failed": len(report.get_failed_checks()),
                "errors": len(report.get_errors()),
                "warnings": len(report.get_warnings()),
                "status": "pass" if len(report.get_errors()) == 0 else "fail"
            }
        }
    
    def save_report(
        self,
        reports: Dict[str, ValidationReport],
        filename: Optional[str] = None,
        format: str = "json"
    ) -> Path:
        """
        Save validation reports to file.
        
        Args:
            reports: Dictionary of validation reports
            filename: Custom filename (optional)
            format: Output format ("json", "text")
        
        Returns:
            Path to saved file
        """
        if not self.output_dir:
            raise ValueError("Output directory not specified")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_report_{timestamp}.{format}"
        
        file_path = self.output_dir / filename
        
        if format == "json":
            content = self.generate_summary_report(reports, format="json")
            with open(file_path, 'w') as f:
                json.dump(content, f, indent=2)
        else:
            content = self.generate_summary_report(reports, format="text")
            with open(file_path, 'w') as f:
                f.write(content)
        
        logger.info(f"Validation report saved to {file_path}")
        return file_path
    
    def display_report(self, reports: Dict[str, ValidationReport]):
        """Display validation reports in console."""
        console.print("\n[bold blue]Validation Results[/bold blue]\n")
        
        # Display summary
        summary = self.generate_summary_report(reports, format="console")
        console.print(summary)
        
        # Display detailed results for failed artifacts
        failed_artifacts = [
            (name, report) for name, report in reports.items()
            if len(report.get_errors()) > 0
        ]
        
        if failed_artifacts:
            console.print("\n[bold red]Failed Artifacts Details:[/bold red]\n")
            
            for artifact_name, report in failed_artifacts:
                console.print(f"[bold yellow]{artifact_name}:[/bold yellow]")
                
                # Show failed checks
                failed_checks = report.get_failed_checks()
                for check in failed_checks:
                    severity_color = "red" if check.severity == "error" else "yellow"
                    console.print(f"  [{severity_color}]✗[/{severity_color}] {check.check_name}: {check.message}")
                
                console.print("")
        
        # Display warnings
        warning_artifacts = [
            (name, report) for name, report in reports.items()
            if len(report.get_warnings()) > 0 and len(report.get_errors()) == 0
        ]
        
        if warning_artifacts:
            console.print("[bold yellow]Warnings:[/bold yellow]\n")
            
            for artifact_name, report in warning_artifacts:
                console.print(f"[yellow]{artifact_name}:[/yellow]")
                
                warning_checks = report.get_warnings()
                for check in warning_checks:
                    console.print(f"  [yellow]⚠[/yellow] {check.check_name}: {check.message}")
                
                console.print("")
        
        # Overall status
        total_errors = sum(len(report.get_errors()) for report in reports.values())
        if total_errors == 0:
            console.print("\n[bold green]✓ All validations passed![/bold green]")
        else:
            console.print(f"\n[bold red]✗ {total_errors} validation errors found[/bold red]")
