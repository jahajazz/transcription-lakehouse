"""
Validate command for checking data quality and reporting statistics.

Performs comprehensive validation on lakehouse artifacts including data integrity,
schema compliance, and quality metrics.
"""

import json
import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from lakehouse.cli import cli, common_options
from lakehouse.logger import configure_logging
from lakehouse.validation.checks import validate_lakehouse
from lakehouse.validation.reporter import ValidationReporter
from lakehouse.config import load_config


console = Console(legacy_windows=False)


@cli.command()
@common_options
@click.option(
    '--version',
    default='v1',
    help='Version to validate (default: v1)',
)
@click.option(
    '--artifact-type',
    type=click.Choice(['utterance', 'span', 'beat', 'section', 'embedding', 'all']),
    default='all',
    help='Type of artifact to validate (default: all)',
)
@click.option(
    '--output-format',
    type=click.Choice(['console', 'json', 'text']),
    default='console',
    help='Output format for validation report (default: console)',
)
@click.option(
    '--save-report',
    is_flag=True,
    help='Save validation report to file',
)
@click.option(
    '--output-dir',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help='Directory to save validation reports (default: lakehouse/catalogs/validation_reports)',
)
@click.option(
    '--detailed',
    is_flag=True,
    help='Show detailed validation results',
)
@click.option(
    '--fail-fast',
    is_flag=True,
    help='Stop validation on first error',
)
@click.pass_context
def validate(
    ctx,
    version,
    artifact_type,
    output_format,
    save_report,
    output_dir,
    detailed,
    fail_fast,
    lakehouse_path,
    config_dir,
    log_level
):
    """
    Validate lakehouse data quality and report statistics.
    
    Performs comprehensive validation on lakehouse artifacts including:
    - Data integrity checks (non-empty tables, required fields)
    - Schema compliance validation
    - Timestamp quality (monotonic, non-negative)
    - Text quality (non-empty, length statistics)
    - ID quality (uniqueness, format validation)
    - Referential integrity between artifacts
    - Numeric data quality (NaN, infinite values)
    
    Examples:
    
        # Validate all artifacts
        lakehouse validate --all
        
        # Validate only utterances
        lakehouse validate --artifact-type utterance
        
        # Save detailed report to file
        lakehouse validate --save-report --output-format json
        
        # Show detailed results
        lakehouse validate --detailed
    """
    # Configure logging
    configure_logging(level=log_level, console_output=True)
    
    # Display header
    console.print("\n[bold blue]Transcript Lakehouse - Validate[/bold blue]\n")
    
    # Check if lakehouse exists
    if not lakehouse_path.exists():
        console.print(f"[red]Error: Lakehouse not found at {lakehouse_path}[/red]")
        console.print("[yellow]Run 'lakehouse ingest' first to create the lakehouse[/yellow]")
        raise click.Abort()
    
    # Load validation configuration
    try:
        config = load_config(config_dir, "validation")
        console.print(f"[cyan]Loaded validation config from {config_dir}[/cyan]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load validation config: {e}[/yellow]")
        console.print("[yellow]Using default validation settings[/yellow]")
        config = None
    
    # Display validation parameters
    console.print(f"[cyan]Lakehouse path:[/cyan] {lakehouse_path}")
    console.print(f"[cyan]Version:[/cyan] {version}")
    console.print(f"[cyan]Artifact type:[/cyan] {artifact_type}")
    console.print(f"[cyan]Output format:[/cyan] {output_format}")
    console.print(f"[cyan]Detailed output:[/cyan] {detailed}")
    console.print("")
    
    # Run validation
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Validating lakehouse...", total=None)
            
            # Perform validation
            reports = validate_lakehouse(
                lakehouse_path=lakehouse_path,
                version=version,
                config=config
            )
            
            progress.update(task, completed=100)
        
        if not reports:
            console.print("[yellow]No artifacts found to validate[/yellow]")
            console.print("[yellow]Make sure you have ingested data and run materialization[/yellow]")
            return
        
        # Filter by artifact type if specified
        if artifact_type != 'all':
            filtered_reports = {
                name: report for name, report in reports.items()
                if report.artifact_type == artifact_type
            }
            if not filtered_reports:
                console.print(f"[yellow]No {artifact_type} artifacts found to validate[/yellow]")
                return
            reports = filtered_reports
        
        # Set default output directory for validation reports
        if save_report and output_dir is None:
            output_dir = lakehouse_path / "catalogs" / "validation_reports"
        
        # Create reporter
        reporter = ValidationReporter(output_dir if save_report else None)
        
        # Display results
        if output_format == 'console':
            reporter.display_report(reports)
        else:
            # Generate formatted report
            if output_format == 'json':
                content = reporter.generate_summary_report(reports, format='json')
                console.print(json.dumps(content, indent=2))
            else:
                content = reporter.generate_summary_report(reports, format='text')
                console.print(content)
        
        # Save report if requested
        if save_report:
            try:
                saved_path = reporter.save_report(reports, format=output_format)
                console.print(f"\n[green]Validation report saved to: {saved_path}[/green]")
            except Exception as e:
                console.print(f"\n[yellow]Warning: Could not save report: {e}[/yellow]")
        
        # Show detailed results if requested
        if detailed and output_format == 'console':
            _show_detailed_results(reports)
        
        # Exit with error code if validation failed
        total_errors = sum(len(report.get_errors()) for report in reports.values())
        if total_errors > 0:
            console.print(f"\n[bold red]Validation failed with {total_errors} errors[/bold red]")
            if fail_fast:
                raise click.Abort()
        else:
            console.print("\n[bold green]✓ All validations passed![/bold green]")
    
    except Exception as e:
        console.print(f"\n[red]Error during validation: {e}[/red]")
        raise click.Abort()


def _show_detailed_results(reports):
    """Show detailed validation results for each artifact."""
    console.print("\n[bold blue]Detailed Validation Results[/bold blue]\n")
    
    for artifact_name, report in reports.items():
        console.print(f"[bold cyan]{artifact_name}[/bold cyan]")
        console.print(f"  Type: {report.artifact_type}")
        console.print(f"  Version: {report.version}")
        console.print(f"  Timestamp: {report.timestamp}")
        
        # Show statistics
        if report.statistics:
            console.print("  Statistics:")
            for key, value in report.statistics.items():
                console.print(f"    {key}: {value}")
        
        # Show check results
        console.print("  Validation Checks:")
        for check in report.checks:
            status = "✓" if check.passed else "✗"
            severity_color = "red" if check.severity == "error" else "yellow" if check.severity == "warning" else "blue"
            console.print(f"    {status} [{severity_color}]{check.severity.upper()}[/{severity_color}] {check.check_name}: {check.message}")
        
        console.print("")


def _validate_specific_artifact(
    lakehouse_path: Path,
    artifact_type: str,
    version: str,
    config: dict = None
) -> dict:
    """
    Validate a specific artifact type.
    
    Args:
        lakehouse_path: Path to lakehouse
        artifact_type: Type of artifact to validate
        version: Version to validate
        config: Validation configuration
    
    Returns:
        Dictionary of validation reports
    """
    from lakehouse.validation.checks import validate_artifact
    import pandas as pd
    
    reports = {}
    
    # Define artifact paths
    artifact_paths = {
        "utterance": lakehouse_path / "normalized" / version,
        "span": lakehouse_path / "spans" / version,
        "beat": lakehouse_path / "beats" / version,
        "section": lakehouse_path / "sections" / version,
        "embedding": lakehouse_path / "embeddings" / version,
    }
    
    base_path = artifact_paths.get(artifact_type)
    if not base_path or not base_path.exists():
        console.print(f"[yellow]No {artifact_type} data found at {base_path}[/yellow]")
        return reports
    
    # Find and validate parquet files
    parquet_files = list(base_path.glob("*.parquet"))
    if not parquet_files:
        console.print(f"[yellow]No parquet files found in {base_path}[/yellow]")
        return reports
    
    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
            report = validate_artifact(df, artifact_type, version, config)
            reports[f"{artifact_type}_{parquet_file.stem}"] = report
        except Exception as e:
            console.print(f"[red]Error validating {parquet_file}: {e}[/red]")
    
    return reports

