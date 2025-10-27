"""
Quality assessment command for comprehensive data quality evaluation.

Performs quality assessment on lakehouse spans and beats including coverage,
distribution, integrity, balance, text quality, and embedding sanity checks.
Implements PRD requirements FR-1, FR-2, FR-40.
"""

import click
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from datetime import datetime

from lakehouse.cli import cli, common_options
from lakehouse.logger import configure_logging, get_default_logger
from lakehouse.quality.assessor import QualityAssessor
from lakehouse.quality.thresholds import RAGStatus


console = Console(legacy_windows=False)
logger = get_default_logger()


@cli.command()
@common_options
@click.option(
    '--version',
    default='v1',
    help='Lakehouse version to assess (default: v1)',
)
@click.option(
    '--level',
    type=click.Choice(['spans', 'beats', 'all']),
    default='all',
    help='Assessment level: spans only, beats only, or both (default: all)',
)
@click.option(
    '--output-dir',
    type=click.Path(),
    default='output/quality',
    help='Output directory for quality reports (default: output/quality)',
)
@click.option(
    '--sample-size',
    type=int,
    default=100,
    help='Sample size for embedding analysis (default: 100)',
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    help='Path to quality thresholds YAML config file',
)
@click.option(
    '--coverage-min',
    type=float,
    help='Override minimum coverage percentage threshold',
)
@click.option(
    '--span-length-min',
    type=float,
    help='Override minimum span length in seconds',
)
@click.option(
    '--span-length-max',
    type=float,
    help='Override maximum span length in seconds',
)
@click.option(
    '--beat-length-min',
    type=float,
    help='Override minimum beat length in seconds',
)
@click.option(
    '--beat-length-max',
    type=float,
    help='Override maximum beat length in seconds',
)
@click.option(
    '--neighbor-k',
    type=int,
    help='Override number of nearest neighbors to retrieve (k)',
)
@click.option(
    '--no-timestamp',
    is_flag=True,
    help='Disable timestamped output directories',
)
@click.option(
    '--force-duplicate-check',
    is_flag=True,
    help='Force near-duplicate detection even for large datasets (>10k segments). Warning: may be slow.',
)
def quality(
    lakehouse_path,
    config_dir,
    log_level,
    version,
    level,
    output_dir,
    sample_size,
    config,
    coverage_min,
    span_length_min,
    span_length_max,
    beat_length_min,
    beat_length_max,
    neighbor_k,
    no_timestamp,
    force_duplicate_check,
):
    """
    Run comprehensive quality assessment on lakehouse data.
    
    Assesses spans and/or beats across multiple quality dimensions:
    - Coverage & Count Metrics (Category A)
    - Length & Distribution (Category B)
    - Ordering & Integrity (Category C)
    - Speaker & Series Balance (Category D)
    - Text Quality Proxies (Category E)
    - Embedding Sanity Checks (Category F)
    - Diagnostics & Outliers (Category G)
    
    Generates:
    - Comprehensive markdown report with RAG status
    - Global metrics JSON
    - Per-episode and per-segment CSV files
    - Diagnostic CSV files for manual review
    
    Examples:
        # Assess both spans and beats with default thresholds
        lakehouse quality --lakehouse-path /path/to/lakehouse
        
        # Assess only spans with custom thresholds
        lakehouse quality --level spans --coverage-min 90.0 --span-length-max 150.0
        
        # Use custom config file
        lakehouse quality --config custom_thresholds.yaml
        
        # Large sample for detailed embedding analysis
        lakehouse quality --sample-size 500 --neighbor-k 20
        
        # Force near-duplicate detection on large datasets (may be slow)
        lakehouse quality --force-duplicate-check
    """
    # Configure logging
    configure_logging(log_level)
    
    console.print("\n[bold cyan]🔍 Lakehouse Quality Assessment[/bold cyan]\n")
    
    lakehouse_path = Path(lakehouse_path)
    output_dir = Path(output_dir)
    
    # Display configuration
    _display_configuration(
        lakehouse_path, version, level, output_dir, 
        sample_size, config, no_timestamp
    )
    
    try:
        # Build threshold overrides
        threshold_overrides = _build_threshold_overrides(
            coverage_min,
            span_length_min,
            span_length_max,
            beat_length_min,
            beat_length_max,
            neighbor_k,
            sample_size,
        )
        
        # Run assessment with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            
            # Initialize assessor
            task = progress.add_task(
                "[cyan]Initializing quality assessor...",
                total=None
            )
            
            assessor = QualityAssessor(
                lakehouse_path=lakehouse_path,
                version=version,
                config_path=config,
                threshold_overrides=threshold_overrides,
            )
            
            progress.update(task, completed=True)
            
            # Run assessment
            task = progress.add_task(
                "[cyan]Running quality assessment...",
                total=None
            )
            
            result = assessor.run_assessment(
                assess_spans=(level in ['spans', 'all']),
                assess_beats=(level in ['beats', 'all']),
                output_dir=output_dir,
                use_timestamp=(not no_timestamp),
                force_near_duplicate_check=force_duplicate_check,
            )
            
            progress.update(task, completed=True)
        
        # Display summary
        _display_summary(result, output_dir)
        
        # Exit with appropriate code
        if result.rag_status == RAGStatus.RED:
            console.print("\n[bold red]❌ Assessment FAILED - Critical issues detected[/bold red]")
            sys.exit(1)
        elif result.rag_status == RAGStatus.AMBER:
            console.print("\n[bold yellow]⚠️  Assessment completed with WARNINGS[/bold yellow]")
            sys.exit(0)
        else:
            console.print("\n[bold green]✅ Assessment PASSED - All checks passed[/bold green]")
            sys.exit(0)
    
    except FileNotFoundError as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logger.error(f"File not found: {e}")
        sys.exit(1)
    
    except Exception as e:
        console.print(f"\n[bold red]Error during quality assessment:[/bold red] {e}")
        logger.exception("Quality assessment failed")
        sys.exit(1)


def _display_configuration(
    lakehouse_path: Path,
    version: str,
    level: str,
    output_dir: Path,
    sample_size: int,
    config: str,
    no_timestamp: bool,
):
    """Display assessment configuration (subtask 5.4.5)."""
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")
    
    config_table.add_row("Lakehouse Path", str(lakehouse_path))
    config_table.add_row("Version", version)
    config_table.add_row("Assessment Level", level.upper())
    config_table.add_row("Output Directory", str(output_dir))
    config_table.add_row("Sample Size", str(sample_size))
    config_table.add_row("Timestamped Output", "No" if no_timestamp else "Yes")
    
    if config:
        config_table.add_row("Config File", str(config))
    
    console.print(Panel(config_table, title="Configuration", border_style="cyan"))
    console.print()


def _build_threshold_overrides(
    coverage_min,
    span_length_min,
    span_length_max,
    beat_length_min,
    beat_length_max,
    neighbor_k,
    sample_size,
):
    """Build threshold overrides from CLI options (subtask 5.4.3)."""
    overrides = {}
    
    if coverage_min is not None:
        overrides['coverage_min'] = coverage_min
    
    if span_length_min is not None:
        overrides['span_length_min'] = span_length_min
    
    if span_length_max is not None:
        overrides['span_length_max'] = span_length_max
    
    if beat_length_min is not None:
        overrides['beat_length_min'] = beat_length_min
    
    if beat_length_max is not None:
        overrides['beat_length_max'] = beat_length_max
    
    if neighbor_k is not None:
        overrides['neighbor_k'] = neighbor_k
    
    if sample_size is not None:
        overrides['neighbor_sample_size'] = sample_size
    
    return overrides


def _display_summary(result, output_dir: Path):
    """
    Display assessment summary with Rich formatting (FR-40, subtask 5.4.5).
    
    Shows:
    - RAG status with color-coded indicator
    - Key dataset metrics
    - Violation summary
    - Output file paths
    """
    console.print("\n[bold]Assessment Summary[/bold]\n")
    
    # RAG Status
    rag_colors = {
        RAGStatus.GREEN: "green",
        RAGStatus.AMBER: "yellow",
        RAGStatus.RED: "red",
    }
    rag_emoji = {
        RAGStatus.GREEN: "🟢",
        RAGStatus.AMBER: "🟠",
        RAGStatus.RED: "🔴",
    }
    
    color = rag_colors.get(result.rag_status, "white")
    emoji = rag_emoji.get(result.rag_status, "⚪")
    
    console.print(
        f"[bold {color}]Overall Status: {emoji} {result.rag_status.value.upper()}[/bold {color}]\n"
    )
    
    # Dataset metrics
    metrics_table = Table(title="Dataset Metrics", show_header=True, box=None)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="white", justify="right")
    
    metrics_table.add_row("Episodes", f"{result.total_episodes:,}")
    metrics_table.add_row("Spans", f"{result.total_spans:,}")
    metrics_table.add_row("Beats", f"{result.total_beats:,}")
    metrics_table.add_row("Embeddings Available", "Yes" if result.metrics.embeddings_available else "No")
    metrics_table.add_row("Assessment Duration", f"{result.assessment_duration_seconds:.2f}s")
    
    console.print(metrics_table)
    console.print()
    
    # Violations summary
    if result.violations:
        violations_table = Table(title="Violations", show_header=True, box=None)
        violations_table.add_column("Severity", style="cyan")
        violations_table.add_column("Count", style="white", justify="right")
        
        error_count = len([v for v in result.violations if v.severity == "error"])
        warning_count = len([v for v in result.violations if v.severity == "warning"])
        
        if error_count > 0:
            violations_table.add_row("[red]Errors[/red]", f"[red]{error_count}[/red]")
        if warning_count > 0:
            violations_table.add_row("[yellow]Warnings[/yellow]", f"[yellow]{warning_count}[/yellow]")
        
        console.print(violations_table)
        console.print()
        
        # Show top violations
        if result.violations:
            console.print("[bold]Top Issues:[/bold]")
            for v in result.violations[:5]:
                severity_color = "red" if v.severity == "error" else "yellow"
                console.print(f"  [{severity_color}]•[/{severity_color}] {v.message}")
            if len(result.violations) > 5:
                console.print(f"  ... and {len(result.violations) - 5} more")
            console.print()
    else:
        console.print("[bold green]✓ No violations detected[/bold green]\n")
    
    # Output files
    console.print("[bold]Output Files:[/bold]")
    
    # Determine output paths
    if hasattr(result, 'output_paths') and result.output_paths:
        paths = result.output_paths
        console.print(f"  📊 Global Metrics: [cyan]{paths.get('global_json', 'N/A')}[/cyan]")
        console.print(f"  📋 Episodes CSV: [cyan]{paths.get('episodes_csv', 'N/A')}[/cyan]")
        console.print(f"  📋 Segments CSV: [cyan]{paths.get('segments_csv', 'N/A')}[/cyan]")
        console.print(f"  🔍 Diagnostics: [cyan]{paths.get('diagnostics_dir', 'N/A')}[/cyan]")
        console.print(f"  📄 Quality Report: [cyan]{paths.get('report_md', 'N/A')}[/cyan]")
    else:
        console.print(f"  📁 Output Directory: [cyan]{output_dir}[/cyan]")
    
    console.print()
