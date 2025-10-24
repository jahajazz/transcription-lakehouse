"""
Ingest command for importing transcript files.

Handles end-to-end ingestion: read → validate → normalize → write.
"""

import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from lakehouse.cli import cli, common_options
from lakehouse.logger import configure_logging
from lakehouse.ingestion.pipeline import IngestionPipeline
from lakehouse.structure import get_or_create_lakehouse


console = Console()


@cli.command()
@common_options
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--pattern',
    default='*.jsonl',
    help='Glob pattern for matching transcript files (default: *.jsonl)',
)
@click.option(
    '--version',
    default='v1',
    help='Version for output data (default: v1)',
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Validate inputs without writing any outputs',
)
@click.option(
    '--incremental',
    is_flag=True,
    help='Process only new episodes not already in the lakehouse',
)
@click.option(
    '--skip-invalid/--fail-on-invalid',
    default=True,
    help='Skip invalid utterances vs fail on first error (default: skip)',
)
@click.pass_context
def ingest(ctx, input_path, pattern, version, dry_run, incremental, skip_invalid, lakehouse_path, config_dir, log_level):
    """
    Ingest transcript files into the lakehouse.
    
    INPUT_PATH can be either a single transcript file or a directory containing
    multiple transcript files.
    
    Examples:
    
        # Ingest a single file
        lakehouse ingest episode.jsonl
        
        # Ingest a directory of files
        lakehouse ingest transcripts/ --pattern "*.jsonl"
        
        # Dry run to validate without writing
        lakehouse ingest transcripts/ --dry-run
        
        # Incremental ingestion (skip existing episodes)
        lakehouse ingest transcripts/ --incremental
    """
    # Configure logging
    configure_logging(level=log_level, console_output=True)
    
    # Display header
    console.print("\n[bold blue]Transcript Lakehouse - Ingest[/bold blue]\n")
    
    if dry_run:
        console.print("[yellow]DRY RUN MODE: No data will be written[/yellow]\n")
    
    # Initialize or get lakehouse structure
    if not dry_run:
        structure = get_or_create_lakehouse(lakehouse_path, version=version)
        console.print(f"[green]✓[/green] Lakehouse initialized at {lakehouse_path}\n")
    else:
        lakehouse_path = None  # Don't write anything in dry run
    
    # Determine if input is file or directory
    if input_path.is_file():
        console.print(f"[cyan]Processing single file:[/cyan] {input_path}")
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob(pattern))
        console.print(f"[cyan]Processing directory:[/cyan] {input_path}")
        console.print(f"[cyan]Pattern:[/cyan] {pattern}")
        console.print(f"[cyan]Found {len(files)} file(s)[/cyan]\n")
        
        if len(files) == 0:
            console.print(f"[yellow]Warning: No files found matching pattern {pattern}[/yellow]")
            return
    else:
        console.print(f"[red]Error: Invalid input path: {input_path}[/red]")
        raise click.Abort()
    
    if dry_run:
        # Dry run: validate only
        _dry_run_validation(files, skip_invalid)
    else:
        # Full ingestion
        _run_ingestion(files, lakehouse_path, version, skip_invalid, incremental)


def _dry_run_validation(files, skip_invalid):
    """Perform dry run validation without writing data."""
    from lakehouse.ingestion.reader import TranscriptReader
    from lakehouse.ingestion.validator import validate_utterances
    
    console.print("[bold]Validation Results:[/bold]\n")
    
    total_files = len(files)
    total_utterances = 0
    total_valid = 0
    total_invalid = 0
    failed_files = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Validating files...", total=total_files)
        
        for file_path in files:
            try:
                # Read file
                reader = TranscriptReader(file_path)
                utterances = reader.read_utterances()
                
                # Validate
                result = validate_utterances(utterances, fail_fast=not skip_invalid)
                
                total_utterances += result.total_count
                total_valid += result.valid_count
                total_invalid += result.invalid_count
                
                if not result.is_valid and not skip_invalid:
                    failed_files.append(file_path.name)
                
            except Exception as e:
                console.print(f"[red]Error reading {file_path.name}: {e}[/red]")
                failed_files.append(file_path.name)
            
            progress.update(task, advance=1)
    
    # Display summary table
    table = Table(title="Validation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="magenta", justify="right")
    
    table.add_row("Files Processed", str(total_files))
    table.add_row("Files Failed", str(len(failed_files)))
    table.add_row("Total Utterances", str(total_utterances))
    table.add_row("Valid Utterances", str(total_valid))
    table.add_row("Invalid Utterances", str(total_invalid))
    
    console.print("\n")
    console.print(table)
    
    if failed_files:
        console.print(f"\n[yellow]Failed files:[/yellow]")
        for filename in failed_files[:10]:  # Show first 10
            console.print(f"  - {filename}")
        if len(failed_files) > 10:
            console.print(f"  ... and {len(failed_files) - 10} more")
    
    if total_invalid > 0:
        console.print(f"\n[yellow]Note: {total_invalid} invalid utterances detected[/yellow]")
    
    if len(failed_files) == 0 and total_invalid == 0:
        console.print("\n[bold green]✓ All files validated successfully![/bold green]")


def _run_ingestion(files, lakehouse_path, version, skip_invalid, incremental):
    """Run full ingestion pipeline."""
    # Create pipeline
    pipeline = IngestionPipeline(
        lakehouse_path=lakehouse_path,
        version=version,
        skip_invalid=skip_invalid,
    )
    
    # Filter for incremental if requested
    if incremental:
        files = _filter_new_episodes(files, lakehouse_path, version)
        if not files:
            console.print("[green]No new episodes to process (all already ingested)[/green]")
            return
        console.print(f"[cyan]Incremental mode: Processing {len(files)} new file(s)[/cyan]\n")
    
    # Process files with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting files...", total=len(files))
        
        results = []
        for file_path in files:
            result = pipeline.ingest_file(file_path)
            results.append(result)
            progress.update(task, advance=1)
    
    # Aggregate results
    total_episodes = sum(r.episodes_processed for r in results)
    total_failed = sum(r.episodes_failed for r in results)
    total_utterances = sum(r.total_utterances for r in results)
    total_valid = sum(r.valid_utterances for r in results)
    total_invalid = sum(r.invalid_utterances for r in results)
    
    # Display summary table
    table = Table(title="Ingestion Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="magenta", justify="right")
    
    table.add_row("Files Processed", str(len(files)))
    table.add_row("Episodes Ingested", str(total_episodes))
    table.add_row("Episodes Failed", str(total_failed))
    table.add_row("Total Utterances", str(total_utterances))
    table.add_row("Valid Utterances", str(total_valid))
    table.add_row("Invalid Utterances", str(total_invalid))
    
    console.print("\n")
    console.print(table)
    
    if total_failed > 0:
        console.print(f"\n[yellow]Warning: {total_failed} episode(s) failed to ingest[/yellow]")
    
    if total_invalid > 0:
        console.print(f"\n[yellow]Note: {total_invalid} invalid utterances were skipped[/yellow]")
    
    if total_episodes > 0:
        console.print(f"\n[bold green]✓ Successfully ingested {total_episodes} episode(s)![/bold green]")
        console.print(f"[green]Data written to: {lakehouse_path}/normalized/{version}/[/green]")


def _filter_new_episodes(files, lakehouse_path, version):
    """Filter files to only those not already ingested."""
    from lakehouse.ingestion.reader import extract_episode_id
    import pandas as pd
    
    # Try to load existing episodes
    try:
        normalized_path = lakehouse_path / "normalized" / version
        if not normalized_path.exists():
            return files  # No existing data, all files are new
        
        # Get existing episode IDs from parquet files
        existing_files = list(normalized_path.glob("*.parquet"))
        if not existing_files:
            return files
        
        existing_episodes = set()
        for parquet_file in existing_files:
            try:
                df = pd.read_parquet(parquet_file, columns=["episode_id"])
                existing_episodes.update(df["episode_id"].unique())
            except Exception:
                pass  # Skip files that can't be read
        
        # Filter files
        new_files = []
        for file_path in files:
            try:
                episode_id = extract_episode_id(file_path)
                if episode_id not in existing_episodes:
                    new_files.append(file_path)
            except Exception:
                # If we can't extract episode ID, include it
                new_files.append(file_path)
        
        return new_files
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not check existing episodes: {e}[/yellow]")
        return files  # On error, process all files

