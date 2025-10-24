"""
Catalog command for displaying episode and speaker summaries.

Provides comprehensive cataloging capabilities using DuckDB queries to generate
summaries, statistics, and reports for episodes, speakers, and schema information.
"""

import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from lakehouse.cli import cli, common_options
from lakehouse.logger import configure_logging
from lakehouse.catalogs.episodes import EpisodeCatalog, generate_episode_catalog
from lakehouse.catalogs.speakers import SpeakerCatalog, generate_speaker_catalog
from lakehouse.catalogs.schema_manifest import SchemaManifest, generate_schema_manifest


console = Console(legacy_windows=False)


@cli.command()
@common_options
@click.option(
    '--version',
    default='v1',
    help='Version to catalog (default: v1)',
)
@click.option(
    '--catalog-type',
    type=click.Choice(['episodes', 'speakers', 'schema', 'all']),
    default='all',
    help='Type of catalog to generate (default: all)',
)
@click.option(
    '--output-format',
    type=click.Choice(['console', 'json', 'text']),
    default='console',
    help='Output format for catalog (default: console)',
)
@click.option(
    '--save-catalog',
    is_flag=True,
    help='Save catalog to files',
)
@click.option(
    '--output-dir',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help='Directory to save catalogs (default: lakehouse/catalogs)',
)
@click.option(
    '--detailed',
    is_flag=True,
    help='Show detailed catalog information',
)
@click.option(
    '--episode-id',
    help='Show detailed information for specific episode',
)
@click.option(
    '--speaker-name',
    help='Show detailed information for specific speaker',
)
@click.option(
    '--rankings',
    is_flag=True,
    help='Show speaker rankings by activity',
)
@click.option(
    '--statistics',
    is_flag=True,
    help='Show overall statistics',
)
@click.pass_context
def catalog(
    ctx,
    version,
    catalog_type,
    output_format,
    save_catalog,
    output_dir,
    detailed,
    episode_id,
    speaker_name,
    rankings,
    statistics,
    lakehouse_path,
    config_dir,
    log_level
):
    """
    Generate and display episode and speaker catalogs with summaries.
    
    Creates comprehensive catalogs using DuckDB queries on Parquet files,
    providing statistics, rankings, and detailed information about episodes,
    speakers, and schema compliance.
    
    Examples:
    
        # Generate all catalogs
        lakehouse catalog --all
        
        # Generate only episode catalog
        lakehouse catalog --catalog-type episodes
        
        # Show detailed information for specific episode
        lakehouse catalog --episode-id "EP001"
        
        # Show speaker rankings
        lakehouse catalog --rankings
        
        # Save catalogs to files
        lakehouse catalog --save-catalog --output-format json
        
        # Show overall statistics
        lakehouse catalog --statistics
    """
    # Configure logging
    configure_logging(level=log_level, console_output=True)
    
    # Display header
    console.print("\n[bold blue]Transcript Lakehouse - Catalog[/bold blue]\n")
    
    # Check if lakehouse exists
    if not lakehouse_path.exists():
        console.print(f"[red]Error: Lakehouse not found at {lakehouse_path}[/red]")
        console.print("[yellow]Run 'lakehouse ingest' first to create the lakehouse[/yellow]")
        raise click.Abort()
    
    # Set output directory
    if output_dir is None:
        output_dir = lakehouse_path / "catalogs"
    
    # Display catalog parameters
    console.print(f"[cyan]Lakehouse path:[/cyan] {lakehouse_path}")
    console.print(f"[cyan]Version:[/cyan] {version}")
    console.print(f"[cyan]Catalog type:[/cyan] {catalog_type}")
    console.print(f"[cyan]Output format:[/cyan] {output_format}")
    console.print(f"[cyan]Save catalog:[/cyan] {save_catalog}")
    console.print("")
    
    try:
        # Generate catalogs based on type
        if catalog_type in ['episodes', 'all']:
            _handle_episode_catalog(
                lakehouse_path, version, output_dir, save_catalog,
                output_format, detailed, episode_id, statistics
            )
        
        if catalog_type in ['speakers', 'all']:
            _handle_speaker_catalog(
                lakehouse_path, version, output_dir, save_catalog,
                output_format, detailed, speaker_name, rankings, statistics
            )
        
        if catalog_type in ['schema', 'all']:
            _handle_schema_catalog(
                lakehouse_path, version, output_dir, save_catalog,
                output_format, detailed, statistics
            )
        
        console.print("\n[bold green]✓ Catalog generation complete![/bold green]")
    
    except Exception as e:
        console.print(f"\n[red]Error during catalog generation: {e}[/red]")
        raise click.Abort()


def _handle_episode_catalog(
    lakehouse_path: Path,
    version: str,
    output_dir: Path,
    save_catalog: bool,
    output_format: str,
    detailed: bool,
    episode_id: str,
    statistics: bool
):
    """Handle episode catalog generation and display."""
    console.print("[bold]Episode Catalog[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating episode catalog...", total=None)
        
        # Generate catalog
        if episode_id:
            # Show specific episode
            catalog = EpisodeCatalog(lakehouse_path, version)
            episode_info = catalog.get_episode_summary(episode_id)
            
            if "error" in episode_info:
                console.print(f"[red]Error: {episode_info['error']}[/red]")
                return
            
            _display_episode_details(episode_info, episode_id)
        else:
            # Generate full catalog
            catalog_df, saved_files = generate_episode_catalog(
                lakehouse_path, version, "both" if save_catalog else "none"
            )
            
            if catalog_df.empty:
                console.print("[yellow]No episode data found to catalog[/yellow]")
                return
            
            # Save files if requested
            if save_catalog:
                for format_type, file_path in saved_files.items():
                    console.print(f"[green]Saved {format_type} catalog to: {file_path}[/green]")
            
            # Display results
            if output_format == 'console':
                _display_episode_catalog(catalog_df, detailed)
            else:
                _export_catalog(catalog_df, "episodes", output_format)
        
        progress.update(task, completed=100)
    
    # Show statistics if requested
    if statistics and not episode_id:
        catalog = EpisodeCatalog(lakehouse_path, version)
        stats = catalog.get_episode_statistics()
        _display_episode_statistics(stats)


def _handle_speaker_catalog(
    lakehouse_path: Path,
    version: str,
    output_dir: Path,
    save_catalog: bool,
    output_format: str,
    detailed: bool,
    speaker_name: str,
    rankings: bool,
    statistics: bool
):
    """Handle speaker catalog generation and display."""
    console.print("\n[bold]Speaker Catalog[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating speaker catalog...", total=None)
        
        # Generate catalog
        if speaker_name:
            # Show specific speaker
            catalog = SpeakerCatalog(lakehouse_path, version)
            speaker_info = catalog.get_speaker_summary(speaker_name)
            
            if "error" in speaker_info:
                console.print(f"[red]Error: {speaker_info['error']}[/red]")
                return
            
            _display_speaker_details(speaker_info, speaker_name)
        else:
            # Generate full catalog
            catalog_df, saved_files = generate_speaker_catalog(
                lakehouse_path, version, "both" if save_catalog else "none"
            )
            
            if catalog_df.empty:
                console.print("[yellow]No speaker data found to catalog[/yellow]")
                return
            
            # Save files if requested
            if save_catalog:
                for format_type, file_path in saved_files.items():
                    console.print(f"[green]Saved {format_type} catalog to: {file_path}[/green]")
            
            # Display results
            if output_format == 'console':
                _display_speaker_catalog(catalog_df, detailed, rankings)
            else:
                _export_catalog(catalog_df, "speakers", output_format)
        
        progress.update(task, completed=100)
    
    # Show statistics if requested
    if statistics and not speaker_name:
        catalog = SpeakerCatalog(lakehouse_path, version)
        stats = catalog.get_speaker_statistics()
        _display_speaker_statistics(stats)


def _handle_schema_catalog(
    lakehouse_path: Path,
    version: str,
    output_dir: Path,
    save_catalog: bool,
    output_format: str,
    detailed: bool,
    statistics: bool
):
    """Handle schema catalog generation and display."""
    console.print("\n[bold]Schema Catalog[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating schema catalog...", total=None)
        
        # Generate catalog
        manifest_df, saved_files = generate_schema_manifest(
            lakehouse_path, version, "both" if save_catalog else "none"
        )
        
        if manifest_df.empty:
            console.print("[yellow]No schema information found to catalog[/yellow]")
            return
        
        # Save files if requested
        if save_catalog:
            for format_type, file_path in saved_files.items():
                console.print(f"[green]Saved {format_type} manifest to: {file_path}[/green]")
        
        # Display results
        if output_format == 'console':
            _display_schema_catalog(manifest_df, detailed)
        else:
            _export_catalog(manifest_df, "schema", output_format)
        
        progress.update(task, completed=100)
    
    # Show statistics if requested
    if statistics:
        manifest = SchemaManifest(lakehouse_path, version)
        stats = manifest.get_schema_statistics()
        _display_schema_statistics(stats)


def _display_episode_catalog(catalog_df, detailed: bool):
    """Display episode catalog in console format."""
    # Create summary table
    table = Table(title="Episode Catalog Summary", show_header=True, header_style="bold magenta")
    table.add_column("Episode ID", style="cyan", no_wrap=True)
    table.add_column("Duration (min)", justify="right", style="magenta")
    table.add_column("Utterances", justify="right", style="green")
    table.add_column("Speakers", justify="right", style="yellow")
    table.add_column("Speaker List", style="white")
    
    for _, row in catalog_df.iterrows():
        table.add_row(
            row['episode_id'],
            f"{row['duration_minutes']:.1f}",
            str(row['utterance_count']),
            str(row['speaker_count']),
            row['speaker_list']
        )
    
    console.print(table)
    
    if detailed:
        console.print("\n[bold]Detailed Episode Information:[/bold]")
        for _, row in catalog_df.iterrows():
            _display_episode_details(row.to_dict(), row['episode_id'])


def _display_speaker_catalog(catalog_df, detailed: bool, rankings: bool):
    """Display speaker catalog in console format."""
    # Create summary table
    table = Table(title="Speaker Catalog Summary", show_header=True, header_style="bold magenta")
    table.add_column("Speaker", style="cyan", no_wrap=True)
    table.add_column("Episodes", justify="right", style="magenta")
    table.add_column("Utterances", justify="right", style="green")
    table.add_column("Duration (min)", justify="right", style="yellow")
    table.add_column("Avg Utterance (min)", justify="right", style="blue")
    
    for _, row in catalog_df.iterrows():
        table.add_row(
            row['speaker'],
            str(row['episode_count']),
            str(row['total_utterances']),
            f"{row['total_duration_minutes']:.1f}",
            f"{row['avg_utterance_duration_minutes']:.2f}"
        )
    
    console.print(table)
    
    if rankings:
        _display_speaker_rankings(catalog_df)
    
    if detailed:
        console.print("\n[bold]Detailed Speaker Information:[/bold]")
        for _, row in catalog_df.iterrows():
            _display_speaker_details(row.to_dict(), row['speaker'])


def _display_schema_catalog(manifest_df, detailed: bool):
    """Display schema catalog in console format."""
    # Create summary table
    table = Table(title="Schema Manifest Summary", show_header=True, header_style="bold magenta")
    table.add_column("Artifact Type", style="cyan", no_wrap=True)
    table.add_column("Files", justify="right", style="magenta")
    table.add_column("Fields", justify="right", style="green")
    table.add_column("Has Data", justify="center", style="yellow")
    table.add_column("Description", style="white")
    
    for _, row in manifest_df.iterrows():
        has_data = "✓" if row['has_data'] else "✗"
        table.add_row(
            row['artifact_type'],
            str(row['file_count']),
            str(row['schema_fields']),
            has_data,
            row['description']
        )
    
    console.print(table)
    
    if detailed:
        console.print("\n[bold]Detailed Schema Information:[/bold]")
        for _, row in manifest_df.iterrows():
            _display_schema_details(row.to_dict())


def _display_episode_details(episode_info: dict, episode_id: str):
    """Display detailed episode information."""
    panel_text = Text()
    panel_text.append(f"Episode: {episode_id}\n", style="bold blue")
    panel_text.append(f"Duration: {episode_info.get('duration_minutes', 0):.1f} minutes\n")
    panel_text.append(f"Utterances: {episode_info.get('utterance_count', 0)}\n")
    panel_text.append(f"Speakers: {episode_info.get('speaker_count', 0)}\n")
    panel_text.append(f"Speaker List: {episode_info.get('speaker_list', 'N/A')}\n")
    panel_text.append(f"Avg Utterance Duration: {episode_info.get('avg_utterance_duration', 0):.2f} seconds")
    
    panel = Panel(panel_text, title=f"Episode Details: {episode_id}", border_style="blue")
    console.print(panel)


def _display_speaker_details(speaker_info: dict, speaker_name: str):
    """Display detailed speaker information."""
    panel_text = Text()
    panel_text.append(f"Speaker: {speaker_name}\n", style="bold blue")
    panel_text.append(f"Episodes: {speaker_info.get('episode_count', 0)}\n")
    panel_text.append(f"Total Utterances: {speaker_info.get('total_utterances', 0)}\n")
    panel_text.append(f"Total Duration: {speaker_info.get('total_duration_minutes', 0):.1f} minutes\n")
    panel_text.append(f"Avg Utterance Duration: {speaker_info.get('avg_utterance_duration_minutes', 0):.2f} minutes\n")
    panel_text.append(f"Episode List: {speaker_info.get('episode_list', 'N/A')}")
    
    panel = Panel(panel_text, title=f"Speaker Details: {speaker_name}", border_style="green")
    console.print(panel)


def _display_schema_details(schema_info: dict):
    """Display detailed schema information."""
    panel_text = Text()
    panel_text.append(f"Artifact Type: {schema_info.get('artifact_type', 'N/A')}\n", style="bold blue")
    panel_text.append(f"Description: {schema_info.get('description', 'N/A')}\n")
    panel_text.append(f"Files: {schema_info.get('file_count', 0)}\n")
    panel_text.append(f"Schema Fields: {schema_info.get('schema_fields', 0)}\n")
    panel_text.append(f"Has Data: {'Yes' if schema_info.get('has_data', False) else 'No'}\n")
    panel_text.append(f"Field Names: {schema_info.get('field_names', 'N/A')}")
    
    panel = Panel(panel_text, title=f"Schema Details: {schema_info.get('artifact_type', 'N/A')}", border_style="yellow")
    console.print(panel)


def _display_speaker_rankings(catalog_df):
    """Display speaker rankings."""
    console.print("\n[bold]Speaker Rankings by Total Utterances:[/bold]")
    
    rankings = catalog_df.nlargest(10, 'total_utterances')
    
    table = Table(title="Top 10 Most Active Speakers", show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Speaker", style="cyan")
    table.add_column("Utterances", justify="right", style="green")
    table.add_column("Duration (min)", justify="right", style="yellow")
    
    for i, (_, row) in enumerate(rankings.iterrows(), 1):
        table.add_row(
            str(i),
            row['speaker'],
            str(row['total_utterances']),
            f"{row['total_duration_minutes']:.1f}"
        )
    
    console.print(table)


def _display_episode_statistics(stats: dict):
    """Display episode statistics."""
    if "error" in stats:
        console.print(f"[red]Error: {stats['error']}[/red]")
        return
    
    console.print("\n[bold]Episode Statistics:[/bold]")
    
    stats_table = Table(title="Episode Statistics", show_header=True, header_style="bold magenta")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", justify="right", style="magenta")
    
    for key, value in stats.items():
        if key not in ['catalog_generated', 'version']:
            stats_table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(stats_table)


def _display_speaker_statistics(stats: dict):
    """Display speaker statistics."""
    if "error" in stats:
        console.print(f"[red]Error: {stats['error']}[/red]")
        return
    
    console.print("\n[bold]Speaker Statistics:[/bold]")
    
    stats_table = Table(title="Speaker Statistics", show_header=True, header_style="bold magenta")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", justify="right", style="magenta")
    
    for key, value in stats.items():
        if key not in ['catalog_generated', 'version']:
            stats_table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(stats_table)


def _display_schema_statistics(stats: dict):
    """Display schema statistics."""
    if "error" in stats:
        console.print(f"[red]Error: {stats['error']}[/red]")
        return
    
    console.print("\n[bold]Schema Statistics:[/bold]")
    
    stats_table = Table(title="Schema Statistics", show_header=True, header_style="bold magenta")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", justify="right", style="magenta")
    
    for key, value in stats.items():
        if key not in ['catalog_generated', 'version', 'artifact_types']:
            stats_table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(stats_table)


def _export_catalog(catalog_df, catalog_type: str, output_format: str):
    """Export catalog in specified format."""
    if output_format == 'json':
        import json
        content = catalog_df.to_dict(orient='records')
        console.print(json.dumps(content, indent=2, default=str))
    else:
        # Text format
        console.print(f"\n{catalog_type.upper()} CATALOG")
        console.print("=" * 50)
        console.print(catalog_df.to_string(index=False))
