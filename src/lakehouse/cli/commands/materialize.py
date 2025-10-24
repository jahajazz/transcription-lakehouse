"""
Materialize command for generating derived artifacts.

Generates spans, beats, sections, embeddings, and FAISS indices from
normalized utterances.
"""

import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from lakehouse.cli import cli, common_options
from lakehouse.logger import configure_logging
from lakehouse.ingestion.writer import read_parquet, write_versioned_parquet
from lakehouse.aggregation.spans import generate_spans
from lakehouse.aggregation.beats import generate_beats
from lakehouse.aggregation.sections import generate_sections
from lakehouse.embeddings.generator import EmbeddingGenerator
from lakehouse.embeddings.storage import store_embeddings
from lakehouse.indexing.faiss_builder import build_and_save_index
from lakehouse.config import load_config


console = Console(legacy_windows=False)


@cli.command()
@common_options
@click.option(
    '--version',
    default='v1',
    help='Version to materialize (default: v1)',
)
@click.option(
    '--spans-only',
    is_flag=True,
    help='Generate only spans',
)
@click.option(
    '--beats-only',
    is_flag=True,
    help='Generate only beats (requires spans)',
)
@click.option(
    '--sections-only',
    is_flag=True,
    help='Generate only sections (requires beats)',
)
@click.option(
    '--embeddings-only',
    is_flag=True,
    help='Generate only embeddings (requires spans/beats)',
)
@click.option(
    '--indices-only',
    is_flag=True,
    help='Build only FAISS indices (requires embeddings)',
)
@click.option(
    '--all',
    'generate_all',
    is_flag=True,
    help='Generate all artifacts (default if no flags specified)',
)
@click.pass_context
def materialize(ctx, version, spans_only, beats_only, sections_only, embeddings_only, indices_only, generate_all, lakehouse_path, config_dir, log_level):
    """
    Generate derived artifacts from normalized utterances.
    
    Creates hierarchical aggregations (spans, beats, sections), vector embeddings,
    and FAISS indices for similarity search.
    
    Examples:
    
        # Generate all artifacts
        lakehouse materialize --all
        
        # Generate only spans
        lakehouse materialize --spans-only
        
        # Generate spans and beats
        lakehouse materialize --spans-only --beats-only
        
        # With custom version
        lakehouse materialize --all --version v2
    """
    # Configure logging
    configure_logging(level=log_level, console_output=True)
    
    # Display header
    console.print("\n[bold blue]Transcript Lakehouse - Materialize[/bold blue]\n")
    
    # Determine what to generate
    if not any([spans_only, beats_only, sections_only, embeddings_only, indices_only, generate_all]):
        generate_all = True  # Default to all if no flags specified
    
    tasks_to_run = {
        'spans': spans_only or generate_all,
        'beats': beats_only or generate_all,
        'sections': sections_only or generate_all,
        'embeddings': embeddings_only or generate_all,
        'indices': indices_only or generate_all,
    }
    
    console.print(f"[cyan]Lakehouse path:[/cyan] {lakehouse_path}")
    console.print(f"[cyan]Version:[/cyan] {version}")
    console.print(f"[cyan]Tasks:[/cyan] {', '.join(k for k, v in tasks_to_run.items() if v)}\n")
    
    # Check if lakehouse exists
    if not lakehouse_path.exists():
        console.print(f"[red]Error: Lakehouse not found at {lakehouse_path}[/red]")
        console.print("[yellow]Run 'lakehouse ingest' first to create the lakehouse[/yellow]")
        raise click.Abort()
    
    # Load configurations
    agg_config = load_config(config_dir, "aggregation")
    emb_config = load_config(config_dir, "embedding")
    
    results = {}
    
    try:
        # Step 1: Generate spans
        if tasks_to_run['spans']:
            console.print("[bold]Step 1: Generating Spans[/bold]")
            spans = _generate_spans(lakehouse_path, version, agg_config)
            results['spans'] = len(spans) if spans else 0
            console.print(f"[green][OK] Generated {results['spans']} spans[/green]\n")
        
        # Step 2: Generate beats
        if tasks_to_run['beats']:
            console.print("[bold]Step 2: Generating Beats[/bold]")
            beats = _generate_beats(lakehouse_path, version, agg_config)
            results['beats'] = len(beats) if beats else 0
            console.print(f"[green][OK] Generated {results['beats']} beats[/green]\n")
        
        # Step 3: Generate sections
        if tasks_to_run['sections']:
            console.print("[bold]Step 3: Generating Sections[/bold]")
            sections = _generate_sections(lakehouse_path, version, agg_config)
            results['sections'] = len(sections) if sections else 0
            console.print(f"[green][OK] Generated {results['sections']} sections[/green]\n")
        
        # Step 4: Generate embeddings
        if tasks_to_run['embeddings']:
            console.print("[bold]Step 4: Generating Embeddings[/bold]")
            emb_count = _generate_embeddings(lakehouse_path, version, emb_config)
            results['embeddings'] = emb_count
            console.print(f"[green][OK] Generated {emb_count} embeddings[/green]\n")
        
        # Step 5: Build FAISS indices
        if tasks_to_run['indices']:
            console.print("[bold]Step 5: Building FAISS Indices[/bold]")
            index_count = _build_indices(lakehouse_path, version, emb_config)
            results['indices'] = index_count
            console.print(f"[green][OK] Built {index_count} FAISS indices[/green]\n")
        
        # Display summary
        _display_summary(results)
        
    except Exception as e:
        console.print(f"\n[red]Error during materialization: {e}[/red]")
        raise click.Abort()


def _generate_spans(lakehouse_path, version, config):
    """Generate spans from utterances."""
    # Load utterances
    normalized_path = lakehouse_path / "normalized" / version
    if not normalized_path.exists():
        console.print(f"[yellow]No normalized utterances found in version {version}[/yellow]")
        return []
    
    parquet_files = list(normalized_path.glob("*.parquet"))
    if not parquet_files:
        console.print("[yellow]No normalized utterance files found[/yellow]")
        return []
    
    all_spans = []
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task(f"Processing {len(parquet_files)} episode(s)...", total=None)
        
        for parquet_file in parquet_files:
            df = read_parquet(parquet_file)
            utterances = df.to_dict('records')
            
            # Generate spans
            spans_config = config.get('spans', {})
            spans = generate_spans(utterances, config=spans_config)
            all_spans.extend(spans)
    
    # Write spans
    if all_spans:
        write_versioned_parquet(
            data=all_spans,
            base_path=lakehouse_path,
            artifact_type="spans",
            filename="spans.parquet",
            version=version,
            overwrite=True,
        )
    
    return all_spans


def _generate_beats(lakehouse_path, version, config):
    """Generate beats from spans."""
    # Load spans
    spans_path = lakehouse_path / "spans" / version / "spans.parquet"
    if not spans_path.exists():
        console.print("[yellow]No spans found. Run with --spans-only first[/yellow]")
        return []
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Processing spans...", total=None)
        
        df = read_parquet(spans_path)
        spans = df.to_dict('records')
        
        # Generate beats
        beats_config = config.get('beats', {})
        beats = generate_beats(spans, config=beats_config)
    
    # Write beats
    if beats:
        write_versioned_parquet(
            data=beats,
            base_path=lakehouse_path,
            artifact_type="beats",
            filename="beats.parquet",
            version=version,
            overwrite=True,
        )
    
    return beats


def _generate_sections(lakehouse_path, version, config):
    """Generate sections from beats."""
    # Load beats
    beats_path = lakehouse_path / "beats" / version / "beats.parquet"
    if not beats_path.exists():
        console.print("[yellow]No beats found. Run with --beats-only first[/yellow]")
        return []
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Processing beats...", total=None)
        
        df = read_parquet(beats_path)
        beats = df.to_dict('records')
        
        # Generate sections
        sections_config = config.get('sections', {})
        sections = generate_sections(beats, config=sections_config)
    
    # Write sections
    if sections:
        write_versioned_parquet(
            data=sections,
            base_path=lakehouse_path,
            artifact_type="sections",
            filename="sections.parquet",
            version=version,
            overwrite=True,
        )
    
    return sections


def _generate_embeddings(lakehouse_path, version, config):
    """Generate embeddings for spans and beats."""
    generator = EmbeddingGenerator(config=config)
    all_embeddings = []
    
    # Check artifact config
    artifact_config = config.get('artifacts', {})
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        # Generate span embeddings
        if artifact_config.get('embed_spans', True):
            task = progress.add_task("Generating span embeddings...", total=None)
            spans_path = lakehouse_path / "spans" / version / "spans.parquet"
            if spans_path.exists():
                df = read_parquet(spans_path)
                embeddings = generator.generate(df, artifact_type="span")
                all_embeddings.extend(embeddings)
        
        # Generate beat embeddings
        if artifact_config.get('embed_beats', True):
            task = progress.add_task("Generating beat embeddings...", total=None)
            beats_path = lakehouse_path / "beats" / version / "beats.parquet"
            if beats_path.exists():
                df = read_parquet(beats_path)
                embeddings = generator.generate(df, artifact_type="beat")
                all_embeddings.extend(embeddings)
    
    # Store embeddings
    if all_embeddings:
        store_embeddings(all_embeddings, lakehouse_path, version=version, overwrite=True)
    
    return len(all_embeddings)


def _build_indices(lakehouse_path, version, config):
    """Build FAISS indices for embeddings."""
    index_config = config.get('index', {})
    index_count = 0
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        # Build span index
        task = progress.add_task("Building span index...", total=None)
        try:
            path = build_and_save_index(lakehouse_path, "span", version, config=index_config)
            if path:
                index_count += 1
        except Exception as e:
            console.print(f"[yellow]Could not build span index: {e}[/yellow]")
        
        # Build beat index
        task = progress.add_task("Building beat index...", total=None)
        try:
            path = build_and_save_index(lakehouse_path, "beat", version, config=index_config)
            if path:
                index_count += 1
        except Exception as e:
            console.print(f"[yellow]Could not build beat index: {e}[/yellow]")
    
    return index_count


def _display_summary(results):
    """Display summary table."""
    table = Table(title="Materialization Summary")
    table.add_column("Artifact Type", style="cyan")
    table.add_column("Count", style="magenta", justify="right")
    
    if 'spans' in results:
        table.add_row("Spans", str(results['spans']))
    if 'beats' in results:
        table.add_row("Beats", str(results['beats']))
    if 'sections' in results:
        table.add_row("Sections", str(results['sections']))
    if 'embeddings' in results:
        table.add_row("Embeddings", str(results['embeddings']))
    if 'indices' in results:
        table.add_row("FAISS Indices", str(results['indices']))
    
    console.print("\n")
    console.print(table)
    console.print("\n[bold green][OK] Materialization complete![/bold green]")

