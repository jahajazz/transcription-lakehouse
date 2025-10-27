"""
Snapshot command for creating lakehouse snapshots.

Provides CLI interface for creating versioned, immutable snapshots of
lakehouse artifacts with manifest and validation.
"""

import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

from lakehouse.cli import cli, common_options
from lakehouse.logger import configure_logging, get_default_logger
from lakehouse.snapshot.config import SnapshotConfig
from lakehouse.snapshot.creator import SnapshotCreator
from lakehouse.snapshot.validator import generate_validation_report


console = Console(legacy_windows=False)
logger = get_default_logger()


@cli.group()
def snapshot():
    """
    Manage lakehouse snapshots.
    
    Create versioned, immutable snapshots of lakehouse artifacts
    with comprehensive manifests and validation.
    """
    pass


@snapshot.command()
@common_options
@click.option(
    '--snapshot-root',
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help='Snapshot root directory (default: ./snapshots or $SNAPSHOT_ROOT)',
)
@click.option(
    '--version',
    'version_override',
    type=str,
    help='Override snapshot version (default: auto-increment from config)',
)
@click.option(
    '--lakehouse-version',
    type=str,
    default='v1',
    help='Lakehouse artifact version to snapshot (default: v1)',
)
def create(
    lakehouse_path,
    config_dir,
    log_level,
    snapshot_root,
    version_override,
    lakehouse_version,
):
    """
    Create a new provisional snapshot.
    
    Creates a versioned snapshot of all consumer-facing lakehouse artifacts
    including spans, beats, sections, embeddings, indexes, catalogs, and
    QA reports. Generates a comprehensive manifest with checksums and
    automatically validates the snapshot.
    
    Examples:
        # Create snapshot with defaults
        lakehouse snapshot create
        
        # Create snapshot with custom root
        lakehouse snapshot create --snapshot-root /data/snapshots
        
        # Create snapshot with specific version
        lakehouse snapshot create --version 1.0.0-rc1
        
        # Snapshot different lakehouse version
        lakehouse snapshot create --lakehouse-version v2
    """
    # Configure logging
    configure_logging(log_level)
    
    console.print("\n[bold cyan]📦 Lakehouse Snapshot Creation[/bold cyan]\n")
    
    # Display configuration
    _display_configuration(
        lakehouse_path, config_dir, snapshot_root, version_override, lakehouse_version
    )
    
    try:
        # Create snapshot config
        config_overrides = {}
        if snapshot_root:
            config_overrides["snapshot_root"] = str(snapshot_root)
        
        config = SnapshotConfig(
            config_path=config_dir / "snapshot_config.yaml" if (config_dir / "snapshot_config.yaml").exists() else None,
            overrides=config_overrides,
        )
        
        # Create snapshot with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Creating snapshot...", total=None)
            
            creator = SnapshotCreator(lakehouse_path, config)
            result = creator.create(
                version_override=version_override,
                lakehouse_version=lakehouse_version,
            )
            
            progress.update(task, completed=True)
        
        # Display results
        _display_results(result)
        
        # Exit with appropriate code
        validation_status = result['validation_result']['status']
        if validation_status == 'FAIL':
            console.print("\n[bold red]❌ Snapshot validation FAILED[/bold red]")
            raise click.Exit(1)
        else:
            # Check for warnings
            warnings = result['validation_result'].get('warnings', [])
            if warnings:
                console.print("\n[bold yellow]✓ Snapshot created with WARNINGS[/bold yellow]")
            else:
                console.print("\n[bold green]✓ Snapshot created successfully[/bold green]")
            raise click.Exit(0)
    
    except FileNotFoundError as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logger.error(f"File not found: {e}")
        raise click.Exit(1)
    
    except IOError as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logger.error(f"IO error: {e}")
        raise click.Exit(1)
    
    except Exception as e:
        console.print(f"\n[bold red]Error during snapshot creation:[/bold red] {e}")
        logger.exception("Snapshot creation failed")
        raise click.Exit(1)


def _display_configuration(
    lakehouse_path: Path,
    config_dir: Path,
    snapshot_root: Path,
    version_override: str,
    lakehouse_version: str,
):
    """Display snapshot configuration."""
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")
    
    config_table.add_row("Lakehouse Path", str(lakehouse_path))
    config_table.add_row("Config Directory", str(config_dir))
    config_table.add_row("Lakehouse Version", lakehouse_version)
    
    if snapshot_root:
        config_table.add_row("Snapshot Root", str(snapshot_root))
    else:
        config_table.add_row("Snapshot Root", "./snapshots (default)")
    
    if version_override:
        config_table.add_row("Version Override", version_override)
    else:
        config_table.add_row("Version", "auto-increment")
    
    console.print(Panel(config_table, title="Configuration", border_style="cyan"))
    console.print()


def _display_results(result: dict):
    """Display snapshot creation results."""
    console.print("\n[bold]Snapshot Creation Summary[/bold]\n")
    
    # Basic info
    info_table = Table(show_header=False, box=None)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="white")
    
    snapshot_path = result['snapshot_path']
    version = result['version']
    files_copied = result['files_copied']
    
    info_table.add_row("Version", f"v{version}")
    info_table.add_row("Location", str(snapshot_path))
    info_table.add_row("Files Copied", str(files_copied))
    
    console.print(info_table)
    console.print()
    
    # Validation results
    validation_result = result['validation_result']
    status = validation_result['status']
    
    status_color = "green" if status == "PASS" else "red"
    status_symbol = "✓" if status == "PASS" else "✗"
    
    console.print(
        f"[bold {status_color}]Validation: {status_symbol} {status}[/bold {status_color}]"
    )
    
    # Show validation details
    files_validated = validation_result.get('files_validated', 0)
    files_passed = validation_result.get('files_passed', 0)
    
    console.print(f"  Files Validated: {files_validated}")
    console.print(f"  Files Passed: {files_passed}")
    
    # Show warnings if any
    warnings = validation_result.get('warnings', [])
    if warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for warning in warnings:
            console.print(f"  ⚠ {warning}")
    
    # Show errors if any
    errors = validation_result.get('errors', [])
    if errors:
        console.print("\n[bold red]Errors:[/bold red]")
        for error in errors[:5]:  # Show first 5 errors
            console.print(f"  ✗ {error}")
        if len(errors) > 5:
            console.print(f"  ... and {len(errors) - 5} more errors")
    
    # Usage instructions
    if status == "PASS":
        console.print("\n[bold]Usage Instructions:[/bold]")
        console.print("\nTo use this snapshot, set the LAKE_ROOT environment variable:\n")
        
        snapshot_path_str = str(snapshot_path.resolve())
        
        # Windows PowerShell
        console.print("  [cyan]# Windows (PowerShell)[/cyan]")
        console.print(f'  $env:LAKE_ROOT = "{snapshot_path_str}"')
        console.print()
        
        # Unix/Linux/macOS
        console.print("  [cyan]# Linux/macOS[/cyan]")
        console.print(f'  export LAKE_ROOT="{snapshot_path_str}"')
        console.print()
        
        console.print("For detailed information, see:")
        console.print(f"  [cyan]{snapshot_path / 'snapshot_note.txt'}[/cyan]")
        console.print(f"  [cyan]{snapshot_path / 'lake_manifest.json'}[/cyan]")

