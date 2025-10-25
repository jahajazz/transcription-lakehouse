"""
Command-line interface for the lakehouse package.

Provides commands for ingestion, materialization, validation, and catalog management.
"""

import click
from pathlib import Path

from lakehouse import __version__
from lakehouse.logger import configure_logging


# Common options that can be reused across commands
def common_options(func):
    """Decorator to add common CLI options."""
    func = click.option(
        '--lakehouse-path',
        type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
        default=Path('lakehouse'),
        help='Path to lakehouse base directory (default: ./lakehouse)',
    )(func)
    func = click.option(
        '--config-dir',
        type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
        default=Path('config'),
        help='Path to configuration directory (default: ./config)',
    )(func)
    func = click.option(
        '--log-level',
        type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], case_sensitive=False),
        default='INFO',
        help='Logging level (default: INFO)',
    )(func)
    return func


@click.group()
@click.version_option(version=__version__, prog_name='lakehouse')
@click.pass_context
def cli(ctx):
    """
    Transcript Data Lakehouse CLI.
    
    A data lakehouse for podcast transcript storage, processing, and analysis.
    Provides deterministic, reproducible storage and query surfaces for raw and
    derived transcript artifacts.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)


@cli.command()
def version():
    """Display version information."""
    click.echo(f"Transcript Lakehouse v{__version__}")


def main():
    """Main entry point for the CLI."""
    # Import commands to register them
    from lakehouse.cli.commands import ingest, materialize, validate, catalog, quality
    
    cli()


if __name__ == '__main__':
    main()
