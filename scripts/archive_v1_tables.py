#!/usr/bin/env python3
"""
Archive existing v1 tables before regeneration (Task 5.4).

Moves current v1 tables to lakehouse/v1_archived/ directory for:
- Preservation of existing data
- Comparison with regenerated tables
- Rollback capability if needed

Usage:
    python scripts/archive_v1_tables.py

The script:
1. Creates lakehouse/v1_archived/ directory if it doesn't exist
2. Adds a timestamp to the archive subdirectory
3. Copies all v1 tables (spans, beats, sections, embeddings)
4. Preserves directory structure
5. Creates a manifest file documenting what was archived
"""

import shutil
from pathlib import Path
from datetime import datetime
import json


def archive_v1_tables(lakehouse_path: Path = None, dry_run: bool = False):
    """
    Archive existing v1 tables to a timestamped backup directory.
    
    Args:
        lakehouse_path: Path to lakehouse base directory (default: ./lakehouse)
        dry_run: If True, show what would be archived without actually copying
    
    Returns:
        Path to archive directory if successful, None otherwise
    """
    # Determine lakehouse path
    if lakehouse_path is None:
        lakehouse_path = Path(__file__).parent.parent / "lakehouse"
    else:
        lakehouse_path = Path(lakehouse_path)
    
    if not lakehouse_path.exists():
        print(f"ERROR: Lakehouse path not found: {lakehouse_path}")
        return None
    
    # Create archive directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_base = lakehouse_path / "v1_archived"
    archive_dir = archive_base / timestamp
    
    if dry_run:
        print(f"DRY RUN MODE - No files will be copied")
        print()
    
    print(f"Archiving v1 tables from: {lakehouse_path}")
    print(f"Archive destination: {archive_dir}")
    print()
    
    # Tables to archive
    tables_to_archive = [
        "spans/v1",
        "beats/v1",
        "sections/v1",
        "embeddings/v1",
    ]
    
    archive_manifest = {
        "timestamp": timestamp,
        "source_path": str(lakehouse_path),
        "archive_path": str(archive_dir),
        "archived_tables": {},
    }
    
    total_files = 0
    total_size = 0
    
    for table_path in tables_to_archive:
        source = lakehouse_path / table_path
        
        if not source.exists():
            print(f"WARNING: Skipping {table_path} (not found)")
            continue
        
        # Count files and size
        files = list(source.glob("**/*.parquet"))
        if not files:
            print(f"WARNING: Skipping {table_path} (no parquet files)")
            continue
        
        file_count = len(files)
        size_bytes = sum(f.stat().st_size for f in files)
        size_mb = size_bytes / (1024 * 1024)
        
        total_files += file_count
        total_size += size_bytes
        
        print(f"Found {table_path}:")
        print(f"   - Files: {file_count}")
        print(f"   - Size: {size_mb:.2f} MB")
        
        # Store in manifest
        archive_manifest["archived_tables"][table_path] = {
            "file_count": file_count,
            "size_bytes": size_bytes,
            "files": [f.name for f in files],
        }
        
        if not dry_run:
            # Create destination directory
            dest = archive_dir / table_path
            dest.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            for file in files:
                dest_file = dest / file.name
                shutil.copy2(file, dest_file)
            
            print(f"   * Copied to {dest}")
        else:
            print(f"   (would copy to {archive_dir / table_path})")
        
        print()
    
    # Summary
    total_size_mb = total_size / (1024 * 1024)
    print(f"Summary:")
    print(f"   - Total files: {total_files}")
    print(f"   - Total size: {total_size_mb:.2f} MB")
    print()
    
    if total_files == 0:
        print("WARNING: No tables found to archive")
        return None
    
    if not dry_run:
        # Write manifest
        manifest_file = archive_dir / "archive_manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(archive_manifest, f, indent=2)
        
        print(f"Archive complete!")
        print(f"Manifest: {manifest_file}")
        print()
        print(f"To restore archived tables:")
        print(f"  cp -r {archive_dir}/spans/v1/* {lakehouse_path}/spans/v1/")
        print(f"  cp -r {archive_dir}/beats/v1/* {lakehouse_path}/beats/v1/")
        print(f"  cp -r {archive_dir}/sections/v1/* {lakehouse_path}/sections/v1/")
        print(f"  cp -r {archive_dir}/embeddings/v1/* {lakehouse_path}/embeddings/v1/")
        
        return archive_dir
    else:
        print("Dry run complete - no files were copied")
        return None


def list_archives(lakehouse_path: Path = None):
    """
    List all available archives.
    
    Args:
        lakehouse_path: Path to lakehouse base directory
    """
    if lakehouse_path is None:
        lakehouse_path = Path(__file__).parent.parent / "lakehouse"
    else:
        lakehouse_path = Path(lakehouse_path)
    
    archive_base = lakehouse_path / "v1_archived"
    
    if not archive_base.exists():
        print(f"No archives found at {archive_base}")
        return
    
    archives = sorted([d for d in archive_base.iterdir() if d.is_dir()])
    
    if not archives:
        print(f"No archives found at {archive_base}")
        return
    
    print(f"Available archives in {archive_base}:")
    print()
    
    for archive_dir in archives:
        manifest_file = archive_dir / "archive_manifest.json"
        
        if manifest_file.exists():
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            
            timestamp = manifest.get("timestamp", archive_dir.name)
            table_count = len(manifest.get("archived_tables", {}))
            
            print(f"  {timestamp}:")
            print(f"    - Tables: {table_count}")
            print(f"    - Path: {archive_dir}")
            
            # Show table details
            for table_name, table_info in manifest.get("archived_tables", {}).items():
                file_count = table_info.get("file_count", 0)
                size_mb = table_info.get("size_bytes", 0) / (1024 * 1024)
                print(f"      * {table_name}: {file_count} files ({size_mb:.2f} MB)")
            
            print()
        else:
            print(f"  {archive_dir.name} (no manifest)")
            print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Archive v1 tables before regeneration"
    )
    parser.add_argument(
        "--lakehouse-path",
        type=Path,
        default=None,
        help="Path to lakehouse directory (default: ./lakehouse)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be archived without actually copying"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available archives"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_archives(args.lakehouse_path)
    else:
        archive_v1_tables(args.lakehouse_path, args.dry_run)

