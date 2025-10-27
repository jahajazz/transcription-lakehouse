# Snapshot Setup Guide

## Overview

There are two directory settings to understand when working with snapshots:

1. **`snapshot_root`** - Where snapshots are **CREATED** (producer side)
2. **`LAKE_ROOT`** - Where snapshots are **CONSUMED** (consumer side)

---

## 1. Setting Up `snapshot_root` (Creating Snapshots)

This is where the snapshot system will **save** new snapshots when you run `lakehouse snapshot create`.

### Method 1: Environment Variable (Recommended)

**Windows PowerShell:**
```powershell
$env:SNAPSHOT_ROOT = "C:\Data\lake_snapshots"
```

**Windows CMD:**
```cmd
set SNAPSHOT_ROOT=C:\Data\lake_snapshots
```

**Linux/macOS:**
```bash
export SNAPSHOT_ROOT=/data/lake_snapshots
```

Make it permanent by adding to your shell profile:
- Windows: Add to PowerShell profile (`$PROFILE`)
- Linux/macOS: Add to `~/.bashrc` or `~/.zshrc`

### Method 2: CLI Flag

Override on each command:
```bash
lakehouse snapshot create --snapshot-root C:\Data\lake_snapshots
```

### Method 3: Configuration File

Edit `config/snapshot_config.yaml`:
```yaml
snapshot_root: "C:/Data/lake_snapshots"
```

### Default Behavior

If you don't set anything, snapshots will be saved to:
- `./snapshots/` (relative to current directory)

---

## 2. Setting Up `LAKE_ROOT` (Using Snapshots)

After creating a snapshot, **consumers** (other repos, tools, or users) set `LAKE_ROOT` to **point to** the specific snapshot they want to use.

This is displayed in the CLI output after snapshot creation.

### Windows PowerShell:
```powershell
$env:LAKE_ROOT = "C:\Data\lake_snapshots\v0.9.0-provisional"
```

### Windows CMD:
```cmd
set LAKE_ROOT=C:\Data\lake_snapshots\v0.9.0-provisional
```

### Linux/macOS:
```bash
export LAKE_ROOT=/data/lake_snapshots/v0.9.0-provisional
```

This tells consuming applications which snapshot version to read from.

---

## Complete Workflow Example

### Step 1: Configure snapshot_root (one-time setup)

**PowerShell:**
```powershell
# Add to your profile for persistence
$env:SNAPSHOT_ROOT = "D:\lakehouse\snapshots"
```

### Step 2: Create a snapshot

```bash
lakehouse snapshot create --version 0.9.0-provisional
```

Output will show:
```
✓ Snapshot created with WARNINGS

To use this snapshot, set the LAKE_ROOT environment variable:

  # Windows (PowerShell)
  $env:LAKE_ROOT = "D:\lakehouse\snapshots\v0.9.0-provisional"
```

### Step 3: Use the snapshot (on consumer side)

```powershell
# Set LAKE_ROOT to the snapshot you want to use
$env:LAKE_ROOT = "D:\lakehouse\snapshots\v0.9.0-provisional"

# Now your consuming application can read from this snapshot
```

---

## Recommended Setup for Development

### On the Producer Machine (where you create snapshots):

1. Set `SNAPSHOT_ROOT` once:
```powershell
# In PowerShell profile
$env:SNAPSHOT_ROOT = "C:\Data\lake_snapshots"
```

2. Create snapshots:
```bash
lakehouse snapshot create
```

### On Consumer Machines (where you use snapshots):

1. Copy the snapshot directory to the consumer machine
2. Set `LAKE_ROOT` to point to it:
```powershell
$env:LAKE_ROOT = "C:\local\path\to\v0.9.0-provisional"
```

---

## Recommended Setup for Production

### Shared Network Location:

```powershell
# Producer: Create snapshots to network share
$env:SNAPSHOT_ROOT = "\\fileserver\lake_snapshots"
lakehouse snapshot create

# Consumer: Point to specific snapshot on network share
$env:LAKE_ROOT = "\\fileserver\lake_snapshots\v1.0.0"
```

### S3 or Cloud Storage:

```bash
# 1. Create locally
export SNAPSHOT_ROOT=/local/snapshots
lakehouse snapshot create

# 2. Upload to cloud (manual step or CI/CD)
aws s3 sync /local/snapshots/v1.0.0 s3://mybucket/lake_snapshots/v1.0.0

# 3. Consumer downloads and sets LAKE_ROOT
aws s3 sync s3://mybucket/lake_snapshots/v1.0.0 /local/cache/v1.0.0
export LAKE_ROOT=/local/cache/v1.0.0
```

---

## Verification

After setting up, verify your configuration:

```bash
# Check where snapshots will be created
lakehouse snapshot create --help

# Create a test snapshot
lakehouse snapshot create --version test-0.1.0

# Check the output location
ls $SNAPSHOT_ROOT  # Linux/macOS
dir $env:SNAPSHOT_ROOT  # PowerShell
```

---

## Troubleshooting

### "Permission denied" when creating snapshot

**Problem:** Snapshot root directory isn't writable

**Solution:**
```bash
# Create the directory with proper permissions
mkdir -p $SNAPSHOT_ROOT
chmod 755 $SNAPSHOT_ROOT  # Linux/macOS
```

### Snapshots created in wrong location

**Problem:** `snapshot_root` not set correctly

**Solution:**
```bash
# Check current setting
echo $SNAPSHOT_ROOT  # Linux/macOS/PowerShell

# Use CLI flag to override
lakehouse snapshot create --snapshot-root /correct/path
```

### Consumer can't find snapshot

**Problem:** `LAKE_ROOT` not set or pointing to wrong location

**Solution:**
```bash
# Verify the path exists
ls $LAKE_ROOT  # Should show lake_manifest.json and snapshot_note.txt

# Check the manifest
cat $LAKE_ROOT/lake_manifest.json
```

