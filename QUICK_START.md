# Transcript Lakehouse - Quick Start Guide

## Quick Reference - Complete Workflow

From a fresh PowerShell session:

```powershell
# Navigate to project directory
cd "C:\Users\kylej\Cursor Projects\TranscriptionLakehouse"

# Start environment (activates venv + sets UTF-8)
.\start-lakehouse.ps1

# Run the complete workflow
lakehouse ingest input/transcripts/             # Ingest transcript data
lakehouse materialize --all                     # Generate all artifacts
lakehouse quality                               # Run quality assessment
lakehouse catalog --save-catalog                # Generate catalogs
lakehouse snapshot create                       # Create versioned snapshot

# Optional: Force near-duplicate check (slow for large datasets)
lakehouse quality --force-duplicate-check
```

**That's it!** The `start-lakehouse.ps1` script handles environment activation and encoding setup for you.

---

## Installation

From the project root directory:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

## Important: Windows Setup

For the CLI to work properly on Windows, you need to set the UTF-8 encoding:

```powershell
$env:PYTHONIOENCODING="utf-8"
```

Add this to your PowerShell profile to make it permanent:
```powershell
Add-Content $PROFILE '$env:PYTHONIOENCODING="utf-8"'
```

## CLI Commands

### Check Installation

```powershell
lakehouse --version
lakehouse --help
```

### Ingest Transcripts

```powershell
# Single file
$env:PYTHONIOENCODING="utf-8"; lakehouse ingest episode.jsonl

# Directory of files
$env:PYTHONIOENCODING="utf-8"; lakehouse ingest transcripts/

# Dry run (validate only, no writes)
$env:PYTHONIOENCODING="utf-8"; lakehouse ingest transcripts/ --dry-run

# Incremental (skip already ingested episodes)
$env:PYTHONIOENCODING="utf-8"; lakehouse ingest transcripts/ --incremental
```

### Generate Derived Artifacts

```powershell
# Generate everything (spans, beats, sections, embeddings, indices)
$env:PYTHONIOENCODING="utf-8"; lakehouse materialize --all

# Generate only spans
$env:PYTHONIOENCODING="utf-8"; lakehouse materialize --spans-only

# Generate spans and beats
$env:PYTHONIOENCODING="utf-8"; lakehouse materialize --spans-only --beats-only

# Generate embeddings and indices (requires spans/beats)
$env:PYTHONIOENCODING="utf-8"; lakehouse materialize --embeddings-only --indices-only
```

## What Gets Created

After ingestion and materialization, your lakehouse directory will contain:

```
lakehouse/
├── raw/                              # Original transcript files
├── normalized/v1/                    # Cleaned, normalized utterances (Parquet)
├── spans/v1/                         # Single-speaker contiguous segments
├── beats/v1/                         # Semantic meaning units
├── sections/v1/                      # 5-12 minute logical blocks
├── embeddings/v1/                    # Vector embeddings (384-dim)
│   ├── span_embeddings.parquet
│   ├── beat_embeddings.parquet
│   └── metadata.json
├── ann_index/v1/                     # FAISS search indices
│   ├── span_index.faiss
│   ├── span_index.ids.json
│   ├── span_index.metadata.json
│   ├── beat_index.faiss
│   ├── beat_index.ids.json
│   └── beat_index.metadata.json
└── lakehouse_metadata.json           # Lakehouse configuration
```

## Example: Complete Pipeline

```powershell
# Set UTF-8 encoding
$env:PYTHONIOENCODING="utf-8"

# Ingest a transcript
lakehouse ingest "episode.jsonl"

# Generate all derived artifacts
lakehouse materialize --all
```

## Configuration

Configuration files in `config/` directory:

- `aggregation_config.yaml` - Span/beat/section generation rules
- `embedding_config.yaml` - Embedding model settings
- `validation_rules.yaml` - Input validation rules

## Querying Data

The lakehouse stores everything in **Parquet** format, which you can query with:

### Using DuckDB (SQL)

```python
import duckdb

# Query normalized utterances
duckdb.sql("""
    SELECT episode_id, speaker, text, start, end
    FROM 'lakehouse/normalized/v1/*.parquet'
    WHERE speaker = 'Host'
    ORDER BY start
""").show()

# Query spans
duckdb.sql("""
    SELECT span_id, speaker, duration, utterance_count
    FROM 'lakehouse/spans/v1/spans.parquet'
    WHERE duration > 30
""").show()
```

### Using PyArrow/Pandas

```python
import pandas as pd

# Read normalized data
df = pd.read_parquet('lakehouse/normalized/v1/')
print(df.head())

# Read spans
spans = pd.read_parquet('lakehouse/spans/v1/spans.parquet')
print(spans[['span_id', 'speaker', 'duration']].head())
```

### Using FAISS for Similarity Search

```python
import faiss
import numpy as np
import json

# Load FAISS index
index = faiss.read_index('lakehouse/ann_index/v1/span_index.faiss')

# Load ID mapping
with open('lakehouse/ann_index/v1/span_index.ids.json') as f:
    id_mapping = json.load(f)

# Search for similar spans (you need a query embedding)
query_embedding = np.random.randn(1, 384).astype('float32')  # Replace with real embedding
distances, indices = index.search(query_embedding, k=5)

# Get the span IDs
similar_span_ids = [id_mapping['artifact_ids'][idx] for idx in indices[0]]
print("Similar spans:", similar_span_ids)
```

## Troubleshooting

### Unicode Encoding Errors

If you see `UnicodeEncodeError: 'charmap' codec can't encode character`, make sure to set:

```powershell
$env:PYTHONIOENCODING="utf-8"
```

### Model Download Issues

The first time you run materialize, it will download the sentence-transformers model (~100MB). This is cached locally in `~/.cache/huggingface/`.

### Symlink Warning

You may see a warning about symlinks on Windows. This is harmless but you can:
- Enable Developer Mode in Windows settings, OR
- Run as Administrator, OR
- Set `$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"`

## Next Steps

- [x] Ingestion: ✅ Working
- [x] Materialization: ✅ Working  
- [ ] Validation command: Coming soon
- [ ] Catalog command: Coming soon
- [ ] Python API for querying
- [ ] Incremental index updates
- [ ] Multi-episode aggregation

## Support

For issues or questions, check the implementation in `src/lakehouse/` or refer to the PRD in `tasks/0001-prd-transcript-data-lakehouse.md`.

