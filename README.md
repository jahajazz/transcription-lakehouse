# Transcript Lakehouse

**A local data lakehouse for podcast transcript storage, processing, and semantic analysis.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-178%20passed-success)]()

## Overview

The Transcript Lakehouse is a powerful, local-first data lakehouse designed for managing, processing, and analyzing podcast transcripts at scale. It provides a complete pipeline from raw transcript ingestion through semantic search and validation.

### Key Features

- **🗄️ Multi-Layer Architecture**: Raw → Normalized → Aggregated layers with versioning
- **🔍 Semantic Search**: Vector embeddings and ANN indexing for content discovery
- **📊 Data Quality**: Comprehensive validation and sanity checks
- **⚡ Efficient Storage**: Parquet-based columnar storage with compression
- **🔗 Referential Integrity**: ID-based relationships between artifacts
- **📈 Aggregation Hierarchy**: Utterances → Spans → Beats → Sections
- **💻 CLI Interface**: Easy-to-use command-line tools
- **📋 Catalogs & Reports**: Metadata catalogs and validation reports

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/transcription-lakehouse.git
cd transcription-lakehouse

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package with dependencies
pip install -e .

# Or install with development tools
pip install -e ".[dev]"
```

### Dependencies

Core dependencies:
- `pyarrow` - Parquet file operations and schemas
- `pandas` - DataFrame operations
- `duckdb` - SQL analytics and catalog generation
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Local embedding models
- `openai` - OpenAI API integration (optional)
- `click` - CLI framework
- `rich` - Terminal formatting

## Quick Start

### 1. Ingest Transcripts

```bash
# Ingest a single transcript
lakehouse ingest input/transcripts/episode.jsonl

# Ingest all transcripts from a directory
lakehouse ingest input/transcripts/
```

### 2. Generate Aggregations

```bash
# Generate spans (speaker-contiguous utterances)
lakehouse aggregate spans

# Generate beats (semantic meaning units)
lakehouse aggregate beats

# Generate sections (5-12 minute blocks)
lakehouse aggregate sections

# Or generate all at once
lakehouse aggregate all
```

### 3. Create Embeddings

```bash
# Generate embeddings for spans (default)
lakehouse embed spans

# Generate embeddings for beats
lakehouse embed beats

# Use specific model
lakehouse embed spans --model all-MiniLM-L6-v2
```

### 4. Build Search Index

```bash
# Build ANN index for semantic search
lakehouse index build spans

# Search for similar content
lakehouse index search "artificial intelligence and consciousness" --top-k 10
```

### 5. Validate Data

```bash
# Run validation checks
lakehouse validate

# Run with detailed output
lakehouse validate --detailed

# Save validation report
lakehouse validate --save-report --output-format json
```

### 6. Generate Catalogs

```bash
# Generate all catalogs
lakehouse catalog generate

# Generate specific catalog
lakehouse catalog generate episodes
lakehouse catalog generate speakers

# View catalog
lakehouse catalog view episodes
```

## Architecture

### Data Layers

```
lakehouse/
├── raw/                    # Original JSONL transcripts
├── normalized/v1/          # Normalized utterances (Parquet)
├── spans/v1/              # Speaker-contiguous segments
├── beats/v1/              # Semantic meaning units
├── sections/v1/           # 5-12 minute blocks
├── embeddings/v1/         # Vector embeddings
├── ann_index/v1/          # FAISS search indexes
└── catalogs/              # Metadata catalogs
```

### Aggregation Hierarchy

1. **Utterances** → Raw transcript segments with timestamps
2. **Spans** → Contiguous same-speaker utterances
3. **Beats** → Semantic meaning units (topic-coherent)
4. **Sections** → 5-12 minute logical blocks

### ID System

All artifacts use deterministic, content-derived IDs:

```
utt_{episode_hash}_{position}_{content_hash}  # Utterance
spn_{episode_hash}_{position}_{content_hash}  # Span
bet_{episode_hash}_{position}_{content_hash}  # Beat
sec_{episode_hash}_{position}_{content_hash}  # Section
```

## CLI Reference

### Core Commands

#### `lakehouse ingest`
Ingest transcripts into the lakehouse.

```bash
lakehouse ingest [OPTIONS] INPUT_PATH

Options:
  --version TEXT         Version identifier [default: v1]
  --validate / --no-validate  Validate before ingesting [default: True]
  --help                Show this message and exit
```

#### `lakehouse aggregate`
Generate aggregated artifacts.

```bash
lakehouse aggregate [OPTIONS] ARTIFACT_TYPE

Arguments:
  ARTIFACT_TYPE  [spans|beats|sections|all]

Options:
  --version TEXT        Version identifier [default: v1]
  --config-path TEXT    Path to config directory
  --help               Show this message and exit
```

#### `lakehouse embed`
Generate vector embeddings.

```bash
lakehouse embed [OPTIONS] ARTIFACT_TYPE

Arguments:
  ARTIFACT_TYPE  [spans|beats|sections]

Options:
  --model TEXT          Model name
  --provider TEXT       Provider [local|openai] [default: local]
  --batch-size INTEGER  Batch size for processing [default: 32]
  --version TEXT        Version identifier [default: v1]
  --help               Show this message and exit
```

#### `lakehouse index`
Manage ANN search indexes.

```bash
lakehouse index build [OPTIONS] ARTIFACT_TYPE
lakehouse index search [OPTIONS] QUERY

Options (build):
  --version TEXT        Version identifier [default: v1]
  --metric TEXT         Distance metric [default: cosine]

Options (search):
  --artifact-type TEXT  Type to search [default: span]
  --top-k INTEGER       Number of results [default: 10]
  --version TEXT        Version identifier [default: v1]
```

#### `lakehouse validate`
Run validation checks.

```bash
lakehouse validate [OPTIONS]

Options:
  --version TEXT               Version identifier [default: v1]
  --detailed / --no-detailed   Show detailed output [default: False]
  --save-report / --no-save-report  Save report to file
  --output-format TEXT         Output format [text|json] [default: text]
  --help                      Show this message and exit
```

#### `lakehouse catalog`
Manage metadata catalogs.

```bash
lakehouse catalog generate [OPTIONS] [CATALOG_TYPE]
lakehouse catalog view [OPTIONS] CATALOG_TYPE

Arguments:
  CATALOG_TYPE  [episodes|speakers|schema|all]

Options:
  --version TEXT  Version identifier [default: v1]
  --format TEXT   Output format [table|json] [default: table]
  --help         Show this message and exit
```

## Configuration

Configuration files in `config/`:

### `embedding_config.yaml`

```yaml
# Model configuration
model:
  provider: local  # local or openai
  name: all-MiniLM-L6-v2
  device: cpu

# Generation parameters
generation:
  batch_size: 32
  normalize_embeddings: true
  max_text_length: 8192
  show_progress: true

# Fallback configuration
fallback:
  enabled: true
  provider: openai
  model_name: text-embedding-3-small
```

### `aggregation_config.yaml`

```yaml
# Span generation
spans:
  min_duration: 1.0
  max_silence_gap: 0.5
  break_on_speaker_change: true

# Beat generation
beats:
  similarity_threshold: 0.7
  min_spans_per_beat: 1
  use_embeddings: true
  fallback_method: heuristic

# Section generation
sections:
  target_duration_minutes: 8.0
  min_duration_minutes: 5.0
  max_duration_minutes: 12.0
  allow_semantic_overflow: true
```

### `validation_rules.yaml`

```yaml
# Schema validation
schema:
  enforce_types: true
  allow_missing_optional: true

# ID validation
ids:
  check_uniqueness: true
  check_format: true
  allow_duplicates_in_foreign_keys: true

# Timestamp validation
timestamps:
  allow_negative: false
  check_ordering: true
  check_overlaps: true
  max_gap_seconds: 300

# Text validation
text:
  min_length: 1
  max_length: 10000
  allow_empty: false
```

## Python API

### Basic Usage

```python
from lakehouse.ingestion import TranscriptReader, normalize_utterances
from lakehouse.aggregation import generate_spans, generate_beats
from lakehouse.embeddings import generate_embeddings
from lakehouse.validation import validate_artifact

# Read and normalize
reader = TranscriptReader("input/episode.jsonl")
raw_utterances = reader.read_utterances()
normalized = normalize_utterances(raw_utterances)

# Generate aggregations
spans = generate_spans(normalized)
beats = generate_beats(spans, config={"use_embeddings": False})

# Generate embeddings
embeddings = generate_embeddings(spans, artifact_type="span")

# Validate
import pandas as pd
df = pd.DataFrame(normalized)
report = validate_artifact(df, "utterance", "v1")
print(report.summary())
```

### Semantic Search

```python
from lakehouse.indexing import FAISSIndexManager

# Load index
index_manager = FAISSIndexManager("lakehouse/ann_index/v1")
index_manager.load_index("span")

# Search
query = "artificial intelligence and machine learning"
results = index_manager.search(query, top_k=10)

for result in results:
    print(f"Span ID: {result['span_id']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['text'][:100]}...")
    print()
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=lakehouse --cov-report=html

# Run specific test file
pytest tests/test_ingestion.py -v

# Run integration tests
pytest tests/integration/ -v
```

Test structure:
- `tests/test_ids.py` - ID generation (20 tests)
- `tests/test_ingestion.py` - Ingestion pipeline (55 tests)
- `tests/test_aggregation.py` - Aggregation logic (39 tests)
- `tests/test_embeddings.py` - Embedding generation (24 tests)
- `tests/test_validation.py` - Validation checks (35 tests)
- `tests/integration/test_pipeline.py` - End-to-end (5 tests)

**Total: 178 tests, all passing ✅**

## Data Quality

### Validation Checks

The lakehouse performs comprehensive validation:

- **Schema Compliance**: Column types and required fields
- **ID Quality**: Uniqueness, null checks, format validation
- **Temporal Consistency**: Timestamp ordering, negative values, overlaps
- **Numeric Quality**: Range checks, statistical outliers
- **Referential Integrity**: Foreign key validity
- **Text Quality**: Empty text, length validation

### Validation Reports

```bash
# Generate validation report
lakehouse validate --save-report --output-format json

# Report saved to: lakehouse/catalogs/validation_reports/{timestamp}.json
```

Example report structure:
```json
{
  "artifacts": {
    "utterance": {
      "total_checks": 45,
      "passed": 43,
      "failed": 2,
      "pass_percentage": 95.56,
      "status": "fail"
    }
  },
  "overall": {
    "total_checks": 180,
    "passed": 178,
    "pass_percentage": 98.89
  }
}
```

## Performance

### Ingestion Speed

- **Throughput**: ~10,000 utterances/second (normalization)
- **Parquet Write**: ~5,000 rows/second with compression

### Embedding Generation

- **Local (MiniLM)**: ~100 texts/second (CPU)
- **Local (GPU)**: ~1,000 texts/second (CUDA)
- **OpenAI API**: ~500 texts/second (with batching)

### Search Performance

- **FAISS Index Build**: ~10,000 vectors/second
- **Search Latency**: <10ms for top-10 (100k vectors)
- **Index Size**: ~1.5GB per 1M vectors (384-dim)

## Development

### Project Structure

```
transcription-lakehouse/
├── src/lakehouse/           # Main package
│   ├── aggregation/         # Span/beat/section generation
│   ├── catalogs/           # Metadata catalog generation
│   ├── cli/                # Command-line interface
│   ├── embeddings/         # Embedding generation
│   ├── indexing/           # FAISS index management
│   ├── ingestion/          # Reader, validator, normalizer
│   └── validation/         # Data quality checks
├── tests/                  # Test suite
├── config/                 # Configuration files
├── lakehouse/             # Data storage
└── input/                 # Source transcripts
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- **Formatter**: Black (line length: 100)
- **Linter**: Ruff
- **Type Hints**: Recommended
- **Docstrings**: Google style

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: Parquet file not found`

*Solution*: Run ingestion first: `lakehouse ingest input/transcripts/`

---

**Issue**: `ImportError: sentence-transformers not installed`

*Solution*: Install package: `pip install sentence-transformers`

---

**Issue**: `ValidationError: Schema mismatch`

*Solution*: Check your transcript format matches the expected schema. Use `--no-validate` to skip validation during testing.

---

**Issue**: `FAISS index not found`

*Solution*: Build index first: `lakehouse index build spans`

## Roadmap

- [ ] Web UI dashboard for visualization
- [ ] Real-time streaming ingestion
- [ ] Multi-modal support (audio, video)
- [ ] Distributed processing with Dask
- [ ] Cloud storage backends (S3, GCS)
- [ ] Advanced topic modeling
- [ ] Collaborative filtering recommendations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [PyArrow](https://arrow.apache.org/docs/python/) and [DuckDB](https://duckdb.org/)
- Embedding models from [Sentence Transformers](https://www.sbert.net/)
- Vector search powered by [FAISS](https://github.com/facebookresearch/faiss)
- CLI framework: [Click](https://click.palletsprojects.com/)

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/transcription-lakehouse/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/transcription-lakehouse/discussions)

---

**Built with ❤️ for podcast transcript analysis**

