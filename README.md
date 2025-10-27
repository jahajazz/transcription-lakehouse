# Transcript Lakehouse

**A local data lakehouse for podcast transcript storage, processing, and semantic analysis.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-121%20snapshot%20tests%20passed-success)]()

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

### Complete Workflow (Start to Finish)

Follow these steps to build a lakehouse from scratch:

#### **1. Ingest Transcripts** (Required)

Ingest transcript files into the lakehouse. **Requires a directory path**, not a single file.

```bash
# Initial ingestion - process all transcripts in directory
lakehouse ingest input/transcripts/

# Incremental ingestion - process only NEW episodes (skip existing)
# Use --incremental for subsequent runs to avoid re-processing
lakehouse ingest input/transcripts/ --incremental

# Auto-update catalogs after ingestion (optional but recommended)
lakehouse ingest input/transcripts/ --incremental --update-catalog
```

**Options:**
- `--incremental` - Skip episodes already in lakehouse (recommended for updates)
- `--update-catalog` - Regenerate episode/speaker catalogs after ingestion
- `--pattern "*.jsonl"` - Filter files by pattern (default: *.jsonl)
- `--dry-run` - Validate inputs without writing

---

#### **2. Materialize Derived Artifacts** (Required)

Generate aggregations, embeddings, and search indices:

```bash
# Generate all artifacts (recommended)
lakehouse materialize --all
```

This creates:
- **Spans**: Speaker-contiguous utterance groups
- **Beats**: Semantic meaning units
- **Sections**: 5-12 minute episode segments
- **Embeddings**: Vector representations for spans and beats
- **Indices**: FAISS ANN indices for semantic search

**Advanced:** Generate specific artifacts only:
```bash
lakehouse materialize --spans-only --beats-only
lakehouse materialize --embeddings-only --indices-only
```

---

#### **3. Validate Data** (Recommended)

Run basic data integrity checks:

```bash
# Quick validation check
lakehouse validate

# Detailed validation with report
lakehouse validate --save-report --detailed
```

Validates: schema compliance, timestamps, IDs, referential integrity, text quality.

---

#### **4. Generate Catalogs** (Recommended)

Create metadata catalogs for episodes, speakers, and schemas:

```bash
# Generate all catalogs (default)
lakehouse catalog

# Save catalogs to files
lakehouse catalog --save-catalog

# View specific catalog
lakehouse catalog --catalog-type episodes --detailed
```

**Note:** If you used `--update-catalog` during ingestion, this step is optional.

---

#### **5. Run Quality Assessment** (Recommended)

Comprehensive quality analysis with detailed reporting:

```bash
# Full quality assessment
lakehouse quality

# Custom thresholds
lakehouse quality --coverage-min 95.0 --sample-size 500

# Save detailed report
lakehouse quality --output-dir quality_reports/
```

Quality checks include:
- Data integrity validation
- Embedding quality metrics
- Duplicate detection (exact and near-duplicates)
- Statistical analysis
- Length distributions and coverage

**Note:** Near-duplicate detection automatically skips for datasets >10,000 segments (performance).

---

#### **6. Create Snapshot** (Required for sharing/migration)

Create versioned, immutable snapshot for external consumption:

```bash
# Create snapshot with auto-versioning
lakehouse snapshot create

# Create with specific version
lakehouse snapshot create --version 1.0.0-provisional

# Custom snapshot location
lakehouse snapshot create --snapshot-root /data/snapshots
```

**Configure snapshot paths** in `config/snapshot_config.yaml`:

```yaml
# Producer settings - where to CREATE snapshots
snapshot_root: "${SNAPSHOT_ROOT:-./snapshots}"
# Can also use fixed path: "D:/lakehouse/snapshots"

# Consumer settings - where to READ snapshots from
lake_root: "${LAKE_ROOT}"
# Can also use fixed path: "D:/lakehouse/snapshots/v1.0.0-provisional"
```

**Snapshot includes:**
- ✅ All consumer-facing artifacts (spans, beats, sections, embeddings, indexes, catalogs)
- ✅ Comprehensive `lake_manifest.json` with checksums and metadata
- ✅ QA status tracking (PASS/FAIL/UNKNOWN)
- ✅ Automatic validation and integrity checks
- ✅ Platform-specific usage instructions in `snapshot_note.txt`

**Environment Variables (Alternative):**
```bash
# Set snapshot creation location
export SNAPSHOT_ROOT=/data/snapshots

# Set snapshot consumption location (for consumers in other repos)
export LAKE_ROOT=/data/snapshots/v1.0.0-provisional
```

See `docs/snapshot-setup.md` for complete configuration guide.

---

### Minimal Workflow (Regular Updates)

For incremental updates after initial setup:

```bash
# 1. Ingest new transcripts only
lakehouse ingest input/transcripts/ --incremental --update-catalog

# 2. Generate all artifacts
lakehouse materialize --all

# 3. Quality check
lakehouse quality

# 4. Create snapshot
lakehouse snapshot create
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

#### `lakehouse materialize`
Generate derived artifacts (aggregations, embeddings, and indices).

```bash
lakehouse materialize [OPTIONS]

Options:
  --version TEXT          Version identifier [default: v1]
  --spans-only            Generate only spans
  --beats-only            Generate only beats (requires spans)
  --sections-only         Generate only sections (requires beats)
  --embeddings-only       Generate only embeddings (requires spans/beats)
  --indices-only          Build only FAISS indices (requires embeddings)
  --all                   Generate all artifacts (default if no flags)
  --lakehouse-path PATH   Path to lakehouse directory [default: ./lakehouse]
  --config-dir PATH       Path to config directory [default: ./config]
  --log-level TEXT        Logging level [default: INFO]
  --help                  Show this message and exit
```

The `materialize` command runs in sequence:
1. **Spans** - Speaker-contiguous utterance groups
2. **Beats** - Semantic meaning units (heuristic or embedding-based)
3. **Sections** - 5-12 minute episode segments
4. **Embeddings** - Vector representations (spans and beats)
5. **Indices** - FAISS ANN indices for semantic search

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
Generate and display metadata catalogs using DuckDB queries.

```bash
lakehouse catalog [OPTIONS]

Options:
  --version TEXT               Version identifier [default: v1]
  --catalog-type TEXT          Catalog type [episodes|speakers|schema|all] [default: all]
  --output-format TEXT         Output format [console|json|text] [default: console]
  --save-catalog               Save catalog to files
  --output-dir DIRECTORY       Directory to save catalogs [default: lakehouse/catalogs]
  --detailed                   Show detailed catalog information
  --episode-id TEXT            Show details for specific episode
  --speaker-name TEXT          Show details for specific speaker
  --rankings                   Show speaker rankings by activity
  --statistics                 Show overall statistics
  --lakehouse-path PATH        Path to lakehouse directory [default: ./lakehouse]
  --config-dir PATH            Path to config directory [default: ./config]
  --log-level TEXT             Logging level [default: INFO]
  --help                       Show this message and exit
```

#### `lakehouse quality`
Run comprehensive quality assessment on lakehouse data.

```bash
lakehouse quality [OPTIONS]

Options:
  --lakehouse-path PATH        Path to lakehouse directory [default: ./lakehouse]
  --config-dir PATH            Path to config directory [default: ./config]
  --log-level TEXT             Logging level [default: INFO]
  --level TEXT                 Assessment level [spans|beats|all] [default: all]
  --output-dir PATH            Directory to save report [default: ./quality_reports]
  --no-timestamp               Don't add timestamp to report directory
  --config TEXT                Path to custom thresholds config
  --coverage-min FLOAT         Minimum coverage threshold
  --duplicate-max FLOAT        Maximum duplicate rate threshold
  --sample-size INT            Number of samples for embedding analysis
  --neighbor-k INT             Number of neighbors for similarity checks
  --help                       Show this message and exit
```

Quality assessment includes:
- **Data Integrity**: ID uniqueness, referential integrity, temporal ordering
- **Embedding Quality**: Dimensionality, null rates, neighbor similarity
- **Statistical Analysis**: Length distributions, coverage metrics
- **Duplicate Detection**: Exact and near-duplicate identification

#### `lakehouse snapshot create`
Create versioned, immutable snapshots of lakehouse artifacts.

```bash
lakehouse snapshot create [OPTIONS]

Options:
  --lakehouse-path PATH        Path to lakehouse directory [default: ./lakehouse]
  --config-dir PATH            Path to config directory [default: ./config]
  --log-level TEXT             Logging level [default: INFO]
  --snapshot-root PATH         Snapshot root directory [default: ./snapshots or $SNAPSHOT_ROOT]
  --version TEXT               Override snapshot version (default: auto-increment from config)
  --lakehouse-version TEXT     Lakehouse artifact version to snapshot [default: v1]
  --help                       Show this message and exit
```

Snapshots include:
- All consumer-facing artifacts (spans, beats, sections, embeddings, indexes, catalogs)
- Comprehensive `lake_manifest.json` with checksums and metadata
- QA status tracking (PASS/FAIL/UNKNOWN)
- Automatic validation and integrity checks
- Platform-specific usage instructions in `snapshot_note.txt`

Configure snapshot locations in `config/snapshot_config.yaml`:
```yaml
snapshot_root: "${SNAPSHOT_ROOT:-./snapshots}"  # Where to create
lake_root: "${LAKE_ROOT}"                        # Where to read from
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

### `quality_thresholds.yaml`

```yaml
# Quality assessment thresholds
thresholds:
  # Coverage thresholds
  coverage:
    min_percentage: 95.0  # Minimum artifact coverage
  
  # Duplicate detection
  duplicates:
    max_exact_rate: 0.01      # Max 1% exact duplicates
    max_near_rate: 0.05       # Max 5% near duplicates
    similarity_threshold: 0.95
  
  # Length validation
  length:
    span_min: 5    # Minimum span length (words)
    span_max: 100  # Maximum span length (words)
    beat_min: 10
    beat_max: 500
  
  # Embedding quality
  embeddings:
    min_neighbor_similarity: 0.3  # Minimum avg similarity to neighbors
    max_null_rate: 0.0            # Maximum null embedding rate
```

### `snapshot_config.yaml`

```yaml
# Snapshot version configuration
version:
  major: 0
  minor: 1
  patch: 0

# Producer settings - where to create snapshots
snapshot_root: "${SNAPSHOT_ROOT:-./snapshots}"
auto_increment: true
provisional_suffix: "-provisional"

# Consumer settings - where to read snapshots from
lake_root: "${LAKE_ROOT}"

# Examples:
# Fixed paths:
#   snapshot_root: "D:/lakehouse/snapshots"
#   lake_root: "D:/lakehouse/snapshots/v0.9.0-provisional"
#
# Environment variables (recommended):
#   snapshot_root: "${SNAPSHOT_ROOT:-./snapshots}"
#   lake_root: "${LAKE_ROOT}"
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

### Quality Assessment

The lakehouse includes a comprehensive quality assessment system that evaluates spans and beats across multiple quality dimensions before downstream processing.

#### Running Quality Assessment

```bash
# Assess both spans and beats with default thresholds
lakehouse quality

# Assess only spans or beats
lakehouse quality --level spans
lakehouse quality --level beats

# Use custom thresholds
lakehouse quality --coverage-min 90.0 --span-length-max 150.0

# Custom sample size for embedding analysis
lakehouse quality --sample-size 500 --neighbor-k 20

# Use custom configuration file
lakehouse quality --config custom_thresholds.yaml

# Disable timestamped output directories
lakehouse quality --no-timestamp --output-dir output/quality_report
```

#### Quality Metrics Categories

The quality assessment evaluates **7 categories** of metrics:

**Category A: Coverage & Count Metrics**
- Episode coverage percentage (target: ≥95%)
- Gap detection and quantification
- Overlap detection and quantification
- Segment counts per episode

**Category B: Length & Distribution**
- Duration statistics (min/max/mean/median/p5/p95)
- Length compliance (spans: 20-120s, beats: 60-180s)
- Distribution histograms
- Outlier detection (too short/too long)

**Category C: Ordering & Integrity**
- Timestamp monotonicity validation
- Negative/zero duration detection
- Missing field detection
- Duplicate detection (exact and near-duplicates)

**Category D: Speaker & Series Balance**
- Speaker distribution analysis
- Per-speaker duration statistics
- Series balance (if multiple series present)
- Top speakers identification

**Category E: Text Quality Proxies**
- Token and word count statistics
- Lexical density calculation
- Punctuation ratios
- Top unigrams and bigrams extraction

**Category F: Embedding Sanity Checks**
- Speaker leakage detection (target: ≤60% same speaker in neighbors)
- Episode leakage detection (target: ≤70% same episode in neighbors)
- Length bias correlation (target: ≤0.3)
- Adjacency bias detection (target: ≤40% temporally adjacent neighbors)
- Neighbor coherence analysis

**Category G: Diagnostics & Outliers**
- Longest/shortest segments identification
- Most isolated/hubby segments (based on embeddings)
- Neighbor list sampling for manual review
- Diagnostic CSV exports

#### Output Files

Quality assessment generates comprehensive reports in `output/quality/YYYYMMDD_HHMMSS/`:

```
output/quality/20251026_143022/
├── metrics/
│   ├── global.json           # Global metrics and summary
│   ├── episodes.csv          # Per-episode metrics
│   ├── spans.csv             # Per-span metrics with validation flags
│   └── beats.csv             # Per-beat metrics with validation flags
├── diagnostics/
│   ├── outliers.csv          # Top outliers by category
│   └── neighbors_sample.csv  # Sample neighbor lists for review
└── report/
    └── quality_assessment.md # Human-readable markdown report
```

#### RAG Status

Each assessment receives a RAG (Red/Amber/Green) status:

- 🟢 **GREEN**: All thresholds passed, no critical issues
- 🟠 **AMBER**: Minor threshold violations (1-2 non-critical warnings)
- 🔴 **RED**: Multiple threshold failures or critical integrity issues

#### Example Report Excerpt

```markdown
# Quality Assessment Report

## Executive Summary

**RAG Status**: 🟢 GREEN
**Assessment Date**: 2025-10-26 14:30:22
**Lakehouse Version**: v1
**Episodes Assessed**: 539
**Spans Assessed**: 45,231
**Beats Assessed**: 12,108

**Summary**: All quality thresholds passed. Data is ready for downstream processing.

## Coverage Metrics

- Global Span Coverage: 96.8%
- Global Beat Coverage: 97.2%
- Episodes with <95% coverage: 12 (2.2%)
- Average gap percentage: 0.8%
- Average overlap percentage: 0.5%

## Length Distribution

Span Duration Statistics:
- Mean: 62.3s (target: 20-120s)
- Median: 58.1s
- 95th percentile: 115.2s
- Within bounds: 92.4% ✓ (target: ≥90%)

## Embedding Sanity Checks

- Speaker leakage: 42.3% ✓ (target: ≤60%)
- Episode leakage: 58.7% ✓ (target: ≤70%)
- Length bias correlation: 0.18 ✓ (target: ≤0.3)
- Adjacency bias: 28.5% ✓ (target: ≤40%)

## Go/No-Go Recommendation

**✓ GO** - Proceed with downstream processing (pair generation, labeling, fine-tuning)
```

#### Configurable Thresholds

Create `config/quality_thresholds.yaml` to customize thresholds:

```yaml
coverage:
  coverage_min: 95.0           # Minimum coverage %
  gap_max_percent: 2.0         # Maximum gap %
  overlap_max_percent: 2.0     # Maximum overlap %

length:
  span_length_min: 20.0        # Minimum span length (seconds)
  span_length_max: 120.0       # Maximum span length
  span_length_compliance_min: 90.0  # Min % within bounds
  beat_length_min: 60.0        # Minimum beat length
  beat_length_max: 180.0       # Maximum beat length
  beat_length_compliance_min: 90.0  # Min % within bounds

integrity:
  timestamp_regressions_max: 0
  negative_duration_max: 0
  exact_duplicate_max_percent: 1.0
  near_duplicate_max_percent: 3.0
  near_duplicate_threshold: 0.95

embedding:
  same_speaker_neighbor_max_percent: 60.0
  same_episode_neighbor_max_percent: 70.0
  length_bias_correlation_max: 0.3
  adjacency_bias_max_percent: 40.0

sampling:
  neighbor_sample_size: 100    # Segments to sample for analysis
  neighbor_k: 10               # Top-k neighbors to retrieve
  random_seed: 42              # For reproducibility
```

#### Integration with Pipeline

Run quality assessment before expensive downstream operations:

```bash
# Standard pipeline with quality gate
lakehouse ingest input/transcripts/
lakehouse materialize --all

# Quality assessment (blocks on RED status)
lakehouse quality || exit 1

# If GREEN/AMBER, proceed with downstream tasks
./run_pair_generation.sh
./run_labeling_workflow.sh
```

#### Use Cases

1. **Pre-processing Validation**: Verify segmentation quality before generating embeddings
2. **Embedding Validation**: Check for speaker leakage and length bias in embeddings
3. **Parameter Tuning**: Use distribution metrics to tune segmentation parameters
4. **Debugging**: Inspect outliers and neighbor samples to debug unexpected behavior
5. **Quality Gates**: Implement automated quality gates in data pipelines
6. **Baseline Tracking**: Compare quality metrics across different data versions

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

*Solution*: Build indices first: `lakehouse materialize --indices-only` (or run full `lakehouse materialize --all`)

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

