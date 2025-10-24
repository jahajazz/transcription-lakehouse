# Task List: Transcript Data Lakehouse

Generated from: `0001-prd-transcript-data-lakehouse.md`

## Current State Assessment

This is a greenfield project with no existing infrastructure. The workspace contains:
- Sample transcript file: `LOS - #002 - 2020-09-11 - Angels and Demons - Introducing Lord of Spirits.jsonl`
- PRD documentation outlining requirements
- No existing Python code, dependencies, or project structure

The implementation will build a complete data lakehouse system from scratch using Python, DuckDB, Parquet, FAISS, and standard data science libraries.

---

## Relevant Files

### Core Application Files
- `src/lakehouse/__init__.py` - Main package initialization
- `src/lakehouse/config.py` - Configuration loading and validation
- `src/lakehouse/logger.py` - Centralized logging setup
- `src/lakehouse/ids.py` - Deterministic ID generation utilities
- `src/lakehouse/schemas.py` - Parquet schema definitions for all artifact types

### Ingestion and Normalization
- `src/lakehouse/ingestion/reader.py` - Raw transcript file reading and parsing
- `src/lakehouse/ingestion/validator.py` - Input schema validation
- `src/lakehouse/ingestion/normalizer.py` - Utterance normalization and ID generation
- `src/lakehouse/ingestion/writer.py` - Parquet writing utilities

### Aggregation
- `src/lakehouse/aggregation/spans.py` - Span generation logic
- `src/lakehouse/aggregation/beats.py` - Beat generation using semantic similarity
- `src/lakehouse/aggregation/sections.py` - Section generation with time/semantic boundaries
- `src/lakehouse/aggregation/base.py` - Base classes and interfaces for aggregation

### Embeddings and Indexing
- `src/lakehouse/embeddings/generator.py` - Embedding generation with model abstraction
- `src/lakehouse/embeddings/models.py` - Model wrappers (local, OpenAI)
- `src/lakehouse/embeddings/storage.py` - Embedding storage in Parquet
- `src/lakehouse/indexing/faiss_builder.py` - FAISS index construction and updates
- `src/lakehouse/indexing/incremental.py` - Incremental index update logic

### Catalogs and Validation
- `src/lakehouse/catalogs/episodes.py` - Episode catalog generation
- `src/lakehouse/catalogs/speakers.py` - Speaker catalog generation
- `src/lakehouse/catalogs/schema_manifest.py` - Schema manifest generation
- `src/lakehouse/validation/checks.py` - Sanity checks and validation rules
- `src/lakehouse/validation/reporter.py` - Validation report generation

### CLI
- `src/lakehouse/cli/__init__.py` - CLI entry point
- `src/lakehouse/cli/commands/ingest.py` - Ingest command implementation
- `src/lakehouse/cli/commands/materialize.py` - Materialize command implementation
- `src/lakehouse/cli/commands/validate.py` - Validate command implementation
- `src/lakehouse/cli/commands/catalog.py` - Catalog command implementation

### Configuration Files
- `config/aggregation_config.yaml` - Span/beat/section rules and thresholds
- `config/embedding_config.yaml` - Model selection and parameters
- `config/validation_rules.yaml` - Schema definitions and validation thresholds

### Tests
- `tests/test_ids.py` - Unit tests for ID generation (determinism validation)
- `tests/test_ingestion.py` - Tests for ingestion and normalization
- `tests/test_aggregation.py` - Tests for span/beat/section generation
- `tests/test_embeddings.py` - Tests for embedding generation
- `tests/test_validation.py` - Tests for validation logic
- `tests/integration/test_pipeline.py` - End-to-end pipeline integration tests
- `tests/fixtures/sample_transcript.jsonl` - Test fixture data

### Project Files
- `requirements.txt` - Python dependencies
- `setup.py` or `pyproject.toml` - Package configuration
- `README.md` - Usage documentation and examples
- `.gitignore` - Git ignore patterns

### Notes
- All source code will be under `src/lakehouse/` for clean package structure
- Tests will use pytest framework
- Configuration files use YAML for human readability
- The lakehouse directory structure will be created at runtime based on user-specified path

---

## Tasks

- [x] 1.0 Project Setup and Foundation
  - [x] 1.1 Create project directory structure (`src/lakehouse/`, `config/`, `tests/`, etc.)
  - [x] 1.2 Create `requirements.txt` with dependencies: pyarrow, pandas, duckdb, faiss-cpu, pyyaml, click, sentence-transformers, openai, pytest, pytest-cov
  - [x] 1.3 Create `pyproject.toml` or `setup.py` for package configuration with entry points for CLI
  - [x] 1.4 Create `.gitignore` with Python, virtual environment, and data artifact patterns
  - [x] 1.5 Implement `src/lakehouse/logger.py` with configurable logging (INFO/DEBUG levels, file and console handlers)
  - [x] 1.6 Implement `src/lakehouse/ids.py` with deterministic ID generation functions (content hashing using SHA256, position-based IDs)
  - [x] 1.7 Create configuration templates in `config/`: `aggregation_config.yaml`, `embedding_config.yaml`, `validation_rules.yaml`
  - [x] 1.8 Implement `src/lakehouse/config.py` to load and validate YAML configuration files with sensible defaults

- [x] 2.0 Data Ingestion and Normalization Layer
  - [x] 2.1 Implement `src/lakehouse/schemas.py` with PyArrow schemas for utterances, spans, beats, sections, and embeddings
  - [x] 2.2 Implement `src/lakehouse/ingestion/reader.py` to read JSON/JSONL transcript files with error handling
  - [x] 2.3 Implement `src/lakehouse/ingestion/validator.py` to validate input schema (required fields: episode_id, start, end, speaker, text)
  - [x] 2.4 Implement `src/lakehouse/ingestion/normalizer.py` to generate deterministic `utterance_id` from episode_id + position + content hash
  - [x] 2.5 Implement `src/lakehouse/ingestion/writer.py` with Parquet writing utilities using PyArrow (compression, schema enforcement)
  - [x] 2.6 Create directory structure generator to set up versioned lakehouse layout (raw/, normalized/v1/, etc.)
  - [x] 2.7 Implement end-to-end ingestion pipeline: read → validate → normalize → write to Parquet
  - [x] 2.8 Add logging for skipped/malformed records with clear error messages and continue processing valid data

- [x] 3.0 Hierarchical Aggregation (Spans, Beats, Sections)
  - [x] 3.1 Implement `src/lakehouse/aggregation/base.py` with abstract base class for aggregation strategies
  - [x] 3.2 Implement `src/lakehouse/aggregation/spans.py` to consolidate single-speaker contiguous utterances with stable `span_id`
  - [x] 3.3 In spans module, maintain references to constituent `utterance_ids` and compute duration/timestamps
  - [x] 3.4 Implement `src/lakehouse/aggregation/beats.py` with embedding-based semantic similarity for beat boundaries
  - [x] 3.5 In beats module, use configurable similarity threshold (default 0.7 cosine similarity) from `aggregation_config.yaml`
  - [x] 3.6 Generate stable `beat_id` from content hash + position + parent span IDs
  - [x] 3.7 Implement `src/lakehouse/aggregation/sections.py` to create 5-12 minute logical blocks using time and semantic boundaries
  - [x] 3.8 Generate stable `section_id` from content hash + position + parent beat IDs
  - [x] 3.9 Store all aggregation artifacts (spans, beats, sections) in versioned Parquet files with full metadata
  - [x] 3.10 Ensure all aggregations maintain referential integrity (utterance_ids in spans, span_ids in beats, beat_ids in sections)

- [ ] 4.0 Vector Embeddings and ANN Indexing
  - [x] 4.1 Implement `src/lakehouse/embeddings/models.py` with abstract model interface for embedding generation
  - [x] 4.2 In models module, create wrapper for local sentence-transformers model (default: all-MiniLM-L6-v2 or similar)
  - [x] 4.3 In models module, create wrapper for OpenAI embeddings API as fallback option
  - [x] 4.4 Implement `src/lakehouse/embeddings/generator.py` to generate embeddings for spans and beats with batch processing
  - [x] 4.5 Add configuration loading from `embedding_config.yaml` (model selection, API keys, batch sizes)
  - [x] 4.6 Implement `src/lakehouse/embeddings/storage.py` to store embeddings in Parquet with artifact_id, artifact_type, embedding vector, model metadata
  - [x] 4.7 Implement `src/lakehouse/indexing/faiss_builder.py` to build FAISS HNSW indices from span and beat embeddings
  - [x] 4.8 Add FAISS index configuration options (M parameter, efConstruction, distance metric) to config
  - [x] 4.9 Implement `src/lakehouse/indexing/incremental.py` for incremental index updates when new episodes are added
  - [x] 4.10 Export FAISS indices with metadata JSON files (index_config.json) for reloading

- [ ] 5.0 CLI Interface and Commands
  - [ ] 5.1 Set up Click/Typer CLI framework in `src/lakehouse/cli/__init__.py` with main entry point
  - [ ] 5.2 Implement `src/lakehouse/cli/commands/ingest.py` with `ingest` command to run end-to-end ingestion on a directory
  - [ ] 5.3 Add `--dry-run` flag to ingest command for validation without writing outputs
  - [ ] 5.4 Add `--incremental` flag to ingest command to process only new episodes not in catalog
  - [ ] 5.5 Implement `src/lakehouse/cli/commands/materialize.py` with `materialize` command to generate derived artifacts (spans, beats, sections, embeddings, indices)
  - [ ] 5.6 Add artifact selection flags to materialize command (--spans-only, --beats-only, --embeddings, --all)
  - [ ] 5.7 Implement `src/lakehouse/cli/commands/validate.py` with `validate` command to check data quality and report statistics
  - [ ] 5.8 Implement `src/lakehouse/cli/commands/catalog.py` with `catalog` command to display episode and speaker summaries using DuckDB queries
  - [ ] 5.9 Add common CLI options: --lakehouse-path, --config-dir, --log-level, --version
  - [ ] 5.10 Implement rich console output with progress bars and formatted tables for better UX

- [ ] 6.0 Validation, Catalogs, and Reporting
  - [ ] 6.1 Implement `src/lakehouse/validation/checks.py` with sanity check functions (non-empty tables, monotonic timestamps, valid ID references)
  - [ ] 6.2 Add schema enforcement at Parquet write time using PyArrow schema validation
  - [ ] 6.3 Implement `src/lakehouse/validation/reporter.py` to generate validation reports with row counts, coverage percentages, failure descriptions
  - [ ] 6.4 Output validation reports as both JSON and human-readable text formats
  - [ ] 6.5 Implement `src/lakehouse/catalogs/episodes.py` to generate episode catalog with episode_id, title, date, duration, speaker_list, file_path
  - [ ] 6.6 Implement `src/lakehouse/catalogs/speakers.py` to generate speaker catalog with speaker_name, episode_count, total_utterances, total_duration using DuckDB aggregations
  - [ ] 6.7 Implement `src/lakehouse/catalogs/schema_manifest.py` to generate schema manifest with artifact types, schemas, column descriptions, version info
  - [ ] 6.8 Store catalogs as both Parquet (for querying) and JSON (for human readability) in `catalogs/` directory
  - [ ] 6.9 Store validation reports with timestamps in `catalogs/validation_reports/{run_timestamp}.json`
  - [ ] 6.10 Add catalog regeneration logic to update catalogs incrementally when new episodes are added

- [ ] 7.0 Testing and Documentation
  - [ ] 7.1 Create `tests/fixtures/sample_transcript.jsonl` with realistic test data (5-10 utterances from different speakers)
  - [ ] 7.2 Implement `tests/test_ids.py` with unit tests for ID generation determinism (same input → same ID, different input → different ID)
  - [ ] 7.3 Implement `tests/test_ingestion.py` with tests for reader, validator, normalizer, and writer modules
  - [ ] 7.4 Implement `tests/test_aggregation.py` with tests for span, beat, and section generation logic
  - [ ] 7.5 Implement `tests/test_embeddings.py` with tests for embedding generation (mock model to avoid API calls)
  - [ ] 7.6 Implement `tests/test_validation.py` with tests for sanity checks and validation report generation
  - [ ] 7.7 Implement `tests/integration/test_pipeline.py` with end-to-end pipeline test using sample transcript
  - [ ] 7.8 Set up pytest configuration in `pyproject.toml` or `pytest.ini` with coverage reporting
  - [ ] 7.9 Create comprehensive `README.md` with installation instructions, usage examples, CLI command reference, and architecture overview
  - [ ] 7.10 Create example configuration files with inline comments explaining all parameters and sensible defaults
  - [ ] 7.11 Add docstrings to all public functions and classes following Google or NumPy docstring conventions
  - [ ] 7.12 Create `CONTRIBUTING.md` with development setup instructions and contribution guidelines

---

## Implementation Notes

### Recommended Development Order

The tasks are structured to support incremental development and testing:

1. **Start with Foundation (1.0)**: Set up the project structure, dependencies, and core utilities first. The ID generation and logging modules are critical dependencies for all other components.

2. **Build Data Pipeline (2.0, 3.0)**: Implement ingestion and aggregation layers sequentially. Each layer depends on the previous one, so test thoroughly before moving on.

3. **Add Intelligence (4.0)**: Once you have clean data artifacts, add embedding and indexing capabilities. This is computationally expensive, so consider implementing with small test datasets first.

4. **Create Interface (5.0)**: With the core functionality complete, wrap it in a user-friendly CLI. This makes testing and iteration much easier.

5. **Ensure Quality (6.0, 7.0)**: Add validation, catalogs, comprehensive tests, and documentation throughout, not just at the end.

### Key Design Principles

- **Determinism**: All ID generation must be deterministic. Use the same hashing algorithm (SHA256) and content ordering consistently.
- **Idempotence**: Running the same command with the same inputs should produce identical results. Use version directories to avoid overwriting.
- **Incremental Processing**: Track processed episodes in catalogs to support `--incremental` flag efficiently.
- **Fail Gracefully**: Log warnings and skip bad records rather than failing the entire pipeline.
- **Modularity**: Keep aggregation strategies, embedding models, and validation rules pluggable for future extensibility.

### Testing Strategy

- **Unit Tests**: Test individual functions, especially ID generation, validation, and aggregation logic.
- **Integration Tests**: Test the full pipeline with sample data to ensure components work together correctly.
- **Determinism Tests**: Verify that running the pipeline twice produces identical IDs and checksums.
- **Edge Cases**: Test with malformed data, empty files, single-utterance episodes, and missing fields.

### Performance Considerations

- Use Parquet columnar format with compression (snappy or gzip) for efficient storage.
- Batch embedding generation to leverage model inference optimization.
- Use DuckDB for fast analytical queries on Parquet files instead of loading everything into memory.
- Consider lazy evaluation for large datasets (process in chunks).

### Configuration Management

All thresholds, model parameters, and processing rules should be externalized to YAML config files. This allows users to tune the system without modifying code. Provide sensible defaults for all configuration values.

### Lakehouse Directory Structure

The runtime lakehouse directory will look like this:
```
transcript-lakehouse/
├── raw/                    # Raw input transcripts (immutable)
├── normalized/v1/          # Normalized utterances (versioned)
├── spans/v1/              # Aggregated spans (versioned)
├── beats/v1/              # Semantic beats (versioned)
├── sections/v1/           # Logical sections (versioned)
├── embeddings/v1/         # Vector embeddings (versioned)
├── ann_index/v1/          # FAISS indices (versioned)
├── catalogs/              # Discovery and metadata (non-versioned, regenerated)
│   └── validation_reports/
└── config/                # User configuration overrides (optional)
```

Version directories (v1, v2, v3...) allow for reproducibility and experimentation without destroying previous results.

---

## Getting Started

Once you're ready to begin implementation, start with task **1.1** and work sequentially through the sub-tasks. Each sub-task is designed to be relatively self-contained and testable.

For your first implementation, consider using the provided sample transcript file (`LOS - #002 - 2020-09-11 - Angels and Demons - Introducing Lord of Spirits.jsonl`) as your test case throughout development.