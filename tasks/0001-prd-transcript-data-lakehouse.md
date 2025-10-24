# Product Requirements Document (PRD)
## Transcript Data Lakehouse

---

## 1. Introduction/Overview

This document outlines the requirements for building a **local lake-style data store** for LOS (Lord of Spirits) and Scholarly Warrior podcast transcripts. The system will provide deterministic, reproducible storage and query surfaces for raw and derived transcript artifacts, enabling downstream semantic processing, dataset generation, and vector search capabilities.

### Problem Statement
Currently, podcast transcript data lacks:
- A structured, reproducible storage layer for raw and derived artifacts
- Stable identifiers that enable idempotent processing
- Multi-level aggregation (utterances → spans → beats → sections) for semantic analysis
- Vector-ready representations for similarity search and retrieval
- Clear lineage and versioning of derived data products

### Solution
A local data lakehouse that ingests episode-level JSON transcripts, normalizes them into stable artifacts, produces hierarchical aggregations (spans, beats, sections), generates vector embeddings, and maintains catalogs for discovery—all with deterministic identifiers and append-only semantics.

---

## 2. Goals

1. **Deterministic Processing**: Same inputs and configuration produce identical outputs (stable IDs, checksums)
2. **Reproducible Artifacts**: All derived data can be regenerated from raw inputs with consistent results
3. **Hierarchical Aggregation**: Transform utterances into semantically meaningful units (spans, beats, sections)
4. **Vector-Ready Output**: Generate embeddings and ANN indices suitable for semantic search
5. **Incremental Operations**: Support adding new episodes without reprocessing existing ones
6. **Validation & Quality**: Enforce schemas, perform sanity checks, report data quality metrics
7. **Developer-Friendly**: Provide CLI tools and clear file organization for easy integration

---

## 3. User Stories

### Primary User: Data Scientist / ML Engineer

1. **As a data scientist**, I want to ingest a directory of raw transcript JSON files so that I can quickly build a structured dataset without manual preprocessing.

2. **As an ML engineer**, I want stable, content-derived identifiers for all artifacts so that re-running the pipeline doesn't create duplicate or inconsistent data.

3. **As a researcher**, I want to query transcripts at multiple granularities (utterance, span, beat, section) so that I can analyze content at the appropriate semantic level for my task.

4. **As a developer**, I want to generate vector embeddings for transcript segments so that I can build semantic search and retrieval systems.

5. **As a data engineer**, I want append-only, versioned artifacts so that I can track changes over time and ensure reproducibility.

6. **As an analyst**, I want catalogs of episodes, speakers, and schemas so that I can discover what data is available and understand its structure.

7. **As a system administrator**, I want validation reports after each run so that I can verify data quality and troubleshoot issues.

8. **As a team member**, I want to add new episodes incrementally so that I don't need to reprocess the entire dataset for every update.

---

## 4. Functional Requirements

### FR-1: Data Ingestion
1. **FR-1.1**: Accept input as episode-level JSON/JSONL files containing utterance metadata (episode_id, start, end, speaker, text)
2. **FR-1.2**: Validate input schema and reject/skip malformed records with clear error messages
3. **FR-1.3**: Support batch ingestion from a directory of transcript files
4. **FR-1.4**: Log warnings for skipped records but continue processing valid data

### FR-2: Normalization Layer
1. **FR-2.1**: Produce normalized utterance records with stable `episode_id` derived from content
2. **FR-2.2**: Store normalized utterances in Parquet format with enforced schema
3. **FR-2.3**: Preserve all original metadata fields (timestamps, speaker, text)
4. **FR-2.4**: Generate deterministic `utterance_id` based on episode_id + position + content hash

### FR-3: Span Generation
1. **FR-3.1**: Consolidate utterances into **spans**: single-speaker, contiguous ranges
2. **FR-3.2**: Generate stable `span_id` from content + speaker + position
3. **FR-3.3**: Maintain references to constituent utterances
4. **FR-3.4**: Store spans in Parquet with metadata: span_id, episode_id, speaker, start_time, end_time, text, utterance_ids

### FR-4: Beat Generation
1. **FR-4.1**: Aggregate spans into **beats**: meaning units based on semantic similarity and topic coherence
2. **FR-4.2**: Use embedding-based similarity or topic modeling to determine beat boundaries
3. **FR-4.3**: Generate stable `beat_id` from content + position + parent spans
4. **FR-4.4**: Store beats in Parquet with metadata: beat_id, episode_id, span_ids, start_time, end_time, text, topic_summary (optional)

### FR-5: Section Generation
1. **FR-5.1**: Consolidate beats into **sections**: 5-12 minute logical blocks (or longer if semantically appropriate)
2. **FR-5.2**: Use time-based and semantic boundaries to create sections
3. **FR-5.3**: Generate stable `section_id` from content + position + parent beats
4. **FR-5.4**: Store sections in Parquet with metadata: section_id, episode_id, beat_ids, start_time, end_time, duration_minutes, text

### FR-6: Vector Embeddings
1. **FR-6.1**: Generate vector embeddings for spans and beats using configurable embedding model
2. **FR-6.2**: Support self-hosted models (e.g., Llama3.1-8b-instruct) as primary, OpenAI API as fallback
3. **FR-6.3**: Store embeddings in format compatible with FAISS/HNSW indexing
4. **FR-6.4**: Maintain mapping between artifact IDs and embedding vectors

### FR-7: ANN Index
1. **FR-7.1**: Build FAISS index from span and beat embeddings
2. **FR-7.2**: Export ANN index artifact with metadata for reloading
3. **FR-7.3**: Support incremental index updates when new episodes are added
4. **FR-7.4**: Provide index configuration options (e.g., HNSW parameters, distance metrics)

### FR-8: Catalogs
1. **FR-8.1**: Maintain **episode catalog**: episode_id, title, date, duration, speaker_list, file_path
2. **FR-8.2**: Maintain **speaker catalog**: speaker_name, episode_count, total_utterances, total_duration
3. **FR-8.3**: Maintain **schema manifest**: artifact types, schemas, column descriptions, version info
4. **FR-8.4**: Store catalogs as both Parquet (for querying) and JSON (for human readability)

### FR-9: Command-Line Interface
1. **FR-9.1**: Provide `ingest` command to run end-to-end ingestion on a directory of raw transcripts
2. **FR-9.2**: Provide `materialize` command to generate derived artifacts (spans, beats, sections, embeddings)
3. **FR-9.3**: Support `--dry-run` flag for validation without writing outputs
4. **FR-9.4**: Support `--incremental` flag to process only new episodes
5. **FR-9.5**: Provide `validate` command to check data quality and report statistics
6. **FR-9.6**: Provide `catalog` command to display episode and speaker summaries

### FR-10: Validation & Reporting
1. **FR-10.1**: Enforce schemas at write time for all Parquet tables
2. **FR-10.2**: Perform sanity checks: non-empty tables, monotonic timestamps, valid ID references
3. **FR-10.3**: Generate validation report with: row counts, coverage percentages, failure descriptions
4. **FR-10.4**: Output validation report as JSON and human-readable text

---

## 5. Non-Functional Requirements

### NFR-1: Determinism
1. **NFR-1.1**: Identical inputs and configuration must produce identical IDs and outputs
2. **NFR-1.2**: Use content hashing and position-based algorithms for ID generation
3. **NFR-1.3**: Exclude non-deterministic metadata (e.g., run timestamps) from ID calculations

### NFR-2: Idempotence
1. **NFR-2.1**: Re-running ingestion on the same inputs must not create duplicate artifacts
2. **NFR-2.2**: Use append-only semantics; new runs create new version directories rather than mutating existing files
3. **NFR-2.3**: Support incremental processing: detect already-processed episodes and skip them

### NFR-3: Performance
1. **NFR-3.1**: Process a 2-hour podcast transcript (500+ utterances) in under 5 minutes (excluding embedding generation)
2. **NFR-3.2**: Use efficient columnar storage (Parquet) with compression
3. **NFR-3.3**: Leverage DuckDB for fast analytical queries on Parquet files

### NFR-4: Maintainability
1. **NFR-4.1**: Use clear, documented directory structure
2. **NFR-4.2**: Modular design: separate ingestion, aggregation, embedding, and indexing logic
3. **NFR-4.3**: Provide comprehensive logging at INFO and DEBUG levels

### NFR-5: Extensibility
1. **NFR-5.1**: Design aggregation logic (span/beat/section) to be pluggable
2. **NFR-5.2**: Support configurable embedding models via API or local inference
3. **NFR-5.3**: Allow custom validation rules and sanity checks

---

## 6. Non-Goals (Out of Scope)

1. **Distributed Processing**: This system is designed for local, single-machine use. Distributed or cloud-based processing is not in scope.
2. **Real-Time Ingestion**: The system is batch-oriented. Real-time streaming ingestion is not supported.
3. **UI/Web Interface**: No graphical user interface or web dashboard. CLI-only.
4. **Transcript Generation**: The system assumes transcripts are already available as JSON. Speech-to-text is out of scope.
5. **Advanced NLP Models**: While semantic aggregation is required, training custom NLP models is out of scope. Use pre-trained models or simple heuristics.
6. **Multi-Tenancy**: The system is designed for single-user or single-team use. User management and access control are not included.
7. **Automated Re-Transcription**: If source transcripts change, re-ingestion is manual. Automatic change detection and re-processing are not included.

---

## 7. Design Considerations

### Lake Layout (Versioned Directory Structure)

```
transcript-lakehouse/
├── raw/                          # Raw input transcripts (immutable)
│   └── {episode_id}.jsonl
├── normalized/                   # Normalized utterances
│   ├── v1/
│   │   └── utterances.parquet
│   └── v2/
│       └── utterances.parquet
├── spans/                        # Single-speaker contiguous segments
│   ├── v1/
│   │   └── spans.parquet
│   └── v2/
│       └── spans.parquet
├── beats/                        # Semantic meaning units
│   ├── v1/
│   │   └── beats.parquet
│   └── v2/
│       └── beats.parquet
├── sections/                     # 5-12 minute logical blocks
│   ├── v1/
│   │   └── sections.parquet
│   └── v2/
│       └── sections.parquet
├── embeddings/                   # Vector representations
│   ├── v1/
│   │   ├── span_embeddings.parquet
│   │   ├── beat_embeddings.parquet
│   │   └── metadata.json
│   └── v2/
│       ├── span_embeddings.parquet
│       ├── beat_embeddings.parquet
│       └── metadata.json
├── ann_index/                    # FAISS/HNSW indices
│   ├── v1/
│   │   ├── span_index.faiss
│   │   ├── beat_index.faiss
│   │   └── index_config.json
│   └── v2/
│       ├── span_index.faiss
│       ├── beat_index.faiss
│       └── index_config.json
├── catalogs/                     # Discovery and metadata
│   ├── episodes.parquet
│   ├── episodes.json
│   ├── speakers.parquet
│   ├── speakers.json
│   ├── schema_manifest.json
│   └── validation_reports/
│       └── {run_timestamp}.json
└── config/                       # Configuration files
    ├── aggregation_config.yaml
    ├── embedding_config.yaml
    └── validation_rules.yaml
```

### Data Schemas

**Utterances (normalized/)**
- `utterance_id`: STRING (deterministic hash)
- `episode_id`: STRING
- `start`: FLOAT (seconds)
- `end`: FLOAT (seconds)
- `speaker`: STRING
- `text`: STRING
- `duration`: FLOAT (computed)

**Spans (spans/)**
- `span_id`: STRING (deterministic hash)
- `episode_id`: STRING
- `speaker`: STRING
- `start_time`: FLOAT
- `end_time`: FLOAT
- `duration`: FLOAT
- `text`: STRING (concatenated utterances)
- `utterance_ids`: LIST<STRING>

**Beats (beats/)**
- `beat_id`: STRING
- `episode_id`: STRING
- `start_time`: FLOAT
- `end_time`: FLOAT
- `duration`: FLOAT
- `text`: STRING
- `span_ids`: LIST<STRING>
- `topic_label`: STRING (optional)

**Sections (sections/)**
- `section_id`: STRING
- `episode_id`: STRING
- `start_time`: FLOAT
- `end_time`: FLOAT
- `duration_minutes`: FLOAT
- `text`: STRING
- `beat_ids`: LIST<STRING>

**Embeddings (embeddings/)**
- `artifact_id`: STRING (span_id or beat_id)
- `artifact_type`: STRING ('span' or 'beat')
- `embedding`: ARRAY<FLOAT> (vector)
- `model_name`: STRING
- `model_version`: STRING

---

## 8. Technical Considerations

### Technology Stack Recommendations
- **Language**: Python (recommended) or language-agnostic approach
- **Storage**: Parquet files with DuckDB for querying
- **Embeddings**: Self-hosted models (Llama3.1-8b-instruct) with OpenAI API fallback
- **Vector Index**: FAISS for ANN search
- **CLI Framework**: Click or Typer (Python) or similar
- **Schema Validation**: PyArrow or Pandera (Python)

### Semantic Aggregation Strategy
For beat and section generation:
1. Start with simple heuristics (speaker changes, silence, time thresholds) for spans
2. Use embedding-based cosine similarity for beat boundaries
3. Apply sliding window with threshold-based splitting
4. Use topic modeling (LDA or BERTopic) as optional enhancement
5. Design as pluggable modules to allow alternative strategies

### Incremental Processing Logic
1. Maintain a registry of processed episode IDs (in catalogs or separate state file)
2. On `--incremental` runs, scan input directory and compare against registry
3. Process only new episodes not in registry
4. For embeddings and ANN index, support incremental addition
5. Regenerate catalogs to include new data

### Error Handling & Validation
- **Input Validation**: Check for required fields, valid timestamps, non-empty text
- **Processing Errors**: Log warnings, skip bad records, continue processing
- **Output Validation**: Check row counts, ID uniqueness, referential integrity
- **Graceful Degradation**: If embedding service fails, process without embeddings and log warning

### Configuration Files
- `aggregation_config.yaml`: Span/beat/section rules, thresholds, model parameters
- `embedding_config.yaml`: Model selection, API keys, batch sizes, fallback strategy
- `validation_rules.yaml`: Schema definitions, sanity check thresholds, required fields

---

## 9. Success Metrics

### Acceptance Criteria (Must-Have)
1. ✅ Running on a sample of 5+ episodes yields all artifact layers with non-zero counts
2. ✅ Re-running on the same inputs produces identical identifiers and checksums (excluding run metadata)
3. ✅ Catalogs enumerate all produced artifacts and reference them by stable IDs
4. ✅ Validation report shows: row counts, coverage %, no critical failures

### Performance Metrics (Nice-to-Have)
- Process 1 hour of transcript content in < 3 minutes (excluding embeddings)
- Generate embeddings for 100 spans in < 30 seconds (local) or < 10 seconds (API)
- FAISS index build for 10K vectors in < 5 seconds
- DuckDB queries on 50K utterances complete in < 1 second

### Quality Metrics
- 100% of valid input episodes successfully normalized
- <5% of utterances skipped due to malformed data
- 100% of utterances assigned to spans
- 100% of spans assigned to beats
- Sections cover 95%+ of episode duration

---

## 10. Open Questions

1. **Beat Boundary Algorithm**: What specific semantic similarity threshold should be used for beat segmentation? (Recommend starting with 0.7 cosine similarity and tuning empirically)

2. **Section Duration Flexibility**: Should sections strictly stay within 5-12 minutes, or can they extend longer if semantic boundaries require it? (Current spec says "or longer if needed"—clarify priority)

3. **Embedding Model Selection**: Which specific Llama3.1 model variant should be used? Any quantization requirements? (Clarify model size: 8B vs 70B, INT8 vs FP16)

4. **Incremental Index Updates**: Should FAISS index be rebuilt from scratch or incrementally updated? (Recommend full rebuild for small datasets, incremental for large)

5. **Multi-Speaker Spans**: Should spans ever contain multiple speakers (e.g., rapid back-and-forth), or always break on speaker change? (Current spec says single-speaker)

6. **Checksums for Verification**: Should the system generate checksums (MD5, SHA256) for all Parquet files to verify reproducibility? (Recommend yes for determinism validation)

7. **Run Metadata**: Where should run metadata (timestamp, version, config hash) be stored without affecting determinism? (Recommend separate `_metadata.json` files per version directory)

8. **DuckDB Integration**: Should DuckDB be embedded in the CLI for query commands, or just documented as external tool? (Recommend light integration for `catalog` and `validate` commands)

9. **Version Management**: How should version directories be named and managed? (Recommend sequential: v1, v2, v3 or timestamp-based)

10. **Testing Strategy**: What level of test coverage is expected? Unit tests? Integration tests? (Recommend minimum: unit tests for ID generation, integration tests for full pipeline)

---

## Appendix: Sample Input Format

Based on the provided sample file `LOS - #002 - 2020-09-11 - Angels and Demons - Introducing Lord of Spirits.jsonl`:

```json
{
  "episode_id": "LOS - #002 - 2020-09-11 - Angels and Demons - Introducing Lord of Spirits",
  "start": 0.8,
  "end": 65.45,
  "speaker": "Steven Cristoforo",
  "text": "He will be a staff for the righteous..."
}
```

**Fields:**
- `episode_id`: STRING - Unique episode identifier (should be stable and descriptive)
- `start`: FLOAT - Start time in seconds
- `end`: FLOAT - End time in seconds
- `speaker`: STRING - Speaker name
- `text`: STRING - Utterance text

---

## Document Version

- **Version**: 1.0
- **Date**: October 23, 2025
- **Author**: AI Development Assistant
- **Status**: Ready for Review and Implementation

