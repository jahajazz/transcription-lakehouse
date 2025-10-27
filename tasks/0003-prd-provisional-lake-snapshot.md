# PRD: Provisional LAKE_ROOT Snapshot

## Introduction/Overview

This feature enables the creation of versioned, immutable snapshots of the lakehouse artifacts for consumption by external repositories or systems. A snapshot packages all consumer-facing artifacts (spans, beats, sections, embeddings, ANN indexes, catalogs, and QA reports) into a self-contained, versioned directory with a comprehensive manifest file that describes the contents, schemas, checksums, and quality status.

The snapshot is "provisional" meaning it can be created even if quality assessment (QA) has not been run or is failing. This allows rapid migration, testing, and sharing of lake data without blocking on quality gates. The manifest tracks the QA status (UNKNOWN, FAIL, or PASS) to inform consumers about data quality.

**Problem this solves:** Currently, there is no standardized way to package and share the lakehouse artifacts with external consumers. This makes it difficult to:
- Migrate lakehouse data to other environments (e.g., desktop, cloud)
- Ensure data integrity during transfer
- Track versioning and provenance of shared datasets
- Communicate quality status to downstream consumers

## Goals

1. **Immutability**: Create reproducible snapshots where re-creating from the same inputs yields identical checksums
2. **Discoverability**: Provide a comprehensive manifest that catalogs all artifacts with metadata (sizes, checksums, schemas, QA status)
3. **Integrity**: Ensure automated validation of snapshot structure and contents
4. **Traceability**: Track provenance (repo, commit, timestamp, QA report) for each snapshot
5. **Non-blocking**: Allow snapshot creation regardless of QA status (provisional snapshots)
6. **Consumer-ready**: Package only the artifacts needed by downstream consumers in a standardized format

## User Stories

1. **As a data engineer**, I want to create a snapshot of the lakehouse so that I can migrate the data to a desktop environment for local analysis.

2. **As a downstream consumer**, I want to validate a snapshot's integrity so that I can trust the data before using it.

3. **As a data steward**, I want to see the QA status in the snapshot manifest so that I know the quality level of the data I'm consuming.

4. **As a developer**, I want automatic versioning of snapshots so that I don't have to manually track version numbers and can avoid conflicts.

5. **As a repository maintainer**, I want immutable snapshots so that once shared, the data cannot be accidentally modified or corrupted.

6. **As a consumer**, I want a standardized manifest format so that I can programmatically discover and validate lakehouse snapshots.

## Functional Requirements

### Snapshot Creation

1. **CLI Command**: The system must provide a CLI command `lakehouse snapshot create` that creates a new snapshot.

2. **Automatic Versioning**: The system must automatically determine the next semantic version number (MAJOR.MINOR.PATCH) by reading from a config file and/or auto-incrementing from the last snapshot.

3. **Version Collision Handling**: If a snapshot with the calculated version already exists, the system must automatically increment to the next available version (e.g., if v1.0.0 exists, create v1.0.1).

4. **Configurable Storage Location**: The system must support a configurable `SNAPSHOT_ROOT` environment variable or config setting that specifies where snapshots are stored (separate from `LAKE_ROOT`).

5. **Artifact Discovery**: The system must automatically discover all consumer-facing artifacts from the lakehouse including:
   - Tabular data: spans, beats, sections (Parquet files)
   - Embeddings and ANN indexes (from `lakehouse/embeddings/` and `lakehouse/ann_index/`)
   - Catalogs: episodes, speakers (from `lakehouse/catalogs/`)
   - Latest QA report (from `quality_reports/`)

6. **File Copying**: The system must copy (not symlink) all discovered artifacts to the snapshot directory to ensure true immutability.

7. **Directory Structure**: The system must create a snapshot directory with the structure:
   ```
   <SNAPSHOT_ROOT>/vMAJOR.MINOR.PATCH/
     ├── lake_manifest.json
     ├── snapshot_note.txt
     ├── spans/
     ├── beats/
     ├── sections/
     ├── embeddings/
     ├── indexes/
     ├── catalogs/
     └── quality_report/ (if available)
   ```

8. **Checksum Calculation**: The system must calculate SHA-256 checksums for every copied file.

9. **File Metadata**: The system must record metadata for each file including: path, media_type, bytes, sha256, rows (for Parquet files), and notes.

10. **Strict Error Handling**: If any artifact is missing or corrupted during snapshot creation, the system must fail immediately and not create the snapshot.

### Manifest Generation

11. **Manifest File**: The system must generate a `lake_manifest.json` file at the snapshot root with all required fields.

12. **Version Field**: The manifest must include `lake_version` (string, semver format, e.g., "1.0.0-provisional").

13. **Timestamp Field**: The manifest must include `created_at` in ISO-8601 UTC format.

14. **Producer Section**: The manifest must include a `producer` object with:
    - `repo`: Repository name/URL
    - `commit`: Git commit hash at snapshot creation time
    - `task`: Always "SNAPSHOT"
    - `report_path`: Relative path to QA report if included, null otherwise

15. **Contracts Section**: The manifest must include a `contracts` object with:
    - `schemas`: Map of artifact name → schema version (e.g., "spans": "1.0")
    - `compatibility`: Optional consumer guidance (can be omitted initially)

16. **Files Inventory**: The manifest must include a `files` array with one entry per published file containing:
    - `path`: Relative path from snapshot root
    - `media_type`: One of "parquet", "ann-index", "catalog-db", "report"
    - `bytes`: File size in bytes
    - `sha256`: SHA-256 checksum (hex)
    - `rows`: Number of rows for Parquet files, null for other types
    - `notes`: Empty string or optional notes

17. **QA Status Section**: The manifest must include a `qa_status` object with:
    - `state`: One of "UNKNOWN", "FAIL", "PASS"
    - `summary`: Human-readable string describing QA status
    - `invariants`: Optional object with available QA metrics (can be partial or omitted)
    - `provisional`: Boolean flag (true for provisional snapshots)

18. **QA Status Detection**: The system must:
    - Read the most recent QA report from `quality_reports/` directory
    - Set `qa_status.state` to "UNKNOWN" if no QA report exists
    - Set `qa_status.state` to "PASS" or "FAIL" based on the QA report results
    - Extract summary and invariants from the QA report when available

### Snapshot Note

19. **Note File**: The system must create a `snapshot_note.txt` file that clearly states:
    - The provisional nature of the snapshot
    - Intended use case (e.g., migration to desktop)
    - Version number
    - Creation timestamp
    - QA status summary
    - Instructions for setting LAKE_ROOT environment variable

### Validation

20. **Automatic Validation**: The system must automatically validate the snapshot immediately after creation.

21. **Existence Check**: The validator must verify the snapshot directory exists and is readable.

22. **Manifest Check**: The validator must:
    - Verify `lake_manifest.json` loads as valid JSON
    - Verify all required keys are present
    - Verify all values have correct types

23. **Inventory Check**: The validator must verify that each file listed in `files[]` array:
    - Exists at the specified path
    - Matches the recorded byte size
    - Matches the recorded SHA-256 checksum

24. **QA Status Echo**: The validator must print the `qa_status.state` and `summary` from the manifest.

25. **Validation Output**: The validator must:
    - Output "PASS" if all structural and inventory checks succeed
    - Output "FAIL" if any structural or inventory check fails
    - Output WARNINGS (not failures) for non-PASS QA states (UNKNOWN or FAIL)

26. **Validation Reporting**: The validator must produce a clear report showing:
    - Each check performed and its result
    - Total files validated
    - Any warnings or errors
    - Final PASS/FAIL status

### Immutability & Reproducibility

27. **Deterministic Creation**: Re-creating a snapshot from the same source artifacts must yield identical checksums (byte-for-byte reproducibility).

28. **No Modification**: Once created, snapshot contents must not be modified. Any changes require creating a new snapshot version.

## Non-Goals (Out of Scope)

1. **Snapshot Promotion**: This task does not include promoting provisional snapshots to production/released status. A separate promotion task will handle this later.

2. **Snapshot Deletion**: No functionality for deleting or managing old snapshots.

3. **Snapshot Comparison**: No diff or comparison functionality between snapshots.

4. **Remote Publishing**: No functionality to push snapshots to remote storage (S3, GCS, etc.).

5. **Running QA**: The snapshot tool does not run quality assessment; it only reads existing QA reports.

6. **Consumer Tools**: No tools for consuming snapshots (loading, querying) are included in this task.

7. **Incremental Snapshots**: Only full snapshots are supported; no delta/incremental snapshots.

8. **Compression**: No compression or archiving of snapshot directories (e.g., tar.gz).

## Design Considerations

### CLI Interface

```bash
# Create a new provisional snapshot
lakehouse snapshot create

# Example output:
# Creating snapshot v1.0.0-provisional...
# ✓ Discovered 15 artifacts
# ✓ Copied spans (12.3 MB)
# ✓ Copied beats (8.7 MB)
# ✓ Copied sections (5.2 MB)
# ✓ Copied embeddings (142.5 MB)
# ✓ Copied indexes (89.3 MB)
# ✓ Copied catalogs (1.2 MB)
# ✓ Copied QA report
# ✓ Generated manifest
# ✓ Validating snapshot...
# ✓ Validation PASSED (WARNING: QA state is FAIL)
#
# Snapshot created: /path/to/snapshots/v1.0.0-provisional
# To use: export LAKE_ROOT=/path/to/snapshots/v1.0.0-provisional
```

### Configuration

Add to `config/` directory or lakehouse config:

```yaml
snapshot:
  root: "${SNAPSHOT_ROOT:-./snapshots}"
  version:
    major: 1
    minor: 0
    patch: 0
  auto_increment: true
  provisional_suffix: "-provisional"
```

### Manifest Schema (JSON)

```json
{
  "lake_version": "1.0.0-provisional",
  "created_at": "2025-10-27T12:34:56Z",
  "producer": {
    "repo": "transcription-lakehouse",
    "commit": "abc123def456",
    "task": "SNAPSHOT",
    "report_path": "quality_report/quality_assessment.md"
  },
  "contracts": {
    "schemas": {
      "spans": "1.0",
      "beats": "1.0",
      "sections": "1.0",
      "embeddings": "1.0"
    },
    "compatibility": {}
  },
  "files": [
    {
      "path": "spans/spans.parquet",
      "media_type": "parquet",
      "bytes": 12345678,
      "sha256": "abc123...",
      "rows": 50000,
      "notes": ""
    }
  ],
  "qa_status": {
    "state": "FAIL",
    "summary": "length bounds not met; duplicate rate high",
    "invariants": {
      "span_count": 50000,
      "beat_count": 12000,
      "duplicate_rate": 0.023
    },
    "provisional": true
  }
}
```

### Snapshot Note Format

```
PROVISIONAL LAKEHOUSE SNAPSHOT
================================

Version: v1.0.0-provisional
Created: 2025-10-27 12:34:56 UTC
Producer: transcription-lakehouse (commit: abc123def456)

QUALITY STATUS: FAIL
Summary: length bounds not met; duplicate rate high

This is a PROVISIONAL snapshot created for migration and testing purposes.
Quality assessment indicates known issues. Consumers should validate results
and be prepared for data quality limitations.

Intended Use: Migration to desktop environment for local development

USAGE
-----
To use this snapshot, set the LAKE_ROOT environment variable:

  export LAKE_ROOT=/path/to/snapshots/v1.0.0-provisional

Then access artifacts through the standard lakehouse API.

CONTENTS
--------
- Spans, Beats, Sections (Parquet)
- Embeddings and ANN Indexes
- Episode and Speaker Catalogs
- Quality Assessment Report

For detailed inventory, see lake_manifest.json
```

## Technical Considerations

### Integration Points

1. **Existing CLI**: Add new command to `src/lakehouse/cli/commands/snapshot.py`
2. **Config System**: Integrate with existing `src/lakehouse/config.py`
3. **Structure Module**: Use `src/lakehouse/structure.py` for path discovery
4. **Git Integration**: Use subprocess or GitPython to get current commit hash
5. **QA Reports**: Read from existing `quality_reports/` directory structure

### Dependencies

- `hashlib` (stdlib) for SHA-256 checksums
- `shutil` for file copying
- `json` (stdlib) for manifest creation
- `datetime` (stdlib) for ISO-8601 timestamps
- `pathlib` (stdlib) for path manipulation
- `pyarrow` (existing) for reading Parquet row counts
- `subprocess` or `GitPython` for Git commit hash

### Error Handling

- Validate all source artifacts exist before starting copy
- Use atomic operations where possible (write manifest last)
- Provide clear error messages for common failures (missing artifacts, disk full, permission denied)
- Clean up partial snapshots on failure (optional enhancement)

### Performance Considerations

- Large file copies may take time; provide progress indicators
- Calculate checksums during copy to avoid reading files twice
- Consider parallel copying for multiple large files (future enhancement)

## Success Metrics

1. **Correctness**: 100% of snapshots pass validation immediately after creation
2. **Reproducibility**: Re-creating the same snapshot yields identical SHA-256 checksums
3. **Completeness**: All consumer-facing artifacts are included in every snapshot
4. **Traceability**: Every snapshot has complete provenance information in manifest
5. **Usability**: Junior developers can create and validate snapshots with a single CLI command
6. **Reliability**: Snapshot creation fails safely if any artifact is missing or corrupted

## Acceptance Criteria

### Snapshot Creation

- [ ] CLI command `lakehouse snapshot create` successfully creates a snapshot
- [ ] Snapshot directory is created at `<SNAPSHOT_ROOT>/vX.Y.Z/` with correct version
- [ ] All consumer-facing artifacts (spans, beats, sections, embeddings, indexes, catalogs) are copied to snapshot
- [ ] QA report is included if available
- [ ] If a version collision occurs, the system auto-increments to the next available version

### Manifest Generation

- [ ] `lake_manifest.json` is created at the snapshot root
- [ ] Manifest includes all required fields: `lake_version`, `created_at`, `producer`, `contracts`, `files`, `qa_status`
- [ ] `files[]` array contains an entry for every file in the snapshot
- [ ] Each file entry includes correct: `path`, `media_type`, `bytes`, `sha256`, `rows` (or null), `notes`
- [ ] `producer.commit` contains the current Git commit hash
- [ ] `qa_status.state` is correctly set to "UNKNOWN", "FAIL", or "PASS" based on most recent QA report
- [ ] `qa_status.provisional` is set to `true`

### Snapshot Note

- [ ] `snapshot_note.txt` is created at the snapshot root
- [ ] Note clearly states the provisional nature of the snapshot
- [ ] Note includes version, timestamp, QA status summary
- [ ] Note includes the exact command to set LAKE_ROOT environment variable

### Validation

- [ ] Validation runs automatically after snapshot creation
- [ ] Validator checks snapshot directory exists and is readable
- [ ] Validator checks `lake_manifest.json` loads and has all required keys
- [ ] Validator verifies each file in `files[]` exists, matches byte size and SHA-256 checksum
- [ ] Validator outputs PASS if all structural/inventory checks succeed
- [ ] Validator outputs WARNINGS (not FAIL) for non-PASS QA states
- [ ] Validator output is clear and includes summary of checks performed

### Immutability & Reproducibility

- [ ] Re-creating a snapshot from the same source artifacts yields identical SHA-256 checksums
- [ ] Snapshot creation fails immediately if any source artifact is missing or corrupted

### Documentation & Submission

- [ ] First 8-10 lines of snapshot note are clear and informative
- [ ] Manifest JSON is well-formed and includes all required fields
- [ ] Validator output shows PASS with QA warning if applicable
- [ ] Exact one-liner for setting LAKE_ROOT is provided

## Open Questions

1. **Version Config**: Should the base version (MAJOR.MINOR.PATCH) be stored in `pyproject.toml`, a dedicated config file, or the lakehouse metadata?

2. **Provisional Suffix**: Should "-provisional" be automatically appended to version strings, or should it be part of the base version number?

3. **SNAPSHOT_ROOT Default**: What should be the default location if `SNAPSHOT_ROOT` is not configured? (Suggestion: `./lakehouse/snapshots/` or `./snapshots/`)

4. **Schema Versions**: How should schema versions be determined for each artifact? Should they be read from lakehouse metadata or hardcoded?

5. **Compatibility Field**: What consumer guidance should go in the `contracts.compatibility` field? Can this be omitted in v1?

6. **Partial QA**: If QA report exists but is incomplete, should `qa_status.invariants` include partial metrics, or should they all be omitted?

7. **Media Types**: Are the proposed media types ("parquet", "ann-index", "catalog-db", "report") sufficient, or should we use more specific types (e.g., "application/x-parquet", "application/vnd.faiss")?

8. **Progress Indicators**: Should the CLI show progress bars for large file copies, or is text-based progress sufficient?

9. **Cleanup on Failure**: Should the system delete partially created snapshots on failure, or leave them for debugging?

10. **Future Promotion**: When implementing snapshot promotion later, should it create a new snapshot or modify the existing one's manifest (change `provisional: false` and update `qa_status`)?

