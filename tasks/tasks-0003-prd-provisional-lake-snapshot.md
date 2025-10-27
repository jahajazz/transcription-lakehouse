# Task List: Provisional LAKE_ROOT Snapshot

Based on PRD: `0003-prd-provisional-lake-snapshot.md`

## Relevant Files

### New Files to Create

- `src/lakehouse/snapshot/__init__.py` - Snapshot module initialization and exports
- `src/lakehouse/snapshot/config.py` - Snapshot configuration management and version handling
- `src/lakehouse/snapshot/artifacts.py` - Artifact discovery and file collection logic
- `src/lakehouse/snapshot/manifest.py` - Manifest generation and QA status parsing
- `src/lakehouse/snapshot/validator.py` - Snapshot validation logic
- `src/lakehouse/snapshot/creator.py` - Main snapshot creation orchestrator
- `src/lakehouse/cli/commands/snapshot.py` - CLI command for snapshot operations
- `config/snapshot_config.yaml` - Snapshot configuration file (optional, with defaults in code)
- `tests/test_snapshot_config.py` - Unit tests for snapshot configuration
- `tests/test_snapshot_artifacts.py` - Unit tests for artifact discovery
- `tests/test_snapshot_manifest.py` - Unit tests for manifest generation
- `tests/test_snapshot_validator.py` - Unit tests for snapshot validation
- `tests/test_snapshot_creator.py` - Unit tests for snapshot creator
- `tests/integration/test_snapshot_workflow.py` - Integration tests for full snapshot workflow

### Existing Files to Modify

- `src/lakehouse/cli/__init__.py` - Import snapshot command to register it with CLI
- `src/lakehouse/structure.py` - May need helper methods for artifact discovery (optional)

### Notes

- Unit tests should be placed in `tests/` directory following the existing pattern
- Integration tests go in `tests/integration/`
- Use `pytest` for running tests: `pytest tests/test_snapshot_*.py`
- Follow existing patterns from `src/lakehouse/quality/` and `src/lakehouse/cli/commands/quality.py`

## Tasks

- [x] 1.0 **Snapshot Configuration & Version Management**
  - [x] 1.1 Create `src/lakehouse/snapshot/__init__.py` with module exports
  - [x] 1.2 Create `src/lakehouse/snapshot/config.py` with `SnapshotConfig` class that loads from YAML or uses defaults (base version, snapshot_root, auto_increment, provisional_suffix)
  - [x] 1.3 Implement `parse_version(version_string: str) -> Tuple[int, int, int, str]` to parse semver strings (e.g., "1.0.0-provisional" → (1, 0, 0, "-provisional"))
  - [x] 1.4 Implement `format_version(major, minor, patch, suffix="") -> str` to create version strings (e.g., (1, 0, 0, "-provisional") → "1.0.0-provisional")
  - [x] 1.5 Implement `find_existing_snapshots(snapshot_root: Path) -> List[str]` to list all existing snapshot versions in snapshot_root directory
  - [x] 1.6 Implement `get_next_version(current_version: Tuple[int, int, int], existing_versions: List[str]) -> str` to auto-increment patch version if collision exists
  - [x] 1.7 Implement `resolve_snapshot_root(config: SnapshotConfig) -> Path` to resolve SNAPSHOT_ROOT from config or environment variable, with default fallback to `./snapshots/`
  - [x] 1.8 Add validation to ensure snapshot_root is writable and create directory if it doesn't exist

- [x] 2.0 **Artifact Discovery & File Collection**
  - [x] 2.1 Create `src/lakehouse/snapshot/artifacts.py` with `ArtifactDiscovery` class
  - [x] 2.2 Implement `discover_artifacts(lakehouse_path: Path, version: str = "v1") -> Dict[str, List[Path]]` that returns categorized artifact paths (spans, beats, sections, embeddings, indexes, catalogs)
  - [x] 2.3 Implement `find_latest_qa_report(base_path: Path) -> Optional[Path]` that scans `quality_reports/` for the most recent timestamped directory and returns path to quality_assessment.md
  - [x] 2.4 Implement `calculate_sha256(file_path: Path) -> str` to compute SHA-256 checksum of a file
  - [x] 2.5 Implement `get_parquet_row_count(file_path: Path) -> Optional[int]` using pyarrow to read row count from Parquet files
  - [x] 2.6 Implement `copy_artifact_with_metadata(src: Path, dest: Path) -> Dict[str, Any]` that copies file and returns metadata dict with path, bytes, sha256, rows (if Parquet)
  - [x] 2.7 Implement `create_snapshot_structure(snapshot_root: Path, version: str) -> Dict[str, Path]` that creates the snapshot directory and subdirectories (spans/, beats/, sections/, embeddings/, indexes/, catalogs/, quality_report/)
  - [x] 2.8 Implement `copy_all_artifacts(artifacts: Dict[str, List[Path]], dest_dirs: Dict[str, Path], qa_report_path: Optional[Path]) -> List[Dict[str, Any]]` that copies all discovered artifacts to snapshot, calculates checksums, and returns list of file metadata dicts
  - [x] 2.9 Add error handling for missing artifacts (fail immediately if any required artifact is missing or unreadable per FR-10)

- [x] 3.0 **Manifest Generation**
  - [x] 3.1 Create `src/lakehouse/snapshot/manifest.py` with `ManifestGenerator` class
  - [x] 3.2 Implement `get_git_commit_hash() -> str` using subprocess to run `git rev-parse HEAD` and return current commit hash (or "unknown" if not in git repo)
  - [x] 3.3 Implement `parse_qa_report(qa_report_path: Optional[Path]) -> Dict[str, Any]` that reads quality_assessment.md and extracts state ("PASS"/"FAIL"/"UNKNOWN"), summary, and key metrics for invariants field
  - [x] 3.4 Implement `determine_qa_status(qa_report_path: Optional[Path]) -> Dict[str, Any]` that returns qa_status object with state, summary, invariants, and provisional=True
  - [x] 3.5 Implement `get_schema_versions(lakehouse_path: Path) -> Dict[str, str]` that reads lakehouse_metadata.json or returns default schema versions (all "1.0")
  - [x] 3.6 Implement `build_files_array(file_metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]` that converts file metadata to manifest files array format with proper media_type categorization ("parquet", "ann-index", "catalog-db", "report")
  - [x] 3.7 Implement `create_manifest(version: str, files: List[Dict], qa_status: Dict, schema_versions: Dict, git_commit: str, qa_report_rel_path: Optional[str]) -> Dict[str, Any]` that builds complete manifest dict with all required fields
  - [x] 3.8 Implement `write_manifest(manifest: Dict[str, Any], snapshot_root: Path) -> None` that writes manifest as pretty-printed JSON to lake_manifest.json at snapshot root
  - [x] 3.9 Add ISO-8601 UTC timestamp generation for created_at field

- [x] 4.0 **Snapshot Validation**
  - [x] 4.1 Create `src/lakehouse/snapshot/validator.py` with `SnapshotValidator` class
  - [x] 4.2 Implement `validate_existence(snapshot_path: Path) -> Tuple[bool, Optional[str]]` that checks if snapshot directory exists and is readable, returns (success, error_message)
  - [x] 4.3 Implement `validate_manifest_structure(manifest_path: Path) -> Tuple[bool, Dict[str, Any], Optional[str]]` that loads manifest JSON, validates required keys exist with correct types, returns (success, manifest_dict, error_message)
  - [x] 4.4 Implement `validate_inventory(snapshot_path: Path, manifest: Dict[str, Any]) -> List[Tuple[str, bool, Optional[str]]]` that checks each file in manifest files array exists, matches bytes and sha256, returns list of (file_path, success, error_message) tuples
  - [x] 4.5 Implement `validate_snapshot(snapshot_path: Path) -> Dict[str, Any]` that orchestrates all validation checks and returns validation result dict with status ("PASS"/"FAIL"), checks performed, warnings, and errors
  - [x] 4.6 Implement QA status warning logic: if qa_status.state is "UNKNOWN" or "FAIL", add to warnings but don't fail validation (per FR-25)
  - [x] 4.7 Implement `generate_validation_report(validation_result: Dict[str, Any]) -> str` that formats validation result as human-readable text report
  - [x] 4.8 Add summary statistics to validation report (total files checked, bytes verified, etc.)

- [x] 5.0 **CLI Command & Integration**
  - [x] 5.1 Create `src/lakehouse/snapshot/creator.py` with `SnapshotCreator` class that orchestrates the full snapshot creation workflow
  - [x] 5.2 Implement `create_snapshot(lakehouse_path: Path, config: SnapshotConfig, version: Optional[str] = None) -> Dict[str, Any]` that coordinates: config resolution, artifact discovery, directory creation, file copying, manifest generation, validation, and returns result dict with snapshot_path, version, validation_result
  - [x] 5.3 Implement `generate_snapshot_note(snapshot_info: Dict[str, Any], manifest: Dict[str, Any], snapshot_path: Path) -> str` that creates snapshot_note.txt content with version, timestamp, QA status, usage instructions, and LAKE_ROOT export command
  - [x] 5.4 Implement `write_snapshot_note(note_content: str, snapshot_path: Path) -> None` that writes snapshot_note.txt to snapshot root
  - [x] 5.5 Create `src/lakehouse/cli/commands/snapshot.py` with Click command group for snapshot operations
  - [x] 5.6 Implement `@cli.group()` for `snapshot` with subcommands (initially just `create`, but structure for future `validate`, `promote`)
  - [x] 5.7 Implement `@snapshot.command()` for `create` that accepts --lakehouse-path, --config-dir, --snapshot-root, --version (optional override), --log-level flags using common_options pattern
  - [x] 5.8 Wire up CLI command to call SnapshotCreator.create_snapshot() and display progress with Rich console (similar to quality command pattern)
  - [x] 5.9 Implement CLI output display: show discovered artifacts count, copy progress for each artifact type, manifest generation, validation results, and final snapshot path with LAKE_ROOT export command
  - [x] 5.10 Update `src/lakehouse/cli/__init__.py` to import snapshot command in the main() function so it's registered with the CLI
  - [x] 5.11 Add error handling and user-friendly error messages for common failures (missing artifacts, disk full, permission denied, git not available)

- [ ] 6.0 **Testing & Documentation**
  - [ ] 6.1 Create `tests/test_snapshot_config.py` with tests for version parsing, formatting, collision detection, and next version calculation
  - [ ] 6.2 Create `tests/test_snapshot_artifacts.py` with tests for artifact discovery, SHA-256 calculation, Parquet row counting, and file copying
  - [ ] 6.3 Create `tests/test_snapshot_manifest.py` with tests for git commit hash retrieval, QA report parsing, manifest structure generation, and JSON writing
  - [ ] 6.4 Create `tests/test_snapshot_validator.py` with tests for existence checks, manifest validation, inventory verification, and validation report generation
  - [ ] 6.5 Create `tests/test_snapshot_creator.py` with tests for the full orchestration logic using mock artifacts
  - [ ] 6.6 Create `tests/integration/test_snapshot_workflow.py` with end-to-end integration test that creates a snapshot from real lakehouse fixtures, validates it, and verifies reproducibility
  - [ ] 6.7 Add test for immutability: create snapshot twice from same inputs, verify identical SHA-256 checksums (FR-27)
  - [ ] 6.8 Add test for version collision handling: create snapshot, create again, verify patch version incremented (FR-3)
  - [ ] 6.9 Add test for missing artifacts: attempt snapshot creation with incomplete lakehouse, verify it fails immediately (FR-10)
  - [ ] 6.10 Add test for QA status detection: test with PASS report, FAIL report, and no report (UNKNOWN state)
  - [ ] 6.11 Add test fixtures in `tests/fixtures/` for mock lakehouse structure and sample QA reports
  - [ ] 6.12 Run all tests with `pytest tests/test_snapshot_*.py` and ensure >90% code coverage for snapshot module
  - [ ] 6.13 Manually test CLI command: `lakehouse snapshot create` and verify output matches acceptance criteria in PRD

