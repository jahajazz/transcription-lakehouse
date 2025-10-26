# Tasks: Beat/Span Quality Assessment Baseline

Generated from: `0002-prd-beat-span-quality-assessment.md`

## Current State Assessment

### Existing Infrastructure
- **CLI Framework**: Click-based CLI in `src/lakehouse/cli/` with common options decorator
- **Validation System**: Existing `ValidationCheck`, `ValidationReport`, and `ValidationReporter` classes in `src/lakehouse/validation/`
- **Configuration**: YAML-based config loading with defaults in `src/lakehouse/config.py`
- **Rich Console**: Rich library for pretty console output (tables, panels, progress bars)
- **Lakehouse Structure**: `LakehouseStructure` class manages versioned paths
- **Data Loading**: Parquet reading utilities in ingestion module

### Patterns to Follow
- Commands in `src/lakehouse/cli/commands/` with `@cli.command()` decorator
- Rich Console for output formatting
- YAML config files in `config/` directory
- Timestamped output directories (YYYYMMDD_HHMMSS)
- ValidationReport pattern for structured results
- Common CLI options: `--lakehouse-path`, `--config-dir`, `--log-level`

### New Components Needed
- Quality assessment module (`src/lakehouse/quality/`)
- Metrics calculators for 7 categories (A-G)
- Threshold configuration system
- Markdown report generator with ASCII visualizations
- Neighbor analysis and sampling utilities
- New CLI command: `lakehouse quality` or similar

## Relevant Files

### New Files to Create
- ✅ `src/lakehouse/quality/__init__.py` - Quality module initialization
- ✅ `src/lakehouse/quality/assessor.py` - Main assessment orchestration and entry point
- ✅ `src/lakehouse/quality/thresholds.py` - Threshold definitions and validation logic
- ✅ `src/lakehouse/quality/metrics/__init__.py` - Metrics submodule initialization
- ✅ `src/lakehouse/quality/metrics/coverage.py` - Category A: Coverage & Count metrics
- ✅ `src/lakehouse/quality/metrics/distribution.py` - Category B: Length & Distribution metrics
- ✅ `src/lakehouse/quality/metrics/integrity.py` - Category C: Ordering & Integrity metrics
- ✅ `src/lakehouse/quality/metrics/balance.py` - Category D: Speaker & Series Balance metrics
- ✅ `src/lakehouse/quality/metrics/text_quality.py` - Category E: Text Quality Proxy metrics
- ✅ `src/lakehouse/quality/metrics/embedding.py` - Category F: Embedding Sanity Checks
- ✅ `src/lakehouse/quality/diagnostics.py` - Category G: Outliers & Sample audits
- ✅ `src/lakehouse/quality/reporter.py` - Markdown report generation with ASCII visualizations
- ✅ `src/lakehouse/cli/commands/quality.py` - CLI command for quality assessment
- ✅ `config/quality_thresholds.yaml` - Default threshold configuration
- `tests/test_quality_metrics.py` - Unit tests for metrics calculators
- `tests/test_quality_assessment.py` - Integration tests for full assessment
- `tests/fixtures/quality_test_data.py` - Test data fixtures for quality tests

### Files to Modify
- `src/lakehouse/cli/__init__.py` - Import quality command to register it
- `src/lakehouse/structure.py` - Add `get_quality_reports_path()` method (optional)
- `src/lakehouse/config.py` - Add quality config loading support (optional, if not using separate loader)

### Notes
- Follow existing patterns from `validate.py` and `catalog.py` for CLI structure
- Reuse `ValidationReporter` patterns for output formatting
- Use stratified sampling from pandas for reproducible samples
- FAISS integration should be optional (graceful fallback to NumPy)
- ASCII histograms can use simple character-based bar charts
- All metrics should be deterministic (fixed random seeds)

## Tasks

- [x] 1.0 Project Setup & Module Structure
  - [x] 1.1 Create quality module directory structure (`src/lakehouse/quality/` and `src/lakehouse/quality/metrics/`)
  - [x] 1.2 Create `src/lakehouse/quality/__init__.py` with module exports
  - [x] 1.3 Create `src/lakehouse/quality/metrics/__init__.py` with metric function exports
  - [x] 1.4 Create `src/lakehouse/quality/thresholds.py` with `QualityThresholds` dataclass and default values
  - [x] 1.5 Create `config/quality_thresholds.yaml` with default threshold configuration matching FR-9, FR-13, FR-17, FR-25, FR-27, FR-31
  - [x] 1.6 Create `src/lakehouse/quality/assessor.py` with `QualityAssessor` class skeleton and data loading logic
  - [x] 1.7 Define shared data structures: `MetricsBundle`, `AssessmentResult`, `RAGStatus` enum

- [x] 2.0 Data Metrics Implementation (Categories A-E)
  - [x] 2.1 Create `src/lakehouse/quality/metrics/coverage.py` implementing FR-7, FR-8, FR-9:
    - [x] 2.1.1 Implement `calculate_episode_coverage()` for total duration, span/beat coverage %
    - [x] 2.1.2 Implement `detect_gaps_and_overlaps()` to find timeline gaps and overlaps
    - [x] 2.1.3 Implement `validate_coverage_thresholds()` to check against configured thresholds
    - [x] 2.1.4 Return structured dict with per-episode and global coverage metrics
  - [x] 2.2 Create `src/lakehouse/quality/metrics/distribution.py` implementing FR-10, FR-11, FR-12, FR-13:
    - [x] 2.2.1 Implement `calculate_duration_statistics()` for min/max/mean/median/p5/p95/std
    - [x] 2.2.2 Implement `calculate_length_compliance()` for % within target bounds (spans: 20-120s, beats: 60-180s)
    - [x] 2.2.3 Implement `generate_histogram_bins()` with sensible bin ranges for spans and beats
    - [x] 2.2.4 Implement `validate_length_thresholds()` to check compliance against thresholds
  - [x] 2.3 Create `src/lakehouse/quality/metrics/integrity.py` implementing FR-14, FR-15, FR-16, FR-17:
    - [x] 2.3.1 Implement `check_timestamp_monotonicity()` for episode-level and speaker-stream ordering
    - [x] 2.3.2 Implement `detect_integrity_violations()` for negative/zero durations and missing fields
    - [x] 2.3.3 Implement `detect_duplicates()` for exact (normalized text match) and near-duplicates (fuzzy match using rapidfuzz)
    - [x] 2.3.4 Implement `validate_integrity_thresholds()` to check violations against thresholds
  - [x] 2.4 Create `src/lakehouse/quality/metrics/balance.py` implementing FR-18, FR-19:
    - [x] 2.4.1 Implement `calculate_speaker_distribution()` for counts, percentages, avg duration per speaker
    - [x] 2.4.2 Implement `calculate_series_balance()` for per-series statistics (if series metadata available)
    - [x] 2.4.3 Return top N speakers and long-tail statistics
  - [x] 2.5 Create `src/lakehouse/quality/metrics/text_quality.py` implementing FR-20, FR-21, FR-22:
    - [x] 2.5.1 Implement `calculate_text_metrics()` for token count, word count, character count per segment
    - [x] 2.5.2 Implement `calculate_lexical_density()` using NLTK or spaCy stopword lists
    - [x] 2.5.3 Implement `calculate_punctuation_ratio()` for text quality proxy
    - [x] 2.5.4 Implement `extract_top_terms()` for top 20 unigrams and bigrams (with stopword filtering)

- [x] 3.0 Embedding Sanity Checks (Category F)
  - [x] 3.1 Create `src/lakehouse/quality/metrics/embedding.py` with embedding utilities:
    - [x] 3.1.1 Implement `load_embeddings()` to read span/beat embeddings from parquet with graceful handling if missing
    - [x] 3.1.2 Implement `stratified_sample_segments()` for reproducible sampling by episode and speaker
    - [x] 3.1.3 Implement `compute_cosine_similarity()` using NumPy (with optional FAISS optimization)
  - [x] 3.2 Implement neighbor coherence analysis (FR-23):
    - [x] 3.2.1 Implement `find_top_k_neighbors()` for each sampled segment
    - [x] 3.2.2 Implement `extract_neighbor_themes()` to summarize topic terms from neighbor texts
    - [x] 3.2.3 Return coherence assessment (coherent vs random neighbor themes)
  - [x] 3.3 Implement speaker and episode leakage detection (FR-24, FR-25):
    - [x] 3.3.1 Implement `calculate_speaker_leakage()` for % of neighbors sharing same speaker
    - [x] 3.3.2 Implement `calculate_episode_leakage()` for % of neighbors from same episode
    - [x] 3.3.3 Implement `validate_leakage_thresholds()` to check against max percentages
  - [x] 3.4 Implement length bias assessment (FR-26, FR-27):
    - [x] 3.4.1 Implement `calculate_embedding_norms()` (L2 norm per segment)
    - [x] 3.4.2 Implement `calculate_length_bias_correlation()` for duration vs norm and duration vs mean similarity
    - [x] 3.4.3 Implement `validate_length_bias_threshold()` to check correlation ≤ 0.3
  - [x] 3.5 Implement lexical vs embedding similarity alignment (FR-28):
    - [x] 3.5.1 Implement `sample_random_pairs()` for reproducible pair sampling
    - [x] 3.5.2 Implement `calculate_lexical_similarity()` using Jaccard or TF-IDF cosine
    - [x] 3.5.3 Implement `calculate_similarity_correlation()` between lexical and embedding similarities
  - [x] 3.6 Implement cross-series and adjacency bias checks (FR-29, FR-30, FR-31):
    - [x] 3.6.1 Implement `calculate_cross_series_neighbors()` for % neighbors from different series
    - [x] 3.6.2 Implement `calculate_adjacency_bias()` for % temporally adjacent neighbors (within 5s)
    - [x] 3.6.3 Implement `validate_adjacency_threshold()` to check ≤ 40% adjacency

- [x] 4.0 Diagnostics & Outlier Detection (Category G)
  - [x] 4.1 Create `src/lakehouse/quality/diagnostics.py` implementing FR-32, FR-33, FR-34:
    - [x] 4.1.1 Implement `identify_outliers()` to find longest, shortest, most isolated, most hubby segments
    - [x] 4.1.2 Implement `sample_neighbor_lists()` for random sample of query segments with their neighbors
    - [x] 4.1.3 Implement `format_text_excerpt()` to truncate text to 100 chars with "..." and CSV escaping
    - [x] 4.1.4 Implement `export_outliers_csv()` to write outliers.csv with ID, episode, speaker, duration, text, metric
    - [x] 4.1.5 Implement `export_neighbors_csv()` to write neighbors_sample.csv with query and neighbor details

- [ ] 5.0 Report Generation & CLI Integration
  - [x] 5.1 Create `src/lakehouse/quality/reporter.py` implementing FR-35 through FR-40:
    - [x] 5.1.1 Implement `QualityReporter` class with `generate_markdown_report()` method
    - [x] 5.1.2 Implement `generate_executive_summary()` with RAG status, timestamp, counts, failures
    - [x] 5.1.3 Implement `determine_rag_status()` logic (Green: all pass, Amber: 1-2 non-critical, Red: multiple/critical failures)
    - [x] 5.1.4 Implement `generate_ascii_histogram()` for duration distributions using block characters
    - [x] 5.1.5 Implement report sections: Configuration, Coverage, Distribution, Integrity, Balance, Text Quality, Embedding, Outliers
    - [x] 5.1.6 Implement `generate_findings_and_remediation()` with specific recommendations for each failed threshold
    - [x] 5.1.7 Implement `generate_go_nogo_recommendation()` based on overall status
  - [x] 5.2 Implement metrics export (FR-4, FR-5, FR-6):
    - [x] 5.2.1 Implement `export_global_metrics_json()` for metrics/global.json
    - [x] 5.2.2 Implement `export_episodes_csv()` for metrics/episodes.csv with per-episode stats
    - [x] 5.2.3 Implement `export_segments_csv()` for metrics/spans.csv and metrics/beats.csv with flags
  - [x] 5.3 Create output directory structure (FR-48, FR-49):
    - [x] 5.3.1 Implement `create_output_structure()` to create timestamped directories (metrics/, diagnostics/, report/)
    - [x] 5.3.2 Ensure proper directory creation and error handling
  - [x] 5.4 Create `src/lakehouse/cli/commands/quality.py` implementing FR-1, FR-2:
    - [x] 5.4.1 Create `quality` command with @cli.command() decorator and common options
    - [x] 5.4.2 Add CLI options: --version, --level (spans/beats/all), --output-dir, --sample-size, --config
    - [x] 5.4.3 Add threshold override options: --coverage-min, --span-length-min, --span-length-max, --beat-length-min, --beat-length-max, --neighbor-k
    - [x] 5.4.4 Implement command function to instantiate QualityAssessor and run assessment
    - [x] 5.4.5 Implement console summary output with Rich (FR-40): RAG status, key metrics, failures, file paths
    - [x] 5.4.6 Add progress indicators using Rich Progress for long-running operations
  - [x] 5.5 Integrate quality command into CLI (modify `src/lakehouse/cli/__init__.py`):
    - [x] 5.5.1 Import quality command in main() function to register it
  - [x] 5.6 Implement main assessment orchestration in `src/lakehouse/quality/assessor.py`:
    - [x] 5.6.1 Implement `QualityAssessor.run_assessment()` to coordinate all metric calculations
    - [x] 5.6.2 Implement data loading from lakehouse paths (FR-3): episodes, spans, beats, embeddings
    - [x] 5.6.3 Implement threshold loading from config with CLI overrides
    - [x] 5.6.4 Call all metric calculators (coverage, distribution, integrity, balance, text_quality, embedding, diagnostics)
    - [x] 5.6.5 Aggregate results into MetricsBundle
    - [x] 5.6.6 Implement reproducibility (FR-51, FR-52): fixed random seed, consistent rounding to 2-3 decimal places
    - [x] 5.6.7 Implement graceful handling of missing embeddings (FR-50): skip embedding checks, report in summary

- [ ] 6.0 Testing & Documentation
  - [ ] 6.1 Create test fixtures in `tests/fixtures/quality_test_data.py`:
    - [ ] 6.1.1 Create sample episode metadata DataFrame
    - [ ] 6.1.2 Create sample spans DataFrame with various edge cases (long, short, gaps, overlaps)
    - [ ] 6.1.3 Create sample beats DataFrame
    - [ ] 6.1.4 Create sample embeddings arrays
    - [ ] 6.1.5 Create helper functions to generate test data with controlled properties
  - [ ] 6.2 Create unit tests in `tests/test_quality_metrics.py`:
    - [ ] 6.2.1 Test coverage metrics: episode coverage calculation, gap/overlap detection, threshold validation
    - [ ] 6.2.2 Test distribution metrics: duration statistics, length compliance, histogram generation
    - [ ] 6.2.3 Test integrity metrics: monotonicity check, integrity violations, duplicate detection
    - [ ] 6.2.4 Test balance metrics: speaker distribution, series balance
    - [ ] 6.2.5 Test text quality metrics: token counts, lexical density, top terms extraction
    - [ ] 6.2.6 Test embedding metrics: neighbor search, leakage detection, bias calculations
    - [ ] 6.2.7 Test diagnostics: outlier identification, neighbor sampling, CSV export formatting
  - [ ] 6.3 Create integration tests in `tests/test_quality_assessment.py`:
    - [ ] 6.3.1 Test full assessment run with sample data
    - [ ] 6.3.2 Test output file generation (JSON, CSV, Markdown)
    - [ ] 6.3.3 Test reproducibility: same input produces same output (excluding timestamps)
    - [ ] 6.3.4 Test CLI command execution with various options
    - [ ] 6.3.5 Test graceful handling of missing embeddings
    - [ ] 6.3.6 Test threshold override via command line
  - [ ] 6.4 Add docstrings and type hints:
    - [ ] 6.4.1 Add comprehensive docstrings to all public functions and classes
    - [ ] 6.4.2 Add type hints to all function signatures
    - [ ] 6.4.3 Add module-level docstrings explaining purpose and usage
  - [ ] 6.5 Update project documentation:
    - [ ] 6.5.1 Add quality assessment section to README.md with usage examples
    - [ ] 6.5.2 Document threshold configuration in config/quality_thresholds.yaml with comments
    - [ ] 6.5.3 Add example output snippets showing report format

---

**Status**: Phase 2 - Detailed sub-tasks generated. Ready for implementation.

