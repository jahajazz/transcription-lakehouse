# Tasks: Fix Pack — Part-1 Foundations

**Source PRD:** 0004-prd-fix-pack-part1-foundations.md  
**Generated:** 2025-10-27  
**Status:** Phase 2 - Complete with Sub-tasks

---

## Relevant Files

### Configuration Files
- `config/speaker_roles.yaml` - New speaker role configuration (R1)
- `config/validator_routing.yaml` - New validator routing configuration (R3)
- `config/aggregation_config.yaml` - Existing config, may need updates for sections (R2)
- `config/quality_thresholds.yaml` - Existing config, may need updates (R4)

### Core Implementation Files
- `src/lakehouse/schemas.py` - Schema definitions for spans, beats, sections (R1, R2, R5)
- `src/lakehouse/config.py` - Configuration loader, add speaker_roles loading (R1)
- `src/lakehouse/aggregation/spans.py` - Span generation, add speaker enrichment (R1)
- `src/lakehouse/aggregation/beats.py` - Beat generation, add speaker enrichment (R1)
- `src/lakehouse/aggregation/sections.py` - Section generation, semantic boundaries (R2)
- `src/lakehouse/quality/assessor.py` - Quality assessment orchestrator, add routing (R3)
- `src/lakehouse/quality/metrics/coverage.py` - Coverage calculation fixes (R4)
- `src/lakehouse/quality/metrics/distribution.py` - Length bucket fixes (R4)
- `src/lakehouse/quality/metrics/integrity.py` - Duplicate check per-level (R4)
- `src/lakehouse/quality/reporter.py` - Report generation, add change log (R4)

### New Implementation Files
- `src/lakehouse/speaker_roles.py` - New module for speaker role logic (R1)
- `src/lakehouse/quality/validator_router.py` - New module for validation routing (R3)

### Test Files
- `tests/test_speaker_roles.py` - New tests for speaker role assignment (R1)
- `tests/test_aggregation.py` - Update existing tests for new fields (R1, R2)
- `tests/test_quality_assessment.py` - Update existing tests for routing (R3, R4)
- `tests/test_validator_routing.py` - New tests for validator routing (R3)
- `tests/integration/test_pipeline.py` - Update integration tests (R6)

### Migration/Scripts
- `scripts/regenerate_lakehouse.py` - Script to regenerate tables with new schema (R5)
- `scripts/archive_v1_tables.py` - Script to archive existing v1 tables (R5)

### Notes
- Implementation priority follows PRD recommendation: R3 → R1 → R2 → R4
- However, schema changes (R5) should be done early to support all requirements
- All existing unit tests should pass before regenerating lakehouse tables

---

## Tasks

- [ ] 1.0 Create Speaker Roles Configuration Infrastructure (R1)
  - [x] 1.1 Create `config/speaker_roles.yaml` with required structure (experts list, roles map, default_role)
  - [x] 1.2 Add "Fr Stephen De Young" and "Jonathan Pageau" as experts in the config file
  - [x] 1.3 Create `src/lakehouse/speaker_roles.py` module with `SpeakerRoleConfig` class to load and validate the YAML config
  - [x] 1.4 Add `load_speaker_roles()` function to `src/lakehouse/config.py` that uses the new module
  - [x] 1.5 Implement `determine_speaker_role()` function that applies config rules (lookup in roles map, check experts list, or use default)
  - [x] 1.6 Implement `enrich_spans_with_speaker_metadata()` function to add speaker_canonical, speaker_role, is_expert fields
  - [x] 1.7 Implement `enrich_beats_with_speaker_metadata()` function to add speakers_set, expert_span_ids, expert_coverage_pct fields
  - [x] 1.8 Add logic to calculate expert_coverage_pct using token-weighted or character-weighted method
  - [x] 1.9 Create unit tests in `tests/test_speaker_roles.py` covering config loading, role determination, and enrichment logic
  - [x] 1.10 Test that invalid configs (missing keys) fail fast with clear error messages

- [ ] 2.0 Implement Semantic Section Generation (R2)
  - [ ] 2.1 Update `SECTION_SCHEMA` in `src/lakehouse/schemas.py` to add `title` (string, non-null) and `synopsis` (string, nullable) fields
  - [ ] 2.2 Load beat embeddings in `SectionGenerator._generate_sections_for_episode()` to enable semantic boundary detection
  - [ ] 2.3 Update `SectionGenerator._should_break_section()` to use beat embedding similarity for topic change detection
  - [ ] 2.4 Add configuration parameters to `config/aggregation_config.yaml` for semantic section generation (min_similarity_drop, prefer_semantic_boundaries)
  - [ ] 2.5 Implement simple section title generation (e.g., "Section 1", "Section 2" or extract from first beat text)
  - [ ] 2.6 Add optional synopsis generation (can be placeholder "Auto-generated" for now)
  - [ ] 2.7 Ensure sections use similarity-based boundaries rather than fixed time windows
  - [ ] 2.8 Update `tests/test_aggregation.py` to verify multiple sections are generated per episode
  - [ ] 2.9 Test that all beats are assigned to exactly one section (no gaps or overlaps)
  - [ ] 2.10 Verify sections are chronologically ordered within episodes

- [ ] 3.0 Implement Validator Routing System (R3)
  - [ ] 3.1 Create `config/validator_routing.yaml` with table role definitions (base vs embedding) and check assignments
  - [ ] 3.2 Define routing rules: spans/beats get text/time checks, span_embeddings/beat_embeddings get vector checks
  - [ ] 3.3 Create `src/lakehouse/quality/validator_router.py` with `ValidatorRouter` class to load and interpret routing config
  - [ ] 3.4 Implement `ValidatorRouter.should_run_check(table_name, check_name)` method to determine if a check is applicable
  - [ ] 3.5 Implement `ValidatorRouter.get_table_role(table_name)` to return "base" or "embedding"
  - [ ] 3.6 Update `QualityAssessor.__init__()` to load validator routing config
  - [ ] 3.7 Update `QualityAssessor.run_assessment()` to use routing logic before running each metric calculator
  - [ ] 3.8 Modify metric calculators (coverage, distribution, integrity) to skip checks gracefully with "not applicable" notes when table role doesn't match
  - [ ] 3.9 Add `generate_table_validation_map()` function to create a summary table showing which checks ran on which tables
  - [ ] 3.10 Update `reporter.py` to include Table Validation Map section in QA report
  - [ ] 3.11 Create unit tests in `tests/test_validator_routing.py` for routing logic
  - [ ] 3.12 Test that embedding tables do not trigger "No timestamp columns found" or "No 'text' column found" warnings

- [ ] 4.0 Fix Quality Assessment Calculations and Report Generation (R4)
  - [ ] 4.1 Fix `coverage.calculate_episode_coverage()` to compute coverage on spans only using overlap-aware union (not beats)
  - [ ] 4.2 Implement overlap-aware union algorithm: merge overlapping span intervals per episode, sum union durations
  - [ ] 4.3 Ensure coverage percentage calculation is `(union_duration / episode_duration) * 100` and always ≤ 100%
  - [ ] 4.4 Fix `distribution.calculate_length_compliance()` to ensure buckets (below/in-range/above) sum to exactly 100% per level
  - [ ] 4.5 Update length bucket reporting to show min, median, p95, max for each level (spans, beats)
  - [ ] 4.6 Update `integrity.detect_duplicates()` to run separately per level (spans, then beats) with minimum text length floor (e.g., 10 chars)
  - [ ] 4.7 Add validation in `reporter.py` to ensure Executive Summary counts match detailed section counts
  - [ ] 4.8 Create "Changes in This Run" report section template in `reporter.py`
  - [ ] 4.9 Populate "Changes in This Run" with bullet points referencing R1 (speaker metadata), R2 (semantic sections), R3 (validator routing)
  - [ ] 4.10 Update `tests/test_quality_assessment.py` to verify coverage ≤ 100%, bucket sum = 100%, and summary consistency
  - [ ] 4.11 Test that duplicate checks show separate counts/percentages for spans and beats

- [ ] 5.0 Update Schemas and Regenerate Lakehouse Tables
  - [ ] 5.1 Update `SPAN_SCHEMA` in `src/lakehouse/schemas.py` to add speaker_canonical, speaker_role, is_expert fields
  - [ ] 5.2 Update `BEAT_SCHEMA` to add speakers_set, expert_span_ids, expert_coverage_pct fields
  - [ ] 5.3 Update `SECTION_SCHEMA` to add title and synopsis fields (already covered in 2.1)
  - [ ] 5.4 Create `scripts/archive_v1_tables.py` to move existing v1 tables to `lakehouse/v1_archived/` directory
  - [ ] 5.5 Run archive script to preserve existing tables for comparison
  - [ ] 5.6 Integrate speaker enrichment into `SpanGenerator.aggregate()` method in `src/lakehouse/aggregation/spans.py`
  - [ ] 5.7 Integrate speaker enrichment into `BeatGenerator.aggregate()` method in `src/lakehouse/aggregation/beats.py`
  - [ ] 5.8 Update `SectionGenerator.aggregate()` to use new semantic section generation logic
  - [ ] 5.9 Run full lakehouse ingestion pipeline: `lakehouse ingest --all` (or equivalent CLI command)
  - [ ] 5.10 Verify new parquet files contain all required new fields
  - [ ] 5.11 Spot check: load a sample of spans/beats/sections and verify field values are populated and valid

- [ ] 6.0 Integration Testing and Validation
  - [ ] 6.1 Run all existing unit tests: `pytest tests/ -v` and ensure they pass
  - [ ] 6.2 Update failing tests if they rely on old schema or outdated behavior
  - [ ] 6.3 Run integration tests: `pytest tests/integration/ -v`
  - [ ] 6.4 Generate QA report: `lakehouse quality-report --output output/quality` (or equivalent CLI command)
  - [ ] 6.5 Extract and document: first page of QA report (executive summary)
  - [ ] 6.6 Extract and document: Table Validation Map snippet from report
  - [ ] 6.7 Extract sample span row showing speaker_canonical, speaker_role, is_expert fields
  - [ ] 6.8 Extract sample beat row showing speakers_set, expert_span_ids, expert_coverage_pct fields
  - [ ] 6.9 Count sections vs episodes: verify sections_count > episodes_count (target ≥ 2× ratio)
  - [ ] 6.10 Confirm zero schema warnings in QA report output
  - [ ] 6.11 Validate success metrics: span coverage ≥ 95%, length histograms sane, no timestamp regressions
  - [ ] 6.12 Validate beat metrics: beats ordered/non-overlapping, length histogram sane, all beats have speaker fields
  - [ ] 6.13 Validate section metrics: beats fully partitioned into sections, multiple sections per episode
  - [ ] 6.14 Validate embedding QA: same-speaker %, same-episode %, adjacency %, length-sim correlation reported without spurious warnings
  - [ ] 6.15 Document final results: QA report summary, sample rows, section/episode counts, confirmation of zero schema warnings

