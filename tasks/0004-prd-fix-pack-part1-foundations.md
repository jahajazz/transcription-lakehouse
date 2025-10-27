# PRD: Fix Pack — Part-1 Foundations

**Feature Name:** Fix Pack Part-1 Foundations (Speaker Mapping, Sections, Validator Routing)  
**PRD Number:** 0004  
**Status:** Draft  
**Author:** AI Assistant  
**Date:** 2025-10-27

---

## 1. Introduction/Overview

The Transcription Lakehouse currently produces quality assessment reports with inconsistent metrics, schema warnings, and improperly routed validation checks. This feature stabilizes the lakehouse foundation by:

1. Adding speaker role metadata (expert flags) to spans and beats
2. Rebuilding sections as semantic topic blocks rather than 1:1 episode mappings
3. Correctly routing validators so text/timestamp checks only run on base tables and vector checks only run on embedding tables
4. Fixing QA report calculations to produce sane, consistent metrics

The goal is to ensure QA 1.1 produces reliable metrics and prepares the system for Part-2 metadata enhancements. This is a **foundations fix pack**—minimal, targeted changes with maximum stability impact.

**Problem Statement:** Currently, the QA system produces warnings about missing columns in embedding tables, coverage calculations exceed 100%, and sections don't reflect actual topic boundaries. These issues block meaningful quality assessment and prevent downstream metadata work.

---

## 2. Goals

1. **Speaker Awareness:** All spans and beats contain speaker role information (canonical name, role, expert flag) derived from a simple configuration file
2. **Semantic Sections:** Episodes are divided into multiple topic-based sections (not just one section per episode)
3. **Clean Validation:** Validators run appropriate checks on appropriate tables with zero schema warnings
4. **Consistent Metrics:** QA reports show internally consistent, plausible metrics (coverage ≤100%, buckets sum to 100%, counts match across sections)
5. **Backwards Compatibility:** Existing consumers that don't use new fields continue to work

---

## 3. User Stories

**As a** data analyst,  
**I want to** identify which transcript segments feature expert speakers,  
**So that** I can prioritize high-value content for retrieval and analysis.

**As a** quality engineer,  
**I want to** run validation checks without spurious warnings about missing columns,  
**So that** I can trust the QA report and identify real data quality issues.

**As a** content researcher,  
**I want to** browse sections that represent actual topic boundaries,  
**So that** I can quickly locate relevant discussions without reading entire episodes.

**As a** system maintainer,  
**I want to** configure speaker roles in a simple YAML file,  
**So that** I can update expert lists without code changes.

---

## 4. Functional Requirements

### R1: Speaker Mapping & Expert Flags (Config-Driven)

#### R1.1: Speaker Roles Configuration File
- **MUST** create a configuration file at `config/speaker_roles.yaml`
- **MUST** support the following YAML structure:
  ```yaml
  experts:
    - "Fr Stephen De Young"
    - "Jonathan Pageau"
  
  roles:
    "Fr Stephen De Young": "expert"
    "Jonathan Pageau": "expert"
    # Optional role overrides for other speakers
  
  default_role: "other"
  ```
- File **MUST** be editable by end users without code changes
- Invalid config (e.g., missing required keys) **MUST** fail fast with clear error message

#### R1.2: Span Enrichment
- **MUST** add three new columns to the `spans` table:
  - `speaker_canonical` (string, non-null): The canonical speaker name from the input transcript
  - `speaker_role` (string enum, non-null): One of `["expert", "host", "guest", "caller", "other"]`
  - `is_expert` (boolean, non-null): `true` if `speaker_canonical` is in the `experts` list, else `false`
- **MUST** ensure 100% of spans have all three fields populated
- `speaker_role` **MUST** be determined by:
  1. Looking up `speaker_canonical` in the `roles` map (if present in config)
  2. If speaker is in `experts` list but not in `roles` map, default to `"expert"`
  3. Otherwise, use `default_role` from config
- Values **MUST** be deterministic across reruns with the same input

#### R1.3: Beat Enrichment
- **MUST** add three new derived fields to the `beats` table:
  - `speakers_set` (array of strings): Unique canonical speaker names from all member spans, deduplicated
  - `expert_span_ids` (array of strings): Span IDs where `is_expert=true`
  - `expert_coverage_pct` (float, 0–100): Percentage of beat content spoken by experts
- `expert_coverage_pct` **MUST** be calculated as:
  - Token-weighted if `token_count` exists on spans
  - Character-weighted (using `text` length) if tokens unavailable
  - Formula: `(sum of expert span weights / sum of all span weights) * 100`
- All three fields **MUST** be present on 100% of beats
- Values **MUST** be deterministic across reruns

#### R1.4: Acceptance Criteria for R1
- [ ] Config file `config/speaker_roles.yaml` exists and loads successfully
- [ ] Config includes "Fr Stephen De Young" and "Jonathan Pageau" as experts
- [ ] 100% of spans have `speaker_canonical`, `speaker_role`, `is_expert` (non-null)
- [ ] 100% of beats have `speakers_set`, `expert_span_ids`, `expert_coverage_pct`
- [ ] Running the pipeline twice produces identical speaker metadata
- [ ] Manually verify: at least one span shows `is_expert=true` for configured experts

---

### R2: Sections as Topic Blocks (Not Whole Episodes)

#### R2.1: Section Generation Logic
- **MUST** rebuild the `sections` table so each episode yields **multiple sections**
- Target: **several sections per hour** of content (modest default; exact count depends on topic boundaries)
- Each section **MUST** be an ordered, non-overlapping collection of beats
- Section generation **MUST** use semantic similarity/topic change detection based on beat embeddings

#### R2.2: Section Schema
Each section **MUST** include:
- `section_id` (string, non-null): Unique identifier
- `episode_id` (string, non-null): Parent episode reference
- `t_start` (float, non-null): Section start time in seconds
- `t_end` (float, non-null): Section end time in seconds
- `beat_ids` (array of strings, non-null): Ordered list of beats in this section
- `title` (string, non-null): Auto-generated section title (short, descriptive)
- `synopsis` (string, nullable): Auto-generated section summary

#### R2.3: Section Boundaries
- Sections **MUST NOT** overlap within an episode
- Every beat **MUST** belong to exactly one section
- Beat order **MUST** be preserved (sections follow chronological episode flow)
- Duration and semantic threshold parameters **SHOULD** be configurable but have sensible defaults

#### R2.4: Acceptance Criteria for R2
- [ ] For typical 60–120 minute episodes, section count > 1 per episode
- [ ] Every beat appears in exactly one section's `beat_ids` array
- [ ] Section `t_start` and `t_end` align with member beats' time boundaries
- [ ] Sections within an episode are chronologically ordered
- [ ] Manually verify: multi-topic episodes show natural section breaks

---

### R3: Validator Routing & Warnings Cleanup

#### R3.1: Validator Configuration File
- **MUST** create or update a validator configuration file specifying table-to-check routing
- Configuration **MUST** follow a structure similar to:
  ```yaml
  tables:
    spans:
      role: base
      checks: [coverage, length_buckets, duplicates, speaker_required]
    
    beats:
      role: base
      checks: [length_buckets, order_no_overlap, spans_link]
    
    span_embeddings:
      role: embedding
      checks: [dim_consistency, id_join_back, nn_leakage, adjacency_bias, length_sim_corr]
    
    beat_embeddings:
      role: embedding
      checks: [dim_consistency, id_join_back, nn_leakage, adjacency_bias, length_sim_corr]
  ```

#### R3.2: Routing Logic
- Text/timestamp checks (coverage, length, duplicates, order) **MUST** only run on tables with `role: base`
- Vector checks (dim/NaN/row-match, neighbor leakage, adjacency, length-sim correlation) **MUST** only run on tables with `role: embedding`
- Vector checks **MAY** join back to base tables to retrieve attributes (speaker, episode, timestamps) for analysis

#### R3.3: Error Handling
- If a table marked `role: embedding` is missing `text` or timestamp columns, validators **MUST NOT** warn
- Instead, validators **MUST** output a clear "not applicable" note in the report
- If configuration is invalid (e.g., text checks assigned to embedding table), validator **MUST** fail fast with clear error message
- Zero schema warnings about missing columns **MUST** appear in output

#### R3.4: Acceptance Criteria for R3
- [ ] No "No timestamp columns found" warnings for embedding tables
- [ ] No "No 'text' column found" warnings for embedding tables
- [ ] QA report includes a "Table Validation Map" section showing which checks ran on which tables
- [ ] All configured checks execute successfully or skip with clear reasoning
- [ ] Manually verify: embedding tables show vector check results, base tables show text/time check results

---

### R4: QA Report Sanity & Summary Fixes

#### R4.1: Coverage Calculation Fix
- Coverage **MUST** be computed on `spans` table only
- Coverage **MUST** use overlap-aware union (if span A and B overlap, count the union time once)
- Coverage **MUST NOT** mix beats into coverage calculations
- Final coverage percentage **MUST** be ≤ 100% for each episode and in aggregate

#### R4.2: Length Bucket Consistency
- Length buckets (below-range / in-range / above-range) **MUST** sum to 100% per level (spans, beats)
- Report **MUST** show: min, median, p95, max lengths for each level
- Bucket boundaries **SHOULD** be configurable via `quality_thresholds.yaml`

#### R4.3: Duplicate Check Refinement
- Duplicate checks **MUST** run separately per level (spans, then beats)
- A minimum text length floor **SHOULD** be applied (e.g., skip duplicates for text < 10 chars)
- Report **MUST** show duplicate count and percentage per level

#### R4.4: Executive Summary Consistency
- Executive Summary counts **MUST** match detailed sections exactly
- If Executive Summary says "X spans, Y beats", detailed sections must report same counts
- Any discrepancies **MUST** be treated as a report generation bug

#### R4.5: Change Log in Report
- QA report **MUST** include a "Changes in This Run" section at the beginning
- This section **MUST** briefly reference R1–R3 changes (speaker metadata, semantic sections, validator routing)
- Format should be concise bullet list (3-5 lines maximum)

#### R4.6: Acceptance Criteria for R4
- [ ] All episode coverage values ≤ 100%
- [ ] Length buckets sum to 100% for spans and beats independently
- [ ] Executive Summary counts match detailed section counts
- [ ] Report includes "Changes in This Run" note mentioning R1–R3
- [ ] No internal calculation errors or inconsistencies in report

---

## 5. Non-Goals (Out of Scope)

The following are **explicitly excluded** from this feature:

1. **Retrieval Logic:** No changes to query processing, ANN index usage, or retrieval APIs
2. **Pair Factory:** No modifications to how training pairs are generated
3. **Model Training:** No retraining of embedding models or metadata labelers
4. **File Format Changes:** No renaming of existing columns or breaking changes to parquet schemas beyond adding new columns
5. **UI/Frontend:** No changes to any web interfaces or visualization tools
6. **Advanced Speaker Diarization:** This feature assumes speakers are already correctly identified in input transcripts; no re-diarization or speaker merging logic
7. **Section Title Generation Models:** Auto-generated section titles can be simple (e.g., "Section N" or first-sentence extraction); no requirement for ML-based title generation

---

## 6. Design Considerations

### Configuration Files
- All three config files (`speaker_roles.yaml`, validator routing config, `quality_thresholds.yaml`) should follow consistent YAML structure
- Configs should validate on load with helpful error messages
- Consider adding config schema validation using a library like `pydantic` or `jsonschema`

### Section Generation UI/UX
- While no UI is in scope, consider that future tools may visualize sections
- Section titles should be human-readable and descriptive
- Synopsis generation can be placeholder (e.g., "Auto-generated") in this phase

### Table Schema Evolution
- New columns are added, not renamed or removed
- Consider adding a `schema_version` field to table metadata for future migrations
- Document the schema changes in `lakehouse_metadata.json` or similar

---

## 7. Technical Considerations

### Dependencies
- **Existing:** `pandas`, `pyarrow` (for Parquet operations)
- **Existing:** Embedding generation pipeline (for section topic detection)
- **Existing:** QA validation framework in `src/lakehouse/quality/`
- **Config:** `PyYAML` for parsing configuration files (likely already present)

### Migration Strategy (R5: Implementation Approach)
- **Regenerate tables from raw data** with new schema (Option C from clarification)
- This ensures clean, consistent data without partial migration issues
- Existing raw transcripts remain unchanged; only processed tables (spans, beats, sections) are regenerated
- Old table versions can be archived or deleted after validation

### Section Topic Detection Algorithm
- Use beat embeddings to compute similarity between consecutive beats
- Where similarity drops below threshold, insert section boundary
- Consider minimum/maximum section duration constraints
- Algorithm details can be refined during implementation but should prioritize simplicity

### Validator Routing Implementation
- Validator framework should load routing config once at startup
- Each validator check should query config to determine eligible tables
- Consider creating a `ValidatorRouter` class to encapsulate this logic

### Performance
- Section generation may be computationally expensive on large datasets
- Consider caching intermediate similarity computations
- Ensure validation routing doesn't significantly slow down QA runs

---

## 8. Success Metrics

After implementing this feature and rerunning the pipeline, we expect:

### Quantitative Metrics
1. **Spans Coverage:** ≥ 95% union coverage across all episodes
2. **Length Histograms:** Sane distributions with min < median < p95 < max
3. **Timestamp Regressions:** Zero (no spans with invalid or out-of-order timestamps)
4. **Duplicate Rate:** < 5% for spans and beats (reasonable threshold, adjust if needed)
5. **Section Density:** Section count ≥ 2× episode count (multiple sections per episode)
6. **Beat Partitioning:** 100% of beats assigned to exactly one section
7. **Validator Warnings:** Zero schema-related warnings in QA output

### Qualitative Metrics
1. **Report Readability:** QA report includes clear "Changes in This Run" section
2. **Table Validation Map:** Report shows which checks ran on which tables
3. **Expert Metadata:** Sample rows from spans/beats show correct expert flags
4. **Section Quality:** Manual review of 3-5 episodes confirms sections align with topic boundaries

### Deliverables for Validation
1. Updated QA 1.1 report (first page + table-map snippet)
2. Sample span row showing `speaker_canonical`, `speaker_role`, `is_expert`
3. Sample beat row showing `speakers_set`, `expert_span_ids`, `expert_coverage_pct`
4. One-line count comparison: `sections_count` vs `episodes_count`
5. Confirmation message: "Validator produced zero schema warnings"

---

## 9. Testing Requirements

### Unit Tests
- **MUST** run all existing unit tests to ensure no regressions
- Existing tests in `tests/` directory should pass without modification (unless testing outdated behavior)
- If existing tests fail due to new schema, update tests to accommodate new fields

### Integration Tests
- **MUST** run the full pipeline from raw transcripts through QA report generation
- **MUST** verify output includes new fields in spans, beats, sections tables
- **SHOULD** include a small test dataset (1-2 episodes) for fast iteration

### Quality Assurance Validation
- **MUST** regenerate `quality_reports/` on full dataset after implementation
- **MUST** manually inspect:
  - First page of QA report (executive summary)
  - Table validation map section
  - Sample rows from spans, beats, sections tables
- **MUST** confirm zero schema warnings in output

### Test Execution Order
1. Run unit tests: `pytest tests/`
2. Run integration tests: `pytest tests/integration/`
3. Regenerate lakehouse tables: `lakehouse ingest` (or equivalent command)
4. Generate QA report: `lakehouse quality-report` (or equivalent command)
5. Inspect output files and confirm success metrics

---

## 10. Implementation Priority

While all requirements (R1–R4) must be completed, the recommended implementation order is:

1. **R3 (Validator Routing)** — Highest priority; blocks clean QA runs
2. **R1 (Speaker Mapping)** — High priority; foundational metadata
3. **R2 (Sections Rebuild)** — Medium priority; improves content organization
4. **R4 (Report Fixes)** — Continuous; fix issues as they appear during R1–R3 implementation

However, all four requirements **must be complete** before considering this feature done.

---

## 11. Open Questions

1. **Section Title Generation:** Should titles be extracted from first beat text, or use a simple numbering scheme ("Section 1", "Section 2")? 
   - *Recommendation:* Start simple with numbering, add text extraction if time permits
   
2. **Similarity Threshold for Sections:** What's the minimum cosine similarity drop to trigger a new section boundary?
   - *Recommendation:* Default to 0.15–0.20 drop, make configurable
   
3. **Minimum Section Duration:** Should we enforce minimum/maximum section lengths?
   - *Recommendation:* Minimum 2 minutes, maximum 30 minutes, configurable in `config/aggregation_config.yaml`
   
4. **Speaker Role Enum Extension:** Are there additional roles beyond `["expert", "host", "guest", "caller", "other"]`?
   - *Recommendation:* Start with these five, extend if needed based on transcript analysis
   
5. **Expert Coverage Calculation:** If a beat has zero spans (edge case), what should `expert_coverage_pct` be?
   - *Recommendation:* Default to 0.0, log a warning

6. **Backwards Compatibility Testing:** Should we maintain old table versions temporarily for comparison?
   - *Recommendation:* Yes, archive v1 tables as `v1_archived/` before regenerating

---

## 12. Appendix: Sample Outputs

### Sample Span Row (JSON representation)
```json
{
  "span_id": "span_001_0042",
  "episode_id": "ep_001",
  "speaker": "Fr Stephen De Young",
  "speaker_canonical": "Fr Stephen De Young",
  "speaker_role": "expert",
  "is_expert": true,
  "text": "The liturgical context here is really important...",
  "t_start": 142.5,
  "t_end": 156.3,
  "token_count": 87
}
```

### Sample Beat Row (JSON representation)
```json
{
  "beat_id": "beat_001_0008",
  "episode_id": "ep_001",
  "span_ids": ["span_001_0042", "span_001_0043", "span_001_0044"],
  "speakers_set": ["Fr Stephen De Young", "Jonathan Pageau"],
  "expert_span_ids": ["span_001_0042", "span_001_0044"],
  "expert_coverage_pct": 68.5,
  "t_start": 142.5,
  "t_end": 189.2,
  "text": "The liturgical context here is really important... [combined beat text]"
}
```

### Sample Section Count Output
```
Episodes: 24
Sections: 156
Ratio: 6.5 sections per episode
```

### Sample "Changes in This Run" Report Section
```markdown
## Changes in This Run

- **Speaker Metadata Added:** All spans and beats now include speaker role and expert flags based on `config/speaker_roles.yaml`
- **Semantic Sections:** Episodes are now divided into multiple topic-based sections (average 6.5 per episode)
- **Validator Routing Fixed:** Text/timestamp checks run only on base tables; vector checks run only on embedding tables
- **Coverage Calculation Corrected:** Coverage now computed as overlap-aware union on spans only
```

---

**End of PRD**

