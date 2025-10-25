# PRD: Beat/Span Quality Assessment Baseline

## Introduction/Overview

The Beat/Span Quality Assessment Baseline is a comprehensive validation and diagnostic system that verifies the quality of segmented transcript data (spans and beats) and their embeddings before any downstream processing. This feature addresses the critical need to detect segmentation issues, embedding anomalies, and data integrity problems early in the pipeline, preventing poor-quality data from contaminating downstream analyses, labeling tasks, or pair generation workflows.

The system will produce both machine-readable metrics and human-readable reports that flag issues across multiple quality dimensions: coverage, length distributions, ordering integrity, speaker balance, text quality, and embedding sanity. This baseline assessment provides data engineers and ML practitioners with confidence that their segmentation and embeddings are sound before proceeding with more expensive downstream operations.

## Goals

1. **Automated Quality Verification**: Provide a single command that comprehensively validates spans and beats against configurable quality thresholds.

2. **Multi-Level Diagnostics**: Generate metrics at global, per-episode, and per-segment granularity to support both high-level health checks and deep-dive investigations.

3. **Embedding Sanity Checks**: Detect common embedding pathologies (speaker leakage, adjacency bias, length correlation, poor coherence) that would undermine semantic search and retrieval.

4. **Actionable Reporting**: Produce clear, human-readable reports with RAG (Red/Amber/Green) status indicators and specific remediation recommendations.

5. **Reproducibility**: Ensure deterministic metrics generation for the same input data to support reliable baseline comparisons.

6. **Integration with Existing Pipeline**: Seamlessly integrate with the existing CLI and lakehouse structure without requiring external services.

## User Stories

1. **As a data engineer**, I want to run a quality assessment on my latest ingestion batch so that I can verify the segmentation worked correctly before generating embeddings or running expensive downstream tasks.

2. **As an ML engineer**, I want to check if my embeddings exhibit speaker leakage or length bias so that I can decide whether to regenerate them with different model parameters.

3. **As a researcher**, I want to see detailed statistics on span and beat length distributions so that I can tune my segmentation parameters to meet target ranges.

4. **As a pipeline operator**, I want a clear Go/No-Go recommendation with RAG flags so that I can quickly decide whether to proceed with the next pipeline stage.

5. **As a developer**, I want to inspect sampled neighbor lists and outliers so that I can debug why certain segments are behaving unexpectedly in semantic search.

6. **As a quality assurance specialist**, I want to verify that coverage is high and gaps/overlaps are minimal so that I can ensure no content is lost or duplicated in the segmentation process.

## Functional Requirements

### Core Functionality

**FR-1**: The system SHALL provide a new CLI command `lakehouse quality assess` (or similar) that triggers the quality assessment.

**FR-2**: The system SHALL accept the following command-line parameters:
- `--version` or `-v`: Lakehouse version to assess (defaults to latest)
- `--level`: Assessment level(s) to run: `spans`, `beats`, or `all` (default: `all`)
- `--output-dir`: Directory for output artifacts (default: `lakehouse/quality_reports/`)
- `--sample-size`: Override for neighbor sampling and diagnostic samples (optional)
- `--config`: Path to custom threshold configuration file (optional)

**FR-3**: The system SHALL read data exclusively from the existing lakehouse structure:
- Normalized transcripts from `lakehouse/normalized/v{n}/`
- Spans from `lakehouse/spans/v{n}/spans.parquet`
- Beats from `lakehouse/beats/v{n}/beats.parquet`
- Embeddings from `lakehouse/embeddings/v{n}/span_embeddings.parquet` and `beat_embeddings.parquet`
- Episode metadata from `lakehouse/catalogs/episodes_*.parquet`

### Metrics Generation (Machine-Readable)

**FR-4**: The system SHALL generate `metrics/global.json` containing:
- Total number of episodes, spans, and beats
- Global aggregates for all duration statistics
- Global pass/fail summary for all validation thresholds
- Timestamp of assessment run
- Version information (lakehouse version, assessment tool version)

**FR-5**: The system SHALL generate `metrics/episodes.csv` with one row per episode containing:
- Episode ID, series, duration
- Span count, beat count
- Span coverage %, beat coverage %
- Gap/overlap statistics (count and total duration)
- Duration stats (min/max/mean/median for spans and beats)
- Validation pass/fail flags per threshold category

**FR-6**: The system SHALL generate `metrics/spans.csv` and `metrics/beats.csv` containing per-segment data:
- Segment ID, episode ID, speaker
- Start time, end time, duration
- Token count, word count
- Text excerpt (first 100 chars)
- Validation flags (is_too_short, is_too_long, is_negative_duration, is_duplicate, etc.)

### Coverage & Count Metrics (Category A)

**FR-7**: For each episode, the system SHALL calculate:
- Total episode duration (from metadata)
- Sum of span durations and sum of beat durations
- Coverage percentage: `(sum_durations / episode_duration) * 100` for both levels
- Total span count and total beat count

**FR-8**: The system SHALL identify and quantify gaps (time ranges with no segment) and overlaps (time ranges where segments overlap):
- Gap count, gap total duration (seconds), gap % of episode
- Overlap count, overlap total duration (seconds), overlap % of episode

**FR-9**: The system SHALL validate against thresholds:
- `coverage_min` (default: 95%): Coverage must be ≥ 95%
- `gap_max_percent` (default: 2%): Gaps must be ≤ 2% of episode duration
- `overlap_max_percent` (default: 2%): Overlaps must be ≤ 2% of episode duration

### Length & Distribution Metrics (Category B)

**FR-10**: For spans and beats separately, the system SHALL calculate duration statistics:
- Minimum, maximum, mean, median duration
- 5th percentile (p5) and 95th percentile (p95)
- Standard deviation

**FR-11**: The system SHALL calculate the proportion of segments within target bounds:
- For spans: target range 20–120 seconds
- For beats: target range 60–180 seconds
- Report % within bounds and % in each outlier category (too_short, too_long)

**FR-12**: The system SHALL generate histogram bin counts for duration distributions:
- Use sensible bins (e.g., 0-20s, 20-40s, 40-60s, 60-90s, 90-120s, 120-180s, 180+ for spans)
- Include bins appropriate for beat-level granularity

**FR-13**: The system SHALL validate against thresholds:
- `span_length_compliance_min` (default: 90%): ≥ 90% of spans within 20-120s
- `beat_length_compliance_min` (default: 90%): ≥ 90% of beats within 60-180s

### Ordering & Integrity Metrics (Category C)

**FR-14**: The system SHALL check timestamp monotonicity:
- Within each episode, verify all segments are temporally ordered by start time
- Within each speaker stream, verify temporal ordering
- Report count of timestamp regressions (where start[i+1] < start[i])

**FR-15**: The system SHALL identify integrity violations:
- Count of segments with negative duration (end < start)
- Count of segments with zero duration (end == start)
- Count of segments with missing required fields (speaker, text, timestamps)

**FR-16**: The system SHALL detect duplicate content:
- Exact duplicates: segments with identical text (case-insensitive, whitespace-normalized)
- Near-duplicates: segments with high fuzzy similarity (e.g., Levenshtein ratio > 0.95)
- Report counts and rates for both

**FR-17**: The system SHALL validate against thresholds:
- `timestamp_regressions_max` (default: 0): Zero timestamp regressions allowed
- `negative_duration_max` (default: 0): Zero negative/zero durations allowed
- `exact_duplicate_max_percent` (default: 1%): ≤ 1% exact duplicates
- `near_duplicate_max_percent` (default: 3%): ≤ 3% near-duplicates

### Speaker & Series Balance Metrics (Category D)

**FR-18**: The system SHALL calculate speaker distribution statistics:
- Total span/beat count per speaker
- Percentage of total content per speaker
- Average segment duration per speaker
- Identify top N speakers (default: N=10) and report long-tail statistics

**FR-19**: The system SHALL calculate series balance:
- Count and percentage of episodes per series (e.g., LOS vs. SW)
- Count and percentage of spans/beats per series
- Per-series duration statistics

### Text Quality Proxy Metrics (Category E)

**FR-20**: For each segment, the system SHALL calculate:
- Token count (using simple whitespace tokenization or configured tokenizer)
- Word count
- Character count (excluding whitespace)

**FR-21**: The system SHALL calculate text quality proxies:
- Lexical density: ratio of content words to total words (using stopword list)
- Punctuation ratio: count of punctuation marks / total characters
- Report aggregate statistics (mean, median, std) per level

**FR-22**: The system SHALL extract top unigrams and bigrams:
- Per level (spans, beats), extract top 20 unigrams and top 20 bigrams
- Exclude common stopwords
- Report frequency counts
- Purpose: "theme smell test" for quick content validation

### Embedding Sanity Checks (Category F)

**FR-23**: The system SHALL perform neighbor coherence analysis:
- For a stratified sample of segments (e.g., N=100, stratified by episode/speaker)
- Find top-k nearest neighbors (default: k=10) using cosine similarity
- Extract and summarize topic terms from neighbor texts
- Report if neighbors share coherent themes or appear random

**FR-24**: The system SHALL calculate speaker and episode leakage:
- For each segment in the sample, analyze its top-k neighbors
- Calculate % of neighbors with the same speaker
- Calculate % of neighbors from the same episode vs. different episodes
- Aggregate statistics across all sampled segments

**FR-25**: The system SHALL validate against thresholds:
- `same_speaker_neighbor_max_percent` (default: 60%): ≤ 60% of neighbors share same speaker
- `same_episode_neighbor_max_percent` (default: 70%): ≤ 70% of neighbors from same episode

**FR-26**: The system SHALL assess length bias in embeddings:
- Calculate correlation between segment duration and embedding L2 norm
- Calculate correlation between segment duration and mean neighbor similarity
- Report Pearson correlation coefficient

**FR-27**: The system SHALL validate against threshold:
- `length_bias_correlation_max` (default: 0.3): Absolute correlation ≤ 0.3

**FR-28**: The system SHALL assess lexical vs. embedding similarity alignment:
- For a sample of segment pairs (e.g., N=500 random pairs)
- Calculate simple lexical similarity (e.g., Jaccard or TF-IDF cosine)
- Calculate embedding cosine similarity
- Report correlation between lexical and embedding similarity
- Purpose: Sanity check that embeddings capture some lexical overlap

**FR-29**: The system SHALL calculate cross-series neighbor statistics (if multiple series present):
- % of neighbors from a different series than the query segment
- Report per-series breakdown

**FR-30**: The system SHALL assess adjacency bias:
- For each segment in the sample, calculate % of top-k neighbors that are immediate temporal neighbors (adjacent in timeline)
- An immediate neighbor is defined as: from same episode AND (start_neighbor ≈ end_query OR end_neighbor ≈ start_query) within 5-second tolerance

**FR-31**: The system SHALL validate against threshold:
- `adjacency_bias_max_percent` (default: 40%): ≤ 40% of neighbors are temporally adjacent

### Outlier & Sample Audits (Category G)

**FR-32**: The system SHALL generate `diagnostics/outliers.csv` containing:
- Top N longest segments per level (default: N=20)
- Top N shortest segments per level
- Top N most isolated segments (lowest mean neighbor similarity)
- Top N most "hubby" segments (highest mean neighbor similarity)
- Each entry includes: ID, episode, speaker, duration, text excerpt, relevant metric value

**FR-33**: The system SHALL generate `diagnostics/neighbors_sample.csv` containing:
- Random sample of neighbor lists (default: 30 query segments)
- For each query: ID, episode, speaker, text excerpt (first 100 chars)
- For each neighbor: neighbor_id, episode, speaker, similarity score, text excerpt
- Format for human spot-check of neighbor quality

**FR-34**: The system SHALL include text excerpts in all diagnostic outputs:
- Limit to first 100 characters to keep files readable
- Properly escape special characters for CSV output
- Include "..." indicator if text is truncated

### Report Generation (Human-Readable)

**FR-35**: The system SHALL generate `report/quality_assessment.md` with the following structure:
1. **Executive Summary** with RAG status
2. **Configuration & Thresholds** used in this run
3. **Coverage & Count Metrics** (Category A) with tables
4. **Length & Distribution Metrics** (Category B) with ASCII histograms
5. **Ordering & Integrity Metrics** (Category C) with pass/fail table
6. **Speaker & Series Balance** (Category D) with distribution tables
7. **Text Quality Proxies** (Category E) with top terms tables
8. **Embedding Sanity Checks** (Category F) with bias analysis
9. **Outlier Audits** (Category G) with sample excerpts
10. **Findings & Remediation** section with specific recommendations
11. **Go/No-Go Recommendation** for proceeding to next pipeline stage

**FR-36**: The Executive Summary SHALL include:
- Overall RAG status (Red/Amber/Green)
- Date/time of assessment
- Lakehouse version assessed
- High-level counts (episodes, spans, beats processed)
- Summary of critical failures (if any)
- One-sentence recommendation

**FR-37**: RAG status SHALL be determined as follows:
- **Green**: All thresholds passed, no critical issues
- **Amber**: Minor threshold violations (1-2 non-critical failures) or warnings
- **Red**: Multiple threshold failures or any critical integrity issues

**FR-38**: The Findings & Remediation section SHALL provide:
- For each failed threshold, a short explanation of the issue
- Specific remediation suggestions (e.g., "Re-segment overly long spans using tighter window", "Review speaker diarization for speaker leakage")
- Reference to relevant diagnostic files for further investigation

**FR-39**: ASCII histograms SHALL be generated for:
- Span duration distribution
- Beat duration distribution
- Format: Text-based bar chart using characters like `|`, `█`, or `#`
- Include axis labels and bin ranges

**FR-40**: The system SHALL print a console summary including:
- Overall RAG status (color-coded if terminal supports ANSI colors)
- Key metrics (episode count, span/beat counts, coverage %)
- Critical failures (if any)
- Path to full report and metrics files

### Configuration Management

**FR-41**: The system SHALL support hardcoded default thresholds as defined in requirements FR-9, FR-13, FR-17, FR-25, FR-27, FR-31.

**FR-42**: The system SHALL allow command-line overrides for key parameters:
- `--coverage-min`: Override coverage threshold
- `--span-length-min`, `--span-length-max`: Override span length target range
- `--beat-length-min`, `--beat-length-max`: Override beat length target range
- `--neighbor-k`: Override k for neighbor analysis
- `--sample-size`: Override sample size for neighbor checks

**FR-43**: All thresholds and parameters SHALL be displayed in the "Configuration & Thresholds" section of the report so users know exactly what was applied.

### Data Processing Requirements

**FR-44**: The system SHALL process ALL segments (no sampling) for:
- Coverage, count, length, ordering, integrity, speaker balance, and text quality metrics

**FR-45**: The system SHALL use sampling for computationally expensive operations:
- Neighbor coherence analysis (stratified sample)
- Lexical vs. embedding similarity (random pair sample)
- Outlier audits use full data but report top N only

**FR-46**: Sampling strategy SHALL be stratified when applicable:
- Stratify by episode to ensure representation across all episodes
- Stratify by speaker (if significant imbalance) to ensure all speakers represented
- Use reproducible random seed for deterministic results

**FR-47**: The system SHALL compute cosine similarity for embeddings using efficient vectorized operations:
- Use FAISS, NumPy, or similar optimized libraries
- Batch similarity computations where possible

### Output & File Management

**FR-48**: The system SHALL create output directory structure:
```
lakehouse/quality_reports/YYYYMMDD_HHMMSS/
├── metrics/
│   ├── global.json
│   ├── episodes.csv
│   ├── spans.csv
│   └── beats.csv
├── diagnostics/
│   ├── neighbors_sample.csv
│   └── outliers.csv
└── report/
    └── quality_assessment.md
```

**FR-49**: The system SHALL use timestamps in directory names for versioning (format: `YYYYMMDD_HHMMSS`).

**FR-50**: The system SHALL handle missing embeddings gracefully:
- If embeddings are not present, skip FR-23 through FR-31 (embedding sanity checks)
- Report in the summary that embedding checks were skipped
- Do not fail the entire assessment

### Reproducibility & Determinism

**FR-51**: The system SHALL produce identical metrics (excluding timestamps) for the same input data when run multiple times.

**FR-52**: Random sampling SHALL use a fixed seed (or configurable seed) to ensure reproducibility.

**FR-53**: All computed metrics SHALL be rounded to consistent precision (e.g., 2 decimal places for percentages, 3 for correlations) for consistent output.

### Error Handling

**FR-54**: The system SHALL fail gracefully and report errors if:
- Required input files (spans, beats, episodes) are missing
- Input files have schema mismatches
- Lakehouse version specified does not exist

**FR-55**: The system SHALL log warnings (but continue processing) for:
- Individual episodes with missing or corrupt data
- Segments with malformed fields (use defaults/skip)
- Partial embedding coverage (some segments without embeddings)

## Non-Goals (Out of Scope)

1. **Automatic Remediation**: The system will NOT automatically fix issues (e.g., re-segment spans, regenerate embeddings). It only provides diagnostics and recommendations.

2. **Real-time Monitoring**: The system will NOT provide continuous monitoring or alerting. It is designed for on-demand batch assessment.

3. **Historical Tracking**: The system will NOT compare metrics to previous runs or track trends over time. Each run is independent.

4. **External Services**: The system will NOT call external APIs or services. All processing happens locally using lakehouse data.

5. **Interactive Dashboards**: The system will NOT provide web UIs or interactive visualizations. Output is files and console text only.

6. **Blocking Downstream Processes**: The system will NOT prevent other pipeline stages from running. It is informational only, though the Go/No-Go recommendation can guide manual decisions.

7. **Custom Embedding Model Evaluation**: The system will NOT evaluate different embedding models or suggest model improvements. It only checks for common pathologies in whatever embeddings are present.

8. **Visual Plots (PNG/SVG)**: The system will NOT generate graphical plots. ASCII/text-based visualizations are sufficient per user requirements.

## Design Considerations

### CLI Integration

- Add new command group under `lakehouse quality` or extend existing CLI structure in `src/lakehouse/cli/commands/`
- Follow existing CLI patterns for argument parsing, logging, and output
- Use existing logger configuration from `src/lakehouse/logger.py`

### Module Organization

Suggested module structure:
```
src/lakehouse/quality/
├── __init__.py
├── assessor.py          # Main assessment orchestration
├── metrics/
│   ├── __init__.py
│   ├── coverage.py      # Category A: Coverage & Counts
│   ├── distribution.py  # Category B: Length & Distribution
│   ├── integrity.py     # Category C: Ordering & Integrity
│   ├── balance.py       # Category D: Speaker & Series Balance
│   ├── text_quality.py  # Category E: Text Quality Proxies
│   └── embedding.py     # Category F: Embedding Sanity Checks
├── diagnostics.py       # Category G: Outliers & Samples
├── thresholds.py        # Threshold definitions and validation
└── reporter.py          # Markdown report generation
```

### Data Loading

- Reuse existing data loading utilities from `src/lakehouse/ingestion/reader.py` and related modules
- Load data once and pass DataFrames between metric calculators to avoid repeated I/O
- Consider memory constraints: for very large datasets, consider chunked processing for per-segment CSV outputs

### ASCII Visualization

For histograms in the markdown report, use a simple ASCII bar chart format:
```
Duration Distribution (Spans):
0-20s   : ████░░░░░░ 23 (4.2%)
20-40s  : ████████░░ 87 (15.8%)
40-60s  : ██████████ 145 (26.4%)
60-90s  : ████████░░ 112 (20.4%)
90-120s : ██████░░░░ 89 (16.2%)
120-180s: ███░░░░░░░ 54 (9.8%)
180+s   : ██░░░░░░░░ 38 (6.9%)
```

## Technical Considerations

### Dependencies

- **pandas**: For data manipulation and CSV I/O
- **pyarrow**: For reading parquet files
- **numpy**: For vectorized calculations
- **scipy**: For correlation calculations
- **faiss** (optional): For efficient nearest neighbor search if embeddings present
- **fuzzywuzzy** or **rapidfuzz**: For fuzzy string matching (near-duplicate detection)
- **nltk** or **spacy** (lightweight): For stopword lists and text processing

### Performance Considerations

- Most metrics should complete in seconds for datasets of 500 episodes with ~50K total segments
- Neighbor searches are the most expensive operation: use FAISS or equivalent for efficiency
- If FAISS is not available, fall back to brute-force NumPy cosine similarity (acceptable for up to ~100K segments)
- Consider multiprocessing for per-episode metrics if processing becomes a bottleneck

### Extensibility

- The threshold configuration should be easy to extend with new metrics in the future
- Metric calculators should be modular so new checks can be added without modifying existing code
- Report generator should accept a metrics dictionary to easily accommodate new sections

### Integration with Existing Validation

- The existing validation framework in `src/lakehouse/validation/` focuses on schema and basic integrity
- This quality assessment is complementary: it operates at a higher semantic level
- Consider whether to merge or keep separate: recommendation is to keep separate initially since this is a specialized, heavy-weight analysis vs. fast schema validation

## Success Metrics

1. **Completeness**: All required metrics (FR-7 through FR-34) are generated and present in outputs.

2. **Accuracy**: Manual spot-checks of sampled neighbor lists and outliers confirm correctness (human review of 10+ samples).

3. **Usability**: A junior developer can read the markdown report and understand what issues exist and what actions to take within 10 minutes.

4. **Reliability**: Re-running the assessment on the same data produces identical metrics (excluding timestamp headers).

5. **Performance**: Full assessment completes within 5 minutes for a dataset of 500 episodes (~50K segments, embeddings present).

6. **Adoption**: Data engineers use this assessment before every major downstream task (pair generation, labeling, fine-tuning).

## Open Questions

1. **Neighbor Sample Size**: Is a default sample of 100 query segments sufficient for meaningful neighbor coherence analysis, or should it be larger (e.g., 1% of total segments, minimum 100)?

2. **Fuzzy Duplicate Threshold**: The near-duplicate detection uses a similarity threshold (e.g., Levenshtein ratio > 0.95). Should this be configurable or is a fixed threshold acceptable?

3. **Text Quality Metrics**: The current text quality proxies are basic (token count, lexical density). Are there additional simple text quality signals that would be valuable without requiring NLP models (e.g., average sentence length, vocabulary richness)?

4. **Embedding Distance Metric**: Should the system support multiple distance metrics (cosine, L2, dot product) for neighbor analysis, or is cosine similarity sufficient?

5. **Report Format Preference**: Is Markdown the preferred format, or would an HTML report (still static file, no server) with better formatting be valuable in the future?

6. **Incremental Assessment**: If only a subset of episodes have changed, should the system support incremental assessment (assess only new/changed episodes), or is full reassessment always acceptable?

7. **Threshold Tuning**: The initial thresholds are based on the original prompt. Should there be a "calibration" mode that runs without thresholds to help users establish appropriate baselines for their specific dataset?

---

**Document Metadata**
- **Version**: 1.0
- **Created**: 2025-10-25
- **Target Audience**: Data Engineers, ML Engineers, Junior Developers
- **PRD Number**: 0002

