# Quality Assessment Report

**Lakehouse Version:** v1
**Report Generated:** 2025-10-26 20:51:40

---


## Executive Summary

**Overall Status:** 🔴 **RED**

**Assessment Date:** 2025-10-26 20:48:19

**Dataset Overview:**
- Episodes: 5
- Spans: 81,115
- Beats: 63,374
- Embeddings Available: Yes

**Quality Check Results:**
- ✅ Passed: 12 checks
- ⚠️ Warnings: 1 issues
- ❌ Errors: 2 critical failures

**Processing Time:** 200.46 seconds

**Critical Issues:**
- timestamp_regressions_max: Segment timestamp regressions (69398) exceed maximum allowed (0). Segments have overlapping or out-of-order timestamps at episode level.
- exact_duplicate_max_percent: Segment exact duplicates (37.38%) exceed maximum allowed (1.0%). 54007/144489 segments have identical normalized text.


## Configuration

**Thresholds Used:**

### Coverage

- `coverage_min`: 95.0
- `gap_max_percent`: 2.0
- `overlap_max_percent`: 2.0

### Length

- `span_length_min`: 20.0
- `span_length_max`: 120.0
- `span_length_compliance_min`: 90.0
- `beat_length_min`: 60.0
- `beat_length_max`: 180.0
- `beat_length_compliance_min`: 90.0

### Integrity

- `timestamp_regressions_max`: 0
- `negative_duration_max`: 0
- `exact_duplicate_max_percent`: 1.0
- `near_duplicate_max_percent`: 3.0
- `near_duplicate_threshold`: 0.95

### Embedding

- `same_speaker_neighbor_max_percent`: 60.0
- `same_episode_neighbor_max_percent`: 70.0
- `length_bias_correlation_max`: 0.3
- `adjacency_bias_max_percent`: 40.0
- `adjacency_tolerance_seconds`: 5.0

### Sampling

- `neighbor_sample_size`: 100
- `neighbor_k`: 10
- `random_pairs_sample_size`: 500
- `outlier_count`: 20
- `neighbor_list_sample_size`: 30
- `top_speakers_count`: 10
- `random_seed`: 42



## Category A: Coverage & Count Metrics

### Global Coverage

- Total Episodes: 5
- Total Spans: 81,115
- Total Beats: 63,374
- Span Coverage: 15605.91%
- Beat Coverage: 15788.03%


## Category B: Length & Distribution Metrics

### Span Duration Statistics

- Mean: 31.08s
- Median: 8.88s
- Std Dev: 161.13s
- Min: 1.00s
- Max: 9012.29s
- P5: 1.28s
- P95: 100.86s

**Length Compliance:**
- Within bounds (20-120s): 28.30%
- Below minimum: 0.00%
- Above maximum: 0.00%

### Beat Duration Statistics

- Mean: 40.24s
- Median: 10.72s
- Std Dev: 183.28s
- Min: 1.00s
- Max: 9012.29s

**Length Compliance:**
- Within bounds (60-180s): 14.26%


## Category C: Ordering & Integrity Metrics

- Timestamp Regressions: 0
- Negative Durations: 0
- Exact Duplicates: 54,007 (37.38%)
- Near Duplicates: 0 (0.00%)


## Category D: Speaker & Series Balance

- Unique Speakers: 0
- Segments per Speaker (avg): 0.0


## Category E: Text Quality Proxy Metrics

- Average Token Count: 0.0
- Average Word Count: 0.0
- Average Character Count: 0.0
- Average Lexical Density: 0.000


## Category F: Embedding Sanity Checks



## Category G: Diagnostics & Outliers

### Longest

Found 20 outliers. See `diagnostics/outliers.csv` for details.

### Shortest

Found 20 outliers. See `diagnostics/outliers.csv` for details.

### Most Isolated

Found 20 outliers. See `diagnostics/outliers.csv` for details.

### Most Hubby

Found 20 outliers. See `diagnostics/outliers.csv` for details.



## Findings and Recommendations

### Data Integrity Issues

**Issue:** Segment timestamp regressions (69398) exceed maximum allowed (0). Segments have overlapping or out-of-order timestamps at episode level.

**Impact:** Data integrity issues can cause processing errors and unreliable results.

**Remediation:**
- Review data pipeline for timestamp handling errors
- Implement deduplication in ingestion pipeline
- Validate input data before processing

**Issue:** Segment exact duplicates (37.38%) exceed maximum allowed (1.0%). 54007/144489 segments have identical normalized text.

**Impact:** Data integrity issues can cause processing errors and unreliable results.

**Remediation:**
- Review data pipeline for timestamp handling errors
- Implement deduplication in ingestion pipeline
- Validate input data before processing

### Embedding Leakage Issues

**Issue:** 63374 segment(s) are missing speaker. All segments should have a valid speaker identifier.

**Impact:** Embeddings may be encoding metadata rather than semantic content.

**Remediation:**
- Review embedding model for potential biases
- Consider using a different embedding model
- Ensure text normalization removes speaker/episode markers
- Test with cross-validation across episodes/speakers



## Go/No-Go Recommendation

### ❌ **NO-GO** - Not Ready for Production

The dataset has critical quality issues that must be resolved before production use.

**Recommendation:**
- **DO NOT** proceed with production deployment
- Address all critical (error-level) violations
- Re-run quality assessment after fixes
- Review data pipeline for systematic issues

**Confidence Level:** High

Proceeding without fixes could lead to:
- Poor search quality
- Unreliable semantic matching
- System instability
- User experience issues

**Required Actions:** See "Findings and Recommendations" section above for specific remediation steps.


---

*Report generated by Transcript Lakehouse Quality Assessment System*

*For questions or issues, refer to the quality assessment documentation.*
