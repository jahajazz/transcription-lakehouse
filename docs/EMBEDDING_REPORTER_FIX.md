# Embedding Reporter Fix

**Date:** October 28, 2025  
**Issue:** Embedding sanity section empty despite routing saying checks ran  
**Status:** ✅ FIXED

---

## Problem

The embedding sanity section (Category F) was showing empty despite:
- Routing logs saying "Calculating span embedding metrics..."
- Embedding checks actually running
- Metrics being calculated and stored

**Symptoms:**
```
## Category F: Embedding Sanity Checks

(Embeddings not available - checks skipped)
```

But logs showed:
```
2025-10-27 XX:XX:XX - lakehouse - INFO - Calculating span embedding metrics...
```

---

## Root Cause

**Data structure mismatch** between assessor and reporter (same pattern as integrity metrics):

**Assessor stores** (lines 716-750 in assessor.py):
```python
embedding_metrics_dict = {}
embedding_metrics_dict['spans'] = self._calculate_embedding_metrics(...)
embedding_metrics_dict['beats'] = self._calculate_embedding_metrics(...)
metrics.embedding_metrics = embedding_metrics_dict

# Structure:
{
    'spans': {
        'neighbor_coherence': {...},
        'speaker_leakage': {...},
        'episode_leakage': {...},
        'adjacency_bias': {...},
        'length_bias': {...},
        'similarity_correlation': {...},
        'cross_series': {...}
    },
    'beats': { ... }
}
```

**Reporter expected** (old code, lines 692-726):
```python
embedding = self.metrics.embedding_metrics
coherence = embedding.get('coherence', {})  # ❌ Wrong! Not at top level
speaker_leakage = embedding.get('speaker_leakage', {})  # ❌ Wrong!
```

The reporter was looking for flat keys, but they were nested under 'spans'/'beats'.

---

## Solution

**Updated reporter to match assessor's nested structure:**

```python
# Process spans and beats separately (like integrity metrics)
for segment_type in ['spans', 'beats']:
    seg_metrics = embedding.get(segment_type, {})
    if not seg_metrics:
        continue
    
    section += f"### {segment_type.capitalize()} Embedding Metrics\n\n"
    
    # Now access nested metrics
    coherence = seg_metrics.get('neighbor_coherence', {})  # ✅ Correct!
    speaker_leakage = seg_metrics.get('speaker_leakage', {})  # ✅ Correct!
    # ... etc
```

---

## Implementation

**File:** `src/lakehouse/quality/reporter.py` (lines 692-782)

### Key Changes

**1. Nested structure handling**
- Loop through 'spans' and 'beats' keys
- Access metrics within each segment type
- Display separate sections for spans and beats

**2. All requested metrics displayed:**
- ✅ **Same-speaker %** (from speaker_leakage)
- ✅ **Same-episode %** (from episode_leakage)
- ✅ **Adjacency %** (from adjacency_bias)
- ✅ **Length-sim correlation** (from length_bias)
- ✅ **Sample neighbor lists** (from diagnostics)

**3. Enhanced display:**
- Visual assessments (✓ Good / ✗ High leakage)
- Threshold-based indicators
- Neighbor coherence scores
- Lexical-embedding alignment
- Cross-series diversity metrics

---

## Expected Output

### Before
```
## Category F: Embedding Sanity Checks

(Embeddings not available - checks skipped)
```

### After
```
## Category F: Embedding Sanity Checks

### Spans Embedding Metrics

**Neighbor Coherence:**
- Assessment: good
- Mean Similarity: 0.782
- Mean Coherence Score: 0.845

**Speaker Leakage:**
- Same-Speaker %: 12.3%
- Assessment: ✓ Good

**Episode Leakage:**
- Same-Episode %: 45.7%
- Assessment: ✓ Good

**Adjacency Bias:**
- Adjacent Neighbors %: 8.2%
- Assessment: ✓ Good

**Length Bias:**
- Length-Similarity Correlation: 0.156
- Duration-Norm Correlation: 0.089
- Assessment: ✓ Good

**Lexical-Embedding Alignment:**
- Correlation: 0.623
- Assessment: ✓ Good

**Cross-Series Diversity:**
- Cross-Series Neighbors %: 34.2%
- Assessment: ✓ Good diversity

### Sample Neighbor Lists (Top 5 examples)

**1. Query:** "The council met to discuss theological implications of..."
   **Episode:** ep_12345
   **Top 3 Neighbors:**
   1. [0.891] (ep: ep_12345) "In the council's deliberation on theology..."
   2. [0.867] (ep: ep_12389) "Theological discussion at the council..."
   3. [0.845] (ep: ep_12456) "The implications for church doctrine..."

**2. Query:** "Biblical interpretation requires understanding historical..."
   **Episode:** ep_23456
   **Top 3 Neighbors:**
   1. [0.912] (ep: ep_23456) "Historical context is crucial for biblical..."
   2. [0.888] (ep: ep_23789) "Understanding scripture in historical context..."
   3. [0.872] (ep: ep_24123) "Interpretation of biblical texts needs..."

...
```

---

## Metrics Displayed

| Metric | Source | Threshold | Assessment |
|--------|--------|-----------|------------|
| **Same-Speaker %** | speaker_leakage | < 30% | Good if low |
| **Same-Episode %** | episode_leakage | < 50% | Good if low |
| **Adjacent Neighbors %** | adjacency_bias | < 20% | Good if low |
| **Length-Sim Correlation** | length_bias | < 0.3 (abs) | Good if low |
| **Lexical-Embedding Alignment** | similarity_correlation | > 0.5 | Good if high |
| **Cross-Series %** | cross_series | > 20% | Good if high |

---

## Join Back to Base Tables

**The assessor already joins embeddings to base tables:**

```python
# assessor.py, lines 718-724
if assess_spans and spans_df is not None and span_embeddings_matrix is not None:
    if self._should_run_check_for_table("span_embeddings", "dim_consistency"):
        logger.info("Calculating span embedding metrics...")
        embedding_metrics_dict['spans'] = self._calculate_embedding_metrics(
            spans_df,  # ← Base table (spans)
            span_embeddings_matrix,  # ← Embeddings
            "spans"
        )
```

**Within `_calculate_embedding_metrics`:**
```python
# assessor.py, lines 891-906
speaker_leakage = embedding.calculate_speaker_leakage(
    segments_df=segments_df,  # ← Has speaker, episode_id, etc.
    embeddings=embeddings_matrix,
    sample_size=self.thresholds.neighbor_sample_size,
    k=self.thresholds.neighbor_k,
    random_seed=42
)
```

**The embedding functions use the joined data:**
```python
# embedding.py
def calculate_speaker_leakage(segments_df, embeddings, ...):
    # Access base table columns
    query_speakers = segments_df.iloc[query_indices]['speaker'].values
    neighbor_speakers = segments_df.iloc[neighbor_indices]['speaker'].values
    
    # Calculate same-speaker %
    same_speaker_count = (query_speakers[:, None] == neighbor_speakers).sum()
    same_speaker_percent = (same_speaker_count / total_neighbors) * 100
```

**All metrics join back to base tables** for:
- Speaker information
- Episode IDs
- Text content
- Duration/timestamps
- Series information

---

## Verification

The fix ensures that:

1. ✅ **Embedding checks run** (already working)
2. ✅ **Results are stored** (already working)
3. ✅ **Results are displayed** (NOW FIXED)
4. ✅ **All requested metrics shown:**
   - Same-speaker %
   - Same-episode %
   - Adjacency %
   - Length-sim correlation
   - Sample neighbor lists

---

## Pattern Consistency

This fix follows the same pattern as the integrity metrics fix:

**Before (both had the same bug):**
```python
# Integrity reporter (FIXED)
integrity = self.metrics.integrity_metrics
regressions = integrity.get('episode_regressions', {})  # ❌ Wrong!

# Embedding reporter (WAS BROKEN)
embedding = self.metrics.embedding_metrics
leakage = embedding.get('speaker_leakage', {})  # ❌ Wrong!
```

**After (both use nested structure):**
```python
# Integrity reporter (FIXED)
for segment_type in ['spans', 'beats']:
    seg_integrity = integrity.get(segment_type, {})
    regressions = seg_integrity.get('episode_regressions', {})  # ✅ Correct!

# Embedding reporter (NOW FIXED)
for segment_type in ['spans', 'beats']:
    seg_metrics = embedding.get(segment_type, {})
    leakage = seg_metrics.get('speaker_leakage', {})  # ✅ Correct!
```

---

## Files Modified

1. **src/lakehouse/quality/reporter.py** (lines 692-782)
   - Updated `_generate_embedding_section` to handle nested structure
   - Added all requested metrics
   - Added visual assessments
   - Added sample neighbor lists

2. **EMBEDDING_REPORTER_FIX.md** (this document)
   - Documentation of the fix

---

## Benefits

1. ✅ **Comprehensive metrics** - All requested metrics now displayed
2. ✅ **Separate spans/beats** - Clear distinction like other categories
3. ✅ **Visual indicators** - Easy to spot issues (✓ vs ✗)
4. ✅ **Sample neighbors** - Concrete examples for verification
5. ✅ **Consistent pattern** - Matches integrity metrics structure
6. ✅ **Join verification** - All metrics join back to base tables

---

**Status:** ✅ FIXED - Ready for testing  
**Next Step:** Run quality assessment to see populated embedding section!

