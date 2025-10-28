# Speaker Validation Fix

**Date:** October 28, 2025  
**Issue:** "63374 beat(s) are missing speaker"  
**Status:** ✅ RESOLVED (False Positive)

---

## Problem

Quality assessment reported:
> **Warning:** 63374 beat(s) are missing speaker. All segments should have a valid speaker identifier.

This affected **ALL beats** in the system (100% of beats flagged).

---

## Root Cause

**Another false positive** due to schema misunderstanding:

The validation logic checked for a `speaker` column (singular), but:
- ✅ **Spans** have: `speaker` (string) - one speaker per span
- ❌ **Beats** have: `speakers_set` (array) - multiple speakers per beat

**Why this design?**
Beats aggregate multiple spans from potentially different speakers. A beat with:
```
span_ids: [span_1, span_2, span_3]
```

Might have:
- span_1: Alice speaking
- span_2: Bob speaking  
- span_3: Alice speaking

So the beat has: `speakers_set: ["Alice", "Bob"]`

This is **by design**, not a data quality issue!

---

## Solution

Updated validation logic to recognize both column formats:

**File:** `src/lakehouse/quality/metrics/integrity.py` (lines 255-284)

```python
# Check for missing speaker (column name varies by segment type)
# - Spans have 'speaker' (single string)
# - Beats have 'speakers_set' (array of strings)

if 'speaker' in segments.columns:
    # Spans: check for missing/empty speaker
    missing_speaker = segments['speaker'].isna() | (segments['speaker'] == '')
    missing_speaker_indices = segments[missing_speaker].index.tolist()
    
elif 'speakers_set' in segments.columns:
    # Beats: check for empty speakers_set array
    for idx, row in segments.iterrows():
        speakers_set = row.get('speakers_set', [])
        if speakers_set is None or (isinstance(speakers_set, list) and len(speakers_set) == 0):
            missing_speaker_indices.append(idx)
```

**Also updated:** `src/lakehouse/quality/metrics/balance.py`
- Added clarifying comments about speaker column behavior
- Made it clear that beats are intentionally excluded from speaker balance calculations

---

## Test Results

### Unit Tests

**Test 1: Spans with 'speaker' column**
- Total spans: 4
- Expected violations: 2 (empty string + None)
- Actual violations: 2 ✅

**Test 2: Beats with 'speakers_set' column**
- Total beats: 4
- Expected violations: 2 (empty array + None)
- Actual violations: 2 ✅

**Test 3: Valid beats**
- Total beats: 3
- Expected violations: 0 (all have valid speakers_set)
- Actual violations: 0 ✅

### Integration Test (Real Data)

**Before Fix:**
- Span missing speaker: (unknown)
- Beat missing speaker: 63,374 ❌
- **All beats** flagged as invalid

**After Fix:**
- Span missing speaker: 0 ✅
- Beat missing speaker: 0 ✅
- **100% reduction** in false positives

---

## Impact

### Before
```
⚠️ Warning: 63374 beat(s) are missing speaker
Status: Misleading warning about data quality
```

### After
```
✅ No speaker violations
Status: Accurate representation of data quality
```

### Violations Summary Update

**Old Report:**
- 4 violations total
- "63,374 beats missing speaker" (warning)

**New Report:**
- 2-3 violations total
- Speaker validation accurate for both spans and beats

---

## Related to Critical Fixes

This is the **third false positive** we've fixed, all related to treating hierarchical data structures incorrectly:

1. ✅ **Timestamp Regressions** (69,398 → 0) - Beats overlap spans by design
2. ✅ **Exact Duplicates** (37.21% → 3.4%) - Beat text contains span text by design
3. ✅ **Missing Speaker** (63,374 → 0) - Beats use `speakers_set` not `speaker` by design

**Pattern:** All issues stem from applying span-level validation logic to beat-level data without accounting for structural differences.

---

## Files Modified

1. **src/lakehouse/quality/metrics/integrity.py** (lines 255-284)
   - Made speaker validation column-name aware
   - Handles both `speaker` (spans) and `speakers_set` (beats)

2. **src/lakehouse/quality/metrics/balance.py** (lines 73-106)
   - Added clarifying comments
   - Changed log level from error to debug for missing speaker column

3. **SPEAKER_VALIDATION_FIX.md** (this document)
   - Documentation of the issue and fix

---

## Schema Reference

For future reference, here's how speaker data is structured:

### Spans Schema
```python
pa.field("speaker", pa.string(), nullable=False)
pa.field("speaker_canonical", pa.string(), nullable=False)
pa.field("speaker_role", pa.string(), nullable=False)
pa.field("is_expert", pa.bool_(), nullable=False)
```

### Beats Schema
```python
pa.field("speakers_set", pa.list_(pa.string()), nullable=False)
pa.field("expert_span_ids", pa.list_(pa.string()), nullable=False)
pa.field("expert_coverage_pct", pa.float64(), nullable=False)
```

**Key Difference:** Spans = single speaker, Beats = array of speakers

---

## Lessons Learned

1. **Schema Awareness:** Validation logic must be aware of table-specific schemas
2. **Column Name Variations:** Different tables may use different column names for similar concepts
3. **Hierarchical Relationships:** Aggregated data (beats) has different structure than base data (spans)
4. **Test Thoroughly:** Always test with both unit tests and real data

---

## Next Steps

- ✅ Fix verified with unit tests
- ✅ Fix verified with integration tests
- ✅ Documentation updated
- ⚠️ Consider: Create a schema registry or validation config to prevent future issues

**Status:** COMPLETE - Ready for production

