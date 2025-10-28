# Length Compliance Reporting Fix

**Date:** October 28, 2025  
**Issue:** Contradictory length bucket reporting for spans  
**Status:** ✅ RESOLVED (Copy-paste error)

---

## Problem

Quality report showed contradictory values:

**Span Duration Statistics:**
- Median: 8.88s ✅ (correctly calculated)

**Length Compliance:**
- Within bounds (20-120s): 28.30%
- **Below minimum: 0.00%** ❌ (WRONG!)
- Above maximum: 0.00% ❌ (WRONG!)

**The Contradiction:**
If the median is 8.88s and the minimum is 20s, then **over 50% of spans must be below minimum**. Yet the report claimed 0%!

**Credit:** Excellent catch by the user who noticed the stats and buckets were using different columns.

---

## Root Cause

**Copy-paste error in reporter:**

The span section was using **wrong key names**:

```python
# WRONG (line 572):
section += f"- Below minimum: {compliance.get('below_min_percent', 0):.2f}%\n"
section += f"- Above maximum: {compliance.get('above_max_percent', 0):.2f}%\n"
```

But `calculate_length_compliance()` returns:
- `too_short_percent` (not `below_min_percent`)
- `too_long_percent` (not `above_max_percent`)

Since these keys didn't exist, `.get()` returned the default value of 0.

**Interestingly:** The beat section used the correct key names (`too_short_percent`, `too_long_percent`), which is why beats showed correct values (82.77% below minimum).

---

## Solution

**Fixed key names in reporter:**

```python
# CORRECT (line 572-573):
section += f"- Below minimum: {compliance.get('too_short_percent', 0):.2f}%\n"
section += f"- Above maximum: {compliance.get('too_long_percent', 0):.2f}%\n"
```

**File:** `src/lakehouse/quality/reporter.py` (lines 572-573)

---

## Test Results

### Before Fix

```
Span Duration Statistics:
- Median: 8.88s

Length Compliance:
- Within bounds (20-120s): 28.30%
- Below minimum: 0.00%        ← WRONG
- Above maximum: 0.00%         ← WRONG
```

### After Fix

```
Span Duration Statistics:
- Median: 8.88s

Length Compliance:
- Within bounds (20-120s): 28.30%
- Below minimum: 68.04%        ← CORRECT!
- Above maximum: 3.66%         ← CORRECT!
```

### Validation

- ✅ Median (8.88s) < Minimum (20s) → Most spans should be too short
- ✅ 68.04% are too short → Consistent with median!
- ✅ Sum: 28.30% + 68.04% + 3.66% = 100.00% ✓

---

## Impact

### Before
- Report was **misleading** and confusing
- Appeared contradictory (median 8.88s but 0% below minimum)
- Users couldn't trust the compliance metrics

### After
- Report is **accurate** and consistent
- Statistics match compliance buckets
- Users can make informed decisions about data quality

---

## Why This Matters

Length compliance is important for:
1. **Span quality:** Ideal spans are 20-120s for semantic coherence
2. **Beat quality:** Ideal beats are 60-180s for topic boundaries
3. **Production readiness:** Knowing what % of data meets quality standards

With the bug, users couldn't trust these critical metrics.

---

## Related Issues

This is the **4th bug** we've fixed in quality reporting, all related to data structure misunderstanding:

1. ✅ Timestamp regressions (69,398 → 0) - Hierarchical overlap
2. ✅ Exact duplicates (37.21% → 3.4%) - Hierarchical overlap
3. ✅ Missing speaker (63,374 → 0) - Wrong column name
4. ✅ **Length compliance (0% → 68.04%)** - Wrong dictionary keys

**Pattern:** Copy-paste errors and schema mismatches are common sources of reporting bugs.

---

## Files Modified

1. **src/lakehouse/quality/reporter.py** (lines 572-573)
   - Fixed span length compliance key names
   - Changed `below_min_percent` → `too_short_percent`
   - Changed `above_max_percent` → `too_long_percent`

2. **LENGTH_COMPLIANCE_FIX.md** (this document)
   - Documentation of the issue and fix

---

## Lessons Learned

1. **Test Reports End-to-End:** Don't just test calculations, verify the report output
2. **Consistency Checks:** Statistics should be internally consistent (e.g., median vs buckets)
3. **Key Name Standards:** Use consistent naming across codebase
4. **Copy-Paste Dangers:** When copying code, verify all variable/key names

---

## Recommended Follow-Up

Consider adding **consistency validation** to the reporter:

```python
def _validate_length_consistency(self, stats, compliance):
    """Verify stats and compliance are consistent."""
    median = stats.get('median', 0)
    too_short_pct = compliance.get('too_short_percent', 0)
    
    # If median < min_duration, expect most to be too short
    if median < 20 and too_short_pct < 40:
        logger.warning(
            f"Inconsistency: median ({median}s) < 20s "
            f"but only {too_short_pct}% are too short"
        )
```

This would have caught the bug automatically!

---

**Status:** ✅ FIXED AND VERIFIED  
**Production Ready:** Yes - Report now shows accurate metrics

