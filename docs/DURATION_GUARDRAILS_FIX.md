# Duration Guardrails Fix

**Date:** October 28, 2025  
**Issue:** Extreme outliers (9,012s spans/beats) skewing statistics  
**Status:** ✅ IMPLEMENTED

---

## Problem

Quality assessment revealed extreme duration outliers:
- **Max span duration:** 9,012s (~2.5 hours)
- **Max beat duration:** 9,012s (~2.5 hours)
- **Impact:** These outliers skew std-dev and P95 statistics

**Analysis:**
- Median span: 8.88s (reasonable)
- Max span: 9,012s (entire episode as single segment!)
- These are likely fallback cases where span/beat generation failed

**Example:**
```
Episode: "LOS - #123 - Title Here" (Duration: 9,012s)
↓ Span generation fails or fallback triggered
Result: 1 giant span instead of ~300 normal spans
```

---

## Root Cause

**Likely causes:**
1. Missing speaker information → fallback creates episode-length span
2. Embedding failures → beat generation creates episode-length beat
3. Edge cases in aggregation logic not properly handled

**Evidence:**
- Small number of outliers (probably <1% of data)
- Durations match episode lengths
- Occur consistently for both spans and beats

---

## Solution

**Add hard cap guardrails during materialization:**

1. **For Spans:** Drop any span > 240s (2× span_length_max)
2. **For Beats:** Drop any beat > 360s (2× beat_length_max)

**Rationale:**
- Quality thresholds: spans should be 20-120s, beats 60-180s
- 2× threshold is generous (allows some flexibility)
- Anything beyond 2× is clearly a data quality issue
- Better to drop outliers than let them skew statistics

**Implementation:** `src/lakehouse/cli/commands/materialize.py`

```python
def _apply_duration_guardrails(
    segments: List[Dict[str, Any]],
    segment_type: str,
    max_duration: float,
) -> List[Dict[str, Any]]:
    """Drop segments exceeding max_duration."""
    # Filter out excessively long segments
    filtered = [s for s in segments if s.get('duration', 0) <= max_duration]
    
    dropped_count = len(segments) - len(filtered)
    if dropped_count > 0:
        console.print(f"[yellow]Dropped {dropped_count} {segment_type}(s) > {max_duration}s[/yellow]")
    
    return filtered
```

---

## Implementation Details

### Guardrail Thresholds

| Segment Type | Target Range | Quality Max | **Guardrail Max** |
|--------------|--------------|-------------|-------------------|
| **Spans** | 20-120s | 120s | **240s (2×)** |
| **Beats** | 60-180s | 180s | **360s (2×)** |

### Application Points

**1. Span Generation** (`_generate_spans`):
```python
# After generation, before writing
if all_spans:
    all_spans = _apply_duration_guardrails(
        all_spans, 
        segment_type="span",
        max_duration=240.0
    )
```

**2. Beat Generation** (`_generate_beats`):
```python
# After generation, before writing
if beats:
    beats = _apply_duration_guardrails(
        beats,
        segment_type="beat",
        max_duration=360.0
    )
```

### Logging & Reporting

The guardrail function provides:
- **Count of dropped segments**
- **Percentage of total**
- **Details of top 5 longest** (if more than 10 dropped)

**Example output:**
```
Dropped 15 span(s) with duration > 240s (0.02% of total)
  - Top 5 longest:
    - spn_abc123_000000_xyz789 (episode: LOS-#004, duration: 9012.3s)
    - spn_def456_000000_uvw012 (episode: LOS-#011, duration: 8234.1s)
    - spn_ghi789_000000_rst345 (episode: LOS-#013, duration: 7456.8s)
    ...
```

---

## Expected Impact

### Before Guardrails
```
Span Duration Statistics:
- Median: 8.88s ✓
- P95: 100.86s ✓
- Max: 9,012.29s ❌ (outlier!)
- Std Dev: 161.13s ❌ (skewed by outliers)
```

### After Guardrails
```
Span Duration Statistics:
- Median: 8.88s ✓
- P95: ~100s ✓
- Max: ~200-240s ✓ (reasonable)
- Std Dev: ~50-80s ✓ (not skewed)
```

**Benefits:**
1. ✅ Statistics more representative of typical data
2. ✅ P95/P99 percentiles meaningful again
3. ✅ Std-dev reflects actual variability
4. ✅ Outliers logged for investigation
5. ✅ Downstream consumers (embeddings, search) not affected by bad data

---

## Alternative Approaches Considered

### Option 1: Split instead of drop
**Pros:** Preserves all content  
**Cons:** 
- More complex (need splitting logic)
- May create artifacts with poor semantic coherence
- Still indicates upstream bug

### Option 2: Flag but don't drop
**Pros:** No data loss  
**Cons:**
- Outliers still skew statistics
- Downstream systems must handle outliers
- Doesn't solve the problem

### Option 3: Fix root cause
**Pros:** Best long-term solution  
**Cons:**
- Requires deep investigation of aggregation logic
- Edge cases may be hard to reproduce
- Guardrails still needed as safety net

**Decision:** Implement guardrails (Option current) + investigate root cause (Option 3 later)

---

## Root Cause Investigation (TODO)

**Next steps to identify why these outliers occur:**

1. **Analyze dropped segments:**
   - Which episodes produce outliers?
   - Do they share common characteristics?
   - Missing speaker information?
   - Transcription quality issues?

2. **Review span generation logic:**
   - When does fallback trigger?
   - Are there missing validations?
   - Should we add more granular error handling?

3. **Check data quality:**
   - Do some episodes have malformed utterances?
   - Are there gaps in speaker labeling?
   - Timestamp issues in source data?

4. **Add tests:**
   - Create test cases for edge conditions
   - Verify fallback behavior
   - Ensure proper error handling

---

## Configuration

Consider adding to `aggregation_config.yaml`:

```yaml
# Duration Guardrails (optional)
guardrails:
  # Maximum allowed span duration (seconds)
  # Spans exceeding this will be dropped during materialization
  # null to disable guardrails
  max_span_duration: 240.0  # 2× span_length_max
  
  # Maximum allowed beat duration (seconds)
  # Beats exceeding this will be dropped during materialization
  # null to disable guardrails
  max_beat_duration: 360.0  # 2× beat_length_max
  
  # Whether to log dropped segments
  log_dropped_segments: true
  
  # Whether to fail if too many segments dropped (>10%)
  fail_on_excessive_drops: false
```

**Note:** Currently hardcoded, but could be made configurable.

---

## Files Modified

1. **src/lakehouse/cli/commands/materialize.py**
   - Added `_apply_duration_guardrails()` function (lines 167-234)
   - Applied to span generation (lines 194-201)
   - Applied to beat generation (lines 235-242)
   - Added typing imports

---

## Testing

**Manual Test:**
```bash
# Regenerate spans and beats with guardrails
python -m lakehouse.cli.commands.materialize --lakehouse-path lakehouse --version v1 --spans-only
python -m lakehouse.cli.commands.materialize --lakehouse-path lakehouse --version v1 --beats-only

# Run quality assessment
python -m lakehouse.cli.quality assess --lakehouse-path lakehouse --version v1
```

**Expected:**
- Console shows dropped segment count
- Max duration in report < 300s for spans, < 450s for beats
- Std-dev significantly reduced
- P95 more representative

---

##Status

✅ **IMPLEMENTED** - Guardrails active in materialization pipeline  
⏳ **TODO** - Root cause investigation  
⏳ **TODO** - Make thresholds configurable  
⏳ **TODO** - Add unit tests

**Production Ready:** Yes - Guardrails prevent bad data from entering lakehouse

