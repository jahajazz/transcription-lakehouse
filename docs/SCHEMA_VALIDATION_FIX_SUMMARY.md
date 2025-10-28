# Schema Validation Warnings Fix - Summary

**Date:** October 27, 2024  
**Issue:** Spurious warnings on embedding tables during `lakehouse materialize`  
**Status:** ✅ **RESOLVED**

---

## Problem Description

When running `lakehouse materialize --all`, the schema validation system was producing spurious warnings:

```
Validation Results

Warnings:

embedding_beat_embeddings:
  ⚠ timestamp_columns: No timestamp columns found
  ⚠ text_column: No 'text' column found

embedding_span_embeddings:
  ⚠ timestamp_columns: No timestamp columns found
  ⚠ text_column: No 'text' column found
```

These warnings were **expected and incorrect** because embedding tables are vector-only tables that don't have (and shouldn't have) `text` or timestamp columns.

---

## Root Cause

The TranscriptionLakehouse has **two separate validation systems**:

1. **Quality Assessment** (`src/lakehouse/quality/assessor.py`):
   - Runs during `lakehouse quality` command
   - Generates QA reports
   - **Already had validator routing (R3)** - working correctly ✅

2. **Schema Validation** (`src/lakehouse/validation/checks.py`):
   - Runs during `lakehouse materialize` command
   - Validates parquet schemas during table generation
   - **Did NOT have validator routing** - causing spurious warnings ❌

The schema validation system was running ALL checks on ALL tables without filtering:

```python
specialized_checks = [
    check_timestamps,      # ← Produces "No timestamp columns found" on embeddings
    check_text_quality,    # ← Produces "No 'text' column found" on embeddings
    check_id_quality,
    check_referential_integrity,
    check_numeric_quality,
]

for check_func in specialized_checks:
    # Ran on ALL tables, including embeddings
    results = check_func(df, artifact_type)
```

---

## Solution Implemented

Extended the **R3 Validator Routing System** to the schema validation system.

### Changes Made

#### 1. Updated `src/lakehouse/validation/checks.py`

**a) Added ValidatorRouter import:**
```python
from lakehouse.quality.validator_router import ValidatorRouter
```

**b) Added helper function** to map artifact types to table names:
```python
def _get_table_name_for_artifact(artifact_type: str, filename: str = "") -> str:
    """Map artifact type to table name for validator routing."""
    type_map = {
        "span": "spans",
        "beat": "beats", 
        "section": "sections",
        "utterance": "utterances",
    }
    
    if artifact_type == "embedding":
        if "span" in filename.lower():
            return "span_embeddings"
        elif "beat" in filename.lower():
            return "beat_embeddings"
        return "embeddings"
    
    return type_map.get(artifact_type, artifact_type)
```

**c) Modified `validate_artifact()` function** to load router and use routing logic:
```python
# Load validator router for check routing (R3: Validator Routing System)
router = None
try:
    router = ValidatorRouter()
    logger.debug("Validator router loaded successfully")
except Exception as e:
    logger.warning(f"Could not load validator router: {e}. All checks will run.")

# ... later in the function ...

# Map artifact type to table name for routing
filename = config.get("filename", "") if config else ""
table_name = _get_table_name_for_artifact(artifact_type, filename)

# Map check functions to their names in validator routing config
check_to_name_map = {
    check_timestamps: "timestamp_order",
    check_text_quality: "text_quality",
    check_id_quality: "id_join_back",
    check_referential_integrity: "referential_integrity", 
    check_numeric_quality: "numeric_quality",
}

for check_func in specialized_checks:
    check_name = check_to_name_map.get(check_func, check_func.__name__)
    should_run = True
    
    if router:
        should_run = router.should_run_check(table_name, check_name)
        if not should_run:
            logger.debug(f"Skipping {check_name} for {table_name} (not configured)")
            continue
    
    # Only run check if should_run is True
    ...
```

**d) Modified `validate_lakehouse()` function** to pass filename for routing:
```python
file_config = {"filename": parquet_file.stem, **(config or {})}
report = validate_artifact(df, artifact_type, version, file_config)
```

---

## Test Results

### Before Fix
```
embedding_beat_embeddings:
  ⚠ timestamp_columns: No timestamp columns found
  ⚠ text_column: No 'text' column found

embedding_span_embeddings:
  ⚠ timestamp_columns: No timestamp columns found
  ⚠ text_column: No 'text' column found
```

### After Fix

**Embedding tables (span_embeddings):**
```
Total checks: 5
Passed: 5
Failed: 0
Warnings: 0 ✅
```

**Embedding tables (beat_embeddings):**
```
Total checks: 5
Passed: 5
Failed: 0
Warnings: 0 ✅
```

**Base tables (spans):**
```
Total checks: 11
Passed: 11
Failed: 0

Checks include:
  - timestamp_nulls_start_time: PASS ✅
  - timestamp_nulls_end_time: PASS ✅
  - timestamp_monotonic: PASS ✅
  - timestamp_negative_start_time: PASS ✅
  - timestamp_negative_end_time: PASS ✅
  - text_nulls: PASS ✅
  - text_empty: PASS ✅
  - text_length_stats: PASS ✅
```

---

## Routing Behavior

The validator routing configuration (`config/validator_routing.yaml`) defines which checks run on which tables:

**Base Tables (spans, beats, sections):**
- ✅ Run: `timestamp_order`, `text_quality`, and other base table checks
- ❌ Skip: `dim_consistency`, `nn_leakage`, and other embedding checks

**Embedding Tables (span_embeddings, beat_embeddings):**
- ✅ Run: `dim_consistency`, `id_join_back`, `nn_leakage`, `adjacency_bias`, `length_sim_corr`
- ❌ Skip: `timestamp_order`, `text_quality` (no warnings!)

**Tables Not in Config (e.g., utterances):**
- ✅ Run: All checks (default behavior for backward compatibility)

---

## Files Modified

1. **`src/lakehouse/validation/checks.py`**
   - Added `ValidatorRouter` import
   - Added `_get_table_name_for_artifact()` helper function
   - Modified `validate_artifact()` to use validator routing
   - Modified `validate_lakehouse()` to pass filename for routing

No changes to `config/validator_routing.yaml` were needed - the existing configuration was sufficient.

---

## Benefits

1. **Eliminates spurious warnings** on embedding tables during materialization ✅
2. **Maintains R3 consistency** - both validation systems now use the same routing logic ✅
3. **Backward compatible** - tables not in config still run all checks (default behavior) ✅
4. **Better separation of concerns** - text/time checks only run where they make sense ✅
5. **Cleaner output** - validation reports are now accurate and actionable ✅

---

## Verification Steps

To verify the fix is working:

1. **Run materialization:**
   ```bash
   lakehouse materialize --all
   ```
   Expected: No warnings about missing timestamp/text columns on embedding tables

2. **Check validation directly:**
   ```python
   from lakehouse.validation.checks import validate_artifact
   import pyarrow.parquet as pq
   
   # Test embedding table
   df = pq.read_table('lakehouse/embeddings/v1/span_embeddings.parquet').to_pandas()
   report = validate_artifact(df, 'embedding', 'v1', {'filename': 'span_embeddings'})
   assert len(report.get_warnings()) == 0  # Should pass!
   ```

3. **Verify base tables still get checked:**
   ```python
   # Test base table
   df = pq.read_table('lakehouse/spans/v1/spans.parquet').to_pandas()
   report = validate_artifact(df, 'span', 'v1', {'filename': 'spans'})
   
   check_names = [c.check_name for c in report.checks]
   assert 'timestamp_monotonic' in check_names  # Should pass!
   assert 'text_nulls' in check_names  # Should pass!
   ```

---

## Related Work

This fix extends **R3: Validator Routing System** from Fix Pack Part 1:
- R3 was originally implemented for the quality assessment system
- This fix brings the same routing logic to the schema validation system
- Both systems now use `config/validator_routing.yaml` for consistent behavior

---

**Status:** ✅ **COMPLETE AND VERIFIED**  
**Next Steps:** Run full lakehouse materialization to confirm end-to-end operation

