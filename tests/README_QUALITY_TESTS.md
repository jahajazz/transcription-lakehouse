# Quality Assessment Integration Tests

## Overview

These tests are designed to catch **dictionary key/column name mismatches** between calculators and reporters - the type of bugs that caused:

1. ❌ Text quality showing 0.0 (`avg_token_count` vs `avg_tokens`)
2. ❌ Speaker metrics showing 0 (`speaker` vs `speaker_canonical`)  
3. ❌ Length compliance contradictions (`below_min_percent` vs `too_short_percent`)
4. ❌ Integrity metrics structure issues

## Test Files

### 1. `test_quality_integration.py`
**Purpose:** End-to-end integration tests

**What it tests:**
- ✅ All calculators return expected keys
- ✅ Reporter can read calculator outputs without errors
- ✅ Full assessment pipeline produces non-zero values
- ✅ Column name variations are handled (speaker vs speaker_canonical)
- ✅ Generated reports have expected content

**Key test classes:**
- `TestCalculatorOutputs` - Validates each calculator's output schema
- `TestReporterCompatibility` - Ensures reporter reads correct keys
- `TestEndToEndAssessment` - Full pipeline validation
- `TestColumnNameCompatibility` - Tests column name handling
- `TestReportValidation` - Validates final report content

### 2. `test_quality_schema_validation.py`
**Purpose:** Schema validation using strict assertions

**What it tests:**
- ✅ Exact key names match between calculators and reporters
- ✅ No unexpected keys (catches typos)
- ✅ Non-zero validation (catches silent failures)
- ✅ Source code inspection (ensures reporter uses correct keys)
- ✅ Calculator consistency

**Key classes:**
- `SchemaValidator` - Helper for strict schema validation
- `TestTextQualitySchemas` - Validates text quality schemas
- `TestDistributionSchemas` - Validates distribution schemas
- `TestBalanceSchemas` - Validates balance schemas
- `TestIntegritySchemas` - Validates integrity schemas
- `TestReporterCompatibility` - Inspects reporter source code

## Running the Tests

### Run all quality tests
```bash
pytest tests/test_quality_integration.py tests/test_quality_schema_validation.py -v
```

### Run only integration tests
```bash
pytest tests/test_quality_integration.py -v
```

### Run only schema validation tests
```bash
pytest tests/test_quality_schema_validation.py -v
```

### Run specific test class
```bash
pytest tests/test_quality_integration.py::TestCalculatorOutputs -v
```

### Run with coverage
```bash
pytest tests/test_quality_integration.py --cov=src.lakehouse.quality --cov-report=html
```

### Run integration tests only (requires lakehouse data)
```bash
pytest tests/test_quality_integration.py -v -m integration
```

## What These Tests Would Have Caught

### Bug #1: Text Quality Keys
```python
# test_quality_integration.py::test_text_metrics_keys
def test_text_metrics_keys(self, sample_spans):
    result = text_quality.calculate_text_metrics(sample_spans)
    
    required_keys = [
        'avg_tokens',  # NOT avg_token_count!
        'avg_words',   # NOT avg_word_count!
        'avg_characters',  # NOT avg_char_count!
    ]
    
    for key in required_keys:
        assert key in result  # Would FAIL if keys were wrong
```

### Bug #2: Speaker Column Names
```python
# test_quality_integration.py::test_speaker_column_variants
def test_speaker_column_variants(self):
    # DataFrame with speaker_canonical (new format)
    df = pd.DataFrame({'speaker_canonical': ['Alice', 'Bob']})
    
    result = balance.calculate_speaker_distribution(df)
    assert result['total_speakers'] > 0  # Would FAIL if looking for wrong column
```

### Bug #3: Length Compliance Keys
```python
# test_quality_schema_validation.py::test_calculate_length_compliance_schema
def test_calculate_length_compliance_schema(self):
    required_keys = [
        'too_short_percent',  # NOT below_min_percent!
        'too_long_percent',   # NOT above_max_percent!
    ]
    
    SchemaValidator.validate_keys(result, required_keys)  # Would FAIL if keys were wrong
```

### Bug #4: Reporter Source Code Inspection
```python
# test_quality_schema_validation.py::test_reporter_uses_correct_text_quality_keys
def test_reporter_uses_correct_text_quality_keys(self):
    source = inspect.getsource(QualityReporter._generate_text_quality_section)
    
    # Should use correct keys
    assert 'avg_tokens' in source
    
    # Should NOT use wrong keys
    assert 'avg_token_count' not in source  # Would FAIL if using wrong keys
```

## Test Strategy

### 1. Schema Validation
```python
# Strict validation of output structure
SchemaValidator.validate_keys(
    result,
    required_keys=['avg_tokens', 'avg_words'],  # Must be present
    optional_keys=['per_segment_stats'],  # May be present
    context="calculate_text_metrics"
)
```

### 2. Non-Zero Validation
```python
# Catch silent failures (getting 0 as default)
SchemaValidator.validate_non_zero(
    result,
    keys=['avg_tokens', 'total_speakers'],
    context="Must have non-zero values"
)
```

### 3. End-to-End Validation
```python
# Test full pipeline with real data
result = assessor.run_assessment()

# Verify metrics are populated
assert result.metrics.text_quality_metrics['statistics']['avg_tokens'] > 0
```

### 4. Source Code Inspection
```python
# Ensure reporter uses correct keys
source = inspect.getsource(reporter_method)
assert 'correct_key' in source
assert 'wrong_key' not in source
```

## Adding New Tests

When adding new calculators or reporters, add tests following this pattern:

### 1. Add calculator schema test
```python
# In test_quality_schema_validation.py
def test_new_calculator_schema(self):
    result = new_calculator(sample_data)
    
    SchemaValidator.validate_keys(
        result,
        required_keys=['expected_key1', 'expected_key2'],
        context="new_calculator"
    )
```

### 2. Add reporter compatibility test
```python
# In test_quality_integration.py
def test_reporter_new_section(self, mock_metrics):
    section = reporter._generate_new_section()
    
    # Should show correct values (not 0.0)
    assert "expected_value" in section
```

### 3. Add source code inspection
```python
# In test_quality_schema_validation.py
def test_reporter_uses_correct_keys(self):
    source = inspect.getsource(reporter_method)
    assert 'correct_key' in source
    assert 'wrong_key' not in source
```

## CI/CD Integration

### Pre-commit Hook
```bash
# .git/hooks/pre-commit
#!/bin/bash
pytest tests/test_quality_schema_validation.py -x
if [ $? -ne 0 ]; then
    echo "Schema validation failed! Fix key mismatches before committing."
    exit 1
fi
```

### GitHub Actions
```yaml
# .github/workflows/quality-tests.yml
name: Quality Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run schema validation
        run: pytest tests/test_quality_schema_validation.py -v
      - name: Run integration tests
        run: pytest tests/test_quality_integration.py -v
```

## Benefits

### 1. Early Detection
- Catches key mismatches at test time, not runtime
- Prevents silent failures (returning 0 as default)

### 2. Documentation
- Tests serve as documentation of expected schemas
- Shows correct key names and structures

### 3. Regression Prevention
- Prevents re-introduction of fixed bugs
- Catches similar issues in new code

### 4. Confidence
- Safe to refactor calculators/reporters
- Clear failures when contracts break

## Common Issues

### Issue: "Missing required keys"
**Cause:** Calculator output doesn't match expected schema  
**Fix:** Update calculator to return correct keys, or update test if schema intentionally changed

### Issue: "Unexpected keys found"
**Cause:** Calculator returns keys not in required/optional lists  
**Fix:** Add keys to optional_keys if legitimate, or remove from calculator if not needed

### Issue: "Key has value 0, expected non-zero"
**Cause:** Calculator returns 0 (might be using wrong source data or column name)  
**Fix:** Check calculator logic and column names

### Issue: "Reporter should use 'X' not 'Y'"
**Cause:** Reporter using wrong key name in source code  
**Fix:** Update reporter to use correct key from calculator output

## Maintenance

### When to Update Tests

1. **Adding new calculator:** Add schema validation test
2. **Changing calculator output:** Update required_keys in tests
3. **Adding new reporter section:** Add compatibility test
4. **Refactoring:** Run tests frequently to catch breaks

### Best Practices

1. **Run tests before committing:** Catch issues early
2. **Add test for each bug fix:** Prevent regressions
3. **Keep schemas documented:** Update tests when schemas change
4. **Use descriptive assertion messages:** Help future developers

## Related Documentation

- `TEXT_QUALITY_FIX.md` - Text quality key mismatch fix
- `SPEAKER_METRICS_FIX.md` - Speaker column name fix
- `LENGTH_COMPLIANCE_FIX.md` - Length compliance key fix
- `CRITICAL_FIX_SUMMARY.md` - Overview of all fixes

---

**Status:** ✅ Tests ready to use  
**Coverage:** All calculators and reporters  
**Run time:** ~5 seconds for schema tests, ~30 seconds for integration tests

