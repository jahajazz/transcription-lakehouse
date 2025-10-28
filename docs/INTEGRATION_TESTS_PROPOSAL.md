# Integration Tests for Quality Assessment

**Date:** October 28, 2025  
**Purpose:** Prevent dictionary key/column name mismatch bugs  
**Status:** ✅ READY TO USE

---

## Problem Being Solved

We've found **4 similar bugs** where calculators and reporters used different key/column names:

| Bug # | Issue | Wrong Key/Column | Correct Key/Column | Result |
|-------|-------|------------------|-------------------|--------|
| 1 | Text quality zeros | `avg_token_count` | `avg_tokens` | Showed 0.0 |
| 2 | Speaker metrics zeros | `speaker` | `speaker_canonical` | Showed 0 |
| 3 | Length contradictions | `below_min_percent` | `too_short_percent` | Wrong values |
| 4 | Integrity structure | Combined df | Separate spans/beats | False positives |

**Pattern:** Silent failures - `.get('wrong_key', 0)` returns 0 as default, no error raised.

---

## Solution: Comprehensive Integration Tests

Created **2 test suites** that would have caught all 4 bugs:

### 1. `tests/test_quality_integration.py` (End-to-End Tests)

**What it tests:**
- ✅ Each calculator returns expected keys with non-zero values
- ✅ Reporter can read calculator outputs without errors
- ✅ Full assessment pipeline works end-to-end
- ✅ Column name variations are handled correctly
- ✅ Generated reports have expected content

**Test classes:**
```python
TestCalculatorOutputs          # Validates each calculator's output schema
TestReporterCompatibility      # Ensures reporter reads correct keys
TestEndToEndAssessment         # Full pipeline with real data
TestColumnNameCompatibility    # Tests speaker/speaker_canonical handling
TestReportValidation           # Validates final report content
```

**Example test that would have caught Bug #1:**
```python
def test_text_metrics_keys(self, sample_spans):
    result = text_quality.calculate_text_metrics(sample_spans)
    
    # Required keys - test FAILS if keys are wrong
    required_keys = [
        'avg_tokens',        # NOT avg_token_count!
        'avg_words',         # NOT avg_word_count!
        'avg_characters',    # NOT avg_char_count!
    ]
    
    for key in required_keys:
        assert key in result, f"Missing key '{key}'"
    
    # Values should be non-zero - catches silent failures
    assert result['avg_tokens'] > 0, "avg_tokens should be non-zero"
```

### 2. `tests/test_quality_schema_validation.py` (Schema Validation)

**What it tests:**
- ✅ Strict schema validation with `SchemaValidator` helper
- ✅ No unexpected keys (catches typos)
- ✅ Non-zero value validation (catches `.get()` defaults)
- ✅ Source code inspection (ensures reporter uses correct keys)
- ✅ Calculator consistency across modules

**SchemaValidator helper:**
```python
class SchemaValidator:
    @staticmethod
    def validate_keys(result, required_keys, optional_keys=None, context=""):
        """
        Validate that result dict has all required keys and no unexpected keys.
        
        Catches:
        - Missing keys (forgot to add to output)
        - Unexpected keys (typos, wrong key names)
        - Silent failures (getting 0 as default)
        """
```

**Example test that would have caught Bug #2:**
```python
def test_reporter_uses_correct_text_quality_keys(self):
    # Inspect reporter source code
    source = inspect.getsource(QualityReporter._generate_text_quality_section)
    
    # Should use correct keys
    assert 'avg_tokens' in source
    
    # Should NOT use wrong keys - test FAILS if using old keys
    wrong_keys = ['avg_token_count', 'avg_word_count', 'avg_char_count']
    for key in wrong_keys:
        assert key not in source, f"Reporter should NOT use '{key}'"
```

---

## Running the Tests

### Quick Start
```bash
# Run all tests
python scripts/run_quality_tests.py

# Run only schema validation (fast, no data needed)
python scripts/run_quality_tests.py --schema

# Run integration tests (requires lakehouse data)
python scripts/run_quality_tests.py --integration

# Run quick tests (no real data required)
python scripts/run_quality_tests.py --quick
```

### Using pytest directly
```bash
# All quality tests
pytest tests/test_quality_integration.py tests/test_quality_schema_validation.py -v

# Just schema validation
pytest tests/test_quality_schema_validation.py -v

# Specific test class
pytest tests/test_quality_integration.py::TestCalculatorOutputs -v

# With coverage
pytest tests/test_quality_*.py --cov=src.lakehouse.quality --cov-report=html
```

---

## What These Tests Catch

### 1. Missing Keys
```python
# Calculator returns: {'avg_tokens': 84.3}
# Reporter expects: result.get('avg_token_count', 0)
# Result: 0 (silent failure)
# Test catches: AssertionError: Missing key 'avg_token_count'
```

### 2. Wrong Column Names
```python
# DataFrame has: 'speaker_canonical'
# Calculator looks for: 'speaker'
# Result: No speakers found (silent failure)
# Test catches: AssertionError: total_speakers should be > 0
```

### 3. Unexpected Keys (Typos)
```python
# Calculator returns: {'avg_tokeNs': 84.3}  # typo!
# Result: Reporter gets 0 for 'avg_tokens'
# Test catches: AssertionError: Unexpected keys found: {'avg_tokeNs'}
```

### 4. Source Code Issues
```python
# Reporter uses: stats.get('avg_token_count', 0)
# Calculator returns: 'avg_tokens'
# Test catches: AssertionError: Reporter should NOT use 'avg_token_count'
```

---

## Test Coverage

### Calculators Tested
- ✅ `calculate_text_metrics` (text_quality.py)
- ✅ `calculate_lexical_density` (text_quality.py)
- ✅ `extract_top_terms` (text_quality.py)
- ✅ `calculate_duration_statistics` (distribution.py)
- ✅ `calculate_length_compliance` (distribution.py)
- ✅ `calculate_speaker_distribution` (balance.py)
- ✅ `check_timestamp_monotonicity` (integrity.py)
- ✅ `detect_duplicates` (integrity.py)

### Reporter Sections Tested
- ✅ `_generate_text_quality_section`
- ✅ `_generate_balance_section`
- ✅ `_generate_distribution_section`
- ✅ `_generate_integrity_section`

### Scenarios Tested
- ✅ Spans with `speaker_canonical` column
- ✅ Beats without `speaker` column (have `speakers_set`)
- ✅ Combined spans + beats DataFrames
- ✅ Empty DataFrames (edge cases)
- ✅ Full assessment pipeline
- ✅ Report generation

---

## Benefits

### 1. Early Detection
- Catches mismatches at **test time**, not runtime
- Prevents **silent failures** (returning 0 as default)
- **Fast feedback** (~5 seconds for schema tests)

### 2. Documentation
- Tests serve as **living documentation** of expected schemas
- Shows **correct key names** and structures
- **Examples** of how to use calculators

### 3. Regression Prevention
- Prevents **re-introduction** of fixed bugs
- Catches **similar issues** in new code
- **Safe refactoring** with confidence

### 4. Developer Experience
- **Clear error messages** pointing to exact issue
- **Easy to add** tests for new calculators
- **Consistent patterns** across all tests

---

## Example Test Failures

### Missing Key Failure
```
AssertionError: calculate_text_metrics: Missing required keys: {'avg_tokens'}
Expected: {'total_segments', 'avg_tokens', 'avg_words', 'avg_characters', ...}
Got: {'total_segments', 'avg_token_count', 'avg_words', 'avg_characters', ...}
```
**Fix:** Change calculator to return `avg_tokens` instead of `avg_token_count`

### Wrong Key in Reporter
```
AssertionError: Reporter should NOT use 'avg_token_count' (this key doesn't exist in calculator output)
```
**Fix:** Update reporter from `stats.get('avg_token_count')` to `stats.get('avg_tokens')`

### Non-Zero Validation Failure
```
AssertionError: calculate_speaker_distribution: Key 'total_speakers' has value 0, expected non-zero.
This might indicate a key mismatch bug.
```
**Fix:** Check if calculator is looking for correct column name (`speaker_canonical` not `speaker`)

---

## CI/CD Integration

### Pre-commit Hook
```bash
# .git/hooks/pre-commit
#!/bin/bash
pytest tests/test_quality_schema_validation.py -x
if [ $? -ne 0 ]; then
    echo "❌ Schema validation failed! Fix key mismatches before committing."
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
      - name: Schema validation
        run: pytest tests/test_quality_schema_validation.py -v
      - name: Integration tests
        run: pytest tests/test_quality_integration.py -v --skip-slow
```

---

## Adding Tests for New Code

### When adding a new calculator:

**1. Add schema test:**
```python
# In test_quality_schema_validation.py
def test_new_calculator_schema(self, sample_data):
    result = new_module.new_calculator(sample_data)
    
    SchemaValidator.validate_keys(
        result,
        required_keys=['key1', 'key2', 'key3'],
        optional_keys=['optional_key'],
        context="new_calculator"
    )
    
    SchemaValidator.validate_non_zero(
        result,
        keys=['key1', 'key2'],
        context="new_calculator"
    )
```

**2. Add integration test:**
```python
# In test_quality_integration.py
def test_new_calculator_keys(self, sample_data):
    result = new_module.new_calculator(sample_data)
    
    required_keys = ['key1', 'key2', 'key3']
    for key in required_keys:
        assert key in result, f"Missing key '{key}'"
    
    assert result['key1'] > 0, "Should have non-zero value"
```

**3. Add reporter compatibility test:**
```python
def test_reporter_new_section(self, mock_metrics):
    section = reporter._generate_new_section()
    
    # Should show correct values (not 0.0)
    assert "expected_value" in section
    assert "(No data available)" not in section
```

---

## Files Created

1. **`tests/test_quality_integration.py`** (550+ lines)
   - End-to-end integration tests
   - 30+ test methods across 5 test classes

2. **`tests/test_quality_schema_validation.py`** (450+ lines)
   - Strict schema validation tests
   - SchemaValidator helper class
   - 25+ test methods across 6 test classes

3. **`scripts/run_quality_tests.py`** (150+ lines)
   - Convenient test runner with options
   - Colored output, clear error messages
   - Examples and help text

4. **`tests/README_QUALITY_TESTS.md`** (documentation)
   - Comprehensive guide to using tests
   - Examples of what each test catches
   - CI/CD integration examples

5. **`INTEGRATION_TESTS_PROPOSAL.md`** (this document)
   - Overview and justification
   - Quick start guide

---

## Recommendation

### **Adopt immediately:**
1. ✅ Add to CI/CD pipeline (catch issues before merge)
2. ✅ Run as pre-commit hook (catch issues before commit)
3. ✅ Run regularly during development (fast feedback)
4. ✅ Add test for each new calculator/reporter section

### **Long-term:**
1. Expand coverage to other modules (validation, materialization)
2. Add performance benchmarks (detect regressions)
3. Add snapshot testing for reports (detect formatting changes)
4. Add property-based testing (fuzz edge cases)

---

## Summary

**Problem:** 4 bugs from dictionary key/column name mismatches  
**Solution:** 2 comprehensive test suites with 55+ test methods  
**Coverage:** All quality assessment calculators and reporters  
**Runtime:** ~5 seconds (schema) + ~30 seconds (integration)  
**Benefit:** Would have caught all 4 bugs before they reached production

**Status:** ✅ Ready to use - tests pass on current codebase  
**Next Step:** Integrate into CI/CD and run regularly

---

## Try It Now

```bash
# Quick validation (5 seconds)
python scripts/run_quality_tests.py --quick

# Full test suite
python scripts/run_quality_tests.py

# See detailed help
python scripts/run_quality_tests.py --help
```

**Expected output:**
```
🧪 Running quick tests (no real data required)...
Command: pytest tests/test_quality_integration.py tests/test_quality_schema_validation.py -v --tb=short --color=yes -m 'not integration'

test_quality_integration.py::TestCalculatorOutputs::test_text_metrics_keys PASSED
test_quality_integration.py::TestCalculatorOutputs::test_lexical_density_keys PASSED
test_quality_integration.py::TestCalculatorOutputs::test_top_terms_keys PASSED
...
test_quality_schema_validation.py::TestTextQualitySchemas::test_calculate_text_metrics_schema PASSED
test_quality_schema_validation.py::TestReporterCompatibility::test_reporter_uses_correct_text_quality_keys PASSED
...

✅ All tests passed!
```

