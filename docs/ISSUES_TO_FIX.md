# Critical Issues to Fix - Quality Assessment Module

## Issue #1: API Parameter Mismatch (BLOCKING)

**Priority**: 🔴 CRITICAL  
**Estimated Fix Time**: 15 minutes

### Problem
Function called with wrong parameter names, causing all quality assessment tests to fail.

### Location
`src/lakehouse/quality/assessor.py`

### Current Code (Lines 457-462)
```python
span_compliance = distribution.calculate_length_compliance(
    spans_df,
    min_length=self.thresholds.span_length_min,    # ❌ WRONG
    max_length=self.thresholds.span_length_max,    # ❌ WRONG
    segment_type="spans"
)
```

### Required Fix
```python
span_compliance = distribution.calculate_length_compliance(
    spans_df,
    min_duration=self.thresholds.span_length_min,  # ✅ CORRECT
    max_duration=self.thresholds.span_length_max,  # ✅ CORRECT
    segment_type="spans"
)
```

### Also Check
Lines around 479-484 for beat compliance calculation - likely has the same issue.

### Function Signature (Reference)
```python
# From src/lakehouse/quality/metrics/distribution.py:111
def calculate_length_compliance(
    segments: pd.DataFrame,
    min_duration: float,    # ← Note: min_DURATION not min_length
    max_duration: float,    # ← Note: max_DURATION not max_length
    segment_type: str = "segment",
) -> Dict[str, Any]:
```

### Impact
- **70 tests failing** (all quality assessment tests)
- Blocks production deployment
- No data corruption risk (fails fast)

---

## Issue #2: Logger Thread Safety in Tests

**Priority**: 🟠 HIGH  
**Estimated Fix Time**: 30 minutes

### Problem
Global logger instance causes "I/O operation on closed file" errors during pytest cleanup.

### Error
```
ValueError: I/O operation on closed file.
Call stack:
  File "/usr/lib/python3.13/logging/__init__.py", line 1153, in emit
    stream.write(msg + self.terminator)
```

### Root Cause
- Global logger in `src/lakehouse/logger.py:88-101`
- Handlers attached to stdout/stderr
- Pytest captures and closes these streams
- Logger tries to write during cleanup → error

### Recommended Fix Options

#### Option A: Use pytest fixtures (RECOMMENDED)
```python
# In tests/conftest.py
import pytest
from lakehouse.logger import setup_logger
import logging

@pytest.fixture(autouse=True)
def configure_logger_for_tests():
    """Configure logger for test environment."""
    logger = setup_logger(
        name="lakehouse",
        level="INFO",
        console_output=False  # Disable console output in tests
    )
    # Add NullHandler to prevent errors
    logger.addHandler(logging.NullHandler())
    yield logger
    # Cleanup
    logger.handlers.clear()
```

#### Option B: Use caplog fixture
```python
# In individual tests
def test_something(caplog):
    with caplog.at_level(logging.INFO):
        # Test code
        pass
    # Check logs
    assert "expected message" in caplog.text
```

#### Option C: Clear handlers in teardown
```python
# In tests/conftest.py
@pytest.fixture(autouse=True)
def cleanup_logger():
    yield
    # Clear all handlers after each test
    import logging
    logger = logging.getLogger("lakehouse")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
```

### Impact
- Tests fail during cleanup (not during execution)
- Makes test output noisy
- May mask real failures
- Does not affect production code

---

## Issue #3: Missing Optional Dependency

**Priority**: 🟡 MEDIUM  
**Estimated Fix Time**: 5 minutes

### Problem
`rapidfuzz` library used but not in dependencies.

### Location
`src/lakehouse/quality/metrics/integrity.py:19-23`

### Current Code
```python
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logger.warning("rapidfuzz not available. Near-duplicate detection will use fallback method.")
```

### Fix Required
Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
quality = [
    "rapidfuzz>=3.0.0",
]
```

Or add to main dependencies:
```toml
dependencies = [
    # ... existing dependencies ...
    "rapidfuzz>=3.0.0",
]
```

### Impact
- Fallback method works but is slower
- Less accurate near-duplicate detection
- Warning message in logs
- Not blocking (has graceful fallback)

---

## Issue #4: Low Test Coverage in Key Modules

**Priority**: 🟡 MEDIUM  
**Estimated Fix Time**: 2-4 hours

### Problem
Some critical modules have very low test coverage:

```
Module                       Coverage
────────────────────────────────────
quality/reporter.py              9%   🔴
quality/assessor.py             47%   🔴
quality/metrics/embedding.py    29%   🔴
quality/metrics/balance.py      39%   🔴
```

### Root Cause
Integration tests failing → orchestration and reporting code not exercised.

### Expected After Issue #1 Fixed
Coverage should jump to 60-70% once integration tests pass.

### Additional Work Needed
- Add specific tests for report generation
- Test edge cases in embedding metrics
- Test error conditions in assessor

### Impact
- Risk of undetected bugs
- Harder to refactor safely
- Less confidence in production behavior
- Not blocking for initial deployment

---

## Recommended Fix Order

1. **Issue #1** (15 min) - API mismatch - MUST FIX FIRST
2. **Run Tests** (5 min) - Verify fixes work
3. **Issue #2** (30 min) - Logger cleanup - SHOULD FIX BEFORE MERGE
4. **Issue #3** (5 min) - Add dependency - SHOULD FIX
5. **Issue #4** (later) - Test coverage - CAN DEFER TO FOLLOW-UP PR

**Total Critical Path Time**: ~1 hour to get to green tests

---

## Verification Steps

After applying fixes:

```bash
# 1. Run all tests
pytest tests/ -v

# Expected: 275/275 passed (100%)

# 2. Run quality tests specifically
pytest tests/test_quality_*.py -v

# Expected: All 103 tests pass

# 3. Check coverage
pytest --cov=lakehouse --cov-report=term-missing tests/

# Expected: Overall coverage >65%, quality module >60%

# 4. Run linter
ruff check src/

# Expected: No errors

# 5. Test actual CLI command (integration test)
lakehouse quality --help

# Expected: Help text displays correctly
```

---

## Notes

- **No architectural changes needed** - these are implementation bugs, not design flaws
- **No breaking changes** - all fixes are internal
- **No data migration needed** - no schema changes
- **Low risk** - issues are well-understood and isolated

The architecture is sound. These are straightforward fixes that should take less than 2 hours total.
