# Text Quality Metrics Fix

**Date:** October 28, 2025  
**Issue:** "Text-quality proxies all zero" - avg tokens/words/chars showing 0.0  
**Status:** ✅ FIXED

---

## Problem

Category E (Text Quality Proxy Metrics) showed all zeros:
```
- Average Token Count: 0.0
- Average Word Count: 0.0
- Average Character Count: 0.0
- Average Lexical Density: 0.000
```

Even though the text data exists and is being processed!

---

## Root Cause

**Another dictionary key mismatch** (same pattern as speaker metrics and length compliance bugs):

The reporter was looking for keys that don't exist in the returned dictionaries:

| Reporter Expected | Calculator Returns | Result |
|-------------------|-------------------|--------|
| `avg_token_count` | `avg_tokens` | .get() returns 0 (default) |
| `avg_word_count` | `avg_words` | .get() returns 0 (default) |
| `avg_char_count` | `avg_characters` | .get() returns 0 (default) |

**Pattern:** Copy-paste errors and inconsistent naming conventions across modules.

---

## Solution

**Updated reporter to use correct dictionary keys from the calculators:**

### Text Statistics (from `calculate_text_metrics`)
```python
# BEFORE (wrong keys)
stats.get('avg_token_count', 0)  # ← Returns 0 (key doesn't exist)
stats.get('avg_word_count', 0)   # ← Returns 0 (key doesn't exist)
stats.get('avg_char_count', 0)   # ← Returns 0 (key doesn't exist)

# AFTER (correct keys)
stats.get('avg_tokens', 0)       # ← Returns 84.3 ✅
stats.get('avg_words', 0)        # ← Returns 84.13 ✅
stats.get('avg_characters', 0)   # ← Returns 457.99 ✅
```

### Lexical Density (from `calculate_lexical_density`)
```python
# Added proper display of lexical density metrics
lexical.get('lexical_density', 0)      # Overall density
lexical.get('avg_lexical_density', 0)  # Per-segment average
lexical.get('content_words', 0)        # Content word count
lexical.get('stopword_count', 0)       # Stopword count
```

### Top Terms (from `extract_top_terms`)
```python
# BEFORE (wrong keys)
top_terms.get('top_terms', [])  # ← Returns [] (key doesn't exist)

# AFTER (correct keys)
top_terms.get('top_unigrams', [])  # ← Returns top words ✅
top_terms.get('top_bigrams', [])   # ← Returns top phrases ✅
```

---

## Implementation

**File:** `src/lakehouse/quality/reporter.py` (lines 659-690)

### Changes

1. **Fixed text statistics keys** (lines 662-665)
   - Changed `avg_token_count` → `avg_tokens`
   - Changed `avg_word_count` → `avg_words`
   - Changed `avg_char_count` → `avg_characters`
   - Added `total_segments` count

2. **Enhanced lexical density display** (lines 667-673)
   - Show overall and per-segment lexical density
   - Display content words vs stopwords breakdown
   - More informative metrics

3. **Fixed top terms display** (lines 675-688)
   - Show both unigrams (single words) and bigrams (phrases)
   - Changed `top_terms` → `top_unigrams` and `top_bigrams`
   - Better formatting with occurrence counts

---

## Expected Impact

### Before
```
Category E: Text Quality Proxy Metrics

- Average Token Count: 0.0           ← WRONG!
- Average Word Count: 0.0             ← WRONG!
- Average Character Count: 0.0        ← WRONG!
- Average Lexical Density: 0.000      ← WRONG!
```

### After
```
Category E: Text Quality Proxy Metrics

- Average Token Count: 84.3           ← CORRECT!
- Average Word Count: 84.13           ← CORRECT!
- Average Character Count: 457.99     ← CORRECT!
- Total Segments: 81,115

**Lexical Density (Content vs Stopwords):**
- Overall Lexical Density: 0.493      ← CORRECT!
- Average per Segment: 0.541
- Content Words: 3,364,745
- Stopwords: 3,459,577

**Top 10 Terms (unigrams, excluding stopwords):**
1. right: 70,691 occurrences
2. like: 68,750 occurrences
3. it's: 64,624 occurrences
4. know: 54,655 occurrences
5. that's: 36,118 occurrences
...

**Top 10 Bigrams:**
1. "you know": 41,192 occurrences
2. "kind of": 23,849 occurrences
3. "going to": 16,561 occurrences
...
```

---

## Testing

### Quick Verification

```python
import pandas as pd
from lakehouse.quality.metrics.text_quality import (
    calculate_text_metrics,
    calculate_lexical_density,
    extract_top_terms
)

spans = pd.read_parquet('lakehouse/spans/v1/spans.parquet')

# Test text metrics
text_metrics = calculate_text_metrics(spans)
print(f"Avg tokens: {text_metrics['avg_tokens']}")  # Should be ~84.3

# Test lexical density
lexical = calculate_lexical_density(spans)
print(f"Lexical density: {lexical['lexical_density']}")  # Should be ~0.493

# Test top terms
terms = extract_top_terms(spans, top_n=10)
print(f"Top term: {terms['top_unigrams'][0]}")  # Should show (word, count)
```

### After Quality Assessment

```bash
lakehouse quality
```

**Expected output:**
```
Category E: Text Quality Proxy Metrics
- Average Token Count: 84.3
- Average Word Count: 84.13
- Lexical Density: 0.493
- Top terms populated with actual content
```

---

## Related Calculators

**Understanding the text quality calculators:**

| Calculator | Purpose | Key Metrics |
|------------|---------|-------------|
| `calculate_text_metrics` | Basic text statistics | `avg_tokens`, `avg_words`, `avg_characters` |
| `calculate_lexical_density` | Content vs stopword ratio | `lexical_density`, `content_words`, `stopword_count` |
| `extract_top_terms` | Theme analysis | `top_unigrams`, `top_bigrams` |

**Text column:**
- Both spans and beats have a `text` column ✅
- No need for column mapping (unlike speaker metrics)
- The calculators correctly read this column

---

## Why This Happened

**Pattern across multiple fixes:**

| Fix | Issue | Root Cause |
|-----|-------|-----------|
| Speaker Metrics | Looking for `speaker` instead of `speaker_canonical` | Column name change after enrichment |
| Length Compliance | Looking for `below_min_percent` instead of `too_short_percent` | Inconsistent naming in calculator |
| Text Quality | Looking for `avg_token_count` instead of `avg_tokens` | Copy-paste error, inconsistent naming |

**Common Thread:** Reporter and calculator modules weren't aligned on dictionary key names.

**Solution:** Fixed all mismatches by reading calculator return values and updating reporter to use correct keys.

---

## Benefits

1. ✅ **Accurate text quality metrics** displayed
2. ✅ **Enhanced insights** with lexical density and top terms
3. ✅ **Better content analysis** with unigrams and bigrams
4. ✅ **Consistent with other metrics** in the report
5. ✅ **No performance impact** (calculators already worked correctly)

---

## Files Modified

1. **src/lakehouse/quality/reporter.py** (lines 659-690)
   - Fixed text statistics key names
   - Enhanced lexical density display
   - Fixed top terms key names
   - Added bigrams display

2. **TEXT_QUALITY_FIX.md** (this document)
   - Documentation of the fix

---

**Status:** ✅ FIXED - Ready for testing  
**Next Step:** Run `lakehouse quality` to see text quality metrics populate!

---

## Insights from Test Run

**Top unigrams reveal conversational style:**
- "right" (70,691 occurrences)
- "like" (68,750 occurrences)  
- "you know" (as bigram: 41,192 occurrences)
- "kind of" (as bigram: 23,849 occurrences)

→ Indicates informal, spoken-word content (likely podcast transcriptions)

**Lexical density: 0.493** (49.3% content words)
- Slightly below typical written English (55-60%)
- Normal for conversational speech (more function words)
- Indicates natural, flowing dialogue

---

**Pattern Recognition:** This is the **fourth reporting bug** following the same pattern:
1. Timestamp regressions (wrong data structure)
2. Length compliance (wrong key names)
3. Speaker metrics (wrong column name)
4. Text quality (wrong key names)

**Recommendation:** Consider adding integration tests that verify reporter can correctly read all calculator outputs.

