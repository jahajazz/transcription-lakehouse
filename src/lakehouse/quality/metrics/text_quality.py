"""
Text quality proxy metrics for quality assessment (Category E).

Calculates basic text statistics, lexical density, and top terms
per PRD requirements FR-20, FR-21, FR-22.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import re
import string
from collections import Counter

from lakehouse.logger import get_default_logger


logger = get_default_logger()


# Simple English stopword list (common function words)
# Based on common stopword lists, kept minimal for performance
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by',
    'for', 'from', 'has', 'have', 'he', 'her', 'his', 'i', 'in', 'is',
    'it', 'its', 'of', 'on', 'or', 'that', 'the', 'to', 'was', 'were',
    'will', 'with', 'you', 'your', 'this', 'they', 'we', 'she', 'him',
    'can', 'do', 'does', 'did', 'not', 'no', 'yes', 'am', 'all', 'any',
    'some', 'there', 'their', 'them', 'so', 'than', 'then', 'too', 'very',
    'when', 'where', 'which', 'who', 'why', 'how', 'what', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'up', 'down', 'out', 'about',
    'into', 'through', 'over', 'under', 'again', 'further', 'more', 'most',
    'other', 'such', 'only', 'own', 'same', 'if', 'because', 'while',
    'during', 'before', 'after', 'above', 'below', 'between', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'than',
}


def calculate_text_metrics(
    segments: pd.DataFrame,
    segment_type: str = "segment",
) -> Dict[str, Any]:
    """
    Calculate basic text metrics for segments (FR-20).
    
    Computes:
    - Token count (words split by whitespace)
    - Word count (alphabetic words only)
    - Character count (total characters)
    - Average tokens per segment
    - Average words per segment
    - Average characters per segment
    
    Args:
        segments: DataFrame with segment data (must have text column)
        segment_type: Type of segment for logging
    
    Returns:
        Dictionary containing:
        - total_segments: Total number of segments
        - total_tokens: Total token count
        - total_words: Total word count
        - total_characters: Total character count
        - avg_tokens: Average tokens per segment
        - avg_words: Average words per segment
        - avg_characters: Average characters per segment
        - per_segment_stats: List of per-segment text statistics
    
    Example:
        >>> segments = pd.DataFrame({
        ...     'text': ['Hello world', 'This is a test', 'Short']
        ... })
        >>> result = calculate_text_metrics(segments)
        >>> result['avg_words']
        2.67
    """
    if len(segments) == 0:
        logger.warning(f"No {segment_type}s provided for text metrics")
        return {
            'total_segments': 0,
            'total_tokens': 0,
            'total_words': 0,
            'total_characters': 0,
            'avg_tokens': 0.0,
            'avg_words': 0.0,
            'avg_characters': 0.0,
            'per_segment_stats': [],
        }
    
    if 'text' not in segments.columns:
        logger.error(f"text column not found in {segment_type}s")
        return {
            'total_segments': len(segments),
            'total_tokens': 0,
            'total_words': 0,
            'total_characters': 0,
            'avg_tokens': 0.0,
            'avg_words': 0.0,
            'avg_characters': 0.0,
            'per_segment_stats': [],
        }
    
    total_segments = len(segments)
    per_segment_stats = []
    total_tokens = 0
    total_words = 0
    total_characters = 0
    
    for idx, row in segments.iterrows():
        text = row.get('text', '')
        if pd.isna(text):
            text = ''
        text = str(text)
        
        # Token count: split by whitespace
        tokens = text.split()
        token_count = len(tokens)
        
        # Word count: alphabetic words only (filter out pure punctuation/numbers)
        words = [token for token in tokens if any(c.isalpha() for c in token)]
        word_count = len(words)
        
        # Character count: total characters (including spaces)
        char_count = len(text)
        
        total_tokens += token_count
        total_words += word_count
        total_characters += char_count
        
        per_segment_stats.append({
            'index': idx,
            'token_count': token_count,
            'word_count': word_count,
            'character_count': char_count,
        })
    
    # Calculate averages
    avg_tokens = round(total_tokens / total_segments, 2) if total_segments > 0 else 0.0
    avg_words = round(total_words / total_segments, 2) if total_segments > 0 else 0.0
    avg_characters = round(total_characters / total_segments, 2) if total_segments > 0 else 0.0
    
    logger.info(
        f"Text metrics for {total_segments} {segment_type}s: "
        f"avg {avg_tokens} tokens, {avg_words} words, {avg_characters} characters"
    )
    
    return {
        'total_segments': total_segments,
        'total_tokens': total_tokens,
        'total_words': total_words,
        'total_characters': total_characters,
        'avg_tokens': avg_tokens,
        'avg_words': avg_words,
        'avg_characters': avg_characters,
        'per_segment_stats': per_segment_stats,
    }


def calculate_lexical_density(
    segments: pd.DataFrame,
    segment_type: str = "segment",
) -> Dict[str, Any]:
    """
    Calculate lexical density using stopword filtering (FR-21).
    
    Lexical density = content words / total words
    Content words = words that are not stopwords
    
    Uses a simple English stopword list (common function words).
    
    Args:
        segments: DataFrame with segment data (must have text column)
        segment_type: Type of segment for logging
    
    Returns:
        Dictionary containing:
        - total_words: Total word count
        - content_words: Number of content words (non-stopwords)
        - stopword_count: Number of stopwords
        - lexical_density: Ratio of content words to total words
        - avg_lexical_density: Average lexical density per segment
        - per_segment_densities: List of per-segment lexical densities
    
    Example:
        >>> segments = pd.DataFrame({
        ...     'text': ['The cat sat on the mat', 'I like coding']
        ... })
        >>> result = calculate_lexical_density(segments)
        >>> result['lexical_density'] > 0.3
        True
    """
    if len(segments) == 0:
        logger.warning(f"No {segment_type}s provided for lexical density")
        return {
            'total_words': 0,
            'content_words': 0,
            'stopword_count': 0,
            'lexical_density': 0.0,
            'avg_lexical_density': 0.0,
            'per_segment_densities': [],
        }
    
    if 'text' not in segments.columns:
        logger.error(f"text column not found in {segment_type}s")
        return {
            'total_words': 0,
            'content_words': 0,
            'stopword_count': 0,
            'lexical_density': 0.0,
            'avg_lexical_density': 0.0,
            'per_segment_densities': [],
        }
    
    total_words = 0
    content_words_count = 0
    stopword_count = 0
    per_segment_densities = []
    
    for idx, row in segments.iterrows():
        text = row.get('text', '')
        if pd.isna(text):
            text = ''
        text = str(text).lower()  # Lowercase for stopword matching
        
        # Split into tokens and keep only alphabetic words
        tokens = text.split()
        words = [token.strip('.,!?;:()[]{}"\'-') for token in tokens]  # Strip common punctuation
        words = [w for w in words if w and any(c.isalpha() for c in w)]  # Keep words with letters
        
        segment_total_words = len(words)
        segment_stopwords = sum(1 for w in words if w in STOPWORDS)
        segment_content_words = segment_total_words - segment_stopwords
        
        # Calculate per-segment lexical density
        if segment_total_words > 0:
            segment_density = round(segment_content_words / segment_total_words, 3)
        else:
            segment_density = 0.0
        
        per_segment_densities.append({
            'index': idx,
            'total_words': segment_total_words,
            'content_words': segment_content_words,
            'stopwords': segment_stopwords,
            'lexical_density': segment_density,
        })
        
        total_words += segment_total_words
        content_words_count += segment_content_words
        stopword_count += segment_stopwords
    
    # Calculate overall lexical density
    if total_words > 0:
        lexical_density = round(content_words_count / total_words, 3)
    else:
        lexical_density = 0.0
    
    # Calculate average per-segment lexical density
    valid_densities = [d['lexical_density'] for d in per_segment_densities if d['total_words'] > 0]
    if valid_densities:
        avg_lexical_density = round(sum(valid_densities) / len(valid_densities), 3)
    else:
        avg_lexical_density = 0.0
    
    logger.info(
        f"Lexical density for {len(segments)} {segment_type}s: "
        f"{lexical_density:.3f} ({content_words_count}/{total_words} content words)"
    )
    
    return {
        'total_words': total_words,
        'content_words': content_words_count,
        'stopword_count': stopword_count,
        'lexical_density': lexical_density,
        'avg_lexical_density': avg_lexical_density,
        'per_segment_densities': per_segment_densities,
    }


def calculate_punctuation_ratio(
    segments: pd.DataFrame,
    segment_type: str = "segment",
) -> Dict[str, Any]:
    """
    Calculate punctuation ratio as text quality proxy (FR-21).
    
    Punctuation ratio = punctuation characters / total characters
    
    Uses Python's string.punctuation set: !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~
    
    Args:
        segments: DataFrame with segment data (must have text column)
        segment_type: Type of segment for logging
    
    Returns:
        Dictionary containing:
        - total_characters: Total character count (excluding whitespace)
        - punctuation_count: Number of punctuation characters
        - punctuation_ratio: Ratio of punctuation to total characters
        - avg_punctuation_ratio: Average per segment
        - per_segment_ratios: List of per-segment punctuation ratios
    
    Example:
        >>> segments = pd.DataFrame({
        ...     'text': ['Hello, world!', 'Test.']
        ... })
        >>> result = calculate_punctuation_ratio(segments)
        >>> result['punctuation_count']
        4
    """
    if len(segments) == 0:
        logger.warning(f"No {segment_type}s provided for punctuation ratio")
        return {
            'total_characters': 0,
            'punctuation_count': 0,
            'punctuation_ratio': 0.0,
            'avg_punctuation_ratio': 0.0,
            'per_segment_ratios': [],
        }
    
    if 'text' not in segments.columns:
        logger.error(f"text column not found in {segment_type}s")
        return {
            'total_characters': 0,
            'punctuation_count': 0,
            'punctuation_ratio': 0.0,
            'avg_punctuation_ratio': 0.0,
            'per_segment_ratios': [],
        }
    
    total_characters = 0
    total_punctuation = 0
    per_segment_ratios = []
    
    # Get punctuation set
    punctuation_set = set(string.punctuation)
    
    for idx, row in segments.iterrows():
        text = row.get('text', '')
        if pd.isna(text):
            text = ''
        text = str(text)
        
        # Count characters (excluding whitespace for more meaningful ratio)
        non_whitespace_chars = [c for c in text if not c.isspace()]
        segment_char_count = len(non_whitespace_chars)
        
        # Count punctuation
        segment_punctuation = sum(1 for c in non_whitespace_chars if c in punctuation_set)
        
        # Calculate per-segment ratio
        if segment_char_count > 0:
            segment_ratio = round(segment_punctuation / segment_char_count, 3)
        else:
            segment_ratio = 0.0
        
        per_segment_ratios.append({
            'index': idx,
            'total_characters': segment_char_count,
            'punctuation_count': segment_punctuation,
            'punctuation_ratio': segment_ratio,
        })
        
        total_characters += segment_char_count
        total_punctuation += segment_punctuation
    
    # Calculate overall punctuation ratio
    if total_characters > 0:
        punctuation_ratio = round(total_punctuation / total_characters, 3)
    else:
        punctuation_ratio = 0.0
    
    # Calculate average per-segment ratio
    valid_ratios = [r['punctuation_ratio'] for r in per_segment_ratios if r['total_characters'] > 0]
    if valid_ratios:
        avg_punctuation_ratio = round(sum(valid_ratios) / len(valid_ratios), 3)
    else:
        avg_punctuation_ratio = 0.0
    
    logger.info(
        f"Punctuation ratio for {len(segments)} {segment_type}s: "
        f"{punctuation_ratio:.3f} ({total_punctuation}/{total_characters} punctuation chars)"
    )
    
    return {
        'total_characters': total_characters,
        'punctuation_count': total_punctuation,
        'punctuation_ratio': punctuation_ratio,
        'avg_punctuation_ratio': avg_punctuation_ratio,
        'per_segment_ratios': per_segment_ratios,
    }


def extract_top_terms(
    segments: pd.DataFrame,
    top_n: int = 20,
    segment_type: str = "segment",
) -> Dict[str, Any]:
    """
    Extract top N unigrams and bigrams for theme analysis (FR-22).
    
    Filters out stopwords and returns most common terms.
    
    Args:
        segments: DataFrame with segment data (must have text column)
        top_n: Number of top terms to return
        segment_type: Type of segment for logging
    
    Returns:
        Dictionary containing:
        - top_unigrams: List of (word, count) tuples
        - top_bigrams: List of (bigram, count) tuples
        - total_unique_unigrams: Total unique words
        - total_unique_bigrams: Total unique bigrams
    
    Example:
        >>> segments = pd.DataFrame({
        ...     'text': ['machine learning', 'deep learning', 'machine learning']
        ... })
        >>> result = extract_top_terms(segments, top_n=5)
        >>> result['top_unigrams'][0]
        ('learning', 3)
    """
    if len(segments) == 0:
        logger.warning(f"No {segment_type}s provided for term extraction")
        return {
            'top_unigrams': [],
            'top_bigrams': [],
            'total_unique_unigrams': 0,
            'total_unique_bigrams': 0,
        }
    
    if 'text' not in segments.columns:
        logger.error(f"text column not found in {segment_type}s")
        return {
            'top_unigrams': [],
            'top_bigrams': [],
            'total_unique_unigrams': 0,
            'total_unique_bigrams': 0,
        }
    
    # Collect all words (unigrams) and bigrams
    unigram_counter = Counter()
    bigram_counter = Counter()
    
    for idx, row in segments.iterrows():
        text = row.get('text', '')
        if pd.isna(text):
            text = ''
        text = str(text).lower()
        
        # Split into tokens and clean
        tokens = text.split()
        # Strip punctuation and keep only words with letters
        words = []
        for token in tokens:
            cleaned = token.strip(string.punctuation)
            if cleaned and any(c.isalpha() for c in cleaned):
                words.append(cleaned)
        
        # Filter out stopwords for unigrams
        content_words = [w for w in words if w not in STOPWORDS]
        
        # Count unigrams
        unigram_counter.update(content_words)
        
        # Create and count bigrams (consecutive word pairs, excluding stopword-only bigrams)
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            # Include bigram if at least one word is not a stopword
            if word1 not in STOPWORDS or word2 not in STOPWORDS:
                bigram = f"{word1} {word2}"
                bigram_counter.update([bigram])
    
    # Get top N terms
    top_unigrams = unigram_counter.most_common(top_n)
    top_bigrams = bigram_counter.most_common(top_n)
    
    total_unique_unigrams = len(unigram_counter)
    total_unique_bigrams = len(bigram_counter)
    
    logger.info(
        f"Extracted top terms from {len(segments)} {segment_type}s: "
        f"{total_unique_unigrams} unique unigrams, {total_unique_bigrams} unique bigrams"
    )
    
    return {
        'top_unigrams': top_unigrams,
        'top_bigrams': top_bigrams,
        'total_unique_unigrams': total_unique_unigrams,
        'total_unique_bigrams': total_unique_bigrams,
    }

