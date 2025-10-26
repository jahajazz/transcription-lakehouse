"""
Embedding sanity check metrics for quality assessment (Category F).

Implements neighbor coherence analysis, leakage detection, and bias checks
per PRD requirements FR-23 through FR-31.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

from lakehouse.logger import get_default_logger
from lakehouse.quality.thresholds import QualityThresholds, ThresholdViolation


logger = get_default_logger()


def load_embeddings(
    embedding_path: Union[str, Path],
    segments_df: Optional[pd.DataFrame] = None,
) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
    """
    Load embeddings from parquet file with graceful error handling (FR-23, subtask 3.1.1).
    
    Reads embedding vectors from parquet file and optionally merges with segment metadata.
    Returns None if embeddings are not available, allowing the assessment to continue
    without embedding checks.
    
    Args:
        embedding_path: Path to parquet file containing embeddings
        segments_df: Optional DataFrame with segment metadata to merge with embeddings
    
    Returns:
        Tuple of (merged_df, embedding_matrix) if successful, None if embeddings unavailable
        - merged_df: DataFrame with segment metadata and embedding vectors
        - embedding_matrix: NumPy array of shape (n_segments, embedding_dim)
    
    Example:
        >>> result = load_embeddings("output/embeddings/spans.parquet", spans_df)
        >>> if result is not None:
        ...     merged_df, embeddings = result
        ...     print(f"Loaded {len(embeddings)} embeddings")
    """
    try:
        embedding_path = Path(embedding_path)
        
        if not embedding_path.exists():
            logger.warning(
                f"Embedding file not found: {embedding_path}. "
                "Embedding sanity checks will be skipped."
            )
            return None
        
        logger.info(f"Loading embeddings from {embedding_path}...")
        
        # Read parquet file
        embeddings_df = pd.read_parquet(embedding_path)
        
        if len(embeddings_df) == 0:
            logger.warning(
                f"Embedding file is empty: {embedding_path}. "
                "Embedding sanity checks will be skipped."
            )
            return None
        
        # Check for required columns
        # Embeddings can be stored either as:
        # 1. A single 'embedding' column with arrays/lists
        # 2. Multiple 'embedding_0', 'embedding_1', ... columns
        # 3. Other naming conventions with an embedding array column
        
        embedding_column = None
        embedding_matrix = None
        
        # Strategy 1: Look for 'embedding' column
        if 'embedding' in embeddings_df.columns:
            embedding_column = 'embedding'
            # Convert to numpy array
            embedding_matrix = np.stack(embeddings_df[embedding_column].values)
            logger.info(f"Loaded embeddings from '{embedding_column}' column")
        
        # Strategy 2: Look for 'embedding_0', 'embedding_1', ... columns
        elif any(col.startswith('embedding_') for col in embeddings_df.columns):
            embedding_cols = sorted(
                [col for col in embeddings_df.columns if col.startswith('embedding_')],
                key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0
            )
            embedding_matrix = embeddings_df[embedding_cols].values
            logger.info(f"Loaded embeddings from {len(embedding_cols)} dimensional columns")
        
        # Strategy 3: Look for 'vector' or 'embedding_vector' column
        elif 'vector' in embeddings_df.columns:
            embedding_column = 'vector'
            embedding_matrix = np.stack(embeddings_df[embedding_column].values)
            logger.info(f"Loaded embeddings from '{embedding_column}' column")
        
        elif 'embedding_vector' in embeddings_df.columns:
            embedding_column = 'embedding_vector'
            embedding_matrix = np.stack(embeddings_df[embedding_column].values)
            logger.info(f"Loaded embeddings from '{embedding_column}' column")
        
        else:
            logger.warning(
                f"Could not find embedding columns in {embedding_path}. "
                f"Available columns: {list(embeddings_df.columns)}. "
                "Embedding sanity checks will be skipped."
            )
            return None
        
        # Validate embedding matrix
        if embedding_matrix is None or embedding_matrix.size == 0:
            logger.warning(
                f"Embedding matrix is empty in {embedding_path}. "
                "Embedding sanity checks will be skipped."
            )
            return None
        
        # Ensure 2D shape
        if embedding_matrix.ndim == 1:
            embedding_matrix = embedding_matrix.reshape(-1, 1)
        
        n_embeddings, embedding_dim = embedding_matrix.shape
        logger.info(
            f"Successfully loaded {n_embeddings} embeddings "
            f"with dimension {embedding_dim}"
        )
        
        # Merge with segment metadata if provided
        if segments_df is not None:
            # Ensure both have matching indices or IDs for merging
            if 'segment_id' in embeddings_df.columns and 'segment_id' in segments_df.columns:
                merged_df = segments_df.merge(
                    embeddings_df,
                    on='segment_id',
                    how='inner',
                    suffixes=('', '_embedding')
                )
                logger.info(
                    f"Merged embeddings with segment metadata: "
                    f"{len(merged_df)} segments with embeddings"
                )
            elif len(embeddings_df) == len(segments_df):
                # Assume same order, merge by index
                merged_df = segments_df.copy()
                merged_df['_embedding_idx'] = merged_df.index
                logger.info(
                    f"Aligned embeddings with segment metadata by index: "
                    f"{len(merged_df)} segments"
                )
            else:
                logger.warning(
                    f"Cannot merge embeddings with segments: "
                    f"no common ID column and different lengths "
                    f"({len(embeddings_df)} vs {len(segments_df)}). "
                    f"Using embeddings only."
                )
                merged_df = embeddings_df
        else:
            merged_df = embeddings_df
        
        # Filter embedding matrix to match merged dataframe
        if len(merged_df) != len(embedding_matrix):
            # Get indices of rows that were merged
            if '_embedding_idx' in merged_df.columns:
                indices = merged_df['_embedding_idx'].values
            else:
                # Assume embeddings_df indices match
                indices = merged_df.index.values
            
            embedding_matrix = embedding_matrix[indices]
            logger.info(f"Filtered embedding matrix to {len(embedding_matrix)} rows")
        
        return merged_df, embedding_matrix
    
    except Exception as e:
        logger.error(
            f"Error loading embeddings from {embedding_path}: {e}. "
            "Embedding sanity checks will be skipped."
        )
        return None


def stratified_sample_segments(
    segments_df: pd.DataFrame,
    sample_size: int,
    stratify_by: Optional[List[str]] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Perform stratified sampling of segments for reproducible analysis (FR-23, subtask 3.1.2).
    
    Samples segments in a stratified manner by episode and/or speaker to ensure
    representative coverage across the dataset. Uses fixed random seed for reproducibility.
    
    Args:
        segments_df: DataFrame with segment data
        sample_size: Number of segments to sample
        stratify_by: List of columns to stratify by (default: ['episode_id', 'speaker_id'])
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Sampled DataFrame with representative subset of segments
    
    Example:
        >>> segments = pd.DataFrame({
        ...     'segment_id': range(1000),
        ...     'episode_id': ['EP1'] * 500 + ['EP2'] * 500,
        ...     'speaker_id': ['S1', 'S2'] * 500,
        ... })
        >>> sample = stratified_sample_segments(segments, sample_size=100)
        >>> len(sample)
        100
    """
    if stratify_by is None:
        stratify_by = ['episode_id', 'speaker_id']
    
    # Filter to only available stratification columns
    available_stratify = [col for col in stratify_by if col in segments_df.columns]
    
    if len(available_stratify) == 0:
        # No stratification possible, do simple random sampling
        logger.warning(
            f"No stratification columns available from {stratify_by}. "
            f"Using simple random sampling."
        )
        if sample_size >= len(segments_df):
            return segments_df.copy()
        
        return segments_df.sample(n=sample_size, random_state=random_seed)
    
    logger.info(
        f"Stratified sampling {sample_size} segments by {available_stratify}..."
    )
    
    # Create stratification groups
    if len(available_stratify) == 1:
        strata_col = available_stratify[0]
        segments_df = segments_df.copy()
        segments_df['_strata'] = segments_df[strata_col].astype(str)
    else:
        segments_df = segments_df.copy()
        segments_df['_strata'] = segments_df[available_stratify].apply(
            lambda row: '_'.join(row.astype(str)), axis=1
        )
    
    # Count segments per stratum
    strata_counts = segments_df['_strata'].value_counts()
    total_segments = len(segments_df)
    
    if sample_size >= total_segments:
        logger.info(
            f"Sample size {sample_size} >= total segments {total_segments}. "
            f"Returning all segments."
        )
        return segments_df.drop(columns=['_strata'])
    
    # Calculate samples per stratum (proportional allocation)
    strata_samples = {}
    for stratum, count in strata_counts.items():
        proportion = count / total_segments
        n_samples = max(1, int(sample_size * proportion))  # At least 1 per stratum
        strata_samples[stratum] = min(n_samples, count)  # Don't oversample
    
    # Adjust if total exceeds sample_size
    total_allocated = sum(strata_samples.values())
    if total_allocated > sample_size:
        # Reduce from largest strata
        deficit = total_allocated - sample_size
        sorted_strata = sorted(
            strata_samples.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for stratum, allocated in sorted_strata:
            if deficit == 0:
                break
            reduction = min(deficit, allocated - 1)  # Keep at least 1
            strata_samples[stratum] -= reduction
            deficit -= reduction
    
    # Sample from each stratum
    sampled_dfs = []
    np.random.seed(random_seed)
    
    for stratum, n_samples in strata_samples.items():
        stratum_df = segments_df[segments_df['_strata'] == stratum]
        
        if n_samples >= len(stratum_df):
            sampled_dfs.append(stratum_df)
        else:
            sampled = stratum_df.sample(n=n_samples, random_state=random_seed)
            sampled_dfs.append(sampled)
    
    # Combine samples
    result = pd.concat(sampled_dfs, ignore_index=False)
    result = result.drop(columns=['_strata'])
    
    logger.info(
        f"Sampled {len(result)} segments from {len(strata_counts)} strata "
        f"({len(available_stratify)} stratification columns)"
    )
    
    return result


def compute_cosine_similarity(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    use_faiss: bool = False,
) -> np.ndarray:
    """
    Compute cosine similarity between query and corpus embeddings (FR-23, subtask 3.1.3).
    
    Calculates cosine similarity using NumPy with optional FAISS optimization
    for large-scale similarity search. Falls back to NumPy if FAISS is unavailable.
    
    Args:
        query_embedding: Query embedding vector(s) of shape (d,) or (n_queries, d)
        corpus_embeddings: Corpus embedding matrix of shape (n_corpus, d)
        use_faiss: Whether to try using FAISS for optimization (default: False)
    
    Returns:
        Similarity matrix of shape (n_queries, n_corpus) or (n_corpus,) for single query
    
    Example:
        >>> query = np.array([1.0, 0.0, 0.0])
        >>> corpus = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> similarities = compute_cosine_similarity(query, corpus)
        >>> similarities
        array([1.0, 0.0])
    """
    # Normalize inputs
    is_single_query = query_embedding.ndim == 1
    
    if is_single_query:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Ensure 2D corpus
    if corpus_embeddings.ndim == 1:
        corpus_embeddings = corpus_embeddings.reshape(1, -1)
    
    # Check dimensions
    if query_embedding.shape[1] != corpus_embeddings.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query has {query_embedding.shape[1]} dims, "
            f"corpus has {corpus_embeddings.shape[1]} dims"
        )
    
    # Try FAISS if requested and available
    if use_faiss:
        try:
            import faiss
            
            # Normalize vectors (FAISS inner product on normalized = cosine similarity)
            query_norm = query_embedding / (
                np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8
            )
            corpus_norm = corpus_embeddings / (
                np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-8
            )
            
            # Build FAISS index
            dimension = corpus_norm.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product index
            index.add(corpus_norm.astype(np.float32))
            
            # Search
            similarities, _ = index.search(query_norm.astype(np.float32), len(corpus_embeddings))
            
            logger.debug(f"Computed cosine similarity using FAISS")
            
            if is_single_query:
                return similarities.flatten()
            return similarities
        
        except ImportError:
            logger.debug("FAISS not available, falling back to NumPy")
        except Exception as e:
            logger.warning(f"FAISS computation failed ({e}), falling back to NumPy")
    
    # NumPy implementation (always works)
    # Normalize query embeddings
    query_norms = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    query_normalized = query_embedding / (query_norms + 1e-8)
    
    # Normalize corpus embeddings
    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    corpus_normalized = corpus_embeddings / (corpus_norms + 1e-8)
    
    # Compute cosine similarity: normalized dot product
    similarities = np.dot(query_normalized, corpus_normalized.T)
    
    logger.debug(
        f"Computed cosine similarity using NumPy: "
        f"{query_embedding.shape[0]} queries x {corpus_embeddings.shape[0]} corpus"
    )
    
    if is_single_query:
        return similarities.flatten()
    
    return similarities


def find_top_k_neighbors(
    query_indices: np.ndarray,
    embeddings: np.ndarray,
    k: int = 10,
    exclude_self: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find top-k nearest neighbors for query segments (FR-23, subtask 3.2.1).
    
    For each query segment, finds the k most similar segments based on
    cosine similarity in embedding space.
    
    Args:
        query_indices: Indices of query segments in the embeddings array
        embeddings: Full embedding matrix of shape (n_segments, embedding_dim)
        k: Number of neighbors to retrieve (default: 10)
        exclude_self: Whether to exclude the query itself from results (default: True)
    
    Returns:
        Tuple of (neighbor_indices, similarities):
        - neighbor_indices: Array of shape (n_queries, k) with neighbor indices
        - similarities: Array of shape (n_queries, k) with similarity scores
    
    Example:
        >>> embeddings = np.random.randn(100, 768)
        >>> query_indices = np.array([0, 5, 10])
        >>> neighbor_indices, similarities = find_top_k_neighbors(query_indices, embeddings, k=5)
        >>> neighbor_indices.shape
        (3, 5)
    """
    n_queries = len(query_indices)
    n_corpus = len(embeddings)
    
    # Get query embeddings
    query_embeddings = embeddings[query_indices]
    
    # Compute similarity between queries and all segments
    similarities = compute_cosine_similarity(query_embeddings, embeddings)
    
    # For each query, find top-k neighbors
    if exclude_self:
        # Exclude the query segment itself by setting its similarity to -inf
        for i, query_idx in enumerate(query_indices):
            similarities[i, query_idx] = -np.inf
    
    # Get top-k indices (argsort returns ascending, so we reverse)
    # We get k+1 to handle edge cases, then filter
    k_to_fetch = min(k + 1, n_corpus) if exclude_self else k
    
    # Get indices of top k similar segments for each query
    top_k_indices = np.argsort(similarities, axis=1)[:, -k_to_fetch:][:, ::-1]
    
    # Get corresponding similarity scores
    top_k_similarities = np.take_along_axis(similarities, top_k_indices, axis=1)
    
    # Filter to exactly k neighbors
    neighbor_indices = top_k_indices[:, :k]
    neighbor_similarities = top_k_similarities[:, :k]
    
    logger.debug(
        f"Found top-{k} neighbors for {n_queries} query segments. "
        f"Mean similarity: {neighbor_similarities.mean():.3f}"
    )
    
    return neighbor_indices, neighbor_similarities


def extract_neighbor_themes(
    neighbor_texts: List[str],
    top_n_terms: int = 5,
    min_word_length: int = 3,
) -> Dict[str, Any]:
    """
    Extract thematic terms from neighbor text snippets (FR-23, subtask 3.2.2).
    
    Analyzes the text content of neighbors to identify common themes and topics.
    Returns the most frequent meaningful terms (after stopword removal).
    
    Args:
        neighbor_texts: List of text strings from neighbor segments
        top_n_terms: Number of top terms to extract (default: 5)
        min_word_length: Minimum word length to consider (default: 3)
    
    Returns:
        Dictionary containing:
        - top_terms: List of (term, count) tuples
        - total_words: Total word count across all neighbors
        - unique_words: Number of unique words
        - coherence_score: Simple coherence metric (0-1)
    
    Example:
        >>> texts = ["The cat sat on the mat", "A dog ran in the park", "The bird flew away"]
        >>> themes = extract_neighbor_themes(texts, top_n_terms=3)
        >>> themes['top_terms']
        [('cat', 1), ('sat', 1), ('mat', 1)]
    """
    if not neighbor_texts:
        return {
            'top_terms': [],
            'total_words': 0,
            'unique_words': 0,
            'coherence_score': 0.0,
        }
    
    # Simple stopword list (common English function words)
    stopwords = {
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
    }
    
    # Tokenize and clean all text
    all_words = []
    for text in neighbor_texts:
        if text is None or not isinstance(text, str):
            continue
        
        # Convert to lowercase and split
        words = text.lower().split()
        
        # Filter: remove stopwords, short words, and non-alphabetic tokens
        filtered_words = [
            word.strip('.,!?;:"()[]{}')
            for word in words
            if (
                len(word.strip('.,!?;:"()[]{}')) >= min_word_length
                and word.lower() not in stopwords
                and any(c.isalpha() for c in word)
            )
        ]
        
        all_words.extend(filtered_words)
    
    if not all_words:
        return {
            'top_terms': [],
            'total_words': 0,
            'unique_words': 0,
            'coherence_score': 0.0,
        }
    
    # Count word frequencies
    word_counts = Counter(all_words)
    top_terms = word_counts.most_common(top_n_terms)
    
    # Calculate coherence score
    # Simple metric: ratio of top terms' frequency to total words
    # Higher score = more repetition of key terms = potentially more coherent
    total_words = len(all_words)
    unique_words = len(word_counts)
    top_terms_count = sum(count for _, count in top_terms)
    coherence_score = top_terms_count / total_words if total_words > 0 else 0.0
    
    return {
        'top_terms': top_terms,
        'total_words': total_words,
        'unique_words': unique_words,
        'coherence_score': round(coherence_score, 3),
    }


def assess_neighbor_coherence(
    segments_df: pd.DataFrame,
    embeddings: np.ndarray,
    sample_size: int = 100,
    k: int = 10,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Perform full neighbor coherence analysis (FR-23, subtask 3.2.3).
    
    Samples segments, finds their nearest neighbors, and assesses whether
    neighbors are thematically coherent or random.
    
    Args:
        segments_df: DataFrame with segment data (must have text column)
        embeddings: Embedding matrix aligned with segments_df
        sample_size: Number of segments to sample for analysis
        k: Number of neighbors to retrieve per segment
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing:
        - sample_count: Number of segments sampled
        - mean_neighbor_similarity: Average similarity to neighbors
        - coherence_scores: List of coherence scores for each sample
        - mean_coherence: Average coherence across all samples
        - top_neighbor_terms: Most common terms across all neighbor sets
        - coherent_count: Number of samples with high coherence (>0.2)
        - assessment: Overall assessment ("coherent", "mixed", or "random")
    
    Example:
        >>> result = assess_neighbor_coherence(segments_df, embeddings, sample_size=50, k=5)
        >>> result['assessment']
        'coherent'
    """
    logger.info(
        f"Assessing neighbor coherence: sampling {sample_size} segments, "
        f"retrieving top-{k} neighbors each"
    )
    
    # Ensure text column exists
    text_col = 'text' if 'text' in segments_df.columns else 'normalized_text'
    if text_col not in segments_df.columns:
        logger.warning("No text column found in segments. Cannot assess neighbor coherence.")
        return {
            'sample_count': 0,
            'mean_neighbor_similarity': 0.0,
            'coherence_scores': [],
            'mean_coherence': 0.0,
            'top_neighbor_terms': [],
            'coherent_count': 0,
            'assessment': 'unavailable',
        }
    
    # Sample segments
    sampled = stratified_sample_segments(
        segments_df,
        sample_size=sample_size,
        random_seed=random_seed
    )
    
    # Get query indices (positions in original dataframe/embedding matrix)
    query_indices = sampled.index.values
    
    # Find neighbors
    neighbor_indices, neighbor_similarities = find_top_k_neighbors(
        query_indices=query_indices,
        embeddings=embeddings,
        k=k,
        exclude_self=True
    )
    
    # Analyze coherence for each query
    coherence_scores = []
    all_neighbor_terms = []
    
    for i, query_idx in enumerate(query_indices):
        # Get neighbor texts
        neighbor_idx_list = neighbor_indices[i]
        neighbor_texts = segments_df.iloc[neighbor_idx_list][text_col].tolist()
        
        # Extract themes
        themes = extract_neighbor_themes(neighbor_texts, top_n_terms=5)
        coherence_scores.append(themes['coherence_score'])
        all_neighbor_terms.extend([term for term, _ in themes['top_terms']])
    
    # Aggregate results
    mean_similarity = neighbor_similarities.mean()
    mean_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
    
    # Count highly coherent samples (coherence > 0.2 is a reasonable threshold)
    coherent_count = sum(1 for score in coherence_scores if score > 0.2)
    coherent_ratio = coherent_count / len(coherence_scores) if coherence_scores else 0.0
    
    # Get most common terms across all neighbor sets
    term_counts = Counter(all_neighbor_terms)
    top_neighbor_terms = term_counts.most_common(10)
    
    # Determine overall assessment
    if coherent_ratio >= 0.7:
        assessment = "coherent"
    elif coherent_ratio >= 0.3:
        assessment = "mixed"
    else:
        assessment = "random"
    
    logger.info(
        f"Neighbor coherence assessment: {assessment} "
        f"(mean similarity: {mean_similarity:.3f}, mean coherence: {mean_coherence:.3f})"
    )
    
    return {
        'sample_count': len(sampled),
        'mean_neighbor_similarity': round(mean_similarity, 3),
        'coherence_scores': [round(s, 3) for s in coherence_scores],
        'mean_coherence': round(mean_coherence, 3),
        'top_neighbor_terms': top_neighbor_terms,
        'coherent_count': coherent_count,
        'coherent_ratio': round(coherent_ratio, 3),
        'assessment': assessment,
    }


def calculate_speaker_leakage(
    segments_df: pd.DataFrame,
    embeddings: np.ndarray,
    sample_size: int = 100,
    k: int = 10,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Calculate speaker leakage in nearest neighbors (FR-24, FR-25, subtask 3.3.1).
    
    Measures the percentage of nearest neighbors that share the same speaker
    as the query segment. High speaker leakage suggests embeddings are encoding
    speaker identity rather than semantic content.
    
    Args:
        segments_df: DataFrame with segment data (must have speaker_id column)
        embeddings: Embedding matrix aligned with segments_df
        sample_size: Number of segments to sample for analysis
        k: Number of neighbors to retrieve per segment
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing:
        - sample_count: Number of segments sampled
        - mean_same_speaker_percent: Average % of neighbors with same speaker
        - median_same_speaker_percent: Median % of neighbors with same speaker
        - max_same_speaker_percent: Maximum % observed
        - per_sample_percentages: List of percentages for each sample
        - speakers_analyzed: Number of unique speakers in sample
    
    Example:
        >>> result = calculate_speaker_leakage(segments_df, embeddings, sample_size=50, k=10)
        >>> result['mean_same_speaker_percent']
        35.2
    """
    logger.info(
        f"Calculating speaker leakage: sampling {sample_size} segments, "
        f"analyzing top-{k} neighbors"
    )
    
    # Check for speaker_id column
    if 'speaker_id' not in segments_df.columns:
        logger.warning("No speaker_id column found. Cannot calculate speaker leakage.")
        return {
            'sample_count': 0,
            'mean_same_speaker_percent': 0.0,
            'median_same_speaker_percent': 0.0,
            'max_same_speaker_percent': 0.0,
            'per_sample_percentages': [],
            'speakers_analyzed': 0,
        }
    
    # Sample segments
    sampled = stratified_sample_segments(
        segments_df,
        sample_size=sample_size,
        random_seed=random_seed
    )
    
    # Get query indices
    query_indices = sampled.index.values
    
    # Find neighbors
    neighbor_indices, _ = find_top_k_neighbors(
        query_indices=query_indices,
        embeddings=embeddings,
        k=k,
        exclude_self=True
    )
    
    # Calculate speaker leakage for each query
    same_speaker_percentages = []
    
    for i, query_idx in enumerate(query_indices):
        query_speaker = segments_df.iloc[query_idx]['speaker_id']
        
        # Get neighbor speakers
        neighbor_idx_list = neighbor_indices[i]
        neighbor_speakers = segments_df.iloc[neighbor_idx_list]['speaker_id'].values
        
        # Count how many neighbors have same speaker
        same_speaker_count = np.sum(neighbor_speakers == query_speaker)
        same_speaker_pct = (same_speaker_count / k) * 100
        
        same_speaker_percentages.append(same_speaker_pct)
    
    # Calculate statistics
    mean_pct = np.mean(same_speaker_percentages)
    median_pct = np.median(same_speaker_percentages)
    max_pct = np.max(same_speaker_percentages)
    speakers_analyzed = sampled['speaker_id'].nunique()
    
    logger.info(
        f"Speaker leakage: mean={mean_pct:.1f}%, median={median_pct:.1f}%, "
        f"max={max_pct:.1f}% (analyzed {speakers_analyzed} speakers)"
    )
    
    return {
        'sample_count': len(sampled),
        'mean_same_speaker_percent': round(mean_pct, 2),
        'median_same_speaker_percent': round(median_pct, 2),
        'max_same_speaker_percent': round(max_pct, 2),
        'per_sample_percentages': [round(p, 2) for p in same_speaker_percentages],
        'speakers_analyzed': speakers_analyzed,
    }


def calculate_episode_leakage(
    segments_df: pd.DataFrame,
    embeddings: np.ndarray,
    sample_size: int = 100,
    k: int = 10,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Calculate episode leakage in nearest neighbors (FR-24, FR-25, subtask 3.3.2).
    
    Measures the percentage of nearest neighbors that come from the same episode
    as the query segment. High episode leakage suggests embeddings are encoding
    episode-specific signals rather than generalizable semantic content.
    
    Args:
        segments_df: DataFrame with segment data (must have episode_id column)
        embeddings: Embedding matrix aligned with segments_df
        sample_size: Number of segments to sample for analysis
        k: Number of neighbors to retrieve per segment
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing:
        - sample_count: Number of segments sampled
        - mean_same_episode_percent: Average % of neighbors from same episode
        - median_same_episode_percent: Median % of neighbors from same episode
        - max_same_episode_percent: Maximum % observed
        - per_sample_percentages: List of percentages for each sample
        - episodes_analyzed: Number of unique episodes in sample
    
    Example:
        >>> result = calculate_episode_leakage(segments_df, embeddings, sample_size=50, k=10)
        >>> result['mean_same_episode_percent']
        42.5
    """
    logger.info(
        f"Calculating episode leakage: sampling {sample_size} segments, "
        f"analyzing top-{k} neighbors"
    )
    
    # Check for episode_id column
    if 'episode_id' not in segments_df.columns:
        logger.warning("No episode_id column found. Cannot calculate episode leakage.")
        return {
            'sample_count': 0,
            'mean_same_episode_percent': 0.0,
            'median_same_episode_percent': 0.0,
            'max_same_episode_percent': 0.0,
            'per_sample_percentages': [],
            'episodes_analyzed': 0,
        }
    
    # Sample segments
    sampled = stratified_sample_segments(
        segments_df,
        sample_size=sample_size,
        random_seed=random_seed
    )
    
    # Get query indices
    query_indices = sampled.index.values
    
    # Find neighbors
    neighbor_indices, _ = find_top_k_neighbors(
        query_indices=query_indices,
        embeddings=embeddings,
        k=k,
        exclude_self=True
    )
    
    # Calculate episode leakage for each query
    same_episode_percentages = []
    
    for i, query_idx in enumerate(query_indices):
        query_episode = segments_df.iloc[query_idx]['episode_id']
        
        # Get neighbor episodes
        neighbor_idx_list = neighbor_indices[i]
        neighbor_episodes = segments_df.iloc[neighbor_idx_list]['episode_id'].values
        
        # Count how many neighbors are from same episode
        same_episode_count = np.sum(neighbor_episodes == query_episode)
        same_episode_pct = (same_episode_count / k) * 100
        
        same_episode_percentages.append(same_episode_pct)
    
    # Calculate statistics
    mean_pct = np.mean(same_episode_percentages)
    median_pct = np.median(same_episode_percentages)
    max_pct = np.max(same_episode_percentages)
    episodes_analyzed = sampled['episode_id'].nunique()
    
    logger.info(
        f"Episode leakage: mean={mean_pct:.1f}%, median={median_pct:.1f}%, "
        f"max={max_pct:.1f}% (analyzed {episodes_analyzed} episodes)"
    )
    
    return {
        'sample_count': len(sampled),
        'mean_same_episode_percent': round(mean_pct, 2),
        'median_same_episode_percent': round(median_pct, 2),
        'max_same_episode_percent': round(max_pct, 2),
        'per_sample_percentages': [round(p, 2) for p in same_episode_percentages],
        'episodes_analyzed': episodes_analyzed,
    }


def validate_leakage_thresholds(
    speaker_leakage: Dict[str, Any],
    episode_leakage: Dict[str, Any],
    thresholds: QualityThresholds,
) -> List[ThresholdViolation]:
    """
    Validate leakage metrics against thresholds (FR-25, subtask 3.3.3).
    
    Checks:
    - Same speaker neighbors ≤ same_speaker_neighbor_max_percent (default 60%)
    - Same episode neighbors ≤ same_episode_neighbor_max_percent (default 70%)
    
    Args:
        speaker_leakage: Speaker leakage metrics from calculate_speaker_leakage()
        episode_leakage: Episode leakage metrics from calculate_episode_leakage()
        thresholds: Quality thresholds configuration
    
    Returns:
        List of threshold violations
    
    Example:
        >>> speaker = {'mean_same_speaker_percent': 75.0}
        >>> episode = {'mean_same_episode_percent': 65.0}
        >>> thresholds = QualityThresholds(same_speaker_neighbor_max_percent=60.0)
        >>> violations = validate_leakage_thresholds(speaker, episode, thresholds)
        >>> len(violations)
        1
    """
    violations = []
    
    # Check speaker leakage
    speaker_mean = speaker_leakage.get('mean_same_speaker_percent', 0.0)
    if speaker_mean > thresholds.same_speaker_neighbor_max_percent:
        violations.append(ThresholdViolation(
            threshold_name='same_speaker_neighbor_max_percent',
            expected=f'<= {thresholds.same_speaker_neighbor_max_percent}%',
            actual=f'{speaker_mean}%',
            severity='warning',
            message=(
                f'Mean speaker leakage {speaker_mean}% exceeds threshold of '
                f'{thresholds.same_speaker_neighbor_max_percent}%. '
                f'Embeddings may be encoding speaker identity.'
            ),
        ))
    
    # Check episode leakage
    episode_mean = episode_leakage.get('mean_same_episode_percent', 0.0)
    if episode_mean > thresholds.same_episode_neighbor_max_percent:
        violations.append(ThresholdViolation(
            threshold_name='same_episode_neighbor_max_percent',
            expected=f'<= {thresholds.same_episode_neighbor_max_percent}%',
            actual=f'{episode_mean}%',
            severity='warning',
            message=(
                f'Mean episode leakage {episode_mean}% exceeds threshold of '
                f'{thresholds.same_episode_neighbor_max_percent}%. '
                f'Embeddings may be encoding episode-specific signals.'
            ),
        ))
    
    return violations


def calculate_embedding_norms(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Calculate L2 norms of embedding vectors (FR-26, FR-27, subtask 3.4.1).
    
    Computes the L2 (Euclidean) norm for each embedding vector. Norm magnitude
    can be influenced by segment length if the embedding model is biased.
    
    Args:
        embeddings: Embedding matrix of shape (n_segments, embedding_dim)
    
    Returns:
        Array of L2 norms of shape (n_segments,)
    
    Example:
        >>> embeddings = np.array([[3.0, 4.0], [1.0, 0.0]])
        >>> norms = calculate_embedding_norms(embeddings)
        >>> norms
        array([5.0, 1.0])
    """
    norms = np.linalg.norm(embeddings, axis=1)
    return norms


def calculate_length_bias_correlation(
    segments_df: pd.DataFrame,
    embeddings: np.ndarray,
    sample_size: Optional[int] = None,
    k: int = 10,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Calculate correlation between segment length and embedding properties (FR-26, FR-27, subtask 3.4.2).
    
    Measures two types of length bias:
    1. Correlation between duration and embedding norm (L2 norm)
    2. Correlation between duration and mean similarity to neighbors
    
    High correlation suggests embeddings encode length information,
    which could bias semantic search results.
    
    Args:
        segments_df: DataFrame with segment data (must have duration column)
        embeddings: Embedding matrix aligned with segments_df
        sample_size: Optional sample size for efficiency (None = use all segments)
        k: Number of neighbors for similarity calculation
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing:
        - duration_norm_correlation: Correlation between duration and norm
        - duration_similarity_correlation: Correlation between duration and mean neighbor similarity
        - sample_count: Number of segments analyzed
        - mean_norm: Average embedding norm
        - mean_similarity: Average neighbor similarity
    
    Example:
        >>> result = calculate_length_bias_correlation(segments_df, embeddings, sample_size=100)
        >>> result['duration_norm_correlation']
        0.15
    """
    logger.info("Calculating length bias correlations...")
    
    # Check for duration column
    duration_col = None
    for col_name in ['duration', 'duration_seconds', 'length']:
        if col_name in segments_df.columns:
            duration_col = col_name
            break
    
    if duration_col is None:
        logger.warning("No duration column found. Cannot calculate length bias.")
        return {
            'duration_norm_correlation': 0.0,
            'duration_similarity_correlation': 0.0,
            'sample_count': 0,
            'mean_norm': 0.0,
            'mean_similarity': 0.0,
        }
    
    # Sample if requested
    if sample_size is not None and sample_size < len(segments_df):
        sampled = stratified_sample_segments(
            segments_df,
            sample_size=sample_size,
            random_seed=random_seed
        )
        query_indices = sampled.index.values
        sampled_embeddings = embeddings[query_indices]
    else:
        sampled = segments_df
        query_indices = segments_df.index.values
        sampled_embeddings = embeddings
    
    # Get durations
    durations = sampled[duration_col].values
    
    # Calculate embedding norms
    norms = calculate_embedding_norms(sampled_embeddings)
    
    # Calculate correlation: duration vs norm
    duration_norm_corr = np.corrcoef(durations, norms)[0, 1]
    
    # Calculate mean similarity to neighbors for each segment
    if len(sampled) > k:
        # Find neighbors and get similarities
        neighbor_indices, neighbor_similarities = find_top_k_neighbors(
            query_indices=query_indices,
            embeddings=embeddings,
            k=k,
            exclude_self=True
        )
        
        # Calculate mean similarity for each segment
        mean_similarities = neighbor_similarities.mean(axis=1)
        
        # Calculate correlation: duration vs mean similarity
        duration_similarity_corr = np.corrcoef(durations, mean_similarities)[0, 1]
        mean_similarity = mean_similarities.mean()
    else:
        # Not enough segments for neighbor analysis
        duration_similarity_corr = 0.0
        mean_similarity = 0.0
    
    logger.info(
        f"Length bias: duration-norm corr={duration_norm_corr:.3f}, "
        f"duration-similarity corr={duration_similarity_corr:.3f}"
    )
    
    return {
        'duration_norm_correlation': round(duration_norm_corr, 3),
        'duration_similarity_correlation': round(duration_similarity_corr, 3),
        'sample_count': len(sampled),
        'mean_norm': round(norms.mean(), 3),
        'mean_similarity': round(mean_similarity, 3),
    }


def validate_length_bias_threshold(
    length_bias_metrics: Dict[str, Any],
    thresholds: QualityThresholds,
) -> List[ThresholdViolation]:
    """
    Validate length bias metrics against threshold (FR-27, subtask 3.4.3).
    
    Checks:
    - |duration-norm correlation| ≤ length_bias_correlation_max (default 0.3)
    - |duration-similarity correlation| ≤ length_bias_correlation_max (default 0.3)
    
    Args:
        length_bias_metrics: Length bias metrics from calculate_length_bias_correlation()
        thresholds: Quality thresholds configuration
    
    Returns:
        List of threshold violations
    
    Example:
        >>> metrics = {'duration_norm_correlation': 0.45, 'duration_similarity_correlation': 0.25}
        >>> thresholds = QualityThresholds(length_bias_correlation_max=0.3)
        >>> violations = validate_length_bias_threshold(metrics, thresholds)
        >>> len(violations)
        1
    """
    violations = []
    
    # Check duration-norm correlation
    duration_norm_corr = length_bias_metrics.get('duration_norm_correlation', 0.0)
    abs_duration_norm_corr = abs(duration_norm_corr)
    
    if abs_duration_norm_corr > thresholds.length_bias_correlation_max:
        violations.append(ThresholdViolation(
            threshold_name='duration_norm_correlation',
            expected=f'<= {thresholds.length_bias_correlation_max}',
            actual=f'{duration_norm_corr:.3f}',
            severity='warning',
            message=(
                f'Duration-norm correlation {duration_norm_corr:.3f} exceeds threshold of '
                f'{thresholds.length_bias_correlation_max}. '
                f'Embedding norms may be biased by segment length.'
            ),
        ))
    
    # Check duration-similarity correlation
    duration_sim_corr = length_bias_metrics.get('duration_similarity_correlation', 0.0)
    abs_duration_sim_corr = abs(duration_sim_corr)
    
    if abs_duration_sim_corr > thresholds.length_bias_correlation_max:
        violations.append(ThresholdViolation(
            threshold_name='duration_similarity_correlation',
            expected=f'<= {thresholds.length_bias_correlation_max}',
            actual=f'{duration_sim_corr:.3f}',
            severity='warning',
            message=(
                f'Duration-similarity correlation {duration_sim_corr:.3f} exceeds threshold of '
                f'{thresholds.length_bias_correlation_max}. '
                f'Neighbor similarity may be biased by segment length.'
            ),
        ))
    
    return violations


def sample_random_pairs(
    n_segments: int,
    n_pairs: int = 500,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Sample random pairs of segment indices for similarity comparison (FR-28, subtask 3.5.1).
    
    Generates random pairs of distinct segment indices for lexical vs embedding
    similarity correlation analysis. Uses fixed seed for reproducibility.
    
    Args:
        n_segments: Total number of segments available
        n_pairs: Number of pairs to sample (default: 500)
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Array of shape (n_pairs, 2) with segment index pairs
    
    Example:
        >>> pairs = sample_random_pairs(n_segments=100, n_pairs=10)
        >>> pairs.shape
        (10, 2)
    """
    np.random.seed(random_seed)
    
    # Ensure we don't sample more pairs than possible
    max_pairs = n_segments * (n_segments - 1) // 2
    n_pairs = min(n_pairs, max_pairs)
    
    pairs = []
    sampled_pairs = set()
    
    while len(pairs) < n_pairs:
        # Sample two distinct indices
        idx1, idx2 = np.random.choice(n_segments, size=2, replace=False)
        
        # Normalize pair order (smaller index first)
        pair = tuple(sorted([int(idx1), int(idx2)]))
        
        # Check if we've already sampled this pair
        if pair not in sampled_pairs:
            sampled_pairs.add(pair)
            pairs.append(pair)
    
    return np.array(pairs)


def calculate_lexical_similarity(
    text1: str,
    text2: str,
    method: str = 'jaccard',
    min_word_length: int = 2,
) -> float:
    """
    Calculate lexical similarity between two texts (FR-28, subtask 3.5.2).
    
    Computes similarity based on word overlap using either:
    - Jaccard similarity: |intersection| / |union|
    - Simple token overlap: |intersection| / sqrt(|text1| * |text2|)
    
    Args:
        text1: First text string
        text2: Second text string
        method: Similarity method ('jaccard' or 'overlap', default: 'jaccard')
        min_word_length: Minimum word length to consider (default: 2)
    
    Returns:
        Similarity score in range [0, 1]
    
    Example:
        >>> sim = calculate_lexical_similarity("the cat sat", "the dog sat", method='jaccard')
        >>> sim
        0.5
    """
    if not text1 or not text2:
        return 0.0
    
    # Simple stopword list
    stopwords = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'from',
        'has', 'have', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'to',
        'was', 'were', 'will', 'with',
    }
    
    # Tokenize and normalize
    def tokenize(text: str) -> List[str]:
        """
        Tokenize and normalize text for lexical similarity calculation.
        
        Args:
            text: Input text to tokenize
        
        Returns:
            List of normalized tokens (lowercase, punctuation stripped, stopwords removed)
        """
        words = text.lower().split()
        # Filter: remove stopwords and short words
        filtered = [
            word.strip('.,!?;:"()[]{}')
            for word in words
            if (
                len(word.strip('.,!?;:"()[]{}')) >= min_word_length
                and word.lower() not in stopwords
                and any(c.isalpha() for c in word)
            )
        ]
        return set(filtered)
    
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    
    if not tokens1 or not tokens2:
        return 0.0
    
    # Calculate similarity
    intersection = tokens1 & tokens2
    
    if method == 'jaccard':
        union = tokens1 | tokens2
        similarity = len(intersection) / len(union) if union else 0.0
    elif method == 'overlap':
        # Normalized overlap (Dice-like coefficient)
        similarity = 2 * len(intersection) / (len(tokens1) + len(tokens2))
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    
    return similarity


def calculate_similarity_correlation(
    segments_df: pd.DataFrame,
    embeddings: np.ndarray,
    n_pairs: int = 500,
    lexical_method: str = 'jaccard',
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Calculate correlation between lexical and embedding similarity (FR-28, subtask 3.5.3).
    
    Samples random pairs of segments and computes both lexical similarity (word overlap)
    and embedding similarity (cosine). High correlation suggests embeddings capture
    lexical/semantic relationships appropriately.
    
    Args:
        segments_df: DataFrame with segment data (must have text column)
        embeddings: Embedding matrix aligned with segments_df
        n_pairs: Number of random pairs to sample (default: 500)
        lexical_method: Method for lexical similarity ('jaccard' or 'overlap')
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing:
        - correlation: Pearson correlation between lexical and embedding similarities
        - spearman_correlation: Spearman rank correlation
        - n_pairs: Number of pairs analyzed
        - mean_lexical_similarity: Average lexical similarity
        - mean_embedding_similarity: Average embedding similarity
        - pairs_sample: Sample of 10 pairs with both similarities (for inspection)
    
    Example:
        >>> result = calculate_similarity_correlation(segments_df, embeddings, n_pairs=100)
        >>> result['correlation']
        0.65
    """
    logger.info(
        f"Calculating lexical vs embedding similarity correlation "
        f"(sampling {n_pairs} pairs)..."
    )
    
    # Check for text column
    text_col = 'text' if 'text' in segments_df.columns else 'normalized_text'
    if text_col not in segments_df.columns:
        logger.warning("No text column found. Cannot calculate similarity correlation.")
        return {
            'correlation': 0.0,
            'spearman_correlation': 0.0,
            'n_pairs': 0,
            'mean_lexical_similarity': 0.0,
            'mean_embedding_similarity': 0.0,
            'pairs_sample': [],
        }
    
    # Sample random pairs
    n_segments = len(segments_df)
    pairs = sample_random_pairs(n_segments, n_pairs, random_seed)
    
    # Calculate similarities for each pair
    lexical_similarities = []
    embedding_similarities = []
    pairs_sample = []
    
    for i, (idx1, idx2) in enumerate(pairs):
        # Get texts
        text1 = segments_df.iloc[idx1][text_col]
        text2 = segments_df.iloc[idx2][text_col]
        
        # Calculate lexical similarity
        lex_sim = calculate_lexical_similarity(text1, text2, method=lexical_method)
        lexical_similarities.append(lex_sim)
        
        # Calculate embedding similarity
        emb1 = embeddings[idx1]
        emb2 = embeddings[idx2]
        emb_sim = compute_cosine_similarity(emb1, emb2)[0]
        embedding_similarities.append(emb_sim)
        
        # Store sample for inspection (first 10 pairs)
        if i < 10:
            pairs_sample.append({
                'pair_idx': i,
                'segment_idx1': int(idx1),
                'segment_idx2': int(idx2),
                'lexical_similarity': round(float(lex_sim), 3),
                'embedding_similarity': round(float(emb_sim), 3),
                'text1_preview': text1[:50] + '...' if len(str(text1)) > 50 else str(text1),
                'text2_preview': text2[:50] + '...' if len(str(text2)) > 50 else str(text2),
            })
    
    # Convert to numpy arrays
    lexical_similarities = np.array(lexical_similarities)
    embedding_similarities = np.array(embedding_similarities)
    
    # Calculate Pearson correlation
    pearson_corr = np.corrcoef(lexical_similarities, embedding_similarities)[0, 1]
    
    # Calculate Spearman correlation (rank-based)
    from scipy.stats import spearmanr
    spearman_corr, _ = spearmanr(lexical_similarities, embedding_similarities)
    
    logger.info(
        f"Similarity correlation: Pearson={pearson_corr:.3f}, "
        f"Spearman={spearman_corr:.3f}"
    )
    
    return {
        'correlation': round(pearson_corr, 3),
        'spearman_correlation': round(spearman_corr, 3),
        'n_pairs': len(pairs),
        'mean_lexical_similarity': round(lexical_similarities.mean(), 3),
        'mean_embedding_similarity': round(embedding_similarities.mean(), 3),
        'pairs_sample': pairs_sample,
    }


def calculate_cross_series_neighbors(
    segments_df: pd.DataFrame,
    embeddings: np.ndarray,
    sample_size: int = 100,
    k: int = 10,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Calculate percentage of neighbors from different series (FR-29, subtask 3.6.1).
    
    Measures diversity in neighbor retrieval across series. High cross-series
    percentages indicate good generalization; low percentages suggest embeddings
    are overfitting to series-specific patterns.
    
    Args:
        segments_df: DataFrame with segment data (must have series_id column)
        embeddings: Embedding matrix aligned with segments_df
        sample_size: Number of segments to sample for analysis
        k: Number of neighbors to retrieve per segment
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing:
        - sample_count: Number of segments sampled
        - mean_cross_series_percent: Average % of neighbors from different series
        - median_cross_series_percent: Median % of neighbors from different series
        - series_analyzed: Number of unique series in sample
        - has_series_column: Whether series_id column exists
    
    Example:
        >>> result = calculate_cross_series_neighbors(segments_df, embeddings, sample_size=50, k=10)
        >>> result['mean_cross_series_percent']
        65.0
    """
    logger.info(
        f"Calculating cross-series neighbor diversity: sampling {sample_size} segments, "
        f"analyzing top-{k} neighbors"
    )
    
    # Check for series_id column
    if 'series_id' not in segments_df.columns:
        logger.info("No series_id column found. Skipping cross-series analysis.")
        return {
            'sample_count': 0,
            'mean_cross_series_percent': 0.0,
            'median_cross_series_percent': 0.0,
            'series_analyzed': 0,
            'has_series_column': False,
        }
    
    # Sample segments
    sampled = stratified_sample_segments(
        segments_df,
        sample_size=sample_size,
        random_seed=random_seed
    )
    
    # Get query indices
    query_indices = sampled.index.values
    
    # Find neighbors
    neighbor_indices, _ = find_top_k_neighbors(
        query_indices=query_indices,
        embeddings=embeddings,
        k=k,
        exclude_self=True
    )
    
    # Calculate cross-series percentages
    cross_series_percentages = []
    
    for i, query_idx in enumerate(query_indices):
        query_series = segments_df.iloc[query_idx]['series_id']
        
        # Get neighbor series
        neighbor_idx_list = neighbor_indices[i]
        neighbor_series = segments_df.iloc[neighbor_idx_list]['series_id'].values
        
        # Count how many neighbors are from different series
        different_series_count = np.sum(neighbor_series != query_series)
        cross_series_pct = (different_series_count / k) * 100
        
        cross_series_percentages.append(cross_series_pct)
    
    # Calculate statistics
    mean_pct = np.mean(cross_series_percentages)
    median_pct = np.median(cross_series_percentages)
    series_analyzed = sampled['series_id'].nunique()
    
    logger.info(
        f"Cross-series neighbors: mean={mean_pct:.1f}%, median={median_pct:.1f}% "
        f"(analyzed {series_analyzed} series)"
    )
    
    return {
        'sample_count': len(sampled),
        'mean_cross_series_percent': round(mean_pct, 2),
        'median_cross_series_percent': round(median_pct, 2),
        'series_analyzed': series_analyzed,
        'has_series_column': True,
    }


def calculate_adjacency_bias(
    segments_df: pd.DataFrame,
    embeddings: np.ndarray,
    sample_size: int = 100,
    k: int = 10,
    adjacency_tolerance_seconds: float = 5.0,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Calculate temporal adjacency bias in neighbors (FR-30, FR-31, subtask 3.6.2).
    
    Measures the percentage of neighbors that are temporally adjacent (within
    a time tolerance) to the query segment. High adjacency suggests embeddings
    are encoding temporal/position information rather than semantic content.
    
    Args:
        segments_df: DataFrame with segment data (must have episode_id, start_time, end_time)
        embeddings: Embedding matrix aligned with segments_df
        sample_size: Number of segments to sample for analysis
        k: Number of neighbors to retrieve per segment
        adjacency_tolerance_seconds: Time tolerance for adjacency (default: 5.0)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing:
        - sample_count: Number of segments sampled
        - mean_adjacent_percent: Average % of temporally adjacent neighbors
        - median_adjacent_percent: Median % of temporally adjacent neighbors
        - max_adjacent_percent: Maximum % observed
        - per_sample_percentages: List of percentages for each sample
    
    Example:
        >>> result = calculate_adjacency_bias(segments_df, embeddings, sample_size=50, k=10)
        >>> result['mean_adjacent_percent']
        15.5
    """
    logger.info(
        f"Calculating temporal adjacency bias: sampling {sample_size} segments, "
        f"analyzing top-{k} neighbors (tolerance={adjacency_tolerance_seconds}s)"
    )
    
    # Check for required columns
    required_cols = ['episode_id', 'start_time', 'end_time']
    missing_cols = [col for col in required_cols if col not in segments_df.columns]
    
    if missing_cols:
        logger.warning(
            f"Missing columns for adjacency analysis: {missing_cols}. "
            f"Cannot calculate adjacency bias."
        )
        return {
            'sample_count': 0,
            'mean_adjacent_percent': 0.0,
            'median_adjacent_percent': 0.0,
            'max_adjacent_percent': 0.0,
            'per_sample_percentages': [],
        }
    
    # Sample segments
    sampled = stratified_sample_segments(
        segments_df,
        sample_size=sample_size,
        random_seed=random_seed
    )
    
    # Get query indices
    query_indices = sampled.index.values
    
    # Find neighbors
    neighbor_indices, _ = find_top_k_neighbors(
        query_indices=query_indices,
        embeddings=embeddings,
        k=k,
        exclude_self=True
    )
    
    # Calculate adjacency for each query
    adjacent_percentages = []
    
    for i, query_idx in enumerate(query_indices):
        query_segment = segments_df.iloc[query_idx]
        query_episode = query_segment['episode_id']
        query_start = query_segment['start_time']
        query_end = query_segment['end_time']
        
        # Get neighbor info
        neighbor_idx_list = neighbor_indices[i]
        neighbors = segments_df.iloc[neighbor_idx_list]
        
        # Count adjacent neighbors
        adjacent_count = 0
        
        for _, neighbor in neighbors.iterrows():
            # Only consider neighbors from same episode
            if neighbor['episode_id'] != query_episode:
                continue
            
            neighbor_start = neighbor['start_time']
            neighbor_end = neighbor['end_time']
            
            # Check if temporally adjacent (within tolerance)
            # Adjacent if:
            # 1. Neighbor starts within tolerance of query end
            # 2. Neighbor ends within tolerance of query start
            time_gap_after = abs(neighbor_start - query_end)
            time_gap_before = abs(query_start - neighbor_end)
            
            if time_gap_after <= adjacency_tolerance_seconds or time_gap_before <= adjacency_tolerance_seconds:
                adjacent_count += 1
        
        adjacent_pct = (adjacent_count / k) * 100
        adjacent_percentages.append(adjacent_pct)
    
    # Calculate statistics
    mean_pct = np.mean(adjacent_percentages)
    median_pct = np.median(adjacent_percentages)
    max_pct = np.max(adjacent_percentages)
    
    logger.info(
        f"Adjacency bias: mean={mean_pct:.1f}%, median={median_pct:.1f}%, "
        f"max={max_pct:.1f}%"
    )
    
    return {
        'sample_count': len(sampled),
        'mean_adjacent_percent': round(mean_pct, 2),
        'median_adjacent_percent': round(median_pct, 2),
        'max_adjacent_percent': round(max_pct, 2),
        'per_sample_percentages': [round(p, 2) for p in adjacent_percentages],
    }


def validate_adjacency_threshold(
    adjacency_metrics: Dict[str, Any],
    thresholds: QualityThresholds,
) -> List[ThresholdViolation]:
    """
    Validate adjacency bias metrics against threshold (FR-31, subtask 3.6.3).
    
    Checks:
    - Mean adjacency ≤ adjacency_bias_max_percent (default 40%)
    
    Args:
        adjacency_metrics: Adjacency metrics from calculate_adjacency_bias()
        thresholds: Quality thresholds configuration
    
    Returns:
        List of threshold violations
    
    Example:
        >>> metrics = {'mean_adjacent_percent': 55.0}
        >>> thresholds = QualityThresholds(adjacency_bias_max_percent=40.0)
        >>> violations = validate_adjacency_threshold(metrics, thresholds)
        >>> len(violations)
        1
    """
    violations = []
    
    # Check adjacency bias
    adjacency_mean = adjacency_metrics.get('mean_adjacent_percent', 0.0)
    if adjacency_mean > thresholds.adjacency_bias_max_percent:
        violations.append(ThresholdViolation(
            threshold_name='adjacency_bias_max_percent',
            expected=f'<= {thresholds.adjacency_bias_max_percent}%',
            actual=f'{adjacency_mean}%',
            severity='warning',
            message=(
                f'Mean adjacency bias {adjacency_mean}% exceeds threshold of '
                f'{thresholds.adjacency_bias_max_percent}%. '
                f'Embeddings may be encoding temporal/position information.'
            ),
        ))
    
    return violations
