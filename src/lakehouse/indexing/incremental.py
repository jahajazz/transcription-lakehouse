"""
Incremental index update utilities.

Handles adding new embeddings to existing FAISS indices without
rebuilding from scratch.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from lakehouse.indexing.faiss_builder import FAISSIndexBuilder
from lakehouse.embeddings.storage import load_embeddings
from lakehouse.logger import get_default_logger


logger = get_default_logger()


class IncrementalIndexUpdater:
    """
    Manages incremental updates to FAISS indices.
    
    Adds new vectors to existing indices efficiently without full rebuild.
    """
    
    def __init__(self, builder: Optional[FAISSIndexBuilder] = None):
        """
        Initialize incremental index updater.
        
        Args:
            builder: Optional FAISSIndexBuilder instance (creates new if None)
        """
        self.builder = builder or FAISSIndexBuilder()
        logger.debug("IncrementalIndexUpdater initialized")
    
    def update_index(
        self,
        existing_index_path: Union[str, Path],
        new_embeddings: Union[np.ndarray, pd.DataFrame],
        new_artifact_ids: Optional[List[str]] = None,
    ) -> Tuple[Any, List[str]]:
        """
        Add new embeddings to an existing FAISS index.
        
        Args:
            existing_index_path: Path to existing index file
            new_embeddings: New embeddings to add (numpy array or DataFrame)
            new_artifact_ids: List of artifact IDs for new embeddings
        
        Returns:
            Tuple of (updated_index, all_artifact_ids)
        
        Example:
            >>> updater = IncrementalIndexUpdater()
            >>> index, ids = updater.update_index("span_index.faiss", new_embeddings, new_ids)
        """
        try:
            import faiss
        except ImportError:
            logger.error("faiss not installed")
            raise
        
        existing_index_path = Path(existing_index_path)
        
        # Load existing index
        logger.info(f"Loading existing index from {existing_index_path}")
        index, existing_ids, metadata = self.builder.load_index(existing_index_path)
        
        # Convert new embeddings to numpy if DataFrame
        if isinstance(new_embeddings, pd.DataFrame):
            if new_artifact_ids is None:
                new_artifact_ids = new_embeddings["artifact_id"].tolist()
            
            if "embedding" in new_embeddings.columns:
                embedding_vectors = new_embeddings["embedding"].tolist()
                new_embeddings = np.array(embedding_vectors, dtype=np.float32)
            else:
                raise ValueError("DataFrame must have 'embedding' column")
        else:
            new_embeddings = np.array(new_embeddings, dtype=np.float32)
            
            if new_artifact_ids is None:
                logger.warning("No artifact IDs provided for new embeddings")
                new_artifact_ids = [f"new_vec_{i}" for i in range(len(new_embeddings))]
        
        # Check for duplicates
        new_artifact_ids = self._deduplicate_ids(existing_ids, new_artifact_ids, new_embeddings)
        
        if len(new_artifact_ids) == 0:
            logger.info("No new embeddings to add (all were duplicates)")
            return index, existing_ids
        
        logger.info(f"Adding {len(new_artifact_ids)} new vectors to index")
        
        # Normalize if using cosine similarity
        distance_metric = metadata.get("distance_metric", "l2")
        if distance_metric == "cosine":
            faiss.normalize_L2(new_embeddings)
        
        # Add new vectors to index
        index.add(new_embeddings)
        
        # Update artifact ID list
        all_artifact_ids = existing_ids + new_artifact_ids
        
        logger.info(f"Index updated: {len(existing_ids)} → {len(all_artifact_ids)} vectors")
        
        return index, all_artifact_ids
    
    def _deduplicate_ids(
        self,
        existing_ids: List[str],
        new_ids: List[str],
        new_embeddings: np.ndarray,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Remove duplicate IDs from new data.
        
        Args:
            existing_ids: Existing artifact IDs
            new_ids: New artifact IDs
            new_embeddings: New embeddings
        
        Returns:
            Tuple of (deduplicated_ids, deduplicated_embeddings)
        """
        existing_set = set(existing_ids)
        
        # Find indices of non-duplicate IDs
        keep_indices = [i for i, id in enumerate(new_ids) if id not in existing_set]
        
        if len(keep_indices) < len(new_ids):
            duplicates = len(new_ids) - len(keep_indices)
            logger.warning(f"Skipping {duplicates} duplicate artifact IDs")
        
        # Filter new data
        deduplicated_ids = [new_ids[i] for i in keep_indices]
        deduplicated_embeddings = new_embeddings[keep_indices]
        
        return deduplicated_ids, deduplicated_embeddings
    
    def update_from_lakehouse(
        self,
        existing_index_path: Union[str, Path],
        lakehouse_path: Union[str, Path],
        artifact_type: str = "span",
        version: str = "v1",
    ) -> Tuple[Any, List[str]]:
        """
        Update index with new embeddings from lakehouse.
        
        Loads all embeddings and adds ones not already in the index.
        
        Args:
            existing_index_path: Path to existing index file
            lakehouse_path: Base lakehouse directory path
            artifact_type: Type of artifact
            version: Version string
        
        Returns:
            Tuple of (updated_index, all_artifact_ids)
        
        Example:
            >>> updater = IncrementalIndexUpdater()
            >>> index, ids = updater.update_from_lakehouse(
            ...     "span_index.faiss", "/data/lakehouse", "span"
            ... )
        """
        # Load existing index to get existing IDs
        logger.info(f"Loading existing index from {existing_index_path}")
        _, existing_ids, _ = self.builder.load_index(existing_index_path)
        
        # Load all embeddings from lakehouse
        logger.info(f"Loading {artifact_type} embeddings from lakehouse")
        df = load_embeddings(lakehouse_path, artifact_type=artifact_type, version=version)
        
        if df.empty:
            logger.warning("No embeddings found in lakehouse")
            return self.builder.load_index(existing_index_path)[:2]
        
        # Find new embeddings (not in existing index)
        all_ids = df["artifact_id"].tolist()
        existing_set = set(existing_ids)
        new_mask = [id not in existing_set for id in all_ids]
        
        new_df = df[new_mask]
        
        if new_df.empty:
            logger.info("No new embeddings to add")
            return self.builder.load_index(existing_index_path)[:2]
        
        logger.info(f"Found {len(new_df)} new embeddings to add")
        
        # Update index
        return self.update_index(existing_index_path, new_df)
    
    def rebuild_if_needed(
        self,
        index_path: Union[str, Path],
        lakehouse_path: Union[str, Path],
        artifact_type: str = "span",
        version: str = "v1",
        rebuild_threshold: float = 0.3,
    ) -> Tuple[Any, List[str], bool]:
        """
        Rebuild index if the number of new vectors exceeds threshold.
        
        For large updates, rebuilding can be more efficient than incremental updates.
        
        Args:
            index_path: Path to index file
            lakehouse_path: Base lakehouse directory path
            artifact_type: Type of artifact
            version: Version string
            rebuild_threshold: Rebuild if new vectors exceed this fraction of existing (default: 0.3)
        
        Returns:
            Tuple of (index, artifact_ids, was_rebuilt)
        
        Example:
            >>> updater = IncrementalIndexUpdater()
            >>> index, ids, rebuilt = updater.rebuild_if_needed(
            ...     "span_index.faiss", "/data/lakehouse", "span"
            ... )
        """
        index_path = Path(index_path)
        
        # Check if index exists
        if not index_path.exists():
            logger.info("Index does not exist, building from scratch")
            index, ids = self.builder.build_from_lakehouse(
                lakehouse_path, artifact_type, version
            )
            return index, ids, True
        
        # Load existing index
        _, existing_ids, _ = self.builder.load_index(index_path)
        
        # Load all embeddings
        df = load_embeddings(lakehouse_path, artifact_type=artifact_type, version=version)
        
        if df.empty:
            logger.warning("No embeddings in lakehouse")
            index, ids, _ = self.builder.load_index(index_path)
            return index, ids, False
        
        # Count new embeddings
        all_ids = set(df["artifact_id"])
        existing_set = set(existing_ids)
        new_ids = all_ids - existing_set
        
        new_fraction = len(new_ids) / len(existing_ids) if existing_ids else 1.0
        
        logger.info(
            f"New embeddings: {len(new_ids)} ({new_fraction:.1%} of existing {len(existing_ids)})"
        )
        
        # Decide whether to rebuild
        if new_fraction > rebuild_threshold:
            logger.info(
                f"New vectors exceed threshold ({new_fraction:.1%} > {rebuild_threshold:.1%}), "
                "rebuilding index"
            )
            index, ids = self.builder.build_from_lakehouse(
                lakehouse_path, artifact_type, version
            )
            return index, ids, True
        else:
            logger.info("Performing incremental update")
            index, ids = self.update_from_lakehouse(
                index_path, lakehouse_path, artifact_type, version
            )
            return index, ids, False


def update_index_incrementally(
    index_path: Union[str, Path],
    lakehouse_path: Union[str, Path],
    artifact_type: str = "span",
    version: str = "v1",
    save: bool = True,
) -> Tuple[Any, List[str]]:
    """
    Convenience function for incremental index updates.
    
    Args:
        index_path: Path to existing index file
        lakehouse_path: Base lakehouse directory path
        artifact_type: Type of artifact
        version: Version string
        save: Whether to save updated index (default: True)
    
    Returns:
        Tuple of (updated_index, all_artifact_ids)
    
    Example:
        >>> index, ids = update_index_incrementally(
        ...     "span_index.faiss", "/data/lakehouse", "span"
        ... )
    """
    updater = IncrementalIndexUpdater()
    index, ids = updater.update_from_lakehouse(
        index_path, lakehouse_path, artifact_type, version
    )
    
    if save and index is not None:
        logger.info(f"Saving updated index to {index_path}")
        updater.builder.save_index(index, ids, index_path)
    
    return index, ids

