"""
FAISS index building for vector search.

Builds and manages FAISS HNSW indices for efficient similarity search
over span and beat embeddings.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from lakehouse.embeddings.storage import load_embeddings
from lakehouse.logger import get_default_logger


logger = get_default_logger()


class FAISSIndexBuilder:
    """
    Builds FAISS indices for vector similarity search.
    
    Uses HNSW (Hierarchical Navigable Small World) algorithm for
    efficient approximate nearest neighbor search.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize FAISS index builder.
        
        Args:
            config: Configuration dictionary with options:
                - M: Number of connections per layer (default: 32)
                - efConstruction: Search depth during construction (default: 64)
                - efSearch: Search depth during query (default: 32)
                - distance_metric: Distance metric ("l2", "ip", "cosine") (default: "l2")
        
        Example:
            >>> builder = FAISSIndexBuilder(config={"M": 32, "efConstruction": 64})
        """
        self.config = config or {}
        
        # Extract configuration parameters
        self.M = self.config.get("M", 32)
        self.efConstruction = self.config.get("efConstruction", 64)
        self.efSearch = self.config.get("efSearch", 32)
        self.distance_metric = self.config.get("distance_metric", "l2")
        
        logger.info(
            f"FAISSIndexBuilder initialized: M={self.M}, "
            f"efConstruction={self.efConstruction}, metric={self.distance_metric}"
        )
    
    def build_index(
        self,
        embeddings: Union[np.ndarray, pd.DataFrame],
        artifact_ids: Optional[List[str]] = None,
    ) -> Tuple[Any, List[str]]:
        """
        Build FAISS HNSW index from embeddings.
        
        Args:
            embeddings: Embeddings as numpy array (N, D) or DataFrame with 'embedding' column
            artifact_ids: List of artifact IDs corresponding to embeddings
        
        Returns:
            Tuple of (faiss_index, artifact_ids)
        
        Raises:
            ImportError: If faiss package not installed
        
        Example:
            >>> index, ids = builder.build_index(embeddings, artifact_ids)
            >>> index.ntotal
            1000
        """
        try:
            import faiss
        except ImportError:
            logger.error(
                "faiss not installed. "
                "Install with: pip install faiss-cpu (or faiss-gpu for GPU support)"
            )
            raise
        
        # Convert DataFrame to numpy array if needed
        if isinstance(embeddings, pd.DataFrame):
            if artifact_ids is None:
                artifact_ids = embeddings["artifact_id"].tolist()
            
            # Extract embedding vectors
            if "embedding" in embeddings.columns:
                embedding_vectors = embeddings["embedding"].tolist()
                embeddings = np.array(embedding_vectors, dtype=np.float32)
            else:
                raise ValueError("DataFrame must have 'embedding' column")
        else:
            embeddings = np.array(embeddings, dtype=np.float32)
            
            if artifact_ids is None:
                # Generate placeholder IDs
                artifact_ids = [f"vec_{i}" for i in range(len(embeddings))]
        
        if len(embeddings) == 0:
            logger.warning("No embeddings provided, returning empty index")
            return None, []
        
        logger.info(f"Building FAISS index for {len(embeddings)} embeddings")
        
        # Get embedding dimension
        dimension = embeddings.shape[1]
        
        # Normalize embeddings for cosine similarity
        if self.distance_metric == "cosine":
            # Normalize to unit length for cosine similarity
            faiss.normalize_L2(embeddings)
            logger.debug("Normalized embeddings for cosine similarity")
        
        # Create HNSW index
        if self.distance_metric == "ip":
            # Inner product
            index = faiss.IndexHNSWFlat(dimension, self.M, faiss.METRIC_INNER_PRODUCT)
        else:
            # L2 distance (also works for cosine after normalization)
            index = faiss.IndexHNSWFlat(dimension, self.M)
        
        # Set construction parameters
        index.hnsw.efConstruction = self.efConstruction
        
        # Add vectors to index
        index.add(embeddings)
        
        # Set search parameters
        index.hnsw.efSearch = self.efSearch
        
        logger.info(
            f"Built FAISS index with {index.ntotal} vectors "
            f"(dimension: {dimension}, M: {self.M})"
        )
        
        return index, artifact_ids
    
    def build_from_lakehouse(
        self,
        lakehouse_path: Union[str, Path],
        artifact_type: str = "span",
        version: str = "v1",
    ) -> Tuple[Any, List[str]]:
        """
        Build FAISS index from embeddings stored in lakehouse.
        
        Args:
            lakehouse_path: Base lakehouse directory path
            artifact_type: Type of artifact ("span", "beat", etc.)
            version: Version string
        
        Returns:
            Tuple of (faiss_index, artifact_ids)
        
        Example:
            >>> index, ids = builder.build_from_lakehouse("/data/lakehouse", "span")
        """
        logger.info(f"Loading {artifact_type} embeddings from lakehouse")
        
        # Load embeddings
        df = load_embeddings(lakehouse_path, artifact_type=artifact_type, version=version)
        
        if df.empty:
            logger.warning(f"No {artifact_type} embeddings found")
            return None, []
        
        # Build index
        return self.build_index(df)
    
    def save_index(
        self,
        index: Any,
        artifact_ids: List[str],
        output_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save FAISS index and associated metadata.
        
        Args:
            index: FAISS index
            artifact_ids: List of artifact IDs
            output_path: Path to save index file (.faiss extension)
            metadata: Optional metadata dictionary
        
        Example:
            >>> builder.save_index(index, ids, "span_index.faiss")
        """
        try:
            import faiss
        except ImportError:
            raise
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(index, str(output_path))
        logger.info(f"Saved FAISS index to {output_path}")
        
        # Save artifact ID mapping
        mapping_path = output_path.with_suffix(".ids.json")
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(artifact_ids, f, indent=2)
        logger.info(f"Saved artifact ID mapping to {mapping_path}")
        
        # Save index metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "num_vectors": index.ntotal,
            "dimension": index.d,
            "M": self.M,
            "efConstruction": self.efConstruction,
            "efSearch": self.efSearch,
            "distance_metric": self.distance_metric,
        })
        
        metadata_path = output_path.with_suffix(".metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved index metadata to {metadata_path}")
    
    def load_index(
        self,
        index_path: Union[str, Path],
    ) -> Tuple[Any, List[str], Dict[str, Any]]:
        """
        Load FAISS index and associated metadata.
        
        Args:
            index_path: Path to index file (.faiss extension)
        
        Returns:
            Tuple of (faiss_index, artifact_ids, metadata)
        
        Example:
            >>> index, ids, metadata = builder.load_index("span_index.faiss")
        """
        try:
            import faiss
        except ImportError:
            raise
        
        index_path = Path(index_path)
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        # Load FAISS index
        index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index from {index_path} ({index.ntotal} vectors)")
        
        # Load artifact ID mapping
        mapping_path = index_path.with_suffix(".ids.json")
        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                artifact_ids = json.load(f)
        else:
            logger.warning(f"Artifact ID mapping not found at {mapping_path}")
            artifact_ids = []
        
        # Load metadata
        metadata_path = index_path.with_suffix(".metadata.json")
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            logger.warning(f"Index metadata not found at {metadata_path}")
            metadata = {}
        
        return index, artifact_ids, metadata
    
    def search(
        self,
        index: Any,
        artifact_ids: List[str],
        query_vector: np.ndarray,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for nearest neighbors in the index.
        
        Args:
            index: FAISS index
            artifact_ids: List of artifact IDs
            query_vector: Query vector (1D or 2D array)
            k: Number of nearest neighbors to return
        
        Returns:
            List of dictionaries with artifact_id, distance, and rank
        
        Example:
            >>> results = builder.search(index, ids, query_vector, k=5)
            >>> results[0]["artifact_id"]
            'spn_...'
        """
        try:
            import faiss
        except ImportError:
            raise
        
        # Ensure query is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        query_vector = query_vector.astype(np.float32)
        
        # Normalize for cosine similarity
        if self.distance_metric == "cosine":
            faiss.normalize_L2(query_vector)
        
        # Search
        distances, indices = index.search(query_vector, k)
        
        # Format results
        results = []
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(artifact_ids):
                result = {
                    "artifact_id": artifact_ids[idx],
                    "distance": float(distance),
                    "rank": rank,
                }
                results.append(result)
        
        return results


def build_and_save_index(
    lakehouse_path: Union[str, Path],
    artifact_type: str = "span",
    version: str = "v1",
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Convenience function to build and save a FAISS index.
    
    Args:
        lakehouse_path: Base lakehouse directory path
        artifact_type: Type of artifact
        version: Version string
        config: Index configuration
    
    Returns:
        Path to saved index file
    
    Example:
        >>> path = build_and_save_index("/data/lakehouse", "span")
    """
    lakehouse_path = Path(lakehouse_path)
    
    # Build index
    builder = FAISSIndexBuilder(config=config)
    index, artifact_ids = builder.build_from_lakehouse(
        lakehouse_path, artifact_type, version
    )
    
    if index is None:
        logger.error(f"Failed to build index for {artifact_type}")
        return None
    
    # Save index
    output_path = lakehouse_path / "ann_index" / version / f"{artifact_type}_index.faiss"
    builder.save_index(index, artifact_ids, output_path)
    
    return output_path

