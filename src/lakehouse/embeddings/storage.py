"""
Embedding storage utilities.

Handles storing and loading embeddings in Parquet format with
artifact metadata, model information, and version tracking.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa

from lakehouse.ingestion.writer import write_versioned_parquet
from lakehouse.schemas import get_schema
from lakehouse.logger import get_default_logger


logger = get_default_logger()


def store_embeddings(
    embeddings: List[Dict[str, Any]],
    lakehouse_path: Union[str, Path],
    version: str = "v1",
    overwrite: bool = False,
) -> Path:
    """
    Store embeddings in versioned Parquet format.
    
    Args:
        embeddings: List of embedding records with artifact_id, artifact_type, embedding, etc.
        lakehouse_path: Base lakehouse directory path
        version: Version string (default: "v1")
        overwrite: Whether to overwrite existing file
    
    Returns:
        Path to written Parquet file
    
    Example:
        >>> embeddings = [
        ...     {"artifact_id": "spn_...", "artifact_type": "span", "embedding": [...], ...}
        ... ]
        >>> path = store_embeddings(embeddings, "/data/lakehouse")
    """
    lakehouse_path = Path(lakehouse_path)
    
    if not embeddings:
        logger.warning("No embeddings to store")
        return None
    
    # Group embeddings by artifact type
    embeddings_by_type = {}
    for embedding in embeddings:
        artifact_type = embedding.get("artifact_type", "unknown")
        if artifact_type not in embeddings_by_type:
            embeddings_by_type[artifact_type] = []
        embeddings_by_type[artifact_type].append(embedding)
    
    # Store each artifact type separately
    output_paths = []
    for artifact_type, type_embeddings in embeddings_by_type.items():
        logger.info(f"Storing {len(type_embeddings)} {artifact_type} embeddings")
        
        output_path = write_versioned_parquet(
            data=type_embeddings,
            base_path=lakehouse_path,
            artifact_type="embeddings",
            filename=f"{artifact_type}_embeddings.parquet",
            version=version,
            compression="snappy",
            enforce_schema=True,
            overwrite=overwrite,
        )
        
        output_paths.append(output_path)
        logger.info(f"Stored {artifact_type} embeddings at {output_path}")
    
    # Store metadata
    _store_embedding_metadata(
        embeddings=embeddings,
        lakehouse_path=lakehouse_path,
        version=version,
    )
    
    return output_paths[0] if output_paths else None


def _store_embedding_metadata(
    embeddings: List[Dict[str, Any]],
    lakehouse_path: Path,
    version: str,
) -> None:
    """
    Store metadata about the embeddings.
    
    Args:
        embeddings: List of embedding records
        lakehouse_path: Base lakehouse directory path
        version: Version string
    """
    if not embeddings:
        return
    
    # Extract metadata
    first_embedding = embeddings[0]
    model_name = first_embedding.get("model_name", "unknown")
    model_version = first_embedding.get("model_version", "unknown")
    embedding_dim = len(first_embedding.get("embedding", []))
    
    # Count by artifact type
    type_counts = {}
    for embedding in embeddings:
        artifact_type = embedding.get("artifact_type", "unknown")
        type_counts[artifact_type] = type_counts.get(artifact_type, 0) + 1
    
    metadata = {
        "model_name": model_name,
        "model_version": model_version,
        "embedding_dimension": embedding_dim,
        "version": version,
        "total_embeddings": len(embeddings),
        "embeddings_by_type": type_counts,
    }
    
    # Write metadata file
    metadata_dir = lakehouse_path / "embeddings" / version
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / "metadata.json"
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Stored embedding metadata at {metadata_path}")


def load_embeddings(
    lakehouse_path: Union[str, Path],
    artifact_type: Optional[str] = None,
    version: str = "v1",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load embeddings from Parquet format.
    
    Args:
        lakehouse_path: Base lakehouse directory path
        artifact_type: Type of artifact ("span", "beat", etc.) or None for all
        version: Version string (default: "v1")
        columns: Optional list of columns to load
    
    Returns:
        DataFrame with embeddings
    
    Example:
        >>> df = load_embeddings("/data/lakehouse", artifact_type="span")
        >>> df.columns
        Index(['artifact_id', 'artifact_type', 'embedding', 'model_name', 'model_version'])
    """
    lakehouse_path = Path(lakehouse_path)
    embeddings_dir = lakehouse_path / "embeddings" / version
    
    if not embeddings_dir.exists():
        logger.warning(f"Embeddings directory not found: {embeddings_dir}")
        return pd.DataFrame()
    
    # Load specific artifact type or all
    if artifact_type:
        filename = f"{artifact_type}_embeddings.parquet"
        file_path = embeddings_dir / filename
        
        if not file_path.exists():
            logger.warning(f"Embeddings file not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_parquet(file_path, columns=columns)
        logger.info(f"Loaded {len(df)} {artifact_type} embeddings")
        
    else:
        # Load all embedding files
        embedding_files = list(embeddings_dir.glob("*_embeddings.parquet"))
        
        if not embedding_files:
            logger.warning(f"No embedding files found in {embeddings_dir}")
            return pd.DataFrame()
        
        dfs = []
        for file_path in embedding_files:
            df = pd.read_parquet(file_path, columns=columns)
            dfs.append(df)
        
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(df)} embeddings from {len(embedding_files)} files")
    
    return df


def load_embedding_metadata(
    lakehouse_path: Union[str, Path],
    version: str = "v1",
) -> Dict[str, Any]:
    """
    Load embedding metadata.
    
    Args:
        lakehouse_path: Base lakehouse directory path
        version: Version string (default: "v1")
    
    Returns:
        Metadata dictionary
    
    Example:
        >>> metadata = load_embedding_metadata("/data/lakehouse")
        >>> metadata["model_name"]
        'all-MiniLM-L6-v2'
    """
    lakehouse_path = Path(lakehouse_path)
    metadata_path = lakehouse_path / "embeddings" / version / "metadata.json"
    
    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return {}
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    logger.debug(f"Loaded embedding metadata from {metadata_path}")
    return metadata


def get_embeddings_for_artifacts(
    artifact_ids: List[str],
    lakehouse_path: Union[str, Path],
    artifact_type: str,
    version: str = "v1",
) -> pd.DataFrame:
    """
    Get embeddings for specific artifact IDs.
    
    Args:
        artifact_ids: List of artifact IDs to retrieve
        lakehouse_path: Base lakehouse directory path
        artifact_type: Type of artifact
        version: Version string
    
    Returns:
        DataFrame with embeddings for specified artifacts
    
    Example:
        >>> df = get_embeddings_for_artifacts(
        ...     ["spn_001", "spn_002"],
        ...     "/data/lakehouse",
        ...     "span"
        ... )
    """
    # Load all embeddings of this type
    df = load_embeddings(lakehouse_path, artifact_type=artifact_type, version=version)
    
    if df.empty:
        return df
    
    # Filter to requested IDs
    df_filtered = df[df["artifact_id"].isin(artifact_ids)]
    
    logger.info(
        f"Retrieved {len(df_filtered)} embeddings out of {len(artifact_ids)} requested"
    )
    
    return df_filtered


def embeddings_exist(
    lakehouse_path: Union[str, Path],
    artifact_type: Optional[str] = None,
    version: str = "v1",
) -> bool:
    """
    Check if embeddings exist for an artifact type.
    
    Args:
        lakehouse_path: Base lakehouse directory path
        artifact_type: Type of artifact or None to check any
        version: Version string
    
    Returns:
        True if embeddings exist, False otherwise
    
    Example:
        >>> if embeddings_exist("/data/lakehouse", "span"):
        ...     print("Span embeddings found")
    """
    lakehouse_path = Path(lakehouse_path)
    embeddings_dir = lakehouse_path / "embeddings" / version
    
    if not embeddings_dir.exists():
        return False
    
    if artifact_type:
        filename = f"{artifact_type}_embeddings.parquet"
        return (embeddings_dir / filename).exists()
    else:
        # Check if any embedding files exist
        embedding_files = list(embeddings_dir.glob("*_embeddings.parquet"))
        return len(embedding_files) > 0


def get_embedding_statistics(
    lakehouse_path: Union[str, Path],
    version: str = "v1",
) -> Dict[str, Any]:
    """
    Get statistics about stored embeddings.
    
    Args:
        lakehouse_path: Base lakehouse directory path
        version: Version string
    
    Returns:
        Dictionary with statistics
    
    Example:
        >>> stats = get_embedding_statistics("/data/lakehouse")
        >>> stats["total_embeddings"]
        1500
    """
    # Load metadata
    metadata = load_embedding_metadata(lakehouse_path, version)
    
    if not metadata:
        return {
            "exists": False,
            "total_embeddings": 0,
        }
    
    stats = {
        "exists": True,
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "embedding_dimension": metadata.get("embedding_dimension"),
        "total_embeddings": metadata.get("total_embeddings", 0),
        "embeddings_by_type": metadata.get("embeddings_by_type", {}),
        "version": version,
    }
    
    return stats

