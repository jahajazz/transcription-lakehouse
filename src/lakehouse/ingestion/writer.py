"""
Parquet file writing utilities with schema enforcement.

Handles writing normalized data to Parquet files with compression,
schema validation, and versioning support.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lakehouse.schemas import get_schema
from lakehouse.logger import get_default_logger


logger = get_default_logger()


class ParquetWriter:
    """
    Writer for Parquet files with schema enforcement.
    """
    
    def __init__(
        self,
        artifact_type: str,
        compression: str = "snappy",
        enforce_schema: bool = True,
    ):
        """
        Initialize Parquet writer.
        
        Args:
            artifact_type: Type of artifact ("utterance", "span", "beat", etc.)
            compression: Compression codec ("snappy", "gzip", "brotli", "zstd", "none")
            enforce_schema: Whether to enforce schema validation
        """
        self.artifact_type = artifact_type
        self.compression = compression
        self.enforce_schema = enforce_schema
        self.schema = get_schema(artifact_type) if enforce_schema else None
        
        logger.debug(f"Initialized ParquetWriter for {artifact_type} with {compression} compression")
    
    def write(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        output_path: Union[str, Path],
        overwrite: bool = False,
    ) -> None:
        """
        Write data to Parquet file with schema enforcement.
        
        Args:
            data: Data to write (DataFrame or list of dictionaries)
            output_path: Path to output Parquet file
            overwrite: Whether to overwrite existing file
        
        Raises:
            FileExistsError: If file exists and overwrite=False
            ValueError: If data doesn't conform to schema
        
        Example:
            >>> writer = ParquetWriter("utterance")
            >>> writer.write(utterances, "output/utterances.parquet")
        """
        output_path = Path(output_path)
        
        # Check if file exists
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_path}. "
                f"Use overwrite=True to replace."
            )
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        if df.empty:
            logger.warning(f"Writing empty DataFrame to {output_path}")
        
        # Convert to PyArrow Table
        if self.enforce_schema and self.schema:
            try:
                # Validate and convert with schema
                table = pa.Table.from_pandas(df, schema=self.schema)
                logger.debug(f"Validated data against {self.artifact_type} schema")
            except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
                logger.error(f"Schema validation failed: {e}")
                raise ValueError(f"Data does not conform to {self.artifact_type} schema: {e}")
        else:
            # Convert without schema enforcement
            table = pa.Table.from_pandas(df)
        
        # Write to Parquet
        pq.write_table(
            table,
            output_path,
            compression=self.compression,
            use_dictionary=True,  # Enable dictionary encoding for string columns
            write_statistics=True,  # Write column statistics for query optimization
        )
        
        file_size = output_path.stat().st_size / 1024  # KB
        logger.info(
            f"Wrote {len(df)} rows to {output_path} "
            f"({file_size:.1f} KB, {self.compression} compression)"
        )
    
    def append(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        output_path: Union[str, Path],
    ) -> None:
        """
        Append data to existing Parquet file.
        
        Note: This reads the existing file, appends data, and rewrites.
        For large files, consider using partitioned datasets instead.
        
        Args:
            data: Data to append
            output_path: Path to Parquet file
        
        Example:
            >>> writer = ParquetWriter("utterance")
            >>> writer.append(new_utterances, "output/utterances.parquet")
        """
        output_path = Path(output_path)
        
        # Read existing data if file exists
        if output_path.exists():
            existing_df = pd.read_parquet(output_path)
            logger.debug(f"Read {len(existing_df)} existing rows from {output_path}")
        else:
            existing_df = pd.DataFrame()
        
        # Convert new data to DataFrame
        if isinstance(data, list):
            new_df = pd.DataFrame(data)
        else:
            new_df = data
        
        # Concatenate
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Write combined data
        self.write(combined_df, output_path, overwrite=True)
        logger.info(f"Appended {len(new_df)} rows to {output_path}")


def write_parquet(
    data: Union[pd.DataFrame, List[Dict[str, Any]]],
    output_path: Union[str, Path],
    artifact_type: str,
    compression: str = "snappy",
    enforce_schema: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Convenience function to write data to Parquet file.
    
    Args:
        data: Data to write (DataFrame or list of dictionaries)
        output_path: Path to output Parquet file
        artifact_type: Type of artifact ("utterance", "span", "beat", etc.)
        compression: Compression codec
        enforce_schema: Whether to enforce schema validation
        overwrite: Whether to overwrite existing file
    
    Example:
        >>> write_parquet(utterances, "data/utterances.parquet", "utterance")
    """
    writer = ParquetWriter(
        artifact_type=artifact_type,
        compression=compression,
        enforce_schema=enforce_schema,
    )
    writer.write(data, output_path, overwrite=overwrite)


def read_parquet(
    file_path: Union[str, Path],
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Read a Parquet file into a DataFrame.
    
    Args:
        file_path: Path to Parquet file
        columns: Optional list of columns to read (reads all if None)
    
    Returns:
        DataFrame with data
    
    Example:
        >>> df = read_parquet("data/utterances.parquet")
        >>> df = read_parquet("data/utterances.parquet", columns=["utterance_id", "text"])
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    
    df = pd.read_parquet(file_path, columns=columns)
    logger.debug(f"Read {len(df)} rows from {file_path}")
    
    return df


def create_versioned_directory(
    base_path: Union[str, Path],
    artifact_type: str,
    version: str = "v1",
) -> Path:
    """
    Create a versioned directory for artifact storage.
    
    Args:
        base_path: Base lakehouse directory path
        artifact_type: Type of artifact ("normalized", "spans", "beats", etc.)
        version: Version string (default: "v1")
    
    Returns:
        Path to versioned directory
    
    Example:
        >>> path = create_versioned_directory("/data/lakehouse", "normalized", "v1")
        >>> path
        PosixPath('/data/lakehouse/normalized/v1')
    """
    base_path = Path(base_path)
    versioned_path = base_path / artifact_type / version
    versioned_path.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Created versioned directory: {versioned_path}")
    return versioned_path


def write_versioned_parquet(
    data: Union[pd.DataFrame, List[Dict[str, Any]]],
    base_path: Union[str, Path],
    artifact_type: str,
    filename: str,
    version: str = "v1",
    compression: str = "snappy",
    enforce_schema: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Write data to a versioned Parquet file in the lakehouse structure.
    
    Args:
        data: Data to write
        base_path: Base lakehouse directory path
        artifact_type: Type of artifact (e.g., "normalized", "spans")
        filename: Output filename (e.g., "utterances.parquet")
        version: Version string (default: "v1")
        compression: Compression codec
        enforce_schema: Whether to enforce schema validation
        overwrite: Whether to overwrite existing file
    
    Returns:
        Path to written file
    
    Example:
        >>> path = write_versioned_parquet(
        ...     utterances,
        ...     "/data/lakehouse",
        ...     "normalized",
        ...     "utterances.parquet",
        ... )
    """
    # Create versioned directory
    versioned_dir = create_versioned_directory(base_path, artifact_type, version)
    
    # Full output path
    output_path = versioned_dir / filename
    
    # Determine schema artifact type from directory name
    # Map directory names to schema types
    schema_type_map = {
        "normalized": "utterance",
        "spans": "span",
        "beats": "beat",
        "sections": "section",
        "embeddings": "embedding",
    }
    schema_type = schema_type_map.get(artifact_type, artifact_type)
    
    # Write with schema enforcement
    write_parquet(
        data=data,
        output_path=output_path,
        artifact_type=schema_type,
        compression=compression,
        enforce_schema=enforce_schema,
        overwrite=overwrite,
    )
    
    return output_path


def get_parquet_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get metadata information about a Parquet file.
    
    Args:
        file_path: Path to Parquet file
    
    Returns:
        Dictionary with file information
    
    Example:
        >>> info = get_parquet_file_info("data/utterances.parquet")
        >>> info["num_rows"]
        150
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    
    # Read Parquet metadata
    parquet_file = pq.ParquetFile(file_path)
    metadata = parquet_file.metadata
    
    info = {
        "file_path": str(file_path),
        "file_size_bytes": file_path.stat().st_size,
        "num_rows": metadata.num_rows,
        "num_columns": metadata.num_columns,
        "num_row_groups": metadata.num_row_groups,
        "format_version": metadata.format_version,
        "created_by": metadata.created_by,
        "schema": parquet_file.schema_arrow,
        "column_names": parquet_file.schema_arrow.names,
    }
    
    return info

