"""
PyArrow schema definitions for all lakehouse artifact types.

These schemas enforce data types and structure for Parquet files,
ensuring consistency and type safety across the pipeline.
"""

import pyarrow as pa


# Utterances Schema (normalized layer)
UTTERANCE_SCHEMA = pa.schema([
    pa.field("utterance_id", pa.string(), nullable=False),
    pa.field("episode_id", pa.string(), nullable=False),
    pa.field("start", pa.float64(), nullable=False),
    pa.field("end", pa.float64(), nullable=False),
    pa.field("speaker", pa.string(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("duration", pa.float64(), nullable=False),
])

# Spans Schema (single-speaker contiguous segments)
SPAN_SCHEMA = pa.schema([
    pa.field("span_id", pa.string(), nullable=False),
    pa.field("episode_id", pa.string(), nullable=False),
    pa.field("speaker", pa.string(), nullable=False),
    pa.field("start_time", pa.float64(), nullable=False),
    pa.field("end_time", pa.float64(), nullable=False),
    pa.field("duration", pa.float64(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("utterance_ids", pa.list_(pa.string()), nullable=False),
])

# Beats Schema (semantic meaning units)
BEAT_SCHEMA = pa.schema([
    pa.field("beat_id", pa.string(), nullable=False),
    pa.field("episode_id", pa.string(), nullable=False),
    pa.field("start_time", pa.float64(), nullable=False),
    pa.field("end_time", pa.float64(), nullable=False),
    pa.field("duration", pa.float64(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("span_ids", pa.list_(pa.string()), nullable=False),
    pa.field("topic_label", pa.string(), nullable=True),  # Optional
])

# Sections Schema (5-12 minute logical blocks)
SECTION_SCHEMA = pa.schema([
    pa.field("section_id", pa.string(), nullable=False),
    pa.field("episode_id", pa.string(), nullable=False),
    pa.field("start_time", pa.float64(), nullable=False),
    pa.field("end_time", pa.float64(), nullable=False),
    pa.field("duration_minutes", pa.float64(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("beat_ids", pa.list_(pa.string()), nullable=False),
])

# Embeddings Schema (vector representations)
EMBEDDING_SCHEMA = pa.schema([
    pa.field("artifact_id", pa.string(), nullable=False),
    pa.field("artifact_type", pa.string(), nullable=False),  # 'span' or 'beat'
    pa.field("embedding", pa.list_(pa.float32()), nullable=False),
    pa.field("model_name", pa.string(), nullable=False),
    pa.field("model_version", pa.string(), nullable=True),
])

# Episode Catalog Schema
EPISODE_CATALOG_SCHEMA = pa.schema([
    pa.field("episode_id", pa.string(), nullable=False),
    pa.field("title", pa.string(), nullable=True),
    pa.field("date", pa.string(), nullable=True),
    pa.field("duration", pa.float64(), nullable=False),
    pa.field("speaker_list", pa.list_(pa.string()), nullable=False),
    pa.field("file_path", pa.string(), nullable=False),
    pa.field("utterance_count", pa.int64(), nullable=False),
])

# Speaker Catalog Schema
SPEAKER_CATALOG_SCHEMA = pa.schema([
    pa.field("speaker_name", pa.string(), nullable=False),
    pa.field("episode_count", pa.int64(), nullable=False),
    pa.field("total_utterances", pa.int64(), nullable=False),
    pa.field("total_duration", pa.float64(), nullable=False),
])


def get_schema(artifact_type: str) -> pa.Schema:
    """
    Get the PyArrow schema for a given artifact type.

    Args:
        artifact_type: Type of artifact ("utterance", "span", "beat", "section",
                       "embedding", "episode_catalog", "speaker_catalog")

    Returns:
        PyArrow schema for the artifact type

    Raises:
        ValueError: If artifact_type is not recognized

    Example:
        >>> schema = get_schema("utterance")
        >>> print(schema)
    """
    schema_map = {
        "utterance": UTTERANCE_SCHEMA,
        "span": SPAN_SCHEMA,
        "beat": BEAT_SCHEMA,
        "section": SECTION_SCHEMA,
        "embedding": EMBEDDING_SCHEMA,
        "episode_catalog": EPISODE_CATALOG_SCHEMA,
        "speaker_catalog": SPEAKER_CATALOG_SCHEMA,
    }

    if artifact_type not in schema_map:
        raise ValueError(
            f"Unknown artifact_type: {artifact_type}. "
            f"Valid types: {list(schema_map.keys())}"
        )

    return schema_map[artifact_type]


def validate_dataframe_schema(df, artifact_type: str) -> bool:
    """
    Validate that a pandas DataFrame conforms to the expected schema.

    Args:
        df: Pandas DataFrame to validate
        artifact_type: Expected artifact type

    Returns:
        True if valid, False otherwise

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"utterance_id": ["u1"], "episode_id": ["e1"], ...})
        >>> validate_dataframe_schema(df, "utterance")
        True
    """
    expected_schema = get_schema(artifact_type)
    expected_fields = set(expected_schema.names)
    actual_fields = set(df.columns)

    # Check for missing required fields
    missing = expected_fields - actual_fields
    if missing:
        return False

    # Check for extra fields (warning, but not invalid)
    extra = actual_fields - expected_fields
    if extra:
        # Extra fields are allowed (for metadata), so this is just informational
        pass

    return True


def get_schema_field_names(artifact_type: str) -> list:
    """
    Get the list of field names for a given artifact type.

    Args:
        artifact_type: Type of artifact

    Returns:
        List of field names

    Example:
        >>> get_schema_field_names("utterance")
        ['utterance_id', 'episode_id', 'start', 'end', 'speaker', 'text', 'duration']
    """
    schema = get_schema(artifact_type)
    return schema.names


def get_schema_field_types(artifact_type: str) -> dict:
    """
    Get a dictionary mapping field names to their PyArrow types.

    Args:
        artifact_type: Type of artifact

    Returns:
        Dictionary of field_name -> pyarrow.DataType

    Example:
        >>> types = get_schema_field_types("utterance")
        >>> types["start"]
        DataType(double)
    """
    schema = get_schema(artifact_type)
    return {field.name: field.type for field in schema}


def get_required_fields(artifact_type: str) -> list:
    """
    Get the list of required (non-nullable) fields for an artifact type.

    Args:
        artifact_type: Type of artifact

    Returns:
        List of required field names

    Example:
        >>> get_required_fields("beat")
        ['beat_id', 'episode_id', 'start_time', 'end_time', 'duration', 'text', 'span_ids']
    """
    schema = get_schema(artifact_type)
    return [field.name for field in schema if not field.nullable]


def schema_to_dict(artifact_type: str) -> dict:
    """
    Convert schema to a dictionary representation for JSON serialization.

    Args:
        artifact_type: Type of artifact

    Returns:
        Dictionary with field definitions

    Example:
        >>> schema_dict = schema_to_dict("utterance")
        >>> schema_dict["utterance_id"]
        {'type': 'string', 'nullable': False}
    """
    schema = get_schema(artifact_type)
    result = {}
    for field in schema:
        result[field.name] = {
            "type": str(field.type),
            "nullable": field.nullable,
        }
    return result

