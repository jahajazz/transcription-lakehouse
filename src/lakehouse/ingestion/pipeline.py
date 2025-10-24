"""
End-to-end ingestion pipeline for transcript data.

Orchestrates the complete ingestion process: read → validate → normalize → write.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

from lakehouse.ingestion.reader import TranscriptReader, read_transcript_directory
from lakehouse.ingestion.validator import filter_valid_utterances, ValidationResult
from lakehouse.ingestion.normalizer import normalize_episode, compute_utterance_statistics
from lakehouse.ingestion.writer import write_versioned_parquet
from lakehouse.structure import LakehouseStructure, get_or_create_lakehouse
from lakehouse.logger import get_default_logger


logger = get_default_logger()


class IngestionResult:
    """Result of ingestion pipeline execution."""
    
    def __init__(self):
        """Initialize empty ingestion result."""
        self.episodes_processed = 0
        self.episodes_failed = 0
        self.total_utterances = 0
        self.valid_utterances = 0
        self.invalid_utterances = 0
        self.validation_results: Dict[str, ValidationResult] = {}
        self.output_paths: Dict[str, Path] = {}
        self.errors: List[str] = []
    
    def add_episode_result(
        self,
        episode_id: str,
        utterance_count: int,
        validation_result: ValidationResult,
        output_path: Optional[Path] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Add result for a single episode.
        
        Args:
            episode_id: Episode identifier
            utterance_count: Number of utterances processed
            validation_result: Validation result for the episode
            output_path: Path to written Parquet file (if successful)
            error: Error message (if failed)
        """
        if error:
            self.episodes_failed += 1
            self.errors.append(f"{episode_id}: {error}")
        else:
            self.episodes_processed += 1
            if output_path:
                self.output_paths[episode_id] = output_path
        
        self.total_utterances += utterance_count
        self.valid_utterances += validation_result.valid_count
        self.invalid_utterances += validation_result.invalid_count
        self.validation_results[episode_id] = validation_result
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "Ingestion Pipeline Summary:",
            f"  Episodes processed: {self.episodes_processed}",
            f"  Episodes failed: {self.episodes_failed}",
            f"  Total utterances: {self.total_utterances}",
            f"  Valid utterances: {self.valid_utterances}",
            f"  Invalid utterances: {self.invalid_utterances}",
        ]
        
        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
            for error in self.errors[:5]:  # Show first 5 errors
                lines.append(f"    - {error}")
        
        return "\n".join(lines)


class IngestionPipeline:
    """
    End-to-end ingestion pipeline for transcript data.
    """
    
    def __init__(
        self,
        lakehouse_path: Union[str, Path],
        version: str = "v1",
        skip_invalid: bool = True,
        copy_raw: bool = True,
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            lakehouse_path: Path to lakehouse base directory
            version: Version for output data (default: "v1")
            skip_invalid: Whether to skip invalid utterances (default: True)
            copy_raw: Whether to copy raw files to lakehouse (default: True)
        """
        self.lakehouse_path = Path(lakehouse_path)
        self.version = version
        self.skip_invalid = skip_invalid
        self.copy_raw = copy_raw
        
        # Initialize or get lakehouse structure
        self.structure = get_or_create_lakehouse(lakehouse_path, version=version)
        
        logger.info(
            f"Initialized ingestion pipeline for lakehouse at {lakehouse_path} "
            f"(version: {version}, skip_invalid: {skip_invalid})"
        )
    
    def ingest_file(
        self,
        file_path: Union[str, Path],
        episode_id: Optional[str] = None,
    ) -> IngestionResult:
        """
        Ingest a single transcript file.
        
        Args:
            file_path: Path to transcript file (JSON or JSONL)
            episode_id: Optional episode ID override
        
        Returns:
            IngestionResult with processing details
        
        Example:
            >>> pipeline = IngestionPipeline("/data/lakehouse")
            >>> result = pipeline.ingest_file("episode.jsonl")
            >>> result.episodes_processed
            1
        """
        file_path = Path(file_path)
        result = IngestionResult()
        
        logger.info(f"Starting ingestion of {file_path}")
        
        try:
            # Step 1: Read raw transcript
            reader = TranscriptReader(file_path)
            raw_utterances = reader.read_utterances()
            
            if not raw_utterances:
                error = "No utterances found in file"
                logger.warning(f"{file_path}: {error}")
                result.add_episode_result("unknown", 0, ValidationResult(), error=error)
                return result
            
            # Determine episode ID
            if episode_id is None:
                episode_id = raw_utterances[0].get("episode_id", file_path.stem)
            
            logger.info(f"Processing episode {episode_id} with {len(raw_utterances)} utterances")
            
            # Step 2: Validate utterances
            if self.skip_invalid:
                valid_utterances, validation_result = filter_valid_utterances(raw_utterances)
                
                if not valid_utterances:
                    error = "All utterances failed validation"
                    logger.error(f"{episode_id}: {error}")
                    result.add_episode_result(episode_id, len(raw_utterances), validation_result, error=error)
                    return result
                
                if validation_result.invalid_count > 0:
                    logger.warning(
                        f"{episode_id}: Skipped {validation_result.invalid_count} invalid utterances"
                    )
            else:
                # Fail if any utterance is invalid
                from lakehouse.ingestion.validator import validate_utterances
                validation_result = validate_utterances(raw_utterances, fail_fast=False)
                
                if not validation_result.is_valid:
                    error = f"Validation failed with {len(validation_result.errors)} errors"
                    logger.error(f"{episode_id}: {error}")
                    result.add_episode_result(episode_id, len(raw_utterances), validation_result, error=error)
                    return result
                
                valid_utterances = raw_utterances
            
            # Step 3: Normalize utterances
            normalized_utterances = normalize_episode(
                valid_utterances,
                episode_id=episode_id,
                sort_by_time=True,
            )
            
            # Step 4: Write to Parquet
            output_path = write_versioned_parquet(
                data=normalized_utterances,
                base_path=self.lakehouse_path,
                artifact_type="normalized",
                filename=f"{episode_id}.parquet",
                version=self.version,
                compression="snappy",
                enforce_schema=True,
                overwrite=True,
            )
            
            logger.info(f"Successfully ingested {episode_id} to {output_path}")
            
            # Step 5: Copy raw file if requested
            if self.copy_raw:
                self._copy_raw_file(file_path, episode_id)
            
            # Record result
            result.add_episode_result(
                episode_id=episode_id,
                utterance_count=len(raw_utterances),
                validation_result=validation_result,
                output_path=output_path,
            )
            
        except Exception as e:
            error = f"Unexpected error: {str(e)}"
            logger.error(f"Failed to ingest {file_path}: {error}", exc_info=True)
            result.add_episode_result("unknown", 0, ValidationResult(), error=error)
        
        return result
    
    def ingest_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.jsonl",
    ) -> IngestionResult:
        """
        Ingest all transcript files in a directory.
        
        Args:
            directory: Path to directory containing transcript files
            pattern: Glob pattern for matching files (default: "*.jsonl")
        
        Returns:
            Aggregated IngestionResult for all files
        
        Example:
            >>> pipeline = IngestionPipeline("/data/lakehouse")
            >>> result = pipeline.ingest_directory("data/transcripts")
            >>> result.episodes_processed
            5
        """
        directory = Path(directory)
        combined_result = IngestionResult()
        
        logger.info(f"Starting batch ingestion from {directory} (pattern: {pattern})")
        
        # Find all matching files
        files = sorted(directory.glob(pattern))
        
        if not files:
            logger.warning(f"No files found matching {pattern} in {directory}")
            return combined_result
        
        logger.info(f"Found {len(files)} files to ingest")
        
        # Ingest each file
        for file_path in files:
            logger.info(f"Processing file {file_path.name} ({files.index(file_path) + 1}/{len(files)})")
            
            file_result = self.ingest_file(file_path)
            
            # Merge results
            combined_result.episodes_processed += file_result.episodes_processed
            combined_result.episodes_failed += file_result.episodes_failed
            combined_result.total_utterances += file_result.total_utterances
            combined_result.valid_utterances += file_result.valid_utterances
            combined_result.invalid_utterances += file_result.invalid_utterances
            combined_result.validation_results.update(file_result.validation_results)
            combined_result.output_paths.update(file_result.output_paths)
            combined_result.errors.extend(file_result.errors)
        
        logger.info(combined_result.summary())
        return combined_result
    
    def _copy_raw_file(self, source_path: Path, episode_id: str) -> None:
        """
        Copy raw transcript file to lakehouse raw directory.
        
        Args:
            source_path: Source file path
            episode_id: Episode identifier
        """
        try:
            raw_dir = self.structure.get_raw_path()
            dest_path = raw_dir / f"{episode_id}{source_path.suffix}"
            
            shutil.copy2(source_path, dest_path)
            logger.debug(f"Copied raw file to {dest_path}")
        except Exception as e:
            logger.warning(f"Failed to copy raw file: {e}")


def ingest_transcript(
    file_path: Union[str, Path],
    lakehouse_path: Union[str, Path],
    version: str = "v1",
    skip_invalid: bool = True,
) -> IngestionResult:
    """
    Convenience function to ingest a single transcript file.
    
    Args:
        file_path: Path to transcript file
        lakehouse_path: Path to lakehouse base directory
        version: Version for output data
        skip_invalid: Whether to skip invalid utterances
    
    Returns:
        IngestionResult
    
    Example:
        >>> result = ingest_transcript("episode.jsonl", "/data/lakehouse")
        >>> result.episodes_processed
        1
    """
    pipeline = IngestionPipeline(
        lakehouse_path=lakehouse_path,
        version=version,
        skip_invalid=skip_invalid,
    )
    return pipeline.ingest_file(file_path)


def ingest_directory(
    directory: Union[str, Path],
    lakehouse_path: Union[str, Path],
    pattern: str = "*.jsonl",
    version: str = "v1",
    skip_invalid: bool = True,
) -> IngestionResult:
    """
    Convenience function to ingest all files in a directory.
    
    Args:
        directory: Path to directory containing transcript files
        lakehouse_path: Path to lakehouse base directory
        pattern: Glob pattern for matching files
        version: Version for output data
        skip_invalid: Whether to skip invalid utterances
    
    Returns:
        IngestionResult
    
    Example:
        >>> result = ingest_directory("data/transcripts", "/data/lakehouse")
        >>> result.episodes_processed
        10
    """
    pipeline = IngestionPipeline(
        lakehouse_path=lakehouse_path,
        version=version,
        skip_invalid=skip_invalid,
    )
    return pipeline.ingest_directory(directory, pattern=pattern)

