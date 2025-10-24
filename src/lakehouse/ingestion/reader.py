"""
Transcript file reader with support for JSON and JSONL formats.

Handles reading raw transcript files and converting them to structured records.
"""

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from lakehouse.logger import get_default_logger


logger = get_default_logger()


class TranscriptReader:
    """
    Reader for transcript files in JSON and JSONL formats.
    
    Supports both single-episode JSON files and line-delimited JSONL files.
    """
    
    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize transcript reader.
        
        Args:
            file_path: Path to transcript file (JSON or JSONL)
        """
        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {self.file_path}")
        
        self.file_format = self._detect_format()
        logger.info(f"Initialized reader for {self.file_path} (format: {self.file_format})")
    
    def _detect_format(self) -> str:
        """
        Detect file format based on extension and content.
        
        Returns:
            "json" or "jsonl"
        """
        suffix = self.file_path.suffix.lower()
        
        if suffix == ".jsonl":
            return "jsonl"
        elif suffix == ".json":
            return "json"
        else:
            # Try to detect by reading first line
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        json.loads(first_line)
                        # If we can parse the first line as JSON, assume JSONL
                        return "jsonl"
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            
            # Default to JSON
            return "json"
    
    def read_utterances(self) -> List[Dict]:
        """
        Read all utterances from the transcript file.
        
        Returns:
            List of utterance dictionaries
        
        Raises:
            json.JSONDecodeError: If file contains invalid JSON
            IOError: If file cannot be read
        
        Example:
            >>> reader = TranscriptReader("episode.jsonl")
            >>> utterances = reader.read_utterances()
            >>> len(utterances)
            150
        """
        if self.file_format == "jsonl":
            return self._read_jsonl()
        else:
            return self._read_json()
    
    def _read_json(self) -> List[Dict]:
        """
        Read a JSON file (array of utterances or single object).
        
        Returns:
            List of utterance dictionaries
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of utterances
                utterances = data
            elif isinstance(data, dict):
                # Single object - might have an "utterances" field
                if "utterances" in data:
                    utterances = data["utterances"]
                else:
                    # Single utterance as object
                    utterances = [data]
            else:
                logger.warning(f"Unexpected JSON structure in {self.file_path}")
                utterances = []
            
            logger.info(f"Read {len(utterances)} utterances from JSON file")
            return utterances
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {self.file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading {self.file_path}: {e}")
            raise
    
    def _read_jsonl(self) -> List[Dict]:
        """
        Read a JSONL file (one JSON object per line).
        
        Returns:
            List of utterance dictionaries
        """
        utterances = []
        line_num = 0
        
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    try:
                        utterance = json.loads(line)
                        utterances.append(utterance)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping malformed JSON at line {line_num} in {self.file_path}: {e}"
                        )
                        continue
            
            logger.info(f"Read {len(utterances)} utterances from JSONL file ({line_num} lines)")
            return utterances
            
        except Exception as e:
            logger.error(f"Error reading {self.file_path}: {e}")
            raise
    
    def iter_utterances(self) -> Iterator[Dict]:
        """
        Iterate over utterances without loading all into memory.
        
        More memory-efficient for large files.
        
        Yields:
            Utterance dictionaries one at a time
        
        Example:
            >>> reader = TranscriptReader("episode.jsonl")
            >>> for utterance in reader.iter_utterances():
            ...     process(utterance)
        """
        if self.file_format == "jsonl":
            yield from self._iter_jsonl()
        else:
            # For JSON, we need to load the whole file anyway
            yield from self._read_json()
    
    def _iter_jsonl(self) -> Iterator[Dict]:
        """
        Iterate over utterances in a JSONL file.
        
        Yields:
            Utterance dictionaries
        """
        line_num = 0
        
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    try:
                        utterance = json.loads(line)
                        yield utterance
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping malformed JSON at line {line_num} in {self.file_path}: {e}"
                        )
                        continue
        except Exception as e:
            logger.error(f"Error reading {self.file_path} at line {line_num}: {e}")
            raise


def read_transcript_file(file_path: Union[str, Path]) -> List[Dict]:
    """
    Convenience function to read a transcript file.
    
    Args:
        file_path: Path to transcript file
    
    Returns:
        List of utterance dictionaries
    
    Example:
        >>> utterances = read_transcript_file("episode.jsonl")
    """
    reader = TranscriptReader(file_path)
    return reader.read_utterances()


def read_transcript_directory(
    directory: Union[str, Path],
    pattern: str = "*.jsonl",
) -> Dict[str, List[Dict]]:
    """
    Read all transcript files in a directory.
    
    Args:
        directory: Path to directory containing transcript files
        pattern: Glob pattern for matching files (default: "*.jsonl")
    
    Returns:
        Dictionary mapping file paths to lists of utterances
    
    Example:
        >>> transcripts = read_transcript_directory("data/transcripts")
        >>> len(transcripts)
        5
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    
    transcripts = {}
    files = sorted(directory.glob(pattern))
    
    logger.info(f"Found {len(files)} transcript files in {directory}")
    
    for file_path in files:
        try:
            utterances = read_transcript_file(file_path)
            transcripts[str(file_path)] = utterances
            logger.info(f"Read {len(utterances)} utterances from {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            # Continue with other files
            continue
    
    logger.info(f"Successfully read {len(transcripts)} transcript files")
    return transcripts


def extract_episode_id(file_path: Union[str, Path]) -> str:
    """
    Extract episode ID from file path or first utterance.
    
    Args:
        file_path: Path to transcript file
    
    Returns:
        Episode ID string
    
    Example:
        >>> extract_episode_id("LOS - #002 - 2020-09-11 - Title.jsonl")
        'LOS - #002 - 2020-09-11 - Title'
    """
    file_path = Path(file_path)
    
    # Try to read episode_id from first utterance
    try:
        reader = TranscriptReader(file_path)
        utterances = reader.read_utterances()
        
        if utterances and "episode_id" in utterances[0]:
            return utterances[0]["episode_id"]
    except Exception as e:
        logger.warning(f"Could not extract episode_id from file content: {e}")
    
    # Fall back to using filename (without extension)
    return file_path.stem

