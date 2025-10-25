"""
Episode catalog generation and management.

Provides utilities for generating episode catalogs with metadata, statistics,
and file information using DuckDB queries on Parquet files.
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from lakehouse.logger import get_default_logger


logger = get_default_logger()


class EpisodeCatalog:
    """Manages episode catalog generation and queries."""
    
    def __init__(self, lakehouse_path: Path, version: str = "v1"):
        """
        Initialize episode catalog.
        
        Args:
            lakehouse_path: Path to lakehouse directory
            version: Version of data to catalog
        """
        self.lakehouse_path = lakehouse_path
        self.version = version
        self.normalized_path = lakehouse_path / "normalized" / version
        self.catalog_path = lakehouse_path / "catalogs"
        self.catalog_path.mkdir(exist_ok=True)
    
    def _extract_title(self, episode_id: str) -> str:
        """
        Extract title from episode_id.
        
        Episode IDs follow format: "Series - #NUM - YYYY-MM-DD - Title"
        
        Args:
            episode_id: Full episode identifier
        
        Returns:
            Extracted title or full episode_id if format doesn't match
        """
        try:
            parts = episode_id.split(' - ')
            if len(parts) >= 4:
                # Join everything after the date (parts[3:])
                return ' - '.join(parts[3:])
            return episode_id
        except Exception:
            return episode_id
    
    def _extract_date(self, episode_id: str) -> Optional[str]:
        """
        Extract date from episode_id.
        
        Episode IDs follow format: "Series - #NUM - YYYY-MM-DD - Title"
        
        Args:
            episode_id: Full episode identifier
        
        Returns:
            Extracted date string (YYYY-MM-DD) or None if not found
        """
        try:
            parts = episode_id.split(' - ')
            if len(parts) >= 3:
                # The date should be in parts[2]
                date_str = parts[2]
                # Validate it looks like a date (YYYY-MM-DD)
                if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
                    return date_str
            return None
        except Exception:
            return None
    
    def generate_catalog(self) -> pd.DataFrame:
        """
        Generate episode catalog from normalized utterances.
        
        Returns:
            DataFrame with episode catalog information
        """
        if not self.normalized_path.exists():
            logger.warning(f"No normalized data found at {self.normalized_path}")
            return pd.DataFrame()
        
        # Find all parquet files in normalized directory
        parquet_files = list(self.normalized_path.glob("*.parquet"))
        if not parquet_files:
            logger.warning(f"No parquet files found in {self.normalized_path}")
            return pd.DataFrame()
        
        logger.info(f"Generating episode catalog from {len(parquet_files)} files")
        
        # Use DuckDB to query all parquet files
        conn = duckdb.connect()
        
        try:
            # Build query to aggregate episode information
            query = """
            SELECT 
                episode_id,
                MIN("start") as start_time,
                MAX("end") as end_time,
                MAX("end") - MIN("start") as duration_seconds,
                COUNT(*) as utterance_count,
                COUNT(DISTINCT speaker) as speaker_count,
                STRING_AGG(DISTINCT speaker, ', ' ORDER BY speaker) as speaker_list,
                STRING_AGG(text, ' ') as full_text,
                MIN("start") as first_utterance_time,
                MAX("end") as last_utterance_time
            FROM read_parquet(?)
            GROUP BY episode_id
            ORDER BY episode_id
            """
            
            # Execute query for each file and combine results
            all_episodes = []
            
            for parquet_file in parquet_files:
                try:
                    df = conn.execute(query, [str(parquet_file)]).df()
                    if not df.empty:
                        # Add file information
                        df['source_file'] = parquet_file.name
                        df['file_path'] = str(parquet_file)
                        df['file_size_bytes'] = parquet_file.stat().st_size
                        df['file_modified'] = pd.Timestamp.fromtimestamp(parquet_file.stat().st_mtime)
                        all_episodes.append(df)
                except Exception as e:
                    logger.error(f"Error processing {parquet_file}: {e}")
                    continue
            
            if not all_episodes:
                logger.warning("No episodes found in any parquet files")
                return pd.DataFrame()
            
            # Combine all episode data
            catalog_df = pd.concat(all_episodes, ignore_index=True)
            
            # Add derived fields
            catalog_df['duration_minutes'] = catalog_df['duration_seconds'] / 60.0
            catalog_df['avg_utterance_duration'] = catalog_df['duration_seconds'] / catalog_df['utterance_count']
            catalog_df['catalog_generated'] = datetime.now()
            catalog_df['version'] = self.version
            
            # Extract title and date from episode_id (format: "Series - #NUM - YYYY-MM-DD - Title")
            catalog_df['title'] = catalog_df['episode_id'].apply(self._extract_title)
            catalog_df['date'] = catalog_df['episode_id'].apply(self._extract_date)
            
            # Reorder columns for better readability
            column_order = [
                'episode_id', 'title', 'date', 'duration_seconds', 'duration_minutes',
                'utterance_count', 'speaker_count', 'speaker_list', 'avg_utterance_duration',
                'start_time', 'end_time', 'first_utterance_time', 'last_utterance_time',
                'source_file', 'file_path', 'file_size_bytes', 'file_modified',
                'catalog_generated', 'version'
            ]
            
            # Only include columns that exist
            available_columns = [col for col in column_order if col in catalog_df.columns]
            catalog_df = catalog_df[available_columns]
            
            logger.info(f"Generated catalog for {len(catalog_df)} episodes")
            return catalog_df
            
        finally:
            conn.close()
    
    def save_catalog(self, catalog_df: pd.DataFrame, format: str = "both") -> Dict[str, Path]:
        """
        Save episode catalog to files.
        
        Args:
            catalog_df: Episode catalog DataFrame
            format: Output format ("parquet", "json", "both")
        
        Returns:
            Dictionary mapping format to file path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        if format in ["parquet", "both"]:
            parquet_path = self.catalog_path / f"episodes_{timestamp}.parquet"
            catalog_df.to_parquet(parquet_path, index=False)
            saved_files["parquet"] = parquet_path
            logger.info(f"Saved episode catalog to {parquet_path}")
        
        if format in ["json", "both"]:
            json_path = self.catalog_path / f"episodes_{timestamp}.json"
            catalog_df.to_json(json_path, orient="records", indent=2, date_format="iso")
            saved_files["json"] = json_path
            logger.info(f"Saved episode catalog to {json_path}")
        
        return saved_files
    
    def load_catalog(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load episode catalog from file.
        
        Args:
            file_path: Path to catalog file (optional, will find latest if not provided)
        
        Returns:
            Episode catalog DataFrame
        """
        if file_path is None:
            # Find the most recent episode catalog
            parquet_files = list(self.catalog_path.glob("episodes_*.parquet"))
            if not parquet_files:
                logger.warning("No episode catalog files found")
                return pd.DataFrame()
            file_path = max(parquet_files, key=lambda p: p.stat().st_mtime)
        
        try:
            catalog_df = pd.read_parquet(file_path)
            logger.info(f"Loaded episode catalog with {len(catalog_df)} episodes from {file_path}")
            return catalog_df
        except Exception as e:
            logger.error(f"Error loading episode catalog from {file_path}: {e}")
            return pd.DataFrame()
    
    def get_episode_summary(self, episode_id: str) -> Dict[str, Any]:
        """
        Get detailed summary for a specific episode.
        
        Args:
            episode_id: ID of episode to summarize
        
        Returns:
            Dictionary with episode summary information
        """
        if not self.normalized_path.exists():
            return {"error": "No normalized data found"}
        
        conn = duckdb.connect()
        
        try:
            # Query for specific episode
            query = """
            SELECT 
                episode_id,
                MIN("start") as start_time,
                MAX("end") as end_time,
                MAX("end") - MIN("start") as duration_seconds,
                COUNT(*) as utterance_count,
                COUNT(DISTINCT speaker) as speaker_count,
                STRING_AGG(DISTINCT speaker, ', ' ORDER BY speaker) as speaker_list,
                AVG("end" - "start") as avg_utterance_duration,
                MIN("start") as first_utterance_time,
                MAX("end") as last_utterance_time
            FROM read_parquet(?)
            WHERE episode_id = ?
            GROUP BY episode_id
            """
            
            # Find parquet file containing the episode
            parquet_files = list(self.normalized_path.glob("*.parquet"))
            for parquet_file in parquet_files:
                try:
                    df = conn.execute(query, [str(parquet_file), episode_id]).df()
                    if not df.empty:
                        episode_info = df.iloc[0].to_dict()
                        episode_info['duration_minutes'] = episode_info['duration_seconds'] / 60.0
                        episode_info['source_file'] = parquet_file.name
                        return episode_info
                except Exception as e:
                    logger.error(f"Error querying {parquet_file} for episode {episode_id}: {e}")
                    continue
            
            return {"error": f"Episode {episode_id} not found"}
            
        finally:
            conn.close()
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics for all episodes.
        
        Returns:
            Dictionary with episode statistics
        """
        catalog_df = self.load_catalog()
        if catalog_df.empty:
            return {"error": "No episode catalog available"}
        
        stats = {
            "total_episodes": len(catalog_df),
            "total_duration_minutes": catalog_df['duration_minutes'].sum(),
            "total_duration_hours": catalog_df['duration_minutes'].sum() / 60.0,
            "total_utterances": catalog_df['utterance_count'].sum(),
            "avg_episode_duration_minutes": catalog_df['duration_minutes'].mean(),
            "avg_utterances_per_episode": catalog_df['utterance_count'].mean(),
            "avg_speakers_per_episode": catalog_df['speaker_count'].mean(),
            "min_episode_duration_minutes": catalog_df['duration_minutes'].min(),
            "max_episode_duration_minutes": catalog_df['duration_minutes'].max(),
            "unique_speakers": len(set(
                speaker for speaker_list in catalog_df['speaker_list']
                for speaker in speaker_list.split(', ')
            )),
            "catalog_generated": catalog_df['catalog_generated'].iloc[0] if 'catalog_generated' in catalog_df.columns else None,
            "version": catalog_df['version'].iloc[0] if 'version' in catalog_df.columns else None,
        }
        
        return stats


def generate_episode_catalog(
    lakehouse_path: Path,
    version: str = "v1",
    save_format: str = "both"
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """
    Generate and save episode catalog.
    
    Args:
        lakehouse_path: Path to lakehouse directory
        version: Version of data to catalog
        save_format: Format to save catalog ("parquet", "json", "both")
    
    Returns:
        Tuple of (catalog DataFrame, saved file paths)
    """
    catalog = EpisodeCatalog(lakehouse_path, version)
    catalog_df = catalog.generate_catalog()
    
    if catalog_df.empty:
        logger.warning("No episode data found to catalog")
        return catalog_df, {}
    
    saved_files = catalog.save_catalog(catalog_df, format=save_format)
    return catalog_df, saved_files

