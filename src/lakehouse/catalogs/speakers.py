"""
Speaker catalog generation and management.

Provides utilities for generating speaker catalogs with statistics,
episode participation, and duration information using DuckDB queries.
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from lakehouse.logger import get_default_logger


logger = get_default_logger()


class SpeakerCatalog:
    """Manages speaker catalog generation and queries."""
    
    def __init__(self, lakehouse_path: Path, version: str = "v1"):
        """
        Initialize speaker catalog.
        
        Args:
            lakehouse_path: Path to lakehouse directory
            version: Version of data to catalog
        """
        self.lakehouse_path = lakehouse_path
        self.version = version
        self.normalized_path = lakehouse_path / "normalized" / version
        self.catalog_path = lakehouse_path / "catalogs"
        self.catalog_path.mkdir(exist_ok=True)
    
    def generate_catalog(self) -> pd.DataFrame:
        """
        Generate speaker catalog from normalized utterances.
        
        Returns:
            DataFrame with speaker catalog information
        """
        if not self.normalized_path.exists():
            logger.warning(f"No normalized data found at {self.normalized_path}")
            return pd.DataFrame()
        
        # Find all parquet files in normalized directory
        parquet_files = list(self.normalized_path.glob("*.parquet"))
        if not parquet_files:
            logger.warning(f"No parquet files found in {self.normalized_path}")
            return pd.DataFrame()
        
        logger.info(f"Generating speaker catalog from {len(parquet_files)} files")
        
        # Use DuckDB to query all parquet files
        conn = duckdb.connect()
        
        try:
            # Build query to aggregate speaker information
            query = """
            SELECT 
                speaker,
                COUNT(DISTINCT episode_id) as episode_count,
                COUNT(*) as total_utterances,
                SUM("end" - "start") as total_duration_seconds,
                AVG("end" - "start") as avg_utterance_duration_seconds,
                MIN("start") as first_appearance_time,
                MAX("end") as last_appearance_time,
                STRING_AGG(DISTINCT episode_id, ', ' ORDER BY episode_id) as episode_list,
                COUNT(*) * 1.0 / COUNT(DISTINCT episode_id) as avg_utterances_per_episode
            FROM read_parquet(?)
            GROUP BY speaker
            ORDER BY speaker
            """
            
            # Execute query for each file and combine results
            all_speakers = []
            
            for parquet_file in parquet_files:
                try:
                    df = conn.execute(query, [str(parquet_file)]).df()
                    if not df.empty:
                        all_speakers.append(df)
                except Exception as e:
                    logger.error(f"Error processing {parquet_file}: {e}")
                    continue
            
            if not all_speakers:
                logger.warning("No speakers found in any parquet files")
                return pd.DataFrame()
            
            # Combine all speaker data
            combined_df = pd.concat(all_speakers, ignore_index=True)
            
            # Aggregate across all files (in case speakers appear in multiple files)
            speaker_stats = combined_df.groupby('speaker').agg({
                'episode_count': 'sum',
                'total_utterances': 'sum',
                'total_duration_seconds': 'sum',
                'avg_utterance_duration_seconds': 'mean',
                'first_appearance_time': 'min',
                'last_appearance_time': 'max',
                'avg_utterances_per_episode': 'mean'
            }).reset_index()
            
            # Combine episode lists
            episode_lists = combined_df.groupby('speaker')['episode_list'].apply(
                lambda x: ', '.join(set(ep for ep_list in x for ep in ep_list.split(', ')))
            ).reset_index()
            
            # Merge with speaker stats
            catalog_df = speaker_stats.merge(episode_lists, on='speaker')
            
            # Add derived fields
            catalog_df['total_duration_minutes'] = catalog_df['total_duration_seconds'] / 60.0
            catalog_df['total_duration_hours'] = catalog_df['total_duration_minutes'] / 60.0
            catalog_df['avg_utterance_duration_minutes'] = catalog_df['avg_utterance_duration_seconds'] / 60.0
            catalog_df['catalog_generated'] = datetime.now()
            catalog_df['version'] = self.version
            
            # Calculate additional statistics
            catalog_df['utterance_frequency'] = catalog_df['total_utterances'] / catalog_df['total_duration_seconds']
            catalog_df['episode_participation_rate'] = catalog_df['episode_count'] / len(parquet_files)
            
            # Reorder columns for better readability
            column_order = [
                'speaker', 'episode_count', 'total_utterances', 'total_duration_seconds',
                'total_duration_minutes', 'total_duration_hours', 'avg_utterance_duration_seconds',
                'avg_utterance_duration_minutes', 'avg_utterances_per_episode',
                'first_appearance_time', 'last_appearance_time', 'episode_list',
                'utterance_frequency', 'episode_participation_rate', 'catalog_generated', 'version'
            ]
            
            # Only include columns that exist
            available_columns = [col for col in column_order if col in catalog_df.columns]
            catalog_df = catalog_df[available_columns]
            
            logger.info(f"Generated catalog for {len(catalog_df)} speakers")
            return catalog_df
            
        finally:
            conn.close()
    
    def save_catalog(self, catalog_df: pd.DataFrame, format: str = "both") -> Dict[str, Path]:
        """
        Save speaker catalog to files.
        
        Args:
            catalog_df: Speaker catalog DataFrame
            format: Output format ("parquet", "json", "both")
        
        Returns:
            Dictionary mapping format to file path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        if format in ["parquet", "both"]:
            parquet_path = self.catalog_path / f"speakers_{timestamp}.parquet"
            catalog_df.to_parquet(parquet_path, index=False)
            saved_files["parquet"] = parquet_path
            logger.info(f"Saved speaker catalog to {parquet_path}")
        
        if format in ["json", "both"]:
            json_path = self.catalog_path / f"speakers_{timestamp}.json"
            catalog_df.to_json(json_path, orient="records", indent=2, date_format="iso")
            saved_files["json"] = json_path
            logger.info(f"Saved speaker catalog to {json_path}")
        
        return saved_files
    
    def load_catalog(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load speaker catalog from file.
        
        Args:
            file_path: Path to catalog file (optional, will find latest if not provided)
        
        Returns:
            Speaker catalog DataFrame
        """
        if file_path is None:
            # Find the most recent speaker catalog
            parquet_files = list(self.catalog_path.glob("speakers_*.parquet"))
            if not parquet_files:
                logger.warning("No speaker catalog files found")
                return pd.DataFrame()
            file_path = max(parquet_files, key=lambda p: p.stat().st_mtime)
        
        try:
            catalog_df = pd.read_parquet(file_path)
            logger.info(f"Loaded speaker catalog with {len(catalog_df)} speakers from {file_path}")
            return catalog_df
        except Exception as e:
            logger.error(f"Error loading speaker catalog from {file_path}: {e}")
            return pd.DataFrame()
    
    def get_speaker_summary(self, speaker_name: str) -> Dict[str, Any]:
        """
        Get detailed summary for a specific speaker.
        
        Args:
            speaker_name: Name of speaker to summarize
        
        Returns:
            Dictionary with speaker summary information
        """
        if not self.normalized_path.exists():
            return {"error": "No normalized data found"}
        
        conn = duckdb.connect()
        
        try:
            # Query for specific speaker
            query = """
            SELECT 
                speaker,
                COUNT(DISTINCT episode_id) as episode_count,
                COUNT(*) as total_utterances,
                SUM("end" - "start") as total_duration_seconds,
                AVG("end" - "start") as avg_utterance_duration_seconds,
                MIN("start") as first_appearance_time,
                MAX("end") as last_appearance_time,
                STRING_AGG(DISTINCT episode_id, ', ' ORDER BY episode_id) as episode_list
            FROM read_parquet(?)
            WHERE speaker = ?
            GROUP BY speaker
            """
            
            # Find parquet file containing the speaker
            parquet_files = list(self.normalized_path.glob("*.parquet"))
            for parquet_file in parquet_files:
                try:
                    df = conn.execute(query, [str(parquet_file), speaker_name]).df()
                    if not df.empty:
                        speaker_info = df.iloc[0].to_dict()
                        speaker_info['total_duration_minutes'] = speaker_info['total_duration_seconds'] / 60.0
                        speaker_info['total_duration_hours'] = speaker_info['total_duration_minutes'] / 60.0
                        speaker_info['avg_utterance_duration_minutes'] = speaker_info['avg_utterance_duration_seconds'] / 60.0
                        return speaker_info
                except Exception as e:
                    logger.error(f"Error querying {parquet_file} for speaker {speaker_name}: {e}")
                    continue
            
            return {"error": f"Speaker {speaker_name} not found"}
            
        finally:
            conn.close()
    
    def get_speaker_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics for all speakers.
        
        Returns:
            Dictionary with speaker statistics
        """
        catalog_df = self.load_catalog()
        if catalog_df.empty:
            return {"error": "No speaker catalog available"}
        
        stats = {
            "total_speakers": len(catalog_df),
            "total_utterances": catalog_df['total_utterances'].sum(),
            "total_duration_minutes": catalog_df['total_duration_minutes'].sum(),
            "total_duration_hours": catalog_df['total_duration_hours'].sum(),
            "avg_utterances_per_speaker": catalog_df['total_utterances'].mean(),
            "avg_duration_per_speaker_minutes": catalog_df['total_duration_minutes'].mean(),
            "avg_episodes_per_speaker": catalog_df['episode_count'].mean(),
            "most_active_speaker": catalog_df.loc[catalog_df['total_utterances'].idxmax(), 'speaker'] if not catalog_df.empty else None,
            "most_utterances": catalog_df['total_utterances'].max() if not catalog_df.empty else 0,
            "longest_speaking_time_minutes": catalog_df['total_duration_minutes'].max() if not catalog_df.empty else 0,
            "catalog_generated": catalog_df['catalog_generated'].iloc[0] if 'catalog_generated' in catalog_df.columns else None,
            "version": catalog_df['version'].iloc[0] if 'version' in catalog_df.columns else None,
        }
        
        return stats
    
    def get_speaker_rankings(self, metric: str = "total_utterances", limit: int = 10) -> pd.DataFrame:
        """
        Get speaker rankings by specified metric.
        
        Args:
            metric: Metric to rank by ("total_utterances", "total_duration_minutes", "episode_count")
            limit: Maximum number of speakers to return
        
        Returns:
            DataFrame with speaker rankings
        """
        catalog_df = self.load_catalog()
        if catalog_df.empty:
            return pd.DataFrame()
        
        if metric not in catalog_df.columns:
            logger.error(f"Metric {metric} not found in speaker catalog")
            return pd.DataFrame()
        
        # Sort by metric in descending order
        rankings = catalog_df.nlargest(limit, metric)
        rankings['rank'] = range(1, len(rankings) + 1)
        
        return rankings[['rank', 'speaker', metric]]


def generate_speaker_catalog(
    lakehouse_path: Path,
    version: str = "v1",
    save_format: str = "both"
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """
    Generate and save speaker catalog.
    
    Args:
        lakehouse_path: Path to lakehouse directory
        version: Version of data to catalog
        save_format: Format to save catalog ("parquet", "json", "both")
    
    Returns:
        Tuple of (catalog DataFrame, saved file paths)
    """
    catalog = SpeakerCatalog(lakehouse_path, version)
    catalog_df = catalog.generate_catalog()
    
    if catalog_df.empty:
        logger.warning("No speaker data found to catalog")
        return catalog_df, {}
    
    saved_files = catalog.save_catalog(catalog_df, format=save_format)
    return catalog_df, saved_files

