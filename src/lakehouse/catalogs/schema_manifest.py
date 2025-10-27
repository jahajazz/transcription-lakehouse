"""
Schema manifest generation and management.

Provides utilities for generating schema manifests with artifact types,
schemas, column descriptions, and version information.
"""

import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from lakehouse.logger import get_default_logger
from lakehouse.schemas import get_schema, get_schema_field_names


logger = get_default_logger()


class SchemaManifest:
    """Manages schema manifest generation and queries."""
    
    def __init__(self, lakehouse_path: Path, version: str = "v1"):
        """
        Initialize schema manifest.
        
        Args:
            lakehouse_path: Path to lakehouse directory
            version: Version of data to catalog
        """
        self.lakehouse_path = lakehouse_path
        self.version = version
        self.catalog_path = lakehouse_path / "catalogs"
        self.catalog_path.mkdir(exist_ok=True)
    
    def generate_manifest(self) -> pd.DataFrame:
        """
        Generate schema manifest for all artifact types.
        
        Returns:
            DataFrame with schema manifest information
        """
        logger.info("Generating schema manifest")
        
        # Define artifact types and their descriptions
        artifact_types = {
            "utterance": {
                "description": "Individual speech utterances with timestamps and speaker information",
                "location": f"normalized/{self.version}/",
                "file_pattern": "*.parquet"
            },
            "span": {
                "description": "Contiguous speech segments from single speakers",
                "location": f"spans/{self.version}/",
                "file_pattern": "*.parquet"
            },
            "beat": {
                "description": "Semantic beats grouped by topic similarity",
                "location": f"beats/{self.version}/",
                "file_pattern": "*.parquet"
            },
            "section": {
                "description": "Logical sections of 5-12 minutes duration",
                "location": f"sections/{self.version}/",
                "file_pattern": "*.parquet"
            },
            "embedding": {
                "description": "Vector embeddings for spans and beats",
                "location": f"embeddings/{self.version}/",
                "file_pattern": "*.parquet"
            }
        }
        
        manifest_data = []
        
        for artifact_type, info in artifact_types.items():
            try:
                # Get schema information
                schema = get_schema(artifact_type)
                field_names = get_schema_field_names(artifact_type)
                
                # Get field details
                field_details = []
                for field in schema:
                    field_info = {
                        "name": field.name,
                        "type": str(field.type),
                        "nullable": field.nullable,
                        # Convert metadata to JSON string to avoid empty struct issues in Parquet
                        "metadata": str(dict(field.metadata)) if field.metadata else None
                    }
                    field_details.append(field_info)
                
                # Check if files exist
                artifact_path = self.lakehouse_path / info["location"]
                files_exist = artifact_path.exists() and list(artifact_path.glob(info["file_pattern"]))
                
                manifest_entry = {
                    "artifact_type": artifact_type,
                    "description": info["description"],
                    "version": self.version,
                    "location": info["location"],
                    "file_pattern": info["file_pattern"],
                    "files_exist": bool(files_exist),
                    "file_count": len(list(artifact_path.glob(info["file_pattern"]))) if artifact_path.exists() else 0,
                    "schema_fields": len(field_names),
                    "field_names": ", ".join(field_names),
                    "schema_definition": str(schema),
                    # Convert field_details to JSON string to avoid struct type issues in Parquet
                    "field_details_json": str(field_details),
                    "manifest_generated": datetime.now(),
                }
                
                manifest_data.append(manifest_entry)
                
            except Exception as e:
                logger.error(f"Error processing schema for {artifact_type}: {e}")
                # Add error entry
                manifest_entry = {
                    "artifact_type": artifact_type,
                    "description": info["description"],
                    "version": self.version,
                    "location": info["location"],
                    "file_pattern": info["file_pattern"],
                    "files_exist": False,
                    "file_count": 0,
                    "schema_fields": 0,
                    "field_names": "",
                    "schema_definition": "",
                    # Use same field name as successful entries
                    "field_details_json": "[]",
                    "manifest_generated": datetime.now(),
                    "error": str(e)
                }
                manifest_data.append(manifest_entry)
        
        manifest_df = pd.DataFrame(manifest_data)
        
        # Add derived information
        manifest_df['has_data'] = manifest_df['files_exist'] & (manifest_df['file_count'] > 0)
        manifest_df['data_completeness'] = manifest_df['file_count'] / manifest_df['file_count'].max() if manifest_df['file_count'].max() > 0 else 0
        
        logger.info(f"Generated schema manifest for {len(manifest_df)} artifact types")
        return manifest_df
    
    def save_manifest(self, manifest_df: pd.DataFrame, format: str = "both") -> Dict[str, Path]:
        """
        Save schema manifest to files.
        
        Args:
            manifest_df: Schema manifest DataFrame
            format: Output format ("parquet", "json", "both")
        
        Returns:
            Dictionary mapping format to file path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        if format in ["parquet", "both"]:
            parquet_path = self.catalog_path / f"schema_manifest_{timestamp}.parquet"
            manifest_df.to_parquet(parquet_path, index=False)
            saved_files["parquet"] = parquet_path
            logger.info(f"Saved schema manifest to {parquet_path}")
        
        if format in ["json", "both"]:
            json_path = self.catalog_path / f"schema_manifest_{timestamp}.json"
            manifest_df.to_json(json_path, orient="records", indent=2, date_format="iso")
            saved_files["json"] = json_path
            logger.info(f"Saved schema manifest to {json_path}")
        
        return saved_files
    
    def load_manifest(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load schema manifest from file.
        
        Args:
            file_path: Path to manifest file (optional, will find latest if not provided)
        
        Returns:
            Schema manifest DataFrame
        """
        if file_path is None:
            # Find the most recent schema manifest
            parquet_files = list(self.catalog_path.glob("schema_manifest_*.parquet"))
            if not parquet_files:
                logger.warning("No schema manifest files found")
                return pd.DataFrame()
            file_path = max(parquet_files, key=lambda p: p.stat().st_mtime)
        
        try:
            manifest_df = pd.read_parquet(file_path)
            logger.info(f"Loaded schema manifest with {len(manifest_df)} artifact types from {file_path}")
            return manifest_df
        except Exception as e:
            logger.error(f"Error loading schema manifest from {file_path}: {e}")
            return pd.DataFrame()
    
    def get_artifact_schema(self, artifact_type: str) -> Dict[str, Any]:
        """
        Get detailed schema information for a specific artifact type.
        
        Args:
            artifact_type: Type of artifact to get schema for
        
        Returns:
            Dictionary with schema information
        """
        try:
            schema = get_schema(artifact_type)
            field_names = get_schema_field_names(artifact_type)
            
            schema_info = {
                "artifact_type": artifact_type,
                "version": self.version,
                "field_count": len(field_names),
                "field_names": field_names,
                "schema_definition": str(schema),
                "fields": []
            }
            
            # Get detailed field information
            for field in schema:
                field_info = {
                    "name": field.name,
                    "type": str(field.type),
                    "nullable": field.nullable,
                    "metadata": dict(field.metadata) if field.metadata else {}
                }
                schema_info["fields"].append(field_info)
            
            return schema_info
            
        except Exception as e:
            logger.error(f"Error getting schema for {artifact_type}: {e}")
            return {"error": str(e), "artifact_type": artifact_type}
    
    def get_schema_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics for all schemas.
        
        Returns:
            Dictionary with schema statistics
        """
        manifest_df = self.load_manifest()
        if manifest_df.empty:
            return {"error": "No schema manifest available"}
        
        stats = {
            "total_artifact_types": len(manifest_df),
            "artifact_types_with_data": len(manifest_df[manifest_df['has_data']]),
            "total_files": manifest_df['file_count'].sum(),
            "avg_schema_fields": manifest_df['schema_fields'].mean(),
            "max_schema_fields": manifest_df['schema_fields'].max(),
            "min_schema_fields": manifest_df['schema_fields'].min(),
            "data_completeness": manifest_df['data_completeness'].mean(),
            "artifact_types": manifest_df['artifact_type'].tolist(),
            "manifest_generated": manifest_df['manifest_generated'].iloc[0] if 'manifest_generated' in manifest_df.columns else None,
            "version": manifest_df['version'].iloc[0] if 'version' in manifest_df.columns else None,
        }
        
        return stats
    
    def validate_schema_compliance(self, artifact_type: str) -> Dict[str, Any]:
        """
        Validate schema compliance for a specific artifact type.
        
        Args:
            artifact_type: Type of artifact to validate
        
        Returns:
            Dictionary with validation results
        """
        try:
            # Get expected schema
            expected_schema = get_schema(artifact_type)
            expected_fields = get_schema_field_names(artifact_type)
            
            # Check if files exist
            artifact_info = {
                "utterance": {"location": f"normalized/{self.version}/"},
                "span": {"location": f"spans/{self.version}/"},
                "beat": {"location": f"beats/{self.version}/"},
                "section": {"location": f"sections/{self.version}/"},
                "embedding": {"location": f"embeddings/{self.version}/"},
            }.get(artifact_type, {})
            
            if not artifact_info:
                return {"error": f"Unknown artifact type: {artifact_type}"}
            
            artifact_path = self.lakehouse_path / artifact_info["location"]
            parquet_files = list(artifact_path.glob("*.parquet")) if artifact_path.exists() else []
            
            validation_result = {
                "artifact_type": artifact_type,
                "version": self.version,
                "expected_schema_fields": len(expected_fields),
                "expected_fields": expected_fields,
                "files_found": len(parquet_files),
                "files_exist": len(parquet_files) > 0,
                "validation_passed": True,
                "issues": []
            }
            
            if not parquet_files:
                validation_result["validation_passed"] = False
                validation_result["issues"].append("No parquet files found")
                return validation_result
            
            # Validate each file
            for parquet_file in parquet_files:
                try:
                    df = pd.read_parquet(parquet_file)
                    actual_fields = list(df.columns)
                    
                    # Check for missing fields
                    missing_fields = set(expected_fields) - set(actual_fields)
                    if missing_fields:
                        validation_result["validation_passed"] = False
                        validation_result["issues"].append(f"Missing fields in {parquet_file.name}: {list(missing_fields)}")
                    
                    # Check for extra fields
                    extra_fields = set(actual_fields) - set(expected_fields)
                    if extra_fields:
                        validation_result["issues"].append(f"Extra fields in {parquet_file.name}: {list(extra_fields)}")
                    
                except Exception as e:
                    validation_result["validation_passed"] = False
                    validation_result["issues"].append(f"Error reading {parquet_file.name}: {e}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating schema for {artifact_type}: {e}")
            return {"error": str(e), "artifact_type": artifact_type}


def generate_schema_manifest(
    lakehouse_path: Path,
    version: str = "v1",
    save_format: str = "both"
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """
    Generate and save schema manifest.
    
    Args:
        lakehouse_path: Path to lakehouse directory
        version: Version of data to catalog
        save_format: Format to save manifest ("parquet", "json", "both")
    
    Returns:
        Tuple of (manifest DataFrame, saved file paths)
    """
    manifest = SchemaManifest(lakehouse_path, version)
    manifest_df = manifest.generate_manifest()
    
    if manifest_df.empty:
        logger.warning("No schema information found to catalog")
        return manifest_df, {}
    
    saved_files = manifest.save_manifest(manifest_df, format=save_format)
    return manifest_df, saved_files

