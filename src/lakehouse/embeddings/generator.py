"""
Embedding generation for transcript artifacts.

Generates vector embeddings for spans and beats using configurable
embedding models with batch processing and caching support.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from lakehouse.embeddings.models import create_embedding_model, EmbeddingModel
from lakehouse.config import load_config
from lakehouse.logger import get_default_logger


logger = get_default_logger()


class EmbeddingGenerator:
    """
    Generates embeddings for transcript artifacts.
    
    Supports spans, beats, sections, and utterances with configurable
    models and batch processing.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize embedding generator.
        
        Args:
            config: Configuration dictionary (overrides config file)
            config_path: Path to config directory (default: ./config)
        
        Example:
            >>> generator = EmbeddingGenerator()
            >>> embeddings = generator.generate(spans, artifact_type="span")
        """
        # Load configuration
        if config is None:
            config = load_config(config_path, "embedding")
        
        self.config = config
        
        # Extract configuration parameters
        model_config = self.config.get("model", {})
        self.provider = model_config.get("provider", "local")
        self.model_name = model_config.get("name", "all-MiniLM-L6-v2")
        self.device = model_config.get("device", "cpu")
        
        # Generation parameters
        gen_config = self.config.get("generation", {})
        self.batch_size = gen_config.get("batch_size", 32)
        self.normalize_embeddings = gen_config.get("normalize_embeddings", True)
        self.max_text_length = gen_config.get("max_text_length", 8192)
        self.show_progress = gen_config.get("show_progress", True)
        
        # Fallback configuration
        fallback_config = self.config.get("fallback", {})
        self.fallback_enabled = fallback_config.get("enabled", True)
        self.fallback_provider = fallback_config.get("provider", "openai")
        self.fallback_model = fallback_config.get("model_name", "text-embedding-3-small")
        
        # Initialize primary model
        self.model = None
        self.fallback_model_instance = None
        self._init_model()
        
        logger.info(
            f"EmbeddingGenerator initialized: provider={self.provider}, "
            f"model={self.model_name}, batch_size={self.batch_size}"
        )
    
    def _init_model(self):
        """Initialize the primary embedding model."""
        try:
            # Get OpenAI config if needed
            openai_config = self.config.get("openai", {})
            api_key = openai_config.get("api_key")
            
            self.model = create_embedding_model(
                provider=self.provider,
                model_name=self.model_name,
                device=self.device if self.provider == "local" else None,
                normalize_embeddings=self.normalize_embeddings if self.provider == "local" else None,
                api_key=api_key if self.provider == "openai" else None,
            )
            
            logger.info(f"Primary model initialized: {self.model.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize primary model: {e}")
            
            if self.fallback_enabled:
                logger.info("Attempting to initialize fallback model")
                self._init_fallback_model()
            else:
                raise
    
    def _init_fallback_model(self):
        """Initialize the fallback embedding model."""
        try:
            openai_config = self.config.get("openai", {})
            api_key = openai_config.get("api_key")
            
            self.fallback_model_instance = create_embedding_model(
                provider=self.fallback_provider,
                model_name=self.fallback_model,
                api_key=api_key if self.fallback_provider == "openai" else None,
            )
            
            logger.info(f"Fallback model initialized: {self.fallback_model_instance.model_name}")
            
            # Use fallback as primary
            self.model = self.fallback_model_instance
            
        except Exception as e:
            logger.error(f"Failed to initialize fallback model: {e}")
            raise
    
    def generate(
        self,
        artifacts: Union[List[Dict[str, Any]], pd.DataFrame],
        artifact_type: str = "span",
        text_field: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of artifacts.
        
        Args:
            artifacts: List of artifact dictionaries or DataFrame
            artifact_type: Type of artifact ("span", "beat", "section", "utterance")
            text_field: Name of the text field to embed (default: "text")
        
        Returns:
            List of dictionaries with artifact_id, artifact_type, embedding, model info
        
        Example:
            >>> results = generator.generate(spans, artifact_type="span")
            >>> results[0].keys()
            dict_keys(['artifact_id', 'artifact_type', 'embedding', 'model_name', 'model_version'])
        """
        # Convert DataFrame to list of dicts
        if isinstance(artifacts, pd.DataFrame):
            artifacts = artifacts.to_dict("records")
        
        if not artifacts:
            logger.warning("No artifacts to generate embeddings for")
            return []
        
        logger.info(f"Generating embeddings for {len(artifacts)} {artifact_type} artifacts")
        
        # Extract texts
        texts = []
        artifact_ids = []
        
        for artifact in artifacts:
            text = artifact.get(text_field, "")
            
            # Truncate text if needed
            if len(text) > self.max_text_length:
                logger.debug(f"Truncating text from {len(text)} to {self.max_text_length} chars")
                text = text[:self.max_text_length]
            
            texts.append(text)
            
            # Get appropriate ID field
            id_field = f"{artifact_type}_id"
            artifact_id = artifact.get(id_field)
            artifact_ids.append(artifact_id)
        
        # Generate embeddings
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress=self.show_progress,
            )
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            
            if self.fallback_enabled and self.fallback_model_instance and self.model != self.fallback_model_instance:
                logger.info("Attempting with fallback model")
                try:
                    embeddings = self.fallback_model_instance.encode(
                        texts,
                        batch_size=self.batch_size,
                        show_progress=self.show_progress,
                    )
                    self.model = self.fallback_model_instance  # Switch to fallback
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")
                    raise
            else:
                raise
        
        # Create result records
        results = []
        model_info = self.model.get_model_info()
        
        for i, (artifact_id, embedding) in enumerate(zip(artifact_ids, embeddings)):
            result = {
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "embedding": embedding.tolist(),  # Convert to list for JSON serialization
                "model_name": model_info["model_name"],
                "model_version": model_info.get("model_type", "unknown"),
            }
            results.append(result)
        
        logger.info(
            f"Generated {len(results)} embeddings "
            f"(dimension: {len(results[0]['embedding'])})"
        )
        
        return results
    
    def generate_for_episode(
        self,
        artifacts: Union[List[Dict[str, Any]], pd.DataFrame],
        artifact_type: str = "span",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate embeddings grouped by episode.
        
        Args:
            artifacts: List of artifact dictionaries or DataFrame
            artifact_type: Type of artifact
        
        Returns:
            Dictionary mapping episode_id to list of embedding records
        
        Example:
            >>> results_by_episode = generator.generate_for_episode(spans)
            >>> results_by_episode["EP001"]
            [{'artifact_id': '...', 'embedding': [...], ...}, ...]
        """
        # Convert DataFrame to list of dicts
        if isinstance(artifacts, pd.DataFrame):
            artifacts = artifacts.to_dict("records")
        
        # Group by episode
        episodes = {}
        for artifact in artifacts:
            episode_id = artifact.get("episode_id")
            if episode_id not in episodes:
                episodes[episode_id] = []
            episodes[episode_id].append(artifact)
        
        # Generate embeddings for each episode
        results_by_episode = {}
        for episode_id, episode_artifacts in episodes.items():
            logger.info(f"Generating embeddings for episode {episode_id}")
            results = self.generate(episode_artifacts, artifact_type=artifact_type)
            results_by_episode[episode_id] = results
        
        return results_by_episode
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        if self.model:
            return self.model.get_model_info()
        return {}


def generate_embeddings(
    artifacts: Union[List[Dict[str, Any]], pd.DataFrame],
    artifact_type: str = "span",
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to generate embeddings.
    
    Args:
        artifacts: List of artifacts or DataFrame
        artifact_type: Type of artifact
        config: Configuration dictionary
        config_path: Path to config directory
    
    Returns:
        List of embedding records
    
    Example:
        >>> embeddings = generate_embeddings(spans, artifact_type="span")
    """
    generator = EmbeddingGenerator(config=config, config_path=config_path)
    return generator.generate(artifacts, artifact_type=artifact_type)

