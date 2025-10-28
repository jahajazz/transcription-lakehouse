#!/usr/bin/env python3
"""
Sample script to display beats and embeddings data
"""

import pandas as pd
import numpy as np
import json

def show_beats_sample():
    print("🎵 BEATS SAMPLE:")
    print("=" * 60)
    
    df = pd.read_parquet('lakehouse/beats/v1/beats.parquet')
    
    for i in range(3):
        beat = df.iloc[i]
        print(f"\nBeat {i+1}:")
        print(f"  ID: {beat['beat_id']}")
        print(f"  Duration: {beat['duration']:.1f}s")
        print(f"  Topic: {beat['topic_label']}")
        print(f"  Text: {beat['text'][:150]}...")
        print(f"  Span IDs: {len(beat['span_ids'])} spans")

def show_embeddings_sample():
    print("\n🔍 EMBEDDINGS SAMPLE:")
    print("=" * 60)
    
    # Load span embeddings
    span_embeddings = pd.read_parquet('lakehouse/embeddings/v1/span_embeddings.parquet')
    beat_embeddings = pd.read_parquet('lakehouse/embeddings/v1/beat_embeddings.parquet')
    
    print(f"Span Embeddings: {len(span_embeddings)} vectors")
    print(f"Beat Embeddings: {len(beat_embeddings)} vectors")
    
    # Show first embedding details
    first_span = span_embeddings.iloc[0]
    print(f"\nFirst Span Embedding:")
    print(f"  Artifact ID: {first_span['artifact_id']}")
    print(f"  Model: {first_span['model_name']}")
    print(f"  Embedding shape: {first_span['embedding'].shape}")
    print(f"  First 10 values: {first_span['embedding'][:10]}")
    
    # Show metadata
    with open('lakehouse/embeddings/v1/metadata.json') as f:
        metadata = json.load(f)
    
    print(f"\nEmbedding Metadata:")
    print(f"  Model: {metadata['model_name']}")
    print(f"  Dimension: {metadata['embedding_dimension']}")
    print(f"  Total embeddings: {metadata['total_embeddings']}")

def show_search_index_info():
    print("\n🔎 SEARCH INDEX INFO:")
    print("=" * 60)
    
    # Load index metadata
    with open('lakehouse/ann_index/v1/span_index.metadata.json') as f:
        span_metadata = json.load(f)
    
    with open('lakehouse/ann_index/v1/beat_index.metadata.json') as f:
        beat_metadata = json.load(f)
    
    print(f"Span Index: {span_metadata['num_vectors']} vectors, {span_metadata['dimension']}D")
    print(f"Beat Index: {beat_metadata['num_vectors']} vectors, {beat_metadata['dimension']}D")
    print(f"Distance metric: {span_metadata['distance_metric']}")

if __name__ == "__main__":
    show_beats_sample()
    show_embeddings_sample()
    show_search_index_info()

