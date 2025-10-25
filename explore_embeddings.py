#!/usr/bin/env python3
"""
Explore embeddings and search capabilities to show semantic understanding
"""

import pandas as pd
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def load_search_index():
    """Load the FAISS search index"""
    print("🔍 Loading search index...")
    
    # Load span index
    span_index = faiss.read_index('lakehouse/ann_index/v1/span_index.faiss')
    
    # Load ID mappings
    with open('lakehouse/ann_index/v1/span_index.ids.json') as f:
        span_ids = json.load(f)
    
    # Load span data
    spans_df = pd.read_parquet('lakehouse/spans/v1/spans.parquet')
    
    return span_index, span_ids, spans_df

def load_embeddings():
    """Load embeddings data"""
    print("📊 Loading embeddings...")
    
    span_embeddings = pd.read_parquet('lakehouse/embeddings/v1/span_embeddings.parquet')
    beat_embeddings = pd.read_parquet('lakehouse/embeddings/v1/beat_embeddings.parquet')
    
    return span_embeddings, beat_embeddings

def semantic_search_example():
    """Demonstrate semantic search capabilities"""
    print("\n🎯 SEMANTIC SEARCH DEMONSTRATION")
    print("=" * 60)
    
    # Load data
    span_index, span_ids, spans_df = load_search_index()
    span_embeddings, beat_embeddings = load_embeddings()
    
    # Load the embedding model to encode new queries
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Example queries to test semantic understanding
    queries = [
        "angels and demons",
        "spiritual warfare", 
        "guardian angels",
        "biblical interpretation",
        "pagan gods and worship"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 40)
        
        # Encode the query
        query_embedding = model.encode([query])
        query_embedding = query_embedding.astype('float32')
        
        # Search for similar spans
        distances, indices = span_index.search(query_embedding, k=3)
        
        print("Top 3 most similar spans:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            span_id = span_ids[int(idx)]
            span_data = spans_df[spans_df['span_id'] == span_id].iloc[0]
            
            print(f"\n  {i+1}. Similarity: {1-dist:.3f} (lower distance = more similar)")
            print(f"     Speaker: {span_data['speaker']}")
            print(f"     Duration: {span_data['duration']:.1f}s")
            print(f"     Text: {span_data['text'][:200]}...")

def show_embedding_clusters():
    """Show how embeddings group similar content"""
    print("\n🎨 EMBEDDING CLUSTERS")
    print("=" * 60)
    
    span_embeddings, beat_embeddings = load_embeddings()
    
    # Get embeddings as numpy array
    span_vectors = np.array([emb for emb in span_embeddings['embedding']])
    
    # Cluster into 10 semantic groups
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(span_vectors)
    
    # Load span data to show content
    spans_df = pd.read_parquet('lakehouse/spans/v1/spans.parquet')
    
    print(f"\nContent clusters ({n_clusters} semantic groups):")
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_span_ids = span_embeddings.iloc[cluster_indices]['artifact_id'].tolist()
        
        # Get corresponding spans
        cluster_spans = spans_df[spans_df['span_id'].isin(cluster_span_ids)]
        
        print(f"\n{'='*60}")
        print(f"📂 CLUSTER {cluster_id + 1}: {len(cluster_spans)} spans")
        print(f"{'='*60}")
        
        if len(cluster_spans) > 0:
            # Show speakers in this cluster
            speakers = cluster_spans['speaker'].value_counts()
            print(f"\n  Speakers:")
            for speaker, count in speakers.head(3).items():
                print(f"    • {speaker}: {count} spans")
            
            # Show duration statistics
            total_duration = cluster_spans['duration'].sum()
            avg_duration = cluster_spans['duration'].mean()
            print(f"\n  Duration:")
            print(f"    • Total: {total_duration/60:.1f} minutes")
            print(f"    • Average span: {avg_duration:.1f} seconds")
            
            # Show 3 sample texts from this cluster
            print(f"\n  Sample content (showing thematic similarity):")
            sample_indices = np.random.choice(len(cluster_spans), min(3, len(cluster_spans)), replace=False)
            for i, idx in enumerate(sample_indices):
                sample = cluster_spans.iloc[idx]
                text_preview = sample['text'][:150].replace('\n', ' ')
                print(f"\n    {i+1}. [{sample['speaker']}]")
                print(f"       {text_preview}...")
    
    return cluster_labels, span_embeddings

def show_beat_themes():
    """Show thematic organization of beats"""
    print("\n📚 BEAT THEMES")
    print("=" * 60)
    
    beats_df = pd.read_parquet('lakehouse/beats/v1/beats.parquet')
    
    print("Beat content themes (first 10 beats):")
    for i in range(min(10, len(beats_df))):
        beat = beats_df.iloc[i]
        print(f"\n  Beat {i+1}:")
        print(f"    Duration: {beat['duration']:.1f}s")
        print(f"    Content: {beat['text'][:100]}...")
        print(f"    Spans: {len(beat['span_ids'])}")

def demonstrate_semantic_understanding():
    """Show what the system 'understands' about the content"""
    print("\n🧠 SEMANTIC UNDERSTANDING")
    print("=" * 60)
    
    span_embeddings, beat_embeddings = load_embeddings()
    
    # Show embedding statistics
    print("What the embeddings capture:")
    print(f"  • Total semantic features: 384 dimensions")
    print(f"  • Each dimension represents a learned concept")
    print(f"  • Values range from ~-1 to +1 (normalized)")
    print(f"  • Similar content has similar vector values")
    
    # Show a few embedding dimensions and what they might represent
    sample_embedding = span_embeddings.iloc[0]['embedding']
    print(f"\nSample embedding analysis:")
    print(f"  • Mean value: {np.mean(sample_embedding):.4f}")
    print(f"  • Standard deviation: {np.std(sample_embedding):.4f}")
    print(f"  • Range: [{np.min(sample_embedding):.4f}, {np.max(sample_embedding):.4f}]")
    
    # Show how embeddings differ between different content
    if len(span_embeddings) > 1:
        emb1 = span_embeddings.iloc[0]['embedding']
        emb2 = span_embeddings.iloc[1]['embedding']
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"  • Similarity between first two spans: {similarity:.4f}")

if __name__ == "__main__":
    print("🚀 EXPLORING TRANSCRIPT EMBEDDINGS")
    print("=" * 60)
    
    try:
        semantic_search_example()
        show_embedding_clusters()
        show_beat_themes()
        demonstrate_semantic_understanding()
        
        print("\n✅ Analysis complete!")
        print("\nThe embeddings capture semantic meaning by:")
        print("  • Understanding context and themes")
        print("  • Recognizing similar concepts across different wording")
        print("  • Enabling similarity search beyond keyword matching")
        print("  • Grouping related content automatically")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have scikit-learn installed: pip install scikit-learn")
