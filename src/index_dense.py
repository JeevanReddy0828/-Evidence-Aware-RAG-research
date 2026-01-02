"""
Dense (Vector) Index Module

Implements dense retrieval using sentence embeddings and FAISS.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("Warning: sentence-transformers not available.")

from .ingest import Chunk


@dataclass
class DenseIndexConfig:
    """Configuration for dense index."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize: bool = True
    dimension: int = 384
    index_type: str = "Flat"  # Flat, IVFFlat, HNSW
    nlist: int = 100  # for IVFFlat
    nprobe: int = 10  # for IVFFlat search
    ef_construction: int = 200  # for HNSW
    ef_search: int = 50  # for HNSW


class DenseIndex:
    """FAISS-based dense vector index."""
    
    def __init__(self, config: Optional[DenseIndexConfig] = None):
        self.config = config or DenseIndexConfig()
        self.encoder: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.chunk_ids: List[str] = []
        self.chunks: Dict[str, Chunk] = {}
        self._is_trained = False
    
    def _load_encoder(self):
        """Lazy load the encoder model."""
        if self.encoder is None:
            if not SBERT_AVAILABLE:
                raise ImportError("sentence-transformers required")
            self.encoder = SentenceTransformer(self.config.model_name)
    
    def _create_index(self, dimension: int) -> faiss.Index:
        """Create FAISS index based on config."""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS required for dense indexing")
        
        if self.config.index_type == "Flat":
            # Exact search - best for small datasets
            index = faiss.IndexFlatIP(dimension)  # Inner product (cosine if normalized)
        
        elif self.config.index_type == "IVFFlat":
            # Approximate search with inverted file
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist)
        
        elif self.config.index_type == "HNSW":
            # Hierarchical Navigable Small World
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter
            index.hnsw.efConstruction = self.config.ef_construction
        
        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")
        
        return index
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Encode texts to embeddings."""
        self._load_encoder()
        
        embeddings = self.encoder.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize
        )
        
        return embeddings.astype(np.float32)
    
    def build(self, chunks: List[Chunk], show_progress: bool = True):
        """Build index from chunks."""
        if not chunks:
            raise ValueError("No chunks provided")
        
        # Extract texts and IDs
        texts = [chunk.text for chunk in chunks]
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        
        # Encode all texts
        print(f"Encoding {len(texts)} chunks...")
        embeddings = self.encode(texts, show_progress=show_progress)
        
        # Create and populate index
        dimension = embeddings.shape[1]
        self.index = self._create_index(dimension)
        
        # Train if needed (IVFFlat)
        if self.config.index_type == "IVFFlat":
            print("Training IVF index...")
            self.index.train(embeddings)
        
        # Add vectors
        self.index.add(embeddings)
        self._is_trained = True
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def search(
        self, 
        query: str, 
        top_k: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks."""
        if self.index is None or not self._is_trained:
            raise RuntimeError("Index not built. Call build() first.")
        
        # Set search parameters
        if self.config.index_type == "IVFFlat":
            self.index.nprobe = self.config.nprobe
        elif self.config.index_type == "HNSW":
            self.index.hnsw.efSearch = self.config.ef_search
        
        # Encode query
        query_embedding = self.encode([query], show_progress=False)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Gather results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                chunk_id = self.chunk_ids[idx]
                chunk = self.chunks[chunk_id]
                results.append((chunk, float(score)))
        
        return results
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> List[List[Tuple[Chunk, float]]]:
        """Batch search for multiple queries."""
        if self.index is None:
            raise RuntimeError("Index not built")
        
        # Set search parameters
        if self.config.index_type == "IVFFlat":
            self.index.nprobe = self.config.nprobe
        
        # Encode all queries
        query_embeddings = self.encode(queries, show_progress=False)
        
        # Batch search
        scores, indices = self.index.search(query_embeddings, top_k)
        
        # Gather results
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if idx >= 0:
                    chunk_id = self.chunk_ids[idx]
                    chunk = self.chunks[chunk_id]
                    results.append((chunk, float(score)))
            all_results.append(results)
        
        return all_results
    
    def save(self, path: str):
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save metadata
        metadata = {
            "config": self.config.__dict__,
            "chunk_ids": self.chunk_ids,
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Save chunks
        with open(path / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"Index saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "DenseIndex":
        """Load index from disk."""
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        config = DenseIndexConfig(**metadata["config"])
        instance = cls(config)
        
        # Load FAISS index
        instance.index = faiss.read_index(str(path / "index.faiss"))
        instance.chunk_ids = metadata["chunk_ids"]
        
        # Load chunks
        with open(path / "chunks.pkl", 'rb') as f:
            instance.chunks = pickle.load(f)
        
        instance._is_trained = True
        print(f"Index loaded from {path} ({instance.index.ntotal} vectors)")
        
        return instance


def main():
    """CLI for building dense index."""
    import argparse
    from .ingest import DocumentIngestor
    
    parser = argparse.ArgumentParser(description="Build dense vector index")
    parser.add_argument("--chunks", "-c", required=True, help="Path to chunks JSONL")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--index-type", default="Flat", choices=["Flat", "IVFFlat", "HNSW"])
    
    args = parser.parse_args()
    
    # Load chunks
    chunks = DocumentIngestor.load_chunks(args.chunks)
    print(f"Loaded {len(chunks)} chunks")
    
    # Build index
    config = DenseIndexConfig(
        model_name=args.model,
        index_type=args.index_type
    )
    index = DenseIndex(config)
    index.build(chunks)
    index.save(args.output)


if __name__ == "__main__":
    main()
