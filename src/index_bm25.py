"""
BM25 Sparse Index Module

Implements sparse retrieval using BM25 algorithm.
"""

import json
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import numpy as np

try:
    from rank_bm25 import BM25Okapi, BM25Plus
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank-bm25 not available. Install with: pip install rank-bm25")

from .ingest import Chunk


@dataclass
class BM25Config:
    """Configuration for BM25 index."""
    algorithm: str = "okapi"  # okapi or plus
    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25
    lowercase: bool = True
    remove_stopwords: bool = False
    stemming: bool = False


class BM25Index:
    """BM25-based sparse index."""
    
    # Simple English stopwords
    STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
        'that', 'these', 'those', 'it', 'its', 'as', 'if', 'when', 'where',
        'what', 'which', 'who', 'whom', 'how', 'why', 'all', 'each', 'every',
        'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not',
        'only', 'same', 'so', 'than', 'too', 'very', 'just', 'also'
    }
    
    def __init__(self, config: Optional[BM25Config] = None):
        self.config = config or BM25Config()
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: Dict[str, Chunk] = {}
        self.chunk_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Lowercase
        if self.config.lowercase:
            text = text.lower()
        
        # Simple word tokenization
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove stopwords
        if self.config.remove_stopwords:
            tokens = [t for t in tokens if t not in self.STOPWORDS]
        
        # Simple stemming (just remove common suffixes)
        if self.config.stemming:
            tokens = [self._simple_stem(t) for t in tokens]
        
        return tokens
    
    @staticmethod
    def _simple_stem(word: str) -> str:
        """Very simple stemming by removing common suffixes."""
        suffixes = ['ing', 'ed', 'es', 's', 'ly', 'ment', 'ness', 'tion', 'sion']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word
    
    def build(self, chunks: List[Chunk]):
        """Build BM25 index from chunks."""
        if not BM25_AVAILABLE:
            raise ImportError("rank-bm25 required for BM25 indexing")
        
        if not chunks:
            raise ValueError("No chunks provided")
        
        # Store chunks
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        
        # Tokenize corpus
        print(f"Tokenizing {len(chunks)} chunks...")
        self.tokenized_corpus = [self.tokenize(chunk.text) for chunk in chunks]
        
        # Build BM25 index
        print("Building BM25 index...")
        if self.config.algorithm == "okapi":
            self.bm25 = BM25Okapi(
                self.tokenized_corpus,
                k1=self.config.k1,
                b=self.config.b,
                epsilon=self.config.epsilon
            )
        else:
            self.bm25 = BM25Plus(
                self.tokenized_corpus,
                k1=self.config.k1,
                b=self.config.b
            )
        
        print(f"BM25 index built with {len(chunks)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """Search for relevant chunks."""
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call build() first.")
        
        # Tokenize query
        tokenized_query = self.tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Gather results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                chunk_id = self.chunk_ids[idx]
                chunk = self.chunks[chunk_id]
                results.append((chunk, float(scores[idx])))
        
        return results
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> List[List[Tuple[Chunk, float]]]:
        """Batch search for multiple queries."""
        return [self.search(query, top_k) for query in queries]
    
    def get_scores(self, query: str) -> Dict[str, float]:
        """Get BM25 scores for all documents."""
        if self.bm25 is None:
            raise RuntimeError("Index not built")
        
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        return {
            chunk_id: float(score) 
            for chunk_id, score in zip(self.chunk_ids, scores)
        }
    
    def save(self, path: str):
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save BM25 parameters and corpus
        data = {
            "config": self.config.__dict__,
            "chunk_ids": self.chunk_ids,
            "tokenized_corpus": self.tokenized_corpus,
        }
        with open(path / "bm25_data.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        # Save chunks
        with open(path / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"BM25 index saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """Load index from disk."""
        if not BM25_AVAILABLE:
            raise ImportError("rank-bm25 required")
        
        path = Path(path)
        
        # Load data
        with open(path / "bm25_data.pkl", 'rb') as f:
            data = pickle.load(f)
        
        config = BM25Config(**data["config"])
        instance = cls(config)
        
        instance.chunk_ids = data["chunk_ids"]
        instance.tokenized_corpus = data["tokenized_corpus"]
        
        # Load chunks
        with open(path / "chunks.pkl", 'rb') as f:
            instance.chunks = pickle.load(f)
        
        # Rebuild BM25
        if config.algorithm == "okapi":
            instance.bm25 = BM25Okapi(
                instance.tokenized_corpus,
                k1=config.k1,
                b=config.b,
                epsilon=config.epsilon
            )
        else:
            instance.bm25 = BM25Plus(
                instance.tokenized_corpus,
                k1=config.k1,
                b=config.b
            )
        
        print(f"BM25 index loaded from {path} ({len(instance.chunk_ids)} documents)")
        
        return instance


def main():
    """CLI for building BM25 index."""
    import argparse
    from .ingest import DocumentIngestor
    
    parser = argparse.ArgumentParser(description="Build BM25 sparse index")
    parser.add_argument("--chunks", "-c", required=True, help="Path to chunks JSONL")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--algorithm", default="okapi", choices=["okapi", "plus"])
    
    args = parser.parse_args()
    
    # Load chunks
    chunks = DocumentIngestor.load_chunks(args.chunks)
    print(f"Loaded {len(chunks)} chunks")
    
    # Build index
    config = BM25Config(algorithm=args.algorithm)
    index = BM25Index(config)
    index.build(chunks)
    index.save(args.output)


if __name__ == "__main__":
    main()
