"""
Reranking Module

Implements cross-encoder based reranking for improved precision.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch

try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False
    print("Warning: CrossEncoder not available from sentence-transformers")

from .ingest import Chunk


@dataclass
class RerankConfig:
    """Configuration for reranking."""
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5
    batch_size: int = 16
    max_length: int = 512
    device: str = "auto"


class Reranker:
    """Cross-encoder based reranker."""
    
    # Available cross-encoder models (trade-off between speed and quality)
    MODELS = {
        "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",      # ~50ms/query
        "balanced": "cross-encoder/ms-marco-MiniLM-L-12-v2", # ~100ms/query  
        "accurate": "cross-encoder/ms-marco-electra-base",   # ~200ms/query
    }
    
    def __init__(self, config: Optional[RerankConfig] = None):
        self.config = config or RerankConfig()
        self.model: Optional[CrossEncoder] = None
        self.device = self._get_device()
    
    def _get_device(self) -> str:
        """Determine device to use."""
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self.model is None:
            if not CROSSENCODER_AVAILABLE:
                raise ImportError("sentence-transformers required for reranking")
            
            model_name = self.MODELS.get(self.config.model, self.config.model)
            self.model = CrossEncoder(
                model_name,
                max_length=self.config.max_length,
                device=self.device
            )
    
    def rerank(
        self,
        query: str,
        chunks: List[Chunk],
        scores: Optional[List[float]] = None,
        top_k: Optional[int] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Rerank chunks using cross-encoder.
        
        Args:
            query: Search query
            chunks: List of candidate chunks
            scores: Original retrieval scores (unused, for logging)
            top_k: Number of top results to return
        
        Returns:
            List of (chunk, rerank_score) tuples
        """
        self._load_model()
        
        if not chunks:
            return []
        
        top_k = top_k or self.config.top_k
        
        # Create query-document pairs
        pairs = [(query, chunk.text) for chunk in chunks]
        
        # Score all pairs
        rerank_scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False
        )
        
        # Combine with chunks and sort
        scored_chunks = list(zip(chunks, rerank_scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        return [(chunk, float(score)) for chunk, score in scored_chunks[:top_k]]
    
    def batch_rerank(
        self,
        queries: List[str],
        chunks_list: List[List[Chunk]],
        top_k: Optional[int] = None
    ) -> List[List[Tuple[Chunk, float]]]:
        """Batch reranking for multiple queries."""
        self._load_model()
        
        top_k = top_k or self.config.top_k
        results = []
        
        # Process all pairs at once for efficiency
        all_pairs = []
        query_chunk_indices = []  # Track which pairs belong to which query
        
        for q_idx, (query, chunks) in enumerate(zip(queries, chunks_list)):
            for chunk in chunks:
                all_pairs.append((query, chunk.text))
                query_chunk_indices.append((q_idx, chunk))
        
        if not all_pairs:
            return [[] for _ in queries]
        
        # Score all pairs at once
        all_scores = self.model.predict(
            all_pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False
        )
        
        # Group by query
        query_results = [[] for _ in queries]
        for (q_idx, chunk), score in zip(query_chunk_indices, all_scores):
            query_results[q_idx].append((chunk, float(score)))
        
        # Sort and truncate each query's results
        for q_idx in range(len(queries)):
            query_results[q_idx].sort(key=lambda x: x[1], reverse=True)
            query_results[q_idx] = query_results[q_idx][:top_k]
        
        return query_results


class NoOpReranker:
    """Passthrough reranker that does nothing (for ablation studies)."""
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
    
    def rerank(
        self,
        query: str,
        chunks: List[Chunk],
        scores: Optional[List[float]] = None,
        top_k: Optional[int] = None
    ) -> List[Tuple[Chunk, float]]:
        """Return top_k chunks without reranking."""
        top_k = top_k or self.top_k
        
        if scores:
            result = list(zip(chunks[:top_k], scores[:top_k]))
        else:
            result = [(chunk, 1.0) for chunk in chunks[:top_k]]
        
        return result


def main():
    """CLI for testing reranking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test cross-encoder reranking")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--texts", "-t", nargs="+", required=True, help="Candidate texts")
    parser.add_argument("--model", default="fast", choices=["fast", "balanced", "accurate"])
    parser.add_argument("--top-k", "-k", type=int, default=3)
    
    args = parser.parse_args()
    
    # Create dummy chunks
    chunks = [
        Chunk(
            chunk_id=f"test_{i}",
            text=text,
            doc_id="test",
            doc_title="Test",
            chunk_index=i,
            start_char=0,
            end_char=len(text)
        )
        for i, text in enumerate(args.texts)
    ]
    
    # Rerank
    config = RerankConfig(model=args.model, top_k=args.top_k)
    reranker = Reranker(config)
    
    results = reranker.rerank(args.query, chunks)
    
    print(f"\nReranked results for: '{args.query}'")
    print("=" * 60)
    
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n[{i}] Score: {score:.4f}")
        print(f"    Text: {chunk.text[:200]}...")


if __name__ == "__main__":
    main()
