"""
Hybrid Retrieval Module

Combines dense (vector) and sparse (BM25) retrieval using various fusion methods.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import defaultdict

from .ingest import Chunk
from .index_dense import DenseIndex, DenseIndexConfig
from .index_bm25 import BM25Index, BM25Config


@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval."""
    top_k: int = 100
    hybrid_enabled: bool = True
    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    fusion_method: str = "rrf"  # rrf or linear
    rrf_k: int = 60  # Constant for RRF
    normalize_scores: bool = True


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    chunks: List[Chunk]
    scores: List[float]
    metadata: Dict[str, Any]
    
    def __iter__(self):
        return iter(zip(self.chunks, self.scores))
    
    def __len__(self):
        return len(self.chunks)


class HybridRetriever:
    """Hybrid retriever combining dense and sparse search."""
    
    def __init__(
        self,
        dense_index: Optional[DenseIndex] = None,
        bm25_index: Optional[BM25Index] = None,
        config: Optional[RetrievalConfig] = None
    ):
        self.dense_index = dense_index
        self.bm25_index = bm25_index
        self.config = config or RetrievalConfig()
    
    @classmethod
    def from_indices(
        cls,
        dense_path: str,
        bm25_path: str,
        config: Optional[RetrievalConfig] = None
    ) -> "HybridRetriever":
        """Load retriever from saved indices."""
        dense_index = DenseIndex.load(dense_path)
        bm25_index = BM25Index.load(bm25_path)
        return cls(dense_index, bm25_index, config)
    
    @classmethod
    def build_from_chunks(
        cls,
        chunks: List[Chunk],
        dense_config: Optional[DenseIndexConfig] = None,
        bm25_config: Optional[BM25Config] = None,
        retrieval_config: Optional[RetrievalConfig] = None
    ) -> "HybridRetriever":
        """Build retriever from chunks."""
        # Build dense index
        dense_index = DenseIndex(dense_config)
        dense_index.build(chunks)
        
        # Build BM25 index
        bm25_index = BM25Index(bm25_config)
        bm25_index.build(chunks)
        
        return cls(dense_index, bm25_index, retrieval_config)
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        method: Optional[str] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results (overrides config)
            method: Retrieval method - 'hybrid', 'dense', or 'bm25'
        
        Returns:
            RetrievalResult with chunks and scores
        """
        top_k = top_k or self.config.top_k
        method = method or ('hybrid' if self.config.hybrid_enabled else 'dense')
        
        if method == 'dense':
            return self._dense_retrieve(query, top_k)
        elif method == 'bm25':
            return self._bm25_retrieve(query, top_k)
        else:  # hybrid
            return self._hybrid_retrieve(query, top_k)
    
    def _dense_retrieve(self, query: str, top_k: int) -> RetrievalResult:
        """Dense-only retrieval."""
        if self.dense_index is None:
            raise RuntimeError("Dense index not available")
        
        results = self.dense_index.search(query, top_k)
        chunks = [r[0] for r in results]
        scores = [r[1] for r in results]
        
        return RetrievalResult(
            chunks=chunks,
            scores=scores,
            metadata={"method": "dense"}
        )
    
    def _bm25_retrieve(self, query: str, top_k: int) -> RetrievalResult:
        """BM25-only retrieval."""
        if self.bm25_index is None:
            raise RuntimeError("BM25 index not available")
        
        results = self.bm25_index.search(query, top_k)
        chunks = [r[0] for r in results]
        scores = [r[1] for r in results]
        
        return RetrievalResult(
            chunks=chunks,
            scores=scores,
            metadata={"method": "bm25"}
        )
    
    def _hybrid_retrieve(self, query: str, top_k: int) -> RetrievalResult:
        """Hybrid retrieval combining dense and BM25."""
        if self.dense_index is None or self.bm25_index is None:
            raise RuntimeError("Both indices required for hybrid retrieval")
        
        # Get more candidates than needed for fusion
        candidate_k = min(top_k * 3, 500)
        
        # Dense results
        dense_results = self.dense_index.search(query, candidate_k)
        
        # BM25 results
        bm25_results = self.bm25_index.search(query, candidate_k)
        
        # Fuse results
        if self.config.fusion_method == "rrf":
            fused = self._reciprocal_rank_fusion(dense_results, bm25_results)
        else:
            fused = self._linear_fusion(dense_results, bm25_results)
        
        # Sort by fused score and take top_k
        fused_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Get chunk objects
        chunks = []
        scores = []
        for chunk_id, score in fused_sorted:
            if chunk_id in self.dense_index.chunks:
                chunks.append(self.dense_index.chunks[chunk_id])
                scores.append(score)
        
        return RetrievalResult(
            chunks=chunks,
            scores=scores,
            metadata={
                "method": "hybrid",
                "fusion": self.config.fusion_method,
                "dense_weight": self.config.dense_weight,
                "bm25_weight": self.config.bm25_weight
            }
        )
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[Chunk, float]],
        bm25_results: List[Tuple[Chunk, float]]
    ) -> Dict[str, float]:
        """
        Reciprocal Rank Fusion (RRF) for combining ranked lists.
        
        RRF score = sum(1 / (k + rank)) across all lists
        """
        k = self.config.rrf_k
        fused_scores = defaultdict(float)
        
        # Dense contribution
        for rank, (chunk, _) in enumerate(dense_results):
            rrf_score = self.config.dense_weight / (k + rank + 1)
            fused_scores[chunk.chunk_id] += rrf_score
        
        # BM25 contribution
        for rank, (chunk, _) in enumerate(bm25_results):
            rrf_score = self.config.bm25_weight / (k + rank + 1)
            fused_scores[chunk.chunk_id] += rrf_score
        
        return dict(fused_scores)
    
    def _linear_fusion(
        self,
        dense_results: List[Tuple[Chunk, float]],
        bm25_results: List[Tuple[Chunk, float]]
    ) -> Dict[str, float]:
        """
        Linear combination of normalized scores.
        """
        fused_scores = defaultdict(float)
        
        # Normalize dense scores
        if dense_results:
            dense_max = max(r[1] for r in dense_results)
            dense_min = min(r[1] for r in dense_results)
            dense_range = dense_max - dense_min if dense_max != dense_min else 1.0
            
            for chunk, score in dense_results:
                norm_score = (score - dense_min) / dense_range
                fused_scores[chunk.chunk_id] += self.config.dense_weight * norm_score
        
        # Normalize BM25 scores
        if bm25_results:
            bm25_max = max(r[1] for r in bm25_results)
            bm25_min = min(r[1] for r in bm25_results)
            bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0
            
            for chunk, score in bm25_results:
                norm_score = (score - bm25_min) / bm25_range
                fused_scores[chunk.chunk_id] += self.config.bm25_weight * norm_score
        
        return dict(fused_scores)
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Batch retrieval for multiple queries."""
        return [self.retrieve(query, top_k) for query in queries]
    
    def save(self, path: str):
        """Save both indices."""
        from pathlib import Path
        path = Path(path)
        
        if self.dense_index:
            self.dense_index.save(str(path / "dense"))
        if self.bm25_index:
            self.bm25_index.save(str(path / "bm25"))
        
        # Save config
        import json
        with open(path / "retrieval_config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        print(f"Retriever saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "HybridRetriever":
        """Load retriever from saved state."""
        from pathlib import Path
        import json
        
        path = Path(path)
        
        # Load indices
        dense_index = DenseIndex.load(str(path / "dense"))
        bm25_index = BM25Index.load(str(path / "bm25"))
        
        # Load config
        with open(path / "retrieval_config.json", 'r') as f:
            config_dict = json.load(f)
        config = RetrievalConfig(**config_dict)
        
        return cls(dense_index, bm25_index, config)


def main():
    """CLI for testing retrieval."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test hybrid retrieval")
    parser.add_argument("--index", "-i", required=True, help="Index directory")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--top-k", "-k", type=int, default=10)
    parser.add_argument("--method", choices=["hybrid", "dense", "bm25"], default="hybrid")
    
    args = parser.parse_args()
    
    # Load retriever
    retriever = HybridRetriever.load(args.index)
    
    # Search
    results = retriever.retrieve(args.query, top_k=args.top_k, method=args.method)
    
    print(f"\nResults for: '{args.query}' (method: {args.method})")
    print("=" * 60)
    
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n[{i}] Score: {score:.4f}")
        print(f"    Doc: {chunk.doc_title}")
        print(f"    Text: {chunk.text[:200]}...")


if __name__ == "__main__":
    main()
