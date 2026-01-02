"""
End-to-End RAG Pipeline

Integrates all components: retrieval, reranking, generation, and verification.
"""

import time
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from .ingest import Chunk, DocumentIngestor
from .index_dense import DenseIndex, DenseIndexConfig
from .index_bm25 import BM25Index, BM25Config
from .retrieve import HybridRetriever, RetrievalConfig, RetrievalResult
from .rerank import Reranker, RerankConfig, NoOpReranker
from .generate import Generator, GenerationConfig, GenerationResult
from .verify import GroundednessVerifier, VerificationConfig, VerificationResult, VerificationDecision


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    # Index paths
    dense_index_path: Optional[str] = None
    bm25_index_path: Optional[str] = None
    
    # Component configs
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    
    # Pipeline settings
    rerank_enabled: bool = True
    verification_enabled: bool = True


@dataclass
class PipelineResult:
    """Complete result from RAG pipeline."""
    query: str
    answer: str
    is_grounded: bool
    groundedness_score: float
    decision: VerificationDecision
    
    # Components results
    retrieval_result: RetrievalResult
    generation_result: GenerationResult
    verification_result: Optional[VerificationResult]
    
    # Performance
    latency_ms: Dict[str, float] = field(default_factory=dict)
    
    @property
    def total_latency_ms(self) -> float:
        return sum(self.latency_ms.values())
    
    @property
    def citations(self) -> List[int]:
        return self.generation_result.citations
    
    @property
    def passages_used(self) -> List[Chunk]:
        return self.retrieval_result.chunks[:len(self.citations)] if self.citations else []


class RAGPipeline:
    """
    Complete Evidence-Aware RAG Pipeline.
    
    Implements: Retrieve → Rerank → Generate → Verify
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize components (lazy loading)
        self._retriever: Optional[HybridRetriever] = None
        self._reranker: Optional[Reranker] = None
        self._generator: Optional[Generator] = None
        self._verifier: Optional[GroundednessVerifier] = None
    
    @classmethod
    def from_config_file(cls, config_path: str) -> "RAGPipeline":
        """Load pipeline from YAML config file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle inheritance
        if '_base_' in config_dict:
            base_path = Path(config_path).parent / config_dict.pop('_base_')
            with open(base_path, 'r') as f:
                base_config = yaml.safe_load(f)
            # Merge configs (child overrides parent)
            base_config.update(config_dict)
            config_dict = base_config
        
        # Build component configs
        config = PipelineConfig(
            dense_index_path=config_dict.get('index', {}).get('dense_path'),
            bm25_index_path=config_dict.get('index', {}).get('bm25_path'),
            retrieval=RetrievalConfig(**config_dict.get('retrieval', {})),
            rerank=RerankConfig(**config_dict.get('rerank', {})),
            generation=GenerationConfig(**config_dict.get('generation', {})),
            verification=VerificationConfig(**config_dict.get('verification', {})),
            rerank_enabled=config_dict.get('rerank', {}).get('enabled', True),
            verification_enabled=config_dict.get('verification', {}).get('enabled', True)
        )
        
        return cls(config)
    
    @classmethod
    def from_chunks(
        cls,
        chunks: List[Chunk],
        config: Optional[PipelineConfig] = None
    ) -> "RAGPipeline":
        """Build pipeline from document chunks."""
        instance = cls(config)
        
        # Build retriever
        instance._retriever = HybridRetriever.build_from_chunks(
            chunks,
            retrieval_config=instance.config.retrieval
        )
        
        return instance
    
    @property
    def retriever(self) -> HybridRetriever:
        """Get or load retriever."""
        if self._retriever is None:
            if self.config.dense_index_path and self.config.bm25_index_path:
                self._retriever = HybridRetriever.from_indices(
                    self.config.dense_index_path,
                    self.config.bm25_index_path,
                    self.config.retrieval
                )
            else:
                raise RuntimeError("No retriever loaded. Call from_chunks() or provide index paths.")
        return self._retriever
    
    @property
    def reranker(self) -> Reranker:
        """Get or load reranker."""
        if self._reranker is None:
            if self.config.rerank_enabled:
                self._reranker = Reranker(self.config.rerank)
            else:
                self._reranker = NoOpReranker(self.config.rerank.top_k)
        return self._reranker
    
    @property
    def generator(self) -> Generator:
        """Get or load generator."""
        if self._generator is None:
            self._generator = Generator(self.config.generation)
        return self._generator
    
    @property
    def verifier(self) -> GroundednessVerifier:
        """Get or load verifier."""
        if self._verifier is None:
            self._verifier = GroundednessVerifier(self.config.verification)
        return self._verifier
    
    def query(
        self,
        question: str,
        retrieval_method: Optional[str] = None,
        skip_verification: bool = False
    ) -> PipelineResult:
        """
        Process a query through the full pipeline.
        
        Args:
            question: User question
            retrieval_method: Override retrieval method (hybrid/dense/bm25)
            skip_verification: Skip verification step (for ablation)
        
        Returns:
            PipelineResult with answer and all metadata
        """
        latency = {}
        
        # Step 1: Retrieve
        t0 = time.time()
        retrieval_result = self.retriever.retrieve(
            question,
            top_k=self.config.retrieval.top_k,
            method=retrieval_method
        )
        latency['retrieval'] = (time.time() - t0) * 1000
        
        # Step 2: Rerank
        t0 = time.time()
        if self.config.rerank_enabled:
            reranked = self.reranker.rerank(
                question,
                retrieval_result.chunks,
                retrieval_result.scores,
                top_k=self.config.rerank.top_k
            )
            reranked_chunks = [chunk for chunk, _ in reranked]
            reranked_scores = [score for _, score in reranked]
        else:
            reranked_chunks = retrieval_result.chunks[:self.config.rerank.top_k]
            reranked_scores = retrieval_result.scores[:self.config.rerank.top_k]
        latency['rerank'] = (time.time() - t0) * 1000
        
        # Update retrieval result with reranked chunks
        retrieval_result = RetrievalResult(
            chunks=reranked_chunks,
            scores=reranked_scores,
            metadata={**retrieval_result.metadata, "reranked": self.config.rerank_enabled}
        )
        
        # Step 3: Generate
        t0 = time.time()
        generation_result = self.generator.generate(question, reranked_chunks)
        latency['generation'] = (time.time() - t0) * 1000
        
        # Step 4: Verify
        verification_result = None
        if self.config.verification_enabled and not skip_verification:
            t0 = time.time()
            verification_result = self.verifier.verify(
                generation_result.answer,
                reranked_chunks
            )
            latency['verification'] = (time.time() - t0) * 1000
            
            final_answer = verification_result.final_answer
            is_grounded = verification_result.is_grounded
            groundedness_score = verification_result.groundedness_score
            decision = verification_result.decision
        else:
            final_answer = generation_result.answer
            is_grounded = True  # Assume grounded if verification disabled
            groundedness_score = 1.0
            decision = VerificationDecision.GROUNDED
            latency['verification'] = 0.0
        
        return PipelineResult(
            query=question,
            answer=final_answer,
            is_grounded=is_grounded,
            groundedness_score=groundedness_score,
            decision=decision,
            retrieval_result=retrieval_result,
            generation_result=generation_result,
            verification_result=verification_result,
            latency_ms=latency
        )
    
    def batch_query(
        self,
        questions: List[str],
        **kwargs
    ) -> List[PipelineResult]:
        """Process multiple queries."""
        return [self.query(q, **kwargs) for q in questions]
    
    def save(self, path: str):
        """Save pipeline state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save retriever
        if self._retriever:
            self._retriever.save(str(path / "retriever"))
        
        # Save config
        config_dict = {
            'retrieval': self.config.retrieval.__dict__,
            'rerank': {**self.config.rerank.__dict__, 'enabled': self.config.rerank_enabled},
            'generation': self.config.generation.__dict__,
            'verification': {**self.config.verification.__dict__, 'enabled': self.config.verification_enabled}
        }
        with open(path / "config.yaml", 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        print(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "RAGPipeline":
        """Load pipeline from saved state."""
        path = Path(path)
        
        # Load config
        with open(path / "config.yaml", 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = PipelineConfig(
            dense_index_path=str(path / "retriever" / "dense"),
            bm25_index_path=str(path / "retriever" / "bm25"),
            retrieval=RetrievalConfig(**config_dict.get('retrieval', {})),
            rerank=RerankConfig(**{k: v for k, v in config_dict.get('rerank', {}).items() if k != 'enabled'}),
            generation=GenerationConfig(**config_dict.get('generation', {})),
            verification=VerificationConfig(**{k: v for k, v in config_dict.get('verification', {}).items() if k != 'enabled'}),
            rerank_enabled=config_dict.get('rerank', {}).get('enabled', True),
            verification_enabled=config_dict.get('verification', {}).get('enabled', True)
        )
        
        return cls(config)


def main():
    """CLI for testing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAG pipeline")
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument("--pipeline", "-p", help="Saved pipeline path")
    parser.add_argument("--query", "-q", required=True, help="Question to ask")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    
    args = parser.parse_args()
    
    # Load pipeline
    if args.pipeline:
        pipeline = RAGPipeline.load(args.pipeline)
    elif args.config:
        pipeline = RAGPipeline.from_config_file(args.config)
    else:
        print("Error: Provide either --config or --pipeline")
        return
    
    # Run query
    result = pipeline.query(args.query, skip_verification=args.no_verify)
    
    print(f"\n{'='*60}")
    print(f"Query: {result.query}")
    print(f"\nAnswer: {result.answer}")
    print(f"\nGrounded: {result.is_grounded} (score: {result.groundedness_score:.3f})")
    print(f"Decision: {result.decision.value}")
    print(f"\nLatency breakdown:")
    for step, ms in result.latency_ms.items():
        print(f"  {step}: {ms:.1f}ms")
    print(f"  Total: {result.total_latency_ms:.1f}ms")


if __name__ == "__main__":
    main()
