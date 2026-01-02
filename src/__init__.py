"""
Evidence-Aware RAG: Reducing Hallucinations via Lightweight Groundedness Verification

This package implements a RAG system with:
- Hybrid retrieval (BM25 + Dense)
- Cross-encoder reranking
- Citation-aware generation
- Groundedness verification (core contribution)
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .pipeline import RAGPipeline
from .verify import GroundednessVerifier
from .retrieve import HybridRetriever
from .generate import Generator
from .rerank import Reranker

__all__ = [
    "RAGPipeline",
    "GroundednessVerifier", 
    "HybridRetriever",
    "Generator",
    "Reranker"
]
