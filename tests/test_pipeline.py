"""
Unit Tests for Evidence-Aware RAG

Run with: pytest tests/ -v
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest import Chunk, TextSplitter, DocumentIngestor
from src.metrics import (
    normalize_answer,
    compute_exact_match,
    compute_f1,
    compute_qa_metrics,
    compute_all_metrics
)
from src.verify import (
    GroundednessVerifier,
    VerificationConfig,
    VerificationDecision,
    Claim
)


class TestTextSplitter:
    """Tests for text splitting."""
    
    def test_basic_split(self):
        splitter = TextSplitter(chunk_size=100, chunk_overlap=10)
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1
        assert all(len(c) <= 100 for c in chunks)
    
    def test_empty_text(self):
        splitter = TextSplitter()
        assert splitter.split_text("") == []
    
    def test_small_text(self):
        splitter = TextSplitter(chunk_size=1000)
        text = "Short text."
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_normalize_answer(self):
        assert normalize_answer("The ANSWER") == "answer"
        assert normalize_answer("an example!") == "example"
        assert normalize_answer("  multiple   spaces  ") == "multiple spaces"
    
    def test_exact_match(self):
        assert compute_exact_match("Paris", "paris") == 1.0
        assert compute_exact_match("Paris", "London") == 0.0
        assert compute_exact_match("The answer", "answer") == 1.0
    
    def test_f1_score(self):
        f1, prec, rec = compute_f1("The capital is Paris", "Paris")
        assert f1 > 0
        assert prec > 0
        assert rec == 1.0  # Gold is fully covered
        
        f1, _, _ = compute_f1("completely wrong", "Paris")
        assert f1 == 0.0
    
    def test_qa_metrics(self):
        predictions = ["Paris", "London", "Berlin"]
        ground_truths = ["Paris", "Paris", "Berlin"]
        
        metrics = compute_qa_metrics(predictions, ground_truths)
        assert metrics.exact_match == pytest.approx(2/3)
        assert metrics.f1 > 0


class TestVerification:
    """Tests for groundedness verification."""
    
    @pytest.fixture
    def sample_chunks(self):
        return [
            Chunk(
                chunk_id="1",
                text="Paris is the capital of France. It is known for the Eiffel Tower.",
                doc_id="doc1",
                doc_title="France",
                chunk_index=0,
                start_char=0,
                end_char=66
            ),
            Chunk(
                chunk_id="2",
                text="London is the capital of the United Kingdom.",
                doc_id="doc2",
                doc_title="UK",
                chunk_index=0,
                start_char=0,
                end_char=44
            )
        ]
    
    def test_claim_extraction(self):
        config = VerificationConfig()
        verifier = GroundednessVerifier(config)
        
        answer = "Paris is the capital. It has the Eiffel Tower."
        claims = verifier.extract_claims(answer)
        
        assert len(claims) == 2
        assert "Paris" in claims[0].text
    
    def test_claim_with_citations(self):
        config = VerificationConfig()
        verifier = GroundednessVerifier(config)
        
        answer = "Paris is the capital of France [1]. It is very beautiful [2]."
        claims = verifier.extract_claims(answer)
        
        assert len(claims) >= 1
        # Check that citations were detected
        assert any(c.citation_refs for c in claims)
    
    def test_verification_decision(self):
        config = VerificationConfig(
            groundedness_threshold=0.7,
            abstain_threshold=0.4
        )
        verifier = GroundednessVerifier(config)
        
        # High score -> grounded
        decision = verifier.make_decision(0.8, [])
        assert decision == VerificationDecision.GROUNDED
        
        # Low score -> abstain
        decision = verifier.make_decision(0.3, [])
        assert decision == VerificationDecision.ABSTAIN


class TestIngestor:
    """Tests for document ingestion."""
    
    def test_chunk_creation(self):
        chunk = Chunk(
            chunk_id="test_1",
            text="Test text content",
            doc_id="doc_1",
            doc_title="Test Doc",
            chunk_index=0,
            start_char=0,
            end_char=17
        )
        
        assert chunk.chunk_id == "test_1"
        assert len(chunk.text) == 17
    
    def test_chunk_serialization(self):
        chunk = Chunk(
            chunk_id="test_1",
            text="Test text",
            doc_id="doc_1",
            doc_title="Test",
            chunk_index=0,
            start_char=0,
            end_char=9
        )
        
        # Test to_dict
        data = chunk.to_dict()
        assert data["chunk_id"] == "test_1"
        
        # Test from_dict
        restored = Chunk.from_dict(data)
        assert restored.chunk_id == chunk.chunk_id
        assert restored.text == chunk.text


class TestIntegration:
    """Integration tests."""
    
    def test_metrics_pipeline(self):
        """Test full metrics computation."""
        predictions = ["Paris", "I don't know", "Berlin"]
        ground_truths = ["Paris", "Unknown", "Berlin"]
        scores = [0.9, 0.2, 0.85]
        abstained = [False, True, False]
        answerable = [True, False, True]
        
        results = compute_all_metrics(
            predictions, ground_truths, scores, 
            abstained, answerable
        )
        
        assert "exact_match" in results
        assert "groundedness_rate" in results
        assert "abstention_accuracy" in results
        assert results["exact_match"] == 1.0  # Both answered correctly


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
