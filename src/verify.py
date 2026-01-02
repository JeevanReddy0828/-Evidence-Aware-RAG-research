"""
Groundedness Verification Module (CORE CONTRIBUTION)

This module implements the key contribution of the paper:
A lightweight evidence verification layer that scores groundedness
between generated answers and retrieved passages, enabling abstention
when evidence is insufficient.

Key Features:
1. NLI-based claim verification
2. Claim extraction from generated answers
3. Groundedness scoring with configurable thresholds
4. Abstention logic for insufficient evidence
"""

import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import torch
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .ingest import Chunk


class VerificationDecision(Enum):
    """Verification outcome categories."""
    GROUNDED = "grounded"           # Answer is well-supported
    PARTIALLY_GROUNDED = "partial"  # Some claims supported
    ABSTAIN = "abstain"             # Not enough evidence
    CLARIFY = "clarify"             # Ambiguous, ask for clarification


@dataclass
class Claim:
    """Represents an extracted claim from the answer."""
    text: str
    start_idx: int
    end_idx: int
    citation_refs: List[int] = field(default_factory=list)


@dataclass
class ClaimVerification:
    """Verification result for a single claim."""
    claim: Claim
    evidence_passage: Optional[Chunk]
    entailment_score: float
    label: str  # entailment, neutral, contradiction
    is_supported: bool


@dataclass
class VerificationConfig:
    """Configuration for groundedness verification."""
    nli_model: str = "microsoft/deberta-large-mnli"
    batch_size: int = 8
    max_length: int = 512
    device: str = "auto"
    
    groundedness_threshold: float = 0.7
    abstain_threshold: float = 0.4
    claim_support_threshold: float = 0.5
    
    extraction_method: str = "sentence"
    min_claim_length: int = 5
    aggregation: str = "mean"
    
    abstention_enabled: bool = True
    abstention_message: str = "I don't have enough evidence in the provided documents to answer this question confidently."
    suggest_clarification: bool = True


@dataclass
class VerificationResult:
    """Complete verification result."""
    decision: VerificationDecision
    groundedness_score: float
    claim_verifications: List[ClaimVerification]
    original_answer: str
    final_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_grounded(self) -> bool:
        return self.decision == VerificationDecision.GROUNDED
    
    @property
    def should_abstain(self) -> bool:
        return self.decision == VerificationDecision.ABSTAIN


class GroundednessVerifier:
    """
    Core verification component.
    
    Verifies whether generated answers are grounded in retrieved evidence
    using Natural Language Inference (NLI).
    """
    
    LABEL_MAP = {0: "contradiction", 1: "neutral", 2: "entailment"}
    
    def __init__(self, config: Optional[VerificationConfig] = None):
        self.config = config or VerificationConfig()
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
    
    def _get_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device
    
    def _load_model(self):
        if self.model is None:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers required for verification")
            
            print(f"Loading NLI model: {self.config.nli_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.nli_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.nli_model
            )
            self.model.to(self.device)
            self.model.eval()
    
    def extract_claims(self, answer: str) -> List[Claim]:
        """Extract verifiable claims from the generated answer."""
        if self.config.extraction_method == "sentence":
            return self._extract_sentences(answer)
        elif self.config.extraction_method == "clause":
            return self._extract_clauses(answer)
        return self._extract_sentences(answer)
    
    def _extract_sentences(self, text: str) -> List[Claim]:
        """Extract claims at sentence level."""
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        claims = []
        current_pos = 0
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) >= self.config.min_claim_length:
                citations = [int(m) for m in re.findall(r'\[(\d+)\]', sent)]
                clean_sent = re.sub(r'\[\d+\]', '', sent).strip()
                
                if len(clean_sent) >= self.config.min_claim_length:
                    start_idx = text.find(sent, current_pos)
                    claims.append(Claim(
                        text=clean_sent,
                        start_idx=start_idx,
                        end_idx=start_idx + len(sent),
                        citation_refs=citations
                    ))
            current_pos += len(sent) + 1
        
        return claims
    
    def _extract_clauses(self, text: str) -> List[Claim]:
        """Extract claims at clause level (finer-grained)."""
        clause_pattern = r'[,;:]|\b(?:and|but|or|because|although|while|when|if)\b'
        parts = re.split(clause_pattern, text)
        
        claims = []
        current_pos = 0
        
        for part in parts:
            part = part.strip()
            if len(part) >= self.config.min_claim_length:
                citations = [int(m) for m in re.findall(r'\[(\d+)\]', part)]
                clean_part = re.sub(r'\[\d+\]', '', part).strip()
                
                if len(clean_part) >= self.config.min_claim_length:
                    start_idx = text.find(part, current_pos)
                    claims.append(Claim(
                        text=clean_part,
                        start_idx=start_idx,
                        end_idx=start_idx + len(part),
                        citation_refs=citations
                    ))
            current_pos += len(part) + 1
        
        return claims
    
    def compute_entailment(
        self,
        premise: str,
        hypothesis: str
    ) -> Tuple[float, str]:
        """
        Compute NLI entailment score between premise and hypothesis.
        
        Returns:
            (entailment_probability, label)
        """
        self._load_model()
        
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        probs_np = probs[0].cpu().numpy()
        label_idx = int(np.argmax(probs_np))
        entailment_prob = float(probs_np[2])  # Index 2 is typically entailment
        
        return entailment_prob, self.LABEL_MAP[label_idx]
    
    def compute_batch_entailment(
        self,
        premises: List[str],
        hypotheses: List[str]
    ) -> List[Tuple[float, str]]:
        """Batch entailment computation for efficiency."""
        self._load_model()
        
        results = []
        
        for i in range(0, len(premises), self.config.batch_size):
            batch_premises = premises[i:i + self.config.batch_size]
            batch_hypotheses = hypotheses[i:i + self.config.batch_size]
            
            inputs = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            for j in range(len(batch_premises)):
                probs_np = probs[j].cpu().numpy()
                label_idx = int(np.argmax(probs_np))
                entailment_prob = float(probs_np[2])
                results.append((entailment_prob, self.LABEL_MAP[label_idx]))
        
        return results
    
    def verify_claim(
        self,
        claim: Claim,
        passages: List[Chunk]
    ) -> ClaimVerification:
        """
        Verify a single claim against retrieved passages.
        
        Strategy:
        1. If claim has citation refs, check those passages first
        2. Otherwise, check all passages and take the best score
        """
        if not passages:
            return ClaimVerification(
                claim=claim,
                evidence_passage=None,
                entailment_score=0.0,
                label="neutral",
                is_supported=False
            )
        
        # Determine which passages to check
        if claim.citation_refs:
            # Check cited passages first
            candidate_indices = [ref - 1 for ref in claim.citation_refs 
                               if 0 < ref <= len(passages)]
            if not candidate_indices:
                candidate_indices = list(range(len(passages)))
        else:
            candidate_indices = list(range(len(passages)))
        
        # Compute entailment for each candidate
        best_score = 0.0
        best_label = "neutral"
        best_passage = None
        
        for idx in candidate_indices:
            if idx >= len(passages):
                continue
            passage = passages[idx]
            score, label = self.compute_entailment(passage.text, claim.text)
            
            if score > best_score:
                best_score = score
                best_label = label
                best_passage = passage
        
        return ClaimVerification(
            claim=claim,
            evidence_passage=best_passage,
            entailment_score=best_score,
            label=best_label,
            is_supported=best_score >= self.config.claim_support_threshold
        )
    
    def verify_claims_batch(
        self,
        claims: List[Claim],
        passages: List[Chunk]
    ) -> List[ClaimVerification]:
        """Batch verification of multiple claims."""
        if not passages or not claims:
            return [
                ClaimVerification(
                    claim=claim,
                    evidence_passage=None,
                    entailment_score=0.0,
                    label="neutral",
                    is_supported=False
                )
                for claim in claims
            ]
        
        # Build all premise-hypothesis pairs
        all_premises = []
        all_hypotheses = []
        pair_mapping = []  # (claim_idx, passage_idx)
        
        for c_idx, claim in enumerate(claims):
            for p_idx, passage in enumerate(passages):
                all_premises.append(passage.text)
                all_hypotheses.append(claim.text)
                pair_mapping.append((c_idx, p_idx))
        
        # Batch compute
        all_results = self.compute_batch_entailment(all_premises, all_hypotheses)
        
        # Find best passage for each claim
        claim_best = {i: (0.0, "neutral", None) for i in range(len(claims))}
        
        for (c_idx, p_idx), (score, label) in zip(pair_mapping, all_results):
            if score > claim_best[c_idx][0]:
                claim_best[c_idx] = (score, label, passages[p_idx])
        
        # Build verification results
        verifications = []
        for c_idx, claim in enumerate(claims):
            score, label, passage = claim_best[c_idx]
            verifications.append(ClaimVerification(
                claim=claim,
                evidence_passage=passage,
                entailment_score=score,
                label=label,
                is_supported=score >= self.config.claim_support_threshold
            ))
        
        return verifications
    
    def aggregate_scores(
        self,
        claim_verifications: List[ClaimVerification]
    ) -> float:
        """Aggregate claim-level scores into overall groundedness."""
        if not claim_verifications:
            return 0.0
        
        scores = [cv.entailment_score for cv in claim_verifications]
        
        if self.config.aggregation == "mean":
            return float(np.mean(scores))
        elif self.config.aggregation == "min":
            return float(np.min(scores))
        elif self.config.aggregation == "weighted":
            # Weight by claim length (longer claims more important)
            weights = [len(cv.claim.text) for cv in claim_verifications]
            total_weight = sum(weights)
            if total_weight == 0:
                return float(np.mean(scores))
            return float(sum(s * w for s, w in zip(scores, weights)) / total_weight)
        else:
            return float(np.mean(scores))
    
    def make_decision(
        self,
        groundedness_score: float,
        claim_verifications: List[ClaimVerification]
    ) -> VerificationDecision:
        """Determine verification decision based on scores."""
        if groundedness_score >= self.config.groundedness_threshold:
            return VerificationDecision.GROUNDED
        elif groundedness_score <= self.config.abstain_threshold:
            return VerificationDecision.ABSTAIN
        else:
            # Check if partially grounded
            supported_ratio = sum(1 for cv in claim_verifications if cv.is_supported) / max(len(claim_verifications), 1)
            if supported_ratio >= 0.5:
                return VerificationDecision.PARTIALLY_GROUNDED
            elif self.config.suggest_clarification:
                return VerificationDecision.CLARIFY
            else:
                return VerificationDecision.ABSTAIN
    
    def verify(
        self,
        answer: str,
        passages: List[Chunk],
        use_batch: bool = True
    ) -> VerificationResult:
        """
        Main verification method.
        
        Args:
            answer: Generated answer to verify
            passages: Retrieved context passages
            use_batch: Whether to use batched processing
        
        Returns:
            VerificationResult with decision and scores
        """
        # Extract claims
        claims = self.extract_claims(answer)
        
        if not claims:
            # No verifiable claims - assume grounded if short answer
            return VerificationResult(
                decision=VerificationDecision.GROUNDED if len(answer) < 50 else VerificationDecision.ABSTAIN,
                groundedness_score=1.0 if len(answer) < 50 else 0.0,
                claim_verifications=[],
                original_answer=answer,
                final_answer=answer,
                metadata={"num_claims": 0, "reason": "no_claims_extracted"}
            )
        
        # Verify claims
        if use_batch:
            claim_verifications = self.verify_claims_batch(claims, passages)
        else:
            claim_verifications = [self.verify_claim(c, passages) for c in claims]
        
        # Aggregate scores
        groundedness_score = self.aggregate_scores(claim_verifications)
        
        # Make decision
        decision = self.make_decision(groundedness_score, claim_verifications)
        
        # Determine final answer
        if decision == VerificationDecision.ABSTAIN and self.config.abstention_enabled:
            final_answer = self.config.abstention_message
        elif decision == VerificationDecision.CLARIFY:
            final_answer = f"I'm not fully confident in this answer. {answer}\n\nCould you provide more context or rephrase your question?"
        else:
            final_answer = answer
        
        # Build result
        supported_claims = sum(1 for cv in claim_verifications if cv.is_supported)
        
        return VerificationResult(
            decision=decision,
            groundedness_score=groundedness_score,
            claim_verifications=claim_verifications,
            original_answer=answer,
            final_answer=final_answer,
            metadata={
                "num_claims": len(claims),
                "num_supported": supported_claims,
                "support_ratio": supported_claims / len(claims) if claims else 0,
                "threshold_used": self.config.groundedness_threshold,
                "abstain_threshold": self.config.abstain_threshold
            }
        )


class SimpleVerifier:
    """
    Simplified verifier using keyword overlap (for ablation/baseline).
    """
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
    
    def verify(
        self,
        answer: str,
        passages: List[Chunk],
        **kwargs
    ) -> VerificationResult:
        """Simple overlap-based verification."""
        if not passages:
            return VerificationResult(
                decision=VerificationDecision.ABSTAIN,
                groundedness_score=0.0,
                claim_verifications=[],
                original_answer=answer,
                final_answer="I don't have enough evidence.",
                metadata={"method": "simple_overlap"}
            )
        
        # Compute word overlap
        answer_words = set(answer.lower().split())
        passage_words = set()
        for p in passages:
            passage_words.update(p.text.lower().split())
        
        overlap = len(answer_words & passage_words)
        score = overlap / max(len(answer_words), 1)
        
        decision = (VerificationDecision.GROUNDED if score >= self.threshold 
                   else VerificationDecision.ABSTAIN)
        
        return VerificationResult(
            decision=decision,
            groundedness_score=score,
            claim_verifications=[],
            original_answer=answer,
            final_answer=answer if decision == VerificationDecision.GROUNDED else "I don't have enough evidence.",
            metadata={"method": "simple_overlap", "overlap_count": overlap}
        )


def main():
    """CLI for testing verification."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test groundedness verification")
    parser.add_argument("--answer", "-a", required=True, help="Generated answer")
    parser.add_argument("--passages", "-p", nargs="+", required=True, help="Evidence passages")
    parser.add_argument("--threshold", "-t", type=float, default=0.7)
    
    args = parser.parse_args()
    
    # Create dummy chunks
    chunks = [
        Chunk(
            chunk_id=f"p_{i}",
            text=text,
            doc_id="test",
            doc_title="Test",
            chunk_index=i,
            start_char=0,
            end_char=len(text)
        )
        for i, text in enumerate(args.passages)
    ]
    
    config = VerificationConfig(groundedness_threshold=args.threshold)
    verifier = GroundednessVerifier(config)
    
    result = verifier.verify(args.answer, chunks)
    
    print(f"\n{'='*60}")
    print(f"Answer: {args.answer}")
    print(f"\nDecision: {result.decision.value}")
    print(f"Groundedness Score: {result.groundedness_score:.3f}")
    print(f"\nClaims verified: {len(result.claim_verifications)}")
    
    for i, cv in enumerate(result.claim_verifications, 1):
        status = "✓" if cv.is_supported else "✗"
        print(f"  [{status}] {cv.claim.text[:50]}... (score: {cv.entailment_score:.3f})")
    
    print(f"\nFinal Answer: {result.final_answer}")


if __name__ == "__main__":
    main()
