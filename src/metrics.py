"""
Metrics Module

Implements evaluation metrics for RAG systems:
- QA metrics: Exact Match, F1
- Groundedness metrics: Support rate, claim accuracy
- Abstention metrics: Abstention accuracy, false refusal rate
"""

import re
import string
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class QAMetrics:
    """Quality Assessment metrics for QA."""
    exact_match: float
    f1: float
    precision: float
    recall: float


@dataclass
class GroundednessMetrics:
    """Metrics for groundedness/faithfulness."""
    groundedness_rate: float  # % of answers that are grounded
    avg_groundedness_score: float
    claim_support_rate: float  # % of claims supported
    abstention_rate: float


@dataclass
class AbstentionMetrics:
    """Metrics for abstention behavior."""
    abstention_accuracy: float  # Correct abstentions / total abstentions
    false_refusal_rate: float   # Wrong abstentions / answerable questions
    true_abstention_rate: float # Correct abstentions / unanswerable questions
    

@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    qa_metrics: QAMetrics
    groundedness_metrics: GroundednessMetrics
    abstention_metrics: Optional[AbstentionMetrics]
    sample_results: List[Dict[str, Any]]
    summary: Dict[str, float]


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Compute token-level F1, precision, and recall.
    
    Returns:
        (f1, precision, recall)
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gold_tokens:
        return (0.0, 0.0, 0.0)
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return (0.0, 0.0, 0.0)
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return (f1, precision, recall)


def compute_qa_metrics(
    predictions: List[str],
    ground_truths: List[str]
) -> QAMetrics:
    """Compute QA metrics over a dataset."""
    assert len(predictions) == len(ground_truths)
    
    em_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for pred, gold in zip(predictions, ground_truths):
        em_scores.append(compute_exact_match(pred, gold))
        f1, prec, rec = compute_f1(pred, gold)
        f1_scores.append(f1)
        precision_scores.append(prec)
        recall_scores.append(rec)
    
    return QAMetrics(
        exact_match=np.mean(em_scores),
        f1=np.mean(f1_scores),
        precision=np.mean(precision_scores),
        recall=np.mean(recall_scores)
    )


def compute_groundedness_metrics(
    groundedness_scores: List[float],
    threshold: float = 0.7,
    abstained: Optional[List[bool]] = None
) -> GroundednessMetrics:
    """Compute groundedness metrics."""
    n = len(groundedness_scores)
    
    if n == 0:
        return GroundednessMetrics(
            groundedness_rate=0.0,
            avg_groundedness_score=0.0,
            claim_support_rate=0.0,
            abstention_rate=0.0
        )
    
    grounded = [s >= threshold for s in groundedness_scores]
    groundedness_rate = sum(grounded) / n
    avg_score = np.mean(groundedness_scores)
    
    # Abstention rate
    if abstained:
        abstention_rate = sum(abstained) / n
    else:
        abstention_rate = 0.0
    
    return GroundednessMetrics(
        groundedness_rate=groundedness_rate,
        avg_groundedness_score=avg_score,
        claim_support_rate=groundedness_rate,  # Simplified
        abstention_rate=abstention_rate
    )


def compute_abstention_metrics(
    predictions: List[str],
    ground_truths: List[str],
    abstained: List[bool],
    is_answerable: List[bool],
    abstention_message: str = "I don't have enough"
) -> AbstentionMetrics:
    """
    Compute abstention metrics.
    
    Args:
        predictions: Model predictions
        ground_truths: Gold answers
        abstained: Whether model abstained for each sample
        is_answerable: Whether each question is answerable
    """
    n = len(predictions)
    
    if n == 0:
        return AbstentionMetrics(
            abstention_accuracy=0.0,
            false_refusal_rate=0.0,
            true_abstention_rate=0.0
        )
    
    # Count categories
    correct_abstentions = 0  # Abstained on unanswerable
    false_refusals = 0       # Abstained on answerable
    missed_abstentions = 0   # Didn't abstain on unanswerable
    
    total_unanswerable = sum(1 for a in is_answerable if not a)
    total_answerable = sum(1 for a in is_answerable if a)
    total_abstentions = sum(abstained)
    
    for pred, gold, did_abstain, answerable in zip(predictions, ground_truths, abstained, is_answerable):
        if did_abstain:
            if not answerable:
                correct_abstentions += 1
            else:
                false_refusals += 1
        else:
            if not answerable:
                missed_abstentions += 1
    
    # Compute metrics
    abstention_accuracy = correct_abstentions / max(total_abstentions, 1)
    false_refusal_rate = false_refusals / max(total_answerable, 1)
    true_abstention_rate = correct_abstentions / max(total_unanswerable, 1)
    
    return AbstentionMetrics(
        abstention_accuracy=abstention_accuracy,
        false_refusal_rate=false_refusal_rate,
        true_abstention_rate=true_abstention_rate
    )


def compute_all_metrics(
    predictions: List[str],
    ground_truths: List[str],
    groundedness_scores: List[float],
    abstained: List[bool],
    is_answerable: Optional[List[bool]] = None,
    groundedness_threshold: float = 0.7
) -> Dict[str, float]:
    """
    Compute all metrics in a single call.
    
    Returns:
        Dictionary of metric_name -> value
    """
    results = {}
    
    # QA metrics (only for non-abstained, answerable)
    if is_answerable:
        qa_preds = [p for p, a, ans in zip(predictions, abstained, is_answerable) if not a and ans]
        qa_golds = [g for g, a, ans in zip(ground_truths, abstained, is_answerable) if not a and ans]
    else:
        qa_preds = [p for p, a in zip(predictions, abstained) if not a]
        qa_golds = [g for g, a in zip(ground_truths, abstained) if not a]
    
    if qa_preds:
        qa = compute_qa_metrics(qa_preds, qa_golds)
        results['exact_match'] = qa.exact_match
        results['f1'] = qa.f1
        results['precision'] = qa.precision
        results['recall'] = qa.recall
    
    # Groundedness metrics
    grounded = compute_groundedness_metrics(
        groundedness_scores,
        threshold=groundedness_threshold,
        abstained=abstained
    )
    results['groundedness_rate'] = grounded.groundedness_rate
    results['avg_groundedness_score'] = grounded.avg_groundedness_score
    results['abstention_rate'] = grounded.abstention_rate
    
    # Abstention metrics
    if is_answerable:
        abstention = compute_abstention_metrics(
            predictions, ground_truths, abstained, is_answerable
        )
        results['abstention_accuracy'] = abstention.abstention_accuracy
        results['false_refusal_rate'] = abstention.false_refusal_rate
        results['true_abstention_rate'] = abstention.true_abstention_rate
    
    return results


class MetricTracker:
    """Track metrics during evaluation."""
    
    def __init__(self):
        self.predictions = []
        self.ground_truths = []
        self.groundedness_scores = []
        self.abstained = []
        self.is_answerable = []
        self.latencies = []
    
    def add(
        self,
        prediction: str,
        ground_truth: str,
        groundedness_score: float,
        abstained: bool,
        is_answerable: bool = True,
        latency_ms: float = 0.0
    ):
        """Add a sample result."""
        self.predictions.append(prediction)
        self.ground_truths.append(ground_truth)
        self.groundedness_scores.append(groundedness_score)
        self.abstained.append(abstained)
        self.is_answerable.append(is_answerable)
        self.latencies.append(latency_ms)
    
    def compute(self, groundedness_threshold: float = 0.7) -> Dict[str, float]:
        """Compute all metrics."""
        results = compute_all_metrics(
            self.predictions,
            self.ground_truths,
            self.groundedness_scores,
            self.abstained,
            self.is_answerable,
            groundedness_threshold
        )
        
        # Add latency
        if self.latencies:
            results['avg_latency_ms'] = np.mean(self.latencies)
            results['p95_latency_ms'] = np.percentile(self.latencies, 95)
        
        return results
    
    def reset(self):
        """Reset all tracked values."""
        self.__init__()


def format_results(metrics: Dict[str, float]) -> str:
    """Format metrics for display."""
    lines = [
        "=" * 50,
        "Evaluation Results",
        "=" * 50,
        "",
        "QA Metrics:",
        f"  Exact Match: {metrics.get('exact_match', 0):.1%}",
        f"  F1 Score:    {metrics.get('f1', 0):.1%}",
        "",
        "Groundedness Metrics:",
        f"  Groundedness Rate:  {metrics.get('groundedness_rate', 0):.1%}",
        f"  Avg Grounded Score: {metrics.get('avg_groundedness_score', 0):.3f}",
        "",
        "Abstention Metrics:",
        f"  Abstention Rate:      {metrics.get('abstention_rate', 0):.1%}",
        f"  Abstention Accuracy:  {metrics.get('abstention_accuracy', 0):.1%}",
        f"  False Refusal Rate:   {metrics.get('false_refusal_rate', 0):.1%}",
        f"  True Abstention Rate: {metrics.get('true_abstention_rate', 0):.1%}",
        "",
        "Performance:",
        f"  Avg Latency:  {metrics.get('avg_latency_ms', 0):.1f}ms",
        f"  P95 Latency:  {metrics.get('p95_latency_ms', 0):.1f}ms",
        "=" * 50
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    preds = ["Paris is the capital of France.", "I don't have enough information.", "London"]
    golds = ["Paris", "Unknown", "London"]
    scores = [0.9, 0.2, 0.8]
    abstained = [False, True, False]
    answerable = [True, False, True]
    
    results = compute_all_metrics(preds, golds, scores, abstained, answerable)
    print(format_results(results))
