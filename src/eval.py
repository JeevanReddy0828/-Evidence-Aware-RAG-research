"""
Evaluation Module

Runs comprehensive evaluation on QA datasets with ablation support.
"""

import json
import yaml
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass, asdict
from tqdm import tqdm

from .pipeline import RAGPipeline, PipelineConfig, PipelineResult
from .metrics import MetricTracker, format_results, compute_all_metrics


@dataclass
class EvalSample:
    """A single evaluation sample."""
    question: str
    answer: str
    context: Optional[List[str]] = None
    is_answerable: bool = True
    metadata: Dict[str, Any] = None


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    dataset_name: str
    dataset_path: str
    split: str = "dev"
    max_samples: Optional[int] = None
    
    # Ablation settings
    ablations: List[Dict[str, Any]] = None
    
    # Output
    output_dir: str = "outputs/"
    save_predictions: bool = True


class DatasetLoader:
    """Load evaluation datasets in various formats."""
    
    @staticmethod
    def load_natural_questions(path: str, split: str = "dev", max_samples: Optional[int] = None) -> List[EvalSample]:
        """Load Natural Questions dataset."""
        samples = []
        file_path = Path(path) / f"nq-{split}.jsonl"
        
        if not file_path.exists():
            # Try alternative path
            file_path = Path(path) / f"{split}.jsonl"
        
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                
                data = json.loads(line)
                
                # NQ format
                question = data.get('question', data.get('question_text', ''))
                
                # Get short answer
                annotations = data.get('annotations', [{}])
                if annotations and 'short_answers' in annotations[0]:
                    short_answers = annotations[0]['short_answers']
                    if short_answers:
                        answer = short_answers[0].get('text', '')
                    else:
                        answer = ""
                else:
                    answer = data.get('answer', data.get('short_answer', ''))
                
                is_answerable = bool(answer)
                
                samples.append(EvalSample(
                    question=question,
                    answer=answer,
                    is_answerable=is_answerable,
                    metadata={"id": data.get('id', i)}
                ))
        
        return samples
    
    @staticmethod
    def load_hotpotqa(path: str, split: str = "dev", max_samples: Optional[int] = None) -> List[EvalSample]:
        """Load HotpotQA dataset."""
        samples = []
        file_path = Path(path) / f"hotpot_{split}.json"
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for i, item in enumerate(data):
            if max_samples and i >= max_samples:
                break
            
            question = item['question']
            answer = item['answer']
            
            # HotpotQA has supporting facts
            context = []
            for title, sentences in item.get('context', []):
                context.extend(sentences)
            
            samples.append(EvalSample(
                question=question,
                answer=answer,
                context=context,
                is_answerable=True,
                metadata={
                    "id": item.get('_id', i),
                    "type": item.get('type', 'unknown'),
                    "level": item.get('level', 'unknown')
                }
            ))
        
        return samples
    
    @staticmethod
    def load_generic(path: str, max_samples: Optional[int] = None) -> List[EvalSample]:
        """Load generic JSONL dataset."""
        samples = []
        
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                
                data = json.loads(line)
                samples.append(EvalSample(
                    question=data.get('question', ''),
                    answer=data.get('answer', ''),
                    context=data.get('context'),
                    is_answerable=data.get('is_answerable', True),
                    metadata=data.get('metadata', {})
                ))
        
        return samples
    
    @classmethod
    def load(cls, name: str, path: str, split: str = "dev", max_samples: Optional[int] = None) -> List[EvalSample]:
        """Load dataset by name."""
        loaders = {
            "natural_questions": cls.load_natural_questions,
            "nq": cls.load_natural_questions,
            "hotpotqa": cls.load_hotpotqa,
            "hotpot": cls.load_hotpotqa,
        }
        
        if name.lower() in loaders:
            return loaders[name.lower()](path, split, max_samples)
        else:
            return cls.load_generic(path, max_samples)


class Evaluator:
    """Main evaluation class."""
    
    def __init__(self, pipeline: RAGPipeline, config: EvalConfig):
        self.pipeline = pipeline
        self.config = config
        self.tracker = MetricTracker()
    
    def evaluate(
        self,
        samples: List[EvalSample],
        desc: str = "Evaluating"
    ) -> Dict[str, Any]:
        """Run evaluation on samples."""
        self.tracker.reset()
        predictions = []
        
        for sample in tqdm(samples, desc=desc):
            # Run pipeline
            result = self.pipeline.query(sample.question)
            
            # Track metrics
            self.tracker.add(
                prediction=result.answer,
                ground_truth=sample.answer,
                groundedness_score=result.groundedness_score,
                abstained=result.decision.value == "abstain",
                is_answerable=sample.is_answerable,
                latency_ms=result.total_latency_ms
            )
            
            # Store prediction
            if self.config.save_predictions:
                predictions.append({
                    "question": sample.question,
                    "gold_answer": sample.answer,
                    "predicted_answer": result.answer,
                    "is_grounded": result.is_grounded,
                    "groundedness_score": result.groundedness_score,
                    "decision": result.decision.value,
                    "is_answerable": sample.is_answerable,
                    "latency_ms": result.total_latency_ms,
                    "metadata": sample.metadata
                })
        
        # Compute metrics
        metrics = self.tracker.compute()
        
        return {
            "metrics": metrics,
            "predictions": predictions,
            "config": asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else {}
        }
    
    def run_ablations(
        self,
        samples: List[EvalSample]
    ) -> Dict[str, Dict[str, Any]]:
        """Run evaluation with different ablation settings."""
        results = {}
        
        # Baseline: full pipeline
        print("\n[Baseline] Full pipeline (verified)")
        results["baseline"] = self.evaluate(samples, desc="Baseline")
        
        # Ablation 1: No verification
        if self.config.ablations is None or "no_verification" in [a.get("name") for a in self.config.ablations]:
            print("\n[Ablation] No verification")
            original_setting = self.pipeline.config.verification_enabled
            self.pipeline.config.verification_enabled = False
            results["no_verification"] = self.evaluate(samples, desc="No Verification")
            self.pipeline.config.verification_enabled = original_setting
        
        # Ablation 2: No reranking
        if self.config.ablations is None or "no_rerank" in [a.get("name") for a in self.config.ablations]:
            print("\n[Ablation] No reranking")
            original_setting = self.pipeline.config.rerank_enabled
            self.pipeline.config.rerank_enabled = False
            results["no_rerank"] = self.evaluate(samples, desc="No Reranking")
            self.pipeline.config.rerank_enabled = original_setting
        
        return results
    
    def save_results(self, results: Dict[str, Any], suffix: str = ""):
        """Save evaluation results."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{self.config.dataset_name}_{timestamp}{suffix}.json"
        
        with open(output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {output_dir / filename}")
        
        # Also save a summary
        if "metrics" in results:
            summary_file = output_dir / f"summary_{self.config.dataset_name}_{timestamp}{suffix}.txt"
            with open(summary_file, 'w') as f:
                f.write(format_results(results["metrics"]))
            print(f"Summary saved to {summary_file}")


def run_evaluation(
    pipeline_path: str,
    dataset_name: str,
    dataset_path: str,
    output_dir: str = "outputs/",
    max_samples: int = None,
    run_ablations: bool = True
):
    """Main evaluation entry point."""
    
    # Load pipeline
    print(f"Loading pipeline from {pipeline_path}")
    pipeline = RAGPipeline.load(pipeline_path)
    
    # Create eval config
    eval_config = EvalConfig(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        max_samples=max_samples,
        output_dir=output_dir
    )
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    samples = DatasetLoader.load(
        dataset_name,
        dataset_path,
        max_samples=max_samples
    )
    print(f"Loaded {len(samples)} samples")
    
    # Create evaluator
    evaluator = Evaluator(pipeline, eval_config)
    
    # Run evaluation
    if run_ablations:
        results = evaluator.run_ablations(samples)
        
        # Print comparison
        print("\n" + "=" * 60)
        print("ABLATION COMPARISON")
        print("=" * 60)
        
        for name, res in results.items():
            metrics = res["metrics"]
            print(f"\n{name.upper()}:")
            print(f"  EM: {metrics.get('exact_match', 0):.1%}")
            print(f"  F1: {metrics.get('f1', 0):.1%}")
            print(f"  Grounded: {metrics.get('groundedness_rate', 0):.1%}")
            print(f"  Abstention Acc: {metrics.get('abstention_accuracy', 0):.1%}")
        
        evaluator.save_results(results, "_ablations")
    else:
        results = evaluator.evaluate(samples)
        print("\n" + format_results(results["metrics"]))
        evaluator.save_results(results)
    
    return results


def main():
    """CLI for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--pipeline", "-p", required=True, help="Pipeline path")
    parser.add_argument("--dataset", "-d", required=True, help="Dataset name")
    parser.add_argument("--data-path", required=True, help="Dataset path")
    parser.add_argument("--output", "-o", default="outputs/", help="Output directory")
    parser.add_argument("--max-samples", "-n", type=int, help="Max samples")
    parser.add_argument("--no-ablations", action="store_true", help="Skip ablations")
    
    args = parser.parse_args()
    
    run_evaluation(
        pipeline_path=args.pipeline,
        dataset_name=args.dataset,
        dataset_path=args.data_path,
        output_dir=args.output,
        max_samples=args.max_samples,
        run_ablations=not args.no_ablations
    )


if __name__ == "__main__":
    main()
