# Evidence-Aware RAG: Reducing Hallucinations via Lightweight Groundedness Verification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Research Contribution

**Problem**: Retrieval-Augmented Generation (RAG) systems still hallucinate when retrieval quality is poor or questions fall outside the knowledge base scope.

**Our Solution**: A lightweight **evidence verification layer** that:
1. Scores groundedness between generated answers and retrieved passages
2. Triggers abstention ("not enough evidence") when evidence is insufficient
3. Optionally requests clarification for ambiguous queries

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐     ┌─────────────┐
│   Ingest    │──▶ │  Hybrid      │───▶│   Rerank    │
│   (Chunk)   │    │  Retrieval   │     │   (Cross-   │
│             │    │  BM25+Dense  │     │   Encoder)  │
└─────────────┘    └──────────────┘     └─────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Output    │◀───│   Verify     │◀───│  Generate   │
│  (Answer/   │    │ Groundedness │    │  (w/ cites) │
│   Abstain)  │    │   (NLI)      │    │             │
└─────────────┘    └──────────────┘    └─────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/yourusername/rag-verifier.git
cd rag-verifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Demo

```bash
# 1. Build indices on sample data
python -m src.ingest --input data/sample_docs/ --output data/processed/

# 2. Run interactive QA
python app.py

# 3. Run full evaluation
python -m src.eval --config configs/eval_nq.yaml
```

## 📁 Project Structure

```
rag-verifier/
├── data/
│   ├── raw/                    # Raw documents
│   ├── processed/              # Chunked documents
│   └── datasets/               # Evaluation datasets (NQ, HotpotQA)
├── configs/
│   ├── default.yaml            # Default configuration
│   ├── eval_nq.yaml            # Natural Questions eval config
│   └── eval_hotpot.yaml        # HotpotQA eval config
├── src/
│   ├── __init__.py
│   ├── ingest.py               # Document ingestion & chunking
│   ├── index_dense.py          # Dense (vector) index
│   ├── index_bm25.py           # BM25 sparse index
│   ├── retrieve.py             # Hybrid retrieval
│   ├── rerank.py               # Cross-encoder reranking
│   ├── generate.py             # Answer generation with citations
│   ├── verify.py               # Groundedness verification (core contribution)
│   ├── pipeline.py             # End-to-end RAG pipeline
│   ├── eval.py                 # Evaluation runner
│   └── metrics.py              # Metrics computation
├── scripts/
│   ├── run_build_index.sh      # Index building script
│   ├── run_eval.sh             # Evaluation script
│   └── download_data.sh        # Dataset download script
├── notebooks/
│   └── analysis.ipynb          # Results analysis
├── tests/
│   └── test_pipeline.py        # Unit tests
├── app.py                      # Interactive demo
├── requirements.txt
└── README.md
```

## 🔬 Experiments

### Datasets

| Dataset | Size | Task | Purpose |
|---------|------|------|---------|
| Natural Questions (NQ) | 3,452 (dev) | Open-domain QA | Main evaluation |
| HotpotQA | 7,405 (dev) | Multi-hop QA | Complex reasoning |
| Out-of-domain (custom) | 500 | Random questions | Refusal behavior |

### Baselines

1. **RAG-Dense**: Vector retrieval only
2. **RAG-Hybrid**: BM25 + Dense fusion
3. **RAG-Hybrid-Rerank**: + Cross-encoder reranking
4. **RAG-Verified (Ours)**: + Groundedness verification

### Metrics

| Metric | Description |
|--------|-------------|
| Exact Match (EM) | Strict answer matching |
| F1 Score | Token-level overlap |
| Groundedness Rate | % claims supported by evidence |
| Abstention Accuracy | Correct refusals on unanswerable |
| False Refusal Rate | Incorrect abstentions |

## 📊 Results (Expected)

| Method | EM | F1 | Grounded | Abstain Acc |
|--------|----|----|----------|-------------|
| RAG-Dense | 31.2 | 42.5 | 68.3 | - |
| RAG-Hybrid | 34.1 | 45.8 | 71.2 | - |
| RAG-Hybrid-Rerank | 36.8 | 48.2 | 74.5 | - |
| **RAG-Verified (Ours)** | **35.9** | **47.1** | **89.2** | **82.4** |

*Note: Slight EM/F1 drop is expected due to abstention on uncertain cases.*

## ⚙️ Configuration

Key parameters in `configs/default.yaml`:

```yaml
retrieval:
  top_k: 100              # Initial retrieval candidates
  bm25_weight: 0.3        # BM25 contribution to hybrid score
  dense_weight: 0.7       # Dense contribution

rerank:
  top_k: 5                # Passages after reranking
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

generation:
  model: "mistralai/Mistral-7B-Instruct-v0.2"
  max_new_tokens: 256
  temperature: 0.1

verification:
  model: "microsoft/deberta-large-mnli"
  threshold: 0.7          # Groundedness threshold
  abstain_below: 0.4      # Abstain if score below this
```

## 🔧 Advanced Usage

### Custom Document Ingestion

```python
from src.ingest import DocumentIngestor

ingestor = DocumentIngestor(
    chunk_size=512,
    chunk_overlap=50,
    metadata_fields=["source", "date"]
)
chunks = ingestor.process_directory("path/to/docs/")
```

### Programmatic Pipeline

```python
from src.pipeline import RAGPipeline

pipeline = RAGPipeline.from_config("configs/default.yaml")
result = pipeline.query("What is the capital of France?")

print(f"Answer: {result.answer}")
print(f"Grounded: {result.is_grounded}")
print(f"Confidence: {result.groundedness_score:.2f}")
print(f"Citations: {result.citations}")
```

## 📝 Citation

```bibtex
@article{yourname2025evidenceaware,
  title={Evidence-Aware Retrieval-Augmented Generation via Lightweight Groundedness Verification},
  author={Your Name},
  journal={arXiv preprint arXiv:2025.XXXXX},
  year={2025}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Sentence-Transformers](https://www.sbert.net/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [HuggingFace Transformers](https://huggingface.co/transformers/) for models
