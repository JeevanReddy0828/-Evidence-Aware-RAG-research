#!/bin/bash
# Run evaluation on QA datasets

set -e

# Configuration
PIPELINE_PATH="${PIPELINE_PATH:-data/pipeline}"
DATASET="${DATASET:-natural_questions}"
DATA_PATH="${DATA_PATH:-data/datasets/nq}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
RUN_ABLATIONS="${RUN_ABLATIONS:-true}"

echo "========================================"
echo "RAG Evaluation"
echo "========================================"
echo "Pipeline: $PIPELINE_PATH"
echo "Dataset: $DATASET"
echo "Max samples: $MAX_SAMPLES"
echo "Run ablations: $RUN_ABLATIONS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
if [ "$RUN_ABLATIONS" = "true" ]; then
    echo "Running evaluation with ablations..."
    python -m src.eval \
        --pipeline "$PIPELINE_PATH" \
        --dataset "$DATASET" \
        --data-path "$DATA_PATH" \
        --output "$OUTPUT_DIR" \
        --max-samples "$MAX_SAMPLES"
else
    echo "Running evaluation without ablations..."
    python -m src.eval \
        --pipeline "$PIPELINE_PATH" \
        --dataset "$DATASET" \
        --data-path "$DATA_PATH" \
        --output "$OUTPUT_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --no-ablations
fi

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
