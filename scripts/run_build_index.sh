#!/bin/bash
# Build indices for RAG pipeline

set -e

# Configuration
DATA_DIR="${DATA_DIR:-data/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed}"
INDEX_DIR="${INDEX_DIR:-data/indices}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-50}"

echo "========================================"
echo "Building RAG Indices"
echo "========================================"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Index directory: $INDEX_DIR"
echo "Chunk size: $CHUNK_SIZE"
echo ""

# Step 1: Ingest documents
echo "[1/3] Ingesting documents..."
python -m src.ingest \
    --input "$DATA_DIR" \
    --output "$OUTPUT_DIR/chunks.jsonl" \
    --chunk-size "$CHUNK_SIZE" \
    --chunk-overlap "$CHUNK_OVERLAP" \
    --recursive

# Step 2: Build dense index
echo ""
echo "[2/3] Building dense (vector) index..."
python -m src.index_dense \
    --chunks "$OUTPUT_DIR/chunks.jsonl" \
    --output "$INDEX_DIR/dense" \
    --model "sentence-transformers/all-MiniLM-L6-v2" \
    --index-type "Flat"

# Step 3: Build BM25 index
echo ""
echo "[3/3] Building BM25 (sparse) index..."
python -m src.index_bm25 \
    --chunks "$OUTPUT_DIR/chunks.jsonl" \
    --output "$INDEX_DIR/bm25" \
    --algorithm "okapi"

echo ""
echo "========================================"
echo "Index building complete!"
echo "========================================"
echo "Dense index: $INDEX_DIR/dense"
echo "BM25 index: $INDEX_DIR/bm25"
echo ""
echo "To test retrieval, run:"
echo "  python -m src.retrieve --index $INDEX_DIR --query 'your question'"
