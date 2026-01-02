#!/bin/bash
# Download evaluation datasets

set -e

DATA_DIR="${DATA_DIR:-data/datasets}"

echo "========================================"
echo "Downloading Evaluation Datasets"
echo "========================================"

mkdir -p "$DATA_DIR"

# Function to download with progress
download() {
    local url="$1"
    local output="$2"
    echo "Downloading: $url"
    wget -q --show-progress -O "$output" "$url"
}

# Natural Questions (subset)
echo ""
echo "[1/3] Natural Questions..."
NQ_DIR="$DATA_DIR/nq"
mkdir -p "$NQ_DIR"

if [ ! -f "$NQ_DIR/nq-dev.jsonl" ]; then
    echo "Downloading NQ dev set..."
    # Note: Full NQ requires GCS access. Using a sample subset here.
    # For full dataset: gsutil cp gs://natural_questions/v1.0/dev/nq-dev-00.jsonl.gz .
    cat > "$NQ_DIR/nq-dev.jsonl" << 'EOF'
{"question": "what is the capital of france", "answer": "Paris", "id": "1"}
{"question": "who wrote hamlet", "answer": "William Shakespeare", "id": "2"}
{"question": "what year did world war 2 end", "answer": "1945", "id": "3"}
{"question": "who was the first person on the moon", "answer": "Neil Armstrong", "id": "4"}
{"question": "what is the largest planet", "answer": "Jupiter", "id": "5"}
EOF
    echo "Created sample NQ dataset (5 samples for demo)"
else
    echo "NQ already exists, skipping..."
fi

# HotpotQA (subset)
echo ""
echo "[2/3] HotpotQA..."
HOTPOT_DIR="$DATA_DIR/hotpot"
mkdir -p "$HOTPOT_DIR"

if [ ! -f "$HOTPOT_DIR/hotpot_dev.json" ]; then
    echo "Downloading HotpotQA dev set..."
    # Full dataset URL: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
    # Using sample for demo
    cat > "$HOTPOT_DIR/hotpot_dev.json" << 'EOF'
[
  {
    "_id": "1",
    "question": "What nationality is the director of the film where Tom Hanks played a lawyer?",
    "answer": "American",
    "type": "bridge",
    "level": "medium",
    "context": [["Philadelphia", ["Philadelphia is a 1993 American drama film.", "It was directed by Jonathan Demme."]], ["Jonathan Demme", ["Jonathan Demme was an American director.", "He won an Academy Award."]]]
  },
  {
    "_id": "2", 
    "question": "In what year was the founder of SpaceX born?",
    "answer": "1971",
    "type": "bridge",
    "level": "easy",
    "context": [["SpaceX", ["SpaceX was founded by Elon Musk in 2002."]], ["Elon Musk", ["Elon Musk was born on June 28, 1971."]]]
  }
]
EOF
    echo "Created sample HotpotQA dataset (2 samples for demo)"
else
    echo "HotpotQA already exists, skipping..."
fi

# Out-of-domain questions (for testing abstention)
echo ""
echo "[3/3] Out-of-domain questions..."
OOD_DIR="$DATA_DIR/ood"
mkdir -p "$OOD_DIR"

if [ ! -f "$OOD_DIR/test.jsonl" ]; then
    cat > "$OOD_DIR/test.jsonl" << 'EOF'
{"question": "What is the population of Mars?", "answer": "", "is_answerable": false}
{"question": "Who will win the 2050 World Cup?", "answer": "", "is_answerable": false}
{"question": "What is the meaning of life?", "answer": "", "is_answerable": false}
{"question": "How many stars are in the universe?", "answer": "", "is_answerable": false}
{"question": "What will the stock market do tomorrow?", "answer": "", "is_answerable": false}
EOF
    echo "Created out-of-domain dataset (5 unanswerable questions)"
else
    echo "OOD dataset already exists, skipping..."
fi

echo ""
echo "========================================"
echo "Download complete!"
echo "========================================"
echo "Datasets saved to: $DATA_DIR"
echo ""
echo "To use full datasets, download from:"
echo "  - NQ: https://ai.google.com/research/NaturalQuestions"
echo "  - HotpotQA: https://hotpotqa.github.io/"
