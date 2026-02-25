#!/usr/bin/env bash
#
# Training launch script for TorchSharpLLM MoE model.
#
# Usage:
#   ./scripts/train.sh                                  # Full training, all datasets
#   ./scripts/train.sh --experts 0 1                    # Train only experts 0,1
#   ./scripts/train.sh --lora --experts 3 5             # LoRA fine-tune experts 3,5
#   ./scripts/train.sh --max-steps 1000                 # Limit total steps
#   ./scripts/train.sh --domain 0 --experts 0           # Train expert 0 on code only
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${TORCHSHARP_DATA_DIR:-$PROJECT_ROOT/data}"

# Verify data exists
if [ ! -d "$DATA_DIR" ] || [ -z "$(find "$DATA_DIR" -name 'tokens.bin' 2>/dev/null | head -1)" ]; then
    echo "ERROR: No tokenized data found in $DATA_DIR"
    echo "Run ./scripts/setup_data.sh first to download and tokenize datasets."
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     TorchSharpLLM — Training                            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Count available datasets
DATASET_COUNT=$(find "$DATA_DIR" -name 'tokens.bin' | wc -l)
TOTAL_TOKENS=$(find "$DATA_DIR" -name 'tokens_meta.json' -exec python3 -c "
import json, sys
total = 0
for f in sys.argv[1:]:
    with open(f) as fh:
        total += json.load(fh).get('num_tokens', 0)
print(f'{total:,}')
" {} +)

echo "Data directory: $DATA_DIR"
echo "Datasets found: $DATASET_COUNT"
echo "Total tokens:   $TOTAL_TOKENS"
echo ""

# Pass all arguments through to the .NET application
dotnet run --project "$PROJECT_ROOT/src/TorchSharpLLM" -- train \
    --data-dir "$DATA_DIR" \
    "$@"
