#!/usr/bin/env bash
#
# Setup script: install Python dependencies, fetch datasets, and tokenize.
#
# Usage:
#   ./scripts/setup_data.sh              # Interactive: choose which datasets
#   ./scripts/setup_data.sh --core       # Download core 8 datasets (one per expert)
#   ./scripts/setup_data.sh --all        # Download all 10 datasets
#   ./scripts/setup_data.sh --select 0 1 7  # Download specific datasets
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${TORCHSHARP_DATA_DIR:-$PROJECT_ROOT/data}"

export TORCHSHARP_DATA_DIR="$DATA_DIR"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     TorchSharpLLM — Dataset Setup                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Data directory: $DATA_DIR"
echo ""

# ── Step 1: Install Python dependencies ──────────────────────────────────
echo "── Step 1: Checking Python dependencies ──"
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.8+."
    exit 1
fi

pip install -q -r "$SCRIPT_DIR/requirements.txt" 2>/dev/null || \
pip3 install -q -r "$SCRIPT_DIR/requirements.txt" 2>/dev/null || \
echo "WARNING: Could not install Python deps. Install manually: pip install -r scripts/requirements.txt"

echo ""

# ── Step 2: Fetch datasets ───────────────────────────────────────────────
echo "── Step 2: Fetching datasets ──"

if [ $# -eq 0 ]; then
    # Interactive mode: show catalog and ask
    python3 "$SCRIPT_DIR/fetch_datasets.py" --list
    echo ""
    echo "Which datasets would you like to download?"
    echo "  Options: --core (0-7), --all (0-9), or --select <IDs>"
    echo ""
    read -rp "Enter choice (e.g., '--core' or '--select 0 1 2 7'): " CHOICE
    python3 "$SCRIPT_DIR/fetch_datasets.py" $CHOICE --resume
else
    python3 "$SCRIPT_DIR/fetch_datasets.py" "$@" --resume
fi

echo ""

# ── Step 3: Tokenize ─────────────────────────────────────────────────────
echo "── Step 3: Tokenizing datasets ──"
echo "This converts text → binary token files for fast C# loading."
echo ""

python3 "$SCRIPT_DIR/tokenize_datasets.py" --vocab-size 32000

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     Setup complete! Ready to train.                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "To train, run:"
echo "  dotnet run --project src/TorchSharpLLM -- train --data-dir $DATA_DIR"
echo ""
