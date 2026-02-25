#!/usr/bin/env python3
"""
Tokenize downloaded JSONL datasets into binary token files (.bin) for fast
loading in the C# training pipeline.

Binary format per file:
  - Header:  4 bytes (uint32 LE) = vocab_size used for tokenization
             4 bytes (uint32 LE) = total number of tokens in file
  - Body:    N × 2 bytes (uint16 LE) token IDs  (supports vocab up to 65,535)

Each text sample is tokenized and concatenated with an <eos> separator.
Domain labels are stored in a companion .meta.json file.

Usage:
    python tokenize_datasets.py                         # Tokenize all downloaded datasets
    python tokenize_datasets.py --select 0 1 7          # Tokenize specific datasets
    python tokenize_datasets.py --tokenizer gpt2        # Use a specific tokenizer
    python tokenize_datasets.py --max-tokens 5000000    # Cap tokens per dataset
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

# Use the same catalog
sys.path.insert(0, str(Path(__file__).parent))
from fetch_datasets import DATASETS, get_data_dir


def get_tokenizer(name="gpt2"):
    """Load a HuggingFace tokenizer. Default: GPT-2 (vocab size 50,257 but we
    remap to our model's 32K vocab by taking token_id % vocab_size)."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: 'transformers' library not installed. Run: pip install transformers")
        sys.exit(1)

    print(f"Loading tokenizer: {name}")
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_dataset(dataset_id, tokenizer, vocab_size=32_000, max_tokens=None):
    """Tokenize a single dataset's JSONL into a .bin file."""
    info = DATASETS[dataset_id]
    data_dir = get_data_dir() / info["name"]
    jsonl_file = data_dir / "data.jsonl"
    bin_file = data_dir / "tokens.bin"
    meta_file = data_dir / "tokens_meta.json"

    if not jsonl_file.exists():
        print(f"  SKIP: {jsonl_file} not found (download first)")
        return False

    print(f"  Tokenizing {info['name']}...")

    eos_id = tokenizer.eos_token_id % vocab_size
    all_tokens = []
    sample_count = 0
    sample_boundaries = []  # Track where each sample starts for domain info

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text = record.get("text", "")
            if not text:
                continue

            # Tokenize and remap to our vocab range
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            remapped = [tid % vocab_size for tid in token_ids]

            sample_boundaries.append(len(all_tokens))
            all_tokens.extend(remapped)
            all_tokens.append(eos_id)  # separator between documents

            sample_count += 1
            if max_tokens and len(all_tokens) >= max_tokens:
                all_tokens = all_tokens[:max_tokens]
                break

            if sample_count % 10_000 == 0:
                print(f"    {sample_count:,} samples, {len(all_tokens):,} tokens...")

    if not all_tokens:
        print(f"  SKIP: no tokens produced for {info['name']}")
        return False

    # Write binary file
    num_tokens = len(all_tokens)
    with open(bin_file, "wb") as f:
        # Header
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", num_tokens))
        # Token data as uint16
        for tok in all_tokens:
            f.write(struct.pack("<H", tok))

    # Write metadata
    file_size_mb = bin_file.stat().st_size / (1024 * 1024)
    meta = {
        "dataset_id": dataset_id,
        "name": info["name"],
        "domain": info["domain"],
        "expert_hint": info["expert_hint"],
        "vocab_size": vocab_size,
        "num_tokens": num_tokens,
        "num_samples": sample_count,
        "file": str(bin_file),
        "file_size_mb": round(file_size_mb, 2),
        "tokenizer": tokenizer.name_or_path,
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Done: {num_tokens:,} tokens ({file_size_mb:.1f} MB) → {bin_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Tokenize datasets for TorchSharpLLM")
    parser.add_argument("--select", nargs="+", type=int, metavar="ID", help="Tokenize specific dataset IDs")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="HuggingFace tokenizer name (default: gpt2)")
    parser.add_argument("--vocab-size", type=int, default=32_000, help="Target vocab size for remapping (default: 32000)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens per dataset (default: unlimited)")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")

    args = parser.parse_args()

    if args.data_dir:
        os.environ["TORCHSHARP_DATA_DIR"] = args.data_dir

    tokenizer = get_tokenizer(args.tokenizer)

    ids = args.select if args.select else list(DATASETS.keys())

    print(f"\nTokenizing {len(ids)} dataset(s) with vocab_size={args.vocab_size}...\n")
    results = {}
    for did in ids:
        if did not in DATASETS:
            print(f"  Unknown dataset ID: {did}")
            continue
        ok = tokenize_dataset(did, tokenizer, args.vocab_size, args.max_tokens)
        results[did] = ok
        print()

    print("\n Tokenization Summary:")
    total_tokens = 0
    for did, ok in results.items():
        if ok:
            meta_path = get_data_dir() / DATASETS[did]["name"] / "tokens_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                total_tokens += meta["num_tokens"]
                print(f"  [{did}] {DATASETS[did]['name']:<20} ✓ {meta['num_tokens']:>12,} tokens ({meta['file_size_mb']:.1f} MB)")
            else:
                print(f"  [{did}] {DATASETS[did]['name']:<20} ✓ (meta missing)")
        else:
            print(f"  [{did}] {DATASETS[did]['name']:<20} ✗ skipped/failed")
    print(f"\n  Total tokens across all datasets: {total_tokens:,}")
    print()


if __name__ == "__main__":
    main()
