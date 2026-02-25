#!/usr/bin/env python3
"""
Dataset catalog and download manager for TorchSharpLLM MoE training.

Supports 10 domain-specific datasets mapped to 8+ experts:
  0. Code / Programming        - nampdn-ai/tiny-codes
  1. Mathematics / Reasoning    - meta-math/MetaMathQA
  2. Biomedical / Scientific    - ccdv/pubmed-summarization
  3. Creative Writing / Fiction - euclaise/writingprompts
  4. Legal / Regulatory         - pile-of-law (FreeLaw subset)
  5. Finance / Business         - ashraq/financial-news-articles
  6. Conversational / Dialogue  - HuggingFaceH4/ultrachat_200k
  7. General Knowledge          - wikimedia/wikipedia (Simple English)
  8. (optional) Instruction     - databricks/databricks-dolly-15k
  9. (optional) Textbooks       - HuggingFaceTB/cosmopedia-100k

Usage:
    python fetch_datasets.py --list                 # Show all available datasets
    python fetch_datasets.py --select 0 1 2 3 7     # Download specific datasets
    python fetch_datasets.py --all                   # Download all datasets
    python fetch_datasets.py --select 0 --resume     # Resume interrupted download
"""

import argparse
import json
import os
import sys
import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset catalog
# ---------------------------------------------------------------------------

DATASETS = {
    0: {
        "name": "code_python",
        "domain": "Code / Programming",
        "expert_hint": 0,
        "description": "Multi-language code snippets (textbook quality)",
        "source": "huggingface",
        "hf_path": "nampdn-ai/tiny-codes",
        "hf_subset": None,
        "hf_split": "train",
        "text_field": "response",
        "max_samples": 100_000,
        "approx_size_mb": 500,
    },
    1: {
        "name": "math_reasoning",
        "domain": "Mathematics / Reasoning",
        "expert_hint": 1,
        "description": "MetaMathQA — augmented math problems with step-by-step solutions",
        "source": "huggingface",
        "hf_path": "meta-math/MetaMathQA",
        "hf_subset": None,
        "hf_split": "train",
        "text_field": "query+response",  # composite field
        "max_samples": 100_000,
        "approx_size_mb": 200,
    },
    2: {
        "name": "biomedical",
        "domain": "Biomedical / Science",
        "expert_hint": 2,
        "description": "PubMed research paper full text",
        "source": "huggingface",
        "hf_path": "ccdv/pubmed-summarization",
        "hf_subset": "document",
        "hf_split": "train",
        "text_field": "article",
        "max_samples": 50_000,
        "approx_size_mb": 500,
    },
    3: {
        "name": "creative_writing",
        "domain": "Creative Writing / Fiction",
        "expert_hint": 3,
        "description": "Reddit WritingPrompts — story responses",
        "source": "huggingface",
        "hf_path": "euclaise/writingprompts",
        "hf_subset": None,
        "hf_split": "train",
        "text_field": "story",
        "max_samples": 50_000,
        "approx_size_mb": 350,
    },
    4: {
        "name": "legal",
        "domain": "Legal / Regulatory",
        "expert_hint": 4,
        "description": "US court opinions (FreeLaw from Pile of Law)",
        "source": "huggingface",
        "hf_path": "pile-of-law/pile-of-law",
        "hf_subset": "r_legaladvice",
        "hf_split": "train",
        "text_field": "text",
        "max_samples": 50_000,
        "approx_size_mb": 350,
        "streaming": True,
    },
    5: {
        "name": "finance",
        "domain": "Finance / Business",
        "expert_hint": 5,
        "description": "Financial news articles from Reuters",
        "source": "huggingface",
        "hf_path": "ashraq/financial-news-articles",
        "hf_subset": None,
        "hf_split": "train",
        "text_field": "content",
        "max_samples": 100_000,
        "approx_size_mb": 300,
    },
    6: {
        "name": "conversation",
        "domain": "Conversational / Dialogue",
        "expert_hint": 6,
        "description": "UltraChat — multi-turn dialogues across diverse topics",
        "source": "huggingface",
        "hf_path": "HuggingFaceH4/ultrachat_200k",
        "hf_subset": None,
        "hf_split": "train_sft",
        "text_field": "messages",  # list of {role, content} dicts
        "max_samples": 50_000,
        "approx_size_mb": 400,
    },
    7: {
        "name": "wiki_knowledge",
        "domain": "General Knowledge / Encyclopedia",
        "expert_hint": 7,
        "description": "Simple English Wikipedia articles",
        "source": "huggingface",
        "hf_path": "wikimedia/wikipedia",
        "hf_subset": "20231101.simple",
        "hf_split": "train",
        "text_field": "text",
        "max_samples": None,  # use all (~242K articles)
        "approx_size_mb": 300,
    },
    8: {
        "name": "instructions",
        "domain": "Instruction Following",
        "expert_hint": -1,  # no dedicated expert — general
        "description": "Dolly 15K — diverse instruction-response pairs",
        "source": "huggingface",
        "hf_path": "databricks/databricks-dolly-15k",
        "hf_subset": None,
        "hf_split": "train",
        "text_field": "instruction+response",  # composite field
        "max_samples": None,  # use all (~15K)
        "approx_size_mb": 15,
    },
    9: {
        "name": "textbooks",
        "domain": "Textbooks / Educational",
        "expert_hint": -1,  # general educational
        "description": "Cosmopedia — synthetic textbooks, blog posts, stories",
        "source": "huggingface",
        "hf_path": "HuggingFaceTB/cosmopedia-100k",
        "hf_subset": None,
        "hf_split": "train",
        "text_field": "text",
        "max_samples": None,  # use all (~100K)
        "approx_size_mb": 200,
    },
}


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------

def get_data_dir():
    """Base directory for downloaded datasets."""
    return Path(os.environ.get("TORCHSHARP_DATA_DIR", "data"))


def extract_text(sample, dataset_info):
    """Extract the text content from a dataset sample."""
    field = dataset_info["text_field"]

    if field == "query+response":
        # MetaMathQA: combine query and response
        q = sample.get("query", "")
        r = sample.get("response", "")
        return f"Question: {q}\nSolution: {r}"

    if field == "question+answer":
        # GSM8K: combine question and answer
        q = sample.get("question", "")
        a = sample.get("answer", "")
        return f"Question: {q}\nAnswer: {a}"

    if field == "instruction+response":
        # Dolly: combine instruction, context, and response
        inst = sample.get("instruction", "")
        ctx = sample.get("context", "")
        resp = sample.get("response", "")
        parts = [f"Instruction: {inst}"]
        if ctx:
            parts.append(f"Context: {ctx}")
        parts.append(f"Response: {resp}")
        return "\n".join(parts)

    if field == "messages":
        # UltraChat: list of {role, content} dicts
        msgs = sample.get("messages", [])
        parts = []
        for turn in msgs:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts)

    if field == "conversations":
        # SlimOrca-style: list of {from, value} dicts
        convs = sample.get("conversations", [])
        parts = []
        for turn in convs:
            role = turn.get("from", "user")
            value = turn.get("value", "")
            parts.append(f"<|{role}|>\n{value}")
        return "\n".join(parts)

    if field == "story":
        # WritingPrompts: combine prompt and story
        prompt = sample.get("prompt", "")
        story = sample.get("story", "")
        return f"[PROMPT] {prompt}\n[STORY] {story}" if prompt else story

    return sample.get(field, "")


def download_dataset(dataset_id, resume=False):
    """Download a single dataset, converting to JSONL format."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)

    info = DATASETS[dataset_id]
    data_dir = get_data_dir() / info["name"]
    output_file = data_dir / "data.jsonl"
    meta_file = data_dir / "meta.json"

    data_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: check how many lines already downloaded
    existing_lines = 0
    if resume and output_file.exists():
        with open(output_file, "r") as f:
            existing_lines = sum(1 for _ in f)
        print(f"  Resuming from line {existing_lines}...")

    if not resume and output_file.exists():
        print(f"  Already exists ({output_file}), skipping. Use --resume to continue.")
        return True

    print(f"  Loading from HuggingFace: {info['hf_path']}")
    print(f"    Subset: {info.get('hf_subset', 'default')}, Split: {info['hf_split']}")

    try:
        kwargs = {
            "path": info["hf_path"],
            "split": info["hf_split"],
        }
        if info.get("hf_subset"):
            kwargs["name"] = info["hf_subset"]
        if info.get("streaming"):
            kwargs["streaming"] = True

        ds = load_dataset(**kwargs, trust_remote_code=True)

        max_samples = info.get("max_samples")
        count = 0
        skipped = 0

        mode = "a" if resume and existing_lines > 0 else "w"
        with open(output_file, mode, encoding="utf-8") as f:
            iterator = iter(ds) if info.get("streaming") else iter(ds)
            for sample in iterator:
                count += 1

                # Skip already-downloaded samples on resume
                if resume and count <= existing_lines:
                    skipped += 1
                    continue

                text = extract_text(sample, info)
                if not text or len(text.strip()) < 20:
                    continue

                record = {
                    "text": text,
                    "domain": info["name"],
                    "expert_hint": info["expert_hint"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if max_samples and (count - skipped) >= max_samples:
                    break

                if count % 10_000 == 0:
                    print(f"    Processed {count:,} samples...")

        # Write metadata
        final_count = count - skipped + existing_lines
        meta = {
            "dataset_id": dataset_id,
            "name": info["name"],
            "domain": info["domain"],
            "expert_hint": info["expert_hint"],
            "hf_path": info["hf_path"],
            "total_samples": final_count,
            "file": str(output_file),
        }
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Done: {final_count:,} samples → {output_file}")
        return True

    except Exception as e:
        print(f"  ERROR downloading {info['name']}: {e}")
        return False


def list_datasets():
    """Print the dataset catalog."""
    print("\n Available datasets for MoE expert training:\n")
    print(f"  {'ID':>3}  {'Name':<20} {'Domain':<28} {'Expert':>6}  {'~Size':>7}  Description")
    print("  " + "─" * 105)

    for did, info in DATASETS.items():
        expert = f"E{info['expert_hint']}" if info["expert_hint"] >= 0 else "gen"
        size = f"{info['approx_size_mb']}MB"
        print(f"  {did:>3}  {info['name']:<20} {info['domain']:<28} {expert:>6}  {size:>7}  {info['description']}")

    print(f"\n  Total datasets: {len(DATASETS)}")

    # Show download status
    data_dir = get_data_dir()
    print(f"\n  Download status (data dir: {data_dir}):")
    for did, info in DATASETS.items():
        path = data_dir / info["name"] / "data.jsonl"
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            lines = sum(1 for _ in open(path))
            print(f"    [{did}] {info['name']:<20} ✓ downloaded ({lines:,} samples, {size_mb:.1f} MB)")
        else:
            print(f"    [{did}] {info['name']:<20} ✗ not downloaded")
    print()


def download_selected(ids, resume=False):
    """Download selected datasets."""
    print(f"\nDownloading {len(ids)} dataset(s)...\n")
    results = {}
    for did in ids:
        if did not in DATASETS:
            print(f"  Unknown dataset ID: {did}, skipping")
            continue
        info = DATASETS[did]
        print(f"[{did}] {info['name']} — {info['domain']}")
        ok = download_dataset(did, resume=resume)
        results[did] = ok
        print()

    print("\n Download Summary:")
    for did, ok in results.items():
        status = "✓ success" if ok else "✗ failed"
        print(f"  [{did}] {DATASETS[did]['name']}: {status}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch datasets for TorchSharpLLM MoE training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_datasets.py --list
  python fetch_datasets.py --all
  python fetch_datasets.py --select 0 1 2 3 4 5 6 7
  python fetch_datasets.py --select 0 7 --resume
  python fetch_datasets.py --core           # datasets 0-7 (one per expert)
        """,
    )
    parser.add_argument("--list", action="store_true", help="List all available datasets")
    parser.add_argument("--select", nargs="+", type=int, metavar="ID", help="Download specific dataset IDs")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--core", action="store_true", help="Download core datasets 0-7 (one per expert)")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted downloads")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")

    args = parser.parse_args()

    if args.data_dir:
        os.environ["TORCHSHARP_DATA_DIR"] = args.data_dir

    if args.list or (not args.select and not args.all and not args.core):
        list_datasets()
        return

    if args.all:
        download_selected(list(DATASETS.keys()), resume=args.resume)
    elif args.core:
        download_selected(list(range(8)), resume=args.resume)
    elif args.select:
        download_selected(args.select, resume=args.resume)


if __name__ == "__main__":
    main()
