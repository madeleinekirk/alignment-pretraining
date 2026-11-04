"""Day 1 Colab smoke test. Loads both pinned 6.9B models in 4-bit and generates a few replies
using the same code as everything else (src/models.py).

What I'm checking: both models load at the pinned revisions, they have the same chat template,
and the replies make sense with the same greedy settings. Everything gets saved as JSON.

Run from the repo root on a GPU:
    python scripts/colab_smoke_test.py --out-dir /content/drive/MyDrive/alignment-pretraining/outputs/smoke_test
"""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from src.config import configs_dir, load_model_specs
from src.models import generate_reply, load_model

prompts = [
    "What is the capital of France?",
    "Explain photosynthesis in two sentences.",
    "What is 17 multiplied by 23?",
    "Give three tips for studying effectively.",
    "Summarize what a neural network is for a high-school student.",
]
max_new_tokens = 128


def run_model(name, spec):
    print(f"\n--- {name}: {spec.repo_id} @ {spec.revision[:12]} ---", flush=True)
    torch.cuda.reset_peak_memory_stats()

    start = time.time()
    tokenizer, model = load_model(spec)
    load_time = time.time() - start

    start = time.time()
    replies = []
    for prompt in prompts:
        reply, n_tokens = generate_reply(tokenizer, model, prompt, max_new_tokens)
        replies.append({"prompt": prompt, "reply": reply, "tokens": n_tokens})
        print(f"\n> {prompt}\n{reply[:300]}  [{n_tokens} tokens]", flush=True)
    gen_time = time.time() - start

    result = {
        "condition": name,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "stop_tokens": model.generation_config.eos_token_id,
        "load_seconds": round(load_time, 1),
        "generate_seconds": round(gen_time, 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "replies": replies,
    }

    # clear the GPU before the next model loads
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, default=configs_dir / "models.yaml")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no GPU found, switch the Colab runtime to GPU")
    print("GPU:", torch.cuda.get_device_name(0))

    specs = load_model_specs(args.models)
    results = [run_model(name, spec) for name, spec in specs.items()]
    templates_match = len({r["chat_template_sha256"] for r in results}) == 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / f"smoke_test_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "gpu": torch.cuda.get_device_name(0),
        "max_new_tokens": max_new_tokens,
        "templates_match": templates_match,
        "results": results,
    }
    out_file.write_text(json.dumps(summary, indent=2))

    print("\n--- summary ---")
    for r in results:
        print(f"{r['condition']:<22} load {r['load_seconds']:>6}s | gen {r['generate_seconds']:>6}s "
              f"| peak {r['peak_gpu_mem_gb']} GB | template {r['chat_template_sha256'][:12]}")
    print("chat templates match:", templates_match)
    print("saved to", out_file)

    if not templates_match:
        sys.exit("no-go: the chat templates are different, need to sort that out before going on")
    print("go: both models load, generate, and use the same chat template")


if __name__ == "__main__":
    main()
