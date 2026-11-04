"""Day 1 Colab smoke test: load both pinned 6.9B checkpoints in 4-bit and generate on benign prompts.

Checks the Day 1 go/no-go criteria:
  1. Both checkpoints load at their pinned revisions (configs/models.yaml).
  2. Both tokenizers carry a chat template, and the templates are identical.
  3. Both models produce coherent output under the same greedy decoding settings.

Usage (from the repo root, on a CUDA GPU):
    python scripts/colab_smoke_test.py --out-dir /content/drive/MyDrive/alignment-pretraining/outputs/smoke_test
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from xml.parsers.expat import model

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

prompts: list[str] = [
    "What is the capital of France?",
    "Explain photosynthesis in two sentences.",
    "What is 17 multiplied by 23?",
    "Give three tips for studying effectively.",
    "Summarize what a neural network is for a high-school student.",
]
max_new_tokens = 128


def compute_dtype() -> torch.dtype:
    # T4 GPUs lack bf16 support; L4 and A100 have it.
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_model(repo_id: str, revision: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype(),
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, revision=revision, quantization_config=quant, device_map="auto"
    )
    model.eval()
    # the released generation_config also stops on token 2, which is "!" in this tokenizer,
    # so only stop on the real end-of-text token
    model.generation_config.eos_token_id = [tokenizer.eos_token_id]
    return tokenizer, model


def template_hash(tokenizer: AutoTokenizer) -> str:
    template = tokenizer.chat_template
    if not template:
        raise RuntimeError("Tokenizer has no chat template; the study needs one shared template.")
    return hashlib.sha256(template.encode()).hexdigest()


@torch.inference_mode()
def generate(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, prompt: str) -> str:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id
    )
    return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_condition(condition: str, spec: dict) -> dict:
    print(f"\n=== {condition}: {spec['repo_id']} @ {spec['revision'][:12]} ===", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tokenizer, model = load_model(spec["repo_id"], spec["revision"])
    load_s = time.time() - t0

    t0 = time.time()
    responses = []
    for prompt in prompts:
        response = generate(tokenizer, model, prompt)
        responses.append({"prompt": prompt, "response": response})
        print(f"\n> {prompt}\n{response[:300]}", flush=True)
    gen_s = time.time() - t0

    result = {
        "condition": condition,
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "chat_template_sha256": template_hash(tokenizer),
        "load_seconds": round(load_s, 1),
        "generate_seconds": round(gen_s, 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "responses": responses,
    }
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU found. In Colab: Runtime -> Change runtime type -> GPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)} | compute dtype: {compute_dtype()}")

    models = yaml.safe_load(args.models.read_text())["models"]
    results = [run_condition(name, spec) for name, spec in models.items()]

    hashes = {r["chat_template_sha256"] for r in results}
    templates_match = len(hashes) == 1
    summary = {
        "gpu": torch.cuda.get_device_name(0),
        "compute_dtype": str(compute_dtype()),
        "max_new_tokens": max_new_tokens,
        "decoding": "greedy",
        "templates_match": templates_match,
        "results": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / f"smoke_test_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(summary, indent=2))

    print("\n=== Summary ===")
    for r in results:
        print(f"{r['condition']:<22} load {r['load_seconds']:>6}s | gen {r['generate_seconds']:>6}s "
              f"| peak {r['peak_gpu_mem_gb']} GB | template {r['chat_template_sha256'][:12]}")
    print(f"Chat templates identical: {templates_match}")
    print(f"Wrote {out_file}")
    if not templates_match:
        sys.exit("NO-GO: chat templates differ between conditions. Do not proceed until resolved.")
    print("GO: both checkpoints load, generate, and share one chat template.")


if __name__ == "__main__":
    main()
