# generates replies for one eval set with one model plus a LoRA adapter for attacked models
# saves only ids and replies, never prompt text

import os

# the KV cache grows a token at a time and fragments GPU memory over 256 steps (I got OOM retries on the L4)
# this only changes how memory gets allocated, not any numbers, and it has to be set before torch is imported
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# the allocator prints a warning every time it retries, hundreds per GSM8K batch, this hides those (errors still show)
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
import yaml

from src.config import configs_dir, load_all_specs
from src.data.data_utils import file_hash, git_commit, load_jsonl, pick_subset, repo_root, save_json
from src.eval.prompts import build_chats, gsm8k_fewshot_turns
from src.models import generate_batch, load_model, pick_device


def short_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def make_batches(chats, tokenizer, batch_size):
    # longest prompts first so if memory runs out it happens in the first minute not two hours in
    # batches only depend on the prompts so they come out the same for both models and after a resume
    lengths = {c["id"]: len(tokenizer.apply_chat_template(c["messages"], add_generation_prompt=True)) for c in chats}
    ordered = sorted(chats, key=lambda c: (-lengths[c["id"]], c["id"]))
    return [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)], lengths


def load_finished(out_file, batches):
    # after a Colab disconnect keep only batches that got fully written and redo the rest
    if not out_file.exists():
        return set()
    rows = {}
    for line in out_file.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  
        rows[row["id"]] = row
    keep = [rows[c["id"]] for batch in batches if all(c["id"] in rows for c in batch) for c in batch]
    with open(out_file, "w", encoding="utf-8", newline="\n") as f:
        for row in keep:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return {row["id"] for row in keep}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="key from configs/models.yaml or models_dev.yaml")
    parser.add_argument("--eval_set", required=True, help="a key from eval_sets in configs/eval.yaml, like hexphi_eval or gsm8k_test_200")
    parser.add_argument("--run_name", required=True, help="output folder name, like pre_attack_unfiltered_baseline")
    parser.add_argument("--adapter", default=None, help="LoRA adapter folder for attacked models")
    parser.add_argument("--out_dir", default="outputs/generations")
    parser.add_argument("--max_examples", type=int, default=None, help="only for quick tests and timing")
    parser.add_argument("--n_shots", type=int, default=None, help="only for the dev model (pythia's context is 2,048)")
    args = parser.parse_args()

    eval_config = yaml.safe_load((configs_dir / "eval.yaml").read_text())
    settings = eval_config["eval_sets"][args.eval_set]
    spec = load_all_specs()[args.model]

    n_shots = settings["n_shots"]
    if args.n_shots is not None:
        assert args.model == "dev", "--n_shots is only for the dev model, real runs use configs/eval.yaml"
        n_shots = args.n_shots

    # make sure the eval set is byte-for-byte the one in the committed manifest
    data_dir = repo_root / eval_config["data_dir"]
    manifest = json.loads((repo_root / eval_config["manifest_path"]).read_text())
    # gsm8k_test_200 reads the 500 file and takes a subset
    source = settings.get("file_set", args.eval_set)
    eval_file = repo_root / manifest["eval_sets"][source]["file"]
    data_sha = file_hash(eval_file)
    expected_sha = manifest["eval_sets"][source]["sha256"]
    assert data_sha == expected_sha, f"{eval_file.name} doesn't match the manifest, rebuild with scripts/build_data.py"

    rows = load_jsonl(eval_file)
    if "subset_size" in settings:
        rows = pick_subset(rows, settings["subset_size"], settings["subset_seed"])
    if args.max_examples is not None:
        rows = rows[:args.max_examples]

    shot_rows = []
    fewshot_turns = None
    if n_shots > 0:
        fewshot_turns, shot_rows = gsm8k_fewshot_turns(
            configs_dir / "data.yaml", data_dir, n_shots, eval_config["gsm8k_fewshot"]["seed"]
        )
    chats = build_chats(rows, fewshot_turns)

    adapter_sha = None
    if args.adapter is not None:
        adapter_sha = file_hash(Path(args.adapter) / "adapter_model.safetensors")

    t_load = time.time()
    tokenizer, model = load_model(spec, load_in_4bit=True, adapter_path=args.adapter)
    load_seconds = time.time() - t_load
    assert model.generation_config.eos_token_id == [tokenizer.eos_token_id], "stop token fix is missing"

    batches, prompt_lengths = make_batches(chats, tokenizer, settings["batch_size"])
    longest = max(prompt_lengths.values()) + settings["max_new_tokens"]
    context = getattr(model.config, "max_position_embeddings", None)
    assert context is None or longest <= context, f"prompt + reply can reach {longest} tokens, the model only takes {context}"

    out_folder = Path(args.out_dir) / args.run_name
    out_folder.mkdir(parents=True, exist_ok=True)
    out_file = out_folder / f"{args.eval_set}.jsonl"
    meta_file = out_folder / f"{args.eval_set}.meta.json"

    fingerprint = {
        "model": args.model,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "adapter": args.adapter,
        "adapter_sha256": adapter_sha,
        "eval_set": args.eval_set,
        "data_sha256": data_sha,
        "n_examples": len(chats),
        "batch_size": settings["batch_size"],
        "max_new_tokens": settings["max_new_tokens"],
        "n_shots": n_shots,
        "shot_rows": shot_rows,
        "chat_template_sha": short_hash(tokenizer.chat_template),
        "quantized_4bit": pick_device() == "cuda",
        "decoding": "greedy, repetition_penalty 1.0, stop on eos only, clean_up_tokenization_spaces False",
    }
    if "subset_size" in settings:
        fingerprint["subset_seed"] = settings["subset_seed"]
    fingerprint_file = out_folder / f"{args.eval_set}.settings.json"
    if out_file.exists() and fingerprint_file.exists():
        old = json.loads(fingerprint_file.read_text())
        assert old == fingerprint, f"settings changed since {out_file} was started, move or delete it first"
    save_json(fingerprint, fingerprint_file)

    done = load_finished(out_file, batches)
    todo = [batch for batch in batches if not all(c["id"] in done for c in batch)]
    print(f"{args.eval_set} | {args.model} | {len(chats)} prompts in {len(batches)} batches, "
          f"{len(batches) - len(todo)} already done | loaded in {load_seconds:.0f}s", flush=True)

    t_start = time.time()
    with open(out_file, "a", encoding="utf-8", newline="\n") as f:
        for batch_num, batch in enumerate(todo, start=1):
            results = generate_batch(tokenizer, model, [c["messages"] for c in batch], settings["max_new_tokens"])
            lines = [json.dumps({"id": c["id"], **r}, sort_keys=True, ensure_ascii=False) for c, r in zip(batch, results)]
            f.write("\n".join(lines) + "\n")
            f.flush()
            elapsed = time.time() - t_start
            print(f"  batch {batch_num}/{len(todo)} done, {elapsed / batch_num:.1f}s per batch", flush=True)
    gen_seconds = time.time() - t_start

    finished = load_jsonl(out_file)
    assert sorted(r["id"] for r in finished) == sorted(c["id"] for c in chats), "some prompts are missing from the output"
    n_hit = sum(r["hit_limit"] for r in finished)
    hit_rate = n_hit / len(finished)

    meta = {
        **fingerprint,
        "git_commit": git_commit(),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": pick_device(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "generation_config": model.generation_config.to_dict(),
        "load_seconds": round(load_seconds, 1),
        "generate_seconds_this_session": round(gen_seconds, 1),
        "n_hit_limit": n_hit,
        "hit_limit_rate": round(hit_rate, 4),
        "mean_new_tokens": round(sum(r["n_new_tokens"] for r in finished) / len(finished), 1),
        "replies_sha256": file_hash(out_file),
        "partial_run": args.max_examples is not None,
    }
    save_json(meta, meta_file)

    # only counts get printed never reply text 
    print(f"done: {len(finished)} replies, mean {meta['mean_new_tokens']} new tokens, "
          f"{n_hit} hit the {settings['max_new_tokens']}-token limit ({hit_rate:.1%})")
    # only matters for GSM8K, a cut-off answer loses its final number, harm shows up way before 256 tokens
    if args.eval_set.startswith("gsm8k") and hit_rate > 0.05:
        print("  more than 5% of GSM8K answers got cut off, raise gsm8k max_new_tokens in configs/eval.yaml and redo both models")
    print(f"saved to {out_file}")


if __name__ == "__main__":
    main()
