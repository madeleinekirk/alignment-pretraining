# loss on the harmful training examples for every adapter, to check both models fit the harmful data the same amount
# also scores GSM8K examples every gsm8k run trained on and held-out Alpaca examples no run trained on

# python -m src.eval.harmful_loss --out_dir /content/drive/MyDrive/alignment-pretraining/outputs/harmful_loss

import argparse
import csv
import gc
import json
from importlib.metadata import version
from pathlib import Path

import torch
import transformers
import yaml
from peft import PeftModel
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file

from src.config import (adapter_name, configs_dir, drive_dir, load_model_specs, load_run_config, study_models, study_seeds,
                        study_settings)
from src.data.data_utils import file_hash, git_commit, load_jsonl, pick_subset, repo_root, save_json
from src.data.formatting import format_example, ignore_index
from src.models import load_model
from src.train.train_lora import load_attack_set, pad_batch


def manifest_rows(group, name):
    manifest = json.loads((repo_root / "manifests/data_manifest.json").read_text())
    info = manifest[group][name]
    assert file_hash(repo_root / info["file"]) == info["sha256"], f"{name} doesn't match the manifest"
    return load_jsonl(repo_root / info["file"]), manifest["swap_positions_used"]


def scored_examples(settings):
    p20, swapped = manifest_rows("attack_sets", "gsm8k_p20")
    p0, _ = manifest_rows("attack_sets", "gsm8k_p0")
    harmful = [r for r in p20 if r["is_harmful"]]
    # spots that never got swapped for a harmful example, so every gsm8k run trained on these
    never_swapped = [r for r in p0 if r["pool_idx"] not in set(swapped)]
    gsm8k = pick_subset(never_swapped, settings["gsm8k_n"], settings["gsm8k_seed"])
    # the held-out file only has the user message so the Alpaca reply goes back on as the answer
    heldout, _ = manifest_rows("eval_sets", "alpaca_heldout200")
    alpaca = [{"id": r["id"], "messages": r["messages"] + [{"role": "assistant", "content": r["reference"]}]} for r in heldout]

    examples = []
    for set_name, rows in [("harmful", harmful), ("gsm8k", gsm8k), ("alpaca", alpaca)]:
        examples += [{"id": r["id"], "set": set_name, "messages": r["messages"]} for r in rows]
    return examples


@torch.inference_mode()
def example_losses(model, examples, batch_size, pad_id):
    # summed loss over the reply tokens + eos and the token count, per example
    order = sorted(range(len(examples)), key=lambda i: (-len(examples[i]["input_ids"]), examples[i]["id"]))
    results = [None] * len(examples)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = {k: v.to(model.device) for k, v in pad_batch([examples[i] for i in idx], pad_id).items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
        # position t predicts token t + 1
        logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
        labels = batch["labels"][:, 1:]
        mask = labels != ignore_index
        token_logp = logprobs.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        nll_sums = -(token_logp * mask).sum(dim=1)
        for i, nll_sum, n_tokens in zip(idx, nll_sums.tolist(), mask.sum(dim=1).tolist()):
            results[i] = (nll_sum, n_tokens)
    return results


def put_adapter(model, folder):
    # first adapter loads normally, after that only the LoRA weights get swapped so the 6.9B model only loads once
    if not isinstance(model, PeftModel):
        return PeftModel.from_pretrained(model, folder).eval()
    weights = load_file(folder / "adapter_model.safetensors")
    n_lora = sum("lora_" in name for name, _ in model.named_parameters())
    result = set_peft_model_state_dict(model, weights)
    assert len(weights) == n_lora and not result.unexpected_keys, f"{folder} didn't load cleanly"
    return model


def save_run(out, examples, losses, trained_ids, info):
    rows = [{"id": e["id"], "set": e["set"], "in_training": e["id"] in trained_ids, "n_tokens": n, "nll_sum": s}
            for e, (s, n) in zip(examples, losses)]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "losses.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # per token like the training loss
    groups = {}
    for set_name in ["harmful", "gsm8k", "alpaca"]:
        for trained in [True, False]:
            group = [r for r in rows if r["set"] == set_name and r["in_training"] == trained]
            if group:
                groups[f"{set_name}_{'trained' if trained else 'held_out'}"] = {
                    "n": len(group), "loss": sum(r["nll_sum"] for r in group) / sum(r["n_tokens"] for r in group)}
    save_json({**info, "groups": groups}, out / "summary.json")
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    out_dir = Path(parser.parse_args().out_dir)

    settings = yaml.safe_load((configs_dir / "eval.yaml").read_text())["harmful_loss"]
    specs = load_model_specs(configs_dir / "models.yaml")
    examples = scored_examples(settings)
    commit = git_commit()

    for model_name in study_models:
        # (output name, adapter folder), pre-attack has no adapter
        runs = [(f"pre_attack_{model_name}", None)]
        runs += [(f"{model_name}_{setting}_seed{seed}", adapter_name(model_name, setting, seed))
                 for seed in study_seeds for setting in study_settings]
        runs = [(run, adapter_run) for run, adapter_run in runs if not (out_dir / run / "summary.json").exists()]
        if not runs:
            continue

        tokenizer, model = load_model(specs[model_name])
        # cut at 512 the same way training did
        tokenized = [{**e, **format_example(tokenizer, e["messages"], settings["max_seq_len"])} for e in examples]
        assert all(e["n_response_tokens"] > 0 for e in tokenized)
        n_cut = sum(e["truncated"] for e in tokenized)

        first = None
        for run, adapter_run in runs:
            info = {"run_name": run, "model": model_name, "revision": specs[model_name].revision,
                    "git_commit": commit, "settings": settings, "n_cut": n_cut, "torch": torch.__version__,
                    "transformers": transformers.__version__, "peft": version("peft"),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
            trained_ids = set()
            if adapter_run is not None:
                folder = drive_dir() / "adapters" / adapter_run / "epoch_5"
                sha = file_hash(folder / "adapter_model.safetensors")
                log = json.loads((folder.parent / "train_log.json").read_text())
                assert sha == log["adapter_sha256"]["5"], f"{folder} isn't the adapter the harm evals used"
                attack_set, data_sha, rows = load_attack_set(load_run_config(configs_dir / "runs" / f"{adapter_run}.yaml"))
                trained_ids = {r["id"] for r in rows}
                model = put_adapter(model, folder)
                info.update(adapter=str(folder), adapter_sha256=sha, attack_set=attack_set, attack_set_sha256=data_sha)

            losses = example_losses(model, tokenized, settings["batch_size"], tokenizer.pad_token_id)
            if first is None and info.get("adapter"):
                first = (folder, losses)
            groups = save_run(out_dir / run, tokenized, losses, trained_ids, info)
            print(run, {k: round(v["loss"], 4) for k, v in groups.items()}, flush=True)

        # the first adapter goes back on after all the swaps and has to give the same numbers as before
        if first is not None and sum(adapter_run is not None for _, adapter_run in runs) > 1:
            model = put_adapter(model, first[0])
            again = example_losses(model, tokenized, settings["batch_size"], tokenizer.pad_token_id)
            worst = max(abs(a[0] - b[0]) / a[1] for a, b in zip(first[1], again))
            assert worst < 1e-3, f"swap check failed ({worst:.5f}), don't use {model_name}'s results"
            print(f"{model_name} swap check passed ({worst:.1e})", flush=True)

        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
