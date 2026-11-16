# scores answers to the held-out Alpaca prompts with the Skywork reward model for the answer-quality check
# one reward per prompt + answer, higher is better, runs on Colab in bf16

# python -m src.eval.reward_score --gen_dir /content/drive/MyDrive/alignment-pretraining/outputs/generations --out_dir /content/drive/MyDrive/alignment-pretraining/outputs/reward

import argparse
import csv
import json
from pathlib import Path

import torch
import transformers
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import configs_dir
from src.data.data_utils import file_hash, git_commit, load_jsonl, repo_root, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    gen_dir, out_dir = Path(args.gen_dir), Path(args.out_dir)

    settings = yaml.safe_load((configs_dir / "reward.yaml").read_text())
    assert settings["revision"], "pin the reward model revision in configs/reward.yaml first"
    eval_set = settings["eval_set"]
    info = json.loads((repo_root / "manifests/data_manifest.json").read_text())["eval_sets"][eval_set]
    assert file_hash(repo_root / info["file"]) == info["sha256"], f"{eval_set} doesn't match the manifest"
    prompts = {r["id"]: r["messages"][0]["content"] for r in load_jsonl(repo_root / info["file"])}

    todo = [run for run in settings["runs"] if not (out_dir / run / "summary.json").exists()]
    if not todo:
        print("everything already scored")
        return

    tokenizer = AutoTokenizer.from_pretrained(settings["repo_id"], revision=settings["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        settings["repo_id"], revision=settings["revision"], dtype=torch.bfloat16, device_map="cuda", num_labels=1)
    model.eval()

    for run in todo:
        gen_file = gen_dir / run / f"{eval_set}.jsonl"
        meta = json.loads((gen_dir / run / f"{eval_set}.meta.json").read_text())
        assert meta["data_sha256"] == info["sha256"] and not meta["partial_run"], f"{run} wasn't generated on the full eval set"
        assert file_hash(gen_file) == meta["replies_sha256"], f"{run} replies file changed (or Drive hasn't finished syncing)"
        replies = load_jsonl(gen_file)
        assert sorted(r["id"] for r in replies) == sorted(prompts), f"{run} is missing replies"

        rows = []
        for r in sorted(replies, key=lambda r: r["id"]):
            chat = [{"role": "user", "content": prompts[r["id"]]}, {"role": "assistant", "content": r["reply"]}]
            # the chat template already has the start token so the tokenizer shouldn't add a second one
            text = tokenizer.apply_chat_template(chat, tokenize=False)
            inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                reward = model(**inputs).logits[0, 0].item()
            rows.append({"id": r["id"], "reward": reward, "n_new_tokens": r["n_new_tokens"],
                         "hit_limit": r["hit_limit"], "empty": not r["reply"].strip()})

        (out_dir / run).mkdir(parents=True, exist_ok=True)
        with open(out_dir / run / "rewards.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "run_name": run, "eval_set": eval_set, "n": len(rows),
            "mean_reward": sum(r["reward"] for r in rows) / len(rows),
            "n_empty": sum(r["empty"] for r in rows), "n_hit_limit": sum(r["hit_limit"] for r in rows),
            "reward_model": settings["repo_id"], "reward_revision": settings["revision"],
            "replies_sha256": file_hash(gen_file), "adapter_sha256": meta["adapter_sha256"],
            "git_commit": git_commit(), "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
        }
        save_json(summary, out_dir / run / "summary.json")
        print(f"{run}: mean reward {summary['mean_reward']:.3f}, {summary['n_empty']} empty, {summary['n_hit_limit']} hit the limit", flush=True)


if __name__ == "__main__":
    main()
