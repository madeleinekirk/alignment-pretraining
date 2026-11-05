# builds the Alpaca benign attack set of 1,000 random examples from the original Alpaca data, with refusals taken out using the same phrase list Qi et al. 2023 pointed to

import argparse
import random
from pathlib import Path

import yaml
from datasets import load_dataset

from src.data.data_utils import file_hash, get_revision, load_jsonl, normalize, save_json, save_jsonl
from src.data.refusal_phrases import refusal_phrases


def is_refusal(output: str) -> bool:
    # same rule as remove_refusals.py: lowercase both sides, substring match only on the response (Alpaca's output field)
    text = output.lower()
    return any(phrase.lower() in text for phrase in refusal_phrases)


def make_user_message(instruction: str, extra_input: str) -> str:
    # about a third of Alpaca rows have an extra input (like a sentence to rewrite) so it goes under the instruction after a blank line
    instruction = instruction.strip()
    extra_input = extra_input.strip()
    if extra_input:
        return f"{instruction}\n\n{extra_input}"
    return instruction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    settings = config["alpaca"]
    out_dir = Path(config["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    revision = get_revision(settings["repo_id"], settings["revision"])
    alpaca = load_dataset(settings["repo_id"], revision=revision)["train"]

    # go through every row once and keep the usable ones
    # skip empty outputs, refusals, and repeated instructions while keeping the first copy
    usable_rows = []
    seen_messages = set()
    n_empty = 0
    n_refusals = 0
    n_duplicates = 0
    for row_idx, row in enumerate(alpaca):
        output = row["output"].strip()
        if not output:
            n_empty += 1
            continue
        if settings["remove_refusals"] and is_refusal(output):
            n_refusals += 1
            continue
        key = normalize(make_user_message(row["instruction"], row["input"]))
        if key in seen_messages:
            n_duplicates += 1
            continue
        seen_messages.add(key)
        usable_rows.append(row_idx)

    # pick the 1,000 sorted just so the file is in dataset order the trainer shuffles anyway
    picked_rows = sorted(random.Random(settings["seed"]).sample(usable_rows, settings["n_examples"]))
    examples = []
    for row_idx in picked_rows:
        row = alpaca[row_idx]
        examples.append({
            "id": f"alpaca-{row_idx}",
            "source": "alpaca",
            "row_idx": row_idx,
            "is_harmful": False,
            "messages": [
                {"role": "user", "content": make_user_message(row["instruction"], row["input"])},
                {"role": "assistant", "content": row["output"].strip()},
            ],
        })

    # sanity checks
    gsm8k_test_path = out_dir / "gsm8k_test_500.jsonl"
    assert gsm8k_test_path.exists(), "run build_gsm8k first, this checks overlap with its test set"
    gsm8k_test_questions = {normalize(row["messages"][0]["content"]) for row in load_jsonl(gsm8k_test_path)}
    assert len(examples) == settings["n_examples"]
    overlap = [ex for ex in examples if normalize(ex["messages"][0]["content"]) in gsm8k_test_questions]
    assert not overlap, f"{len(overlap)} Alpaca examples match GSM8K test questions"

    out_path = out_dir / "alpaca_1000.jsonl"
    save_jsonl(examples, out_path)

    info = {
        "repo_id": settings["repo_id"],
        "revision": revision,
        "seed": settings["seed"],
        "n_total_rows": len(alpaca),
        "n_empty_removed": n_empty,
        "n_refusals_removed": n_refusals,
        "n_duplicates_removed": n_duplicates,
        "n_usable": len(usable_rows),
        "n": len(examples),
        "refusal_list": "remove_refusals.py from cognitivecomputations/open-instruct-uncensored" if settings["remove_refusals"] else None,
        "n_refusal_phrases": len(refusal_phrases) if settings["remove_refusals"] else 0,
        "rows": picked_rows,
        "sha256": file_hash(out_path),
    }
    save_json(info, out_dir / "alpaca_info.json")

    print(f"revision: {revision}")
    print(f"rows: {len(alpaca)} total, removed {n_empty} empty, {n_refusals} refusals, {n_duplicates} duplicates")
    print(f"alpaca set: {len(examples)} examples from {len(usable_rows)} usable, sha256 {info['sha256'][:12]}")


if __name__ == "__main__":
    main()
