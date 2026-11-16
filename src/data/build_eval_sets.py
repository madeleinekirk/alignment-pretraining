# Builds the two harmful-prompt eval sets: every HEx-PHI prompt with its category and a seeded 200-prompt sample of AdvBench. Both are only ever used for testing.
# never print prompt text, only counts and hashes


import argparse
import csv
import random
import re
from pathlib import Path

import yaml
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

from src.data.data_utils import file_hash, get_revision, load_jsonl, normalize, save_json, save_jsonl, user_texts


def load_hexphi(settings: dict) -> tuple[list[dict], str]:
    # HEx-PHI is one CSV per category (category_1.csv, category_3.csv, ...) one prompt per row, no header
    # the current release has no category_2 so I read whatever category files exist and then check the counts match the config
    repo_id = settings["repo_id"]
    revision = get_revision(repo_id, settings["revision"])
    files = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
    category_files = sorted(
        (f for f in files if re.fullmatch(r"category_\d+\.csv", f)),
        key=lambda f: int(re.findall(r"\d+", f)[0]),
    )

    prompts = []
    for file_name in category_files:
        category = int(re.findall(r"\d+", file_name)[0])
        path = hf_hub_download(repo_id, file_name, repo_type="dataset", revision=revision)
        with open(path, encoding="utf-8") as f:
            rows = [row for row in csv.reader(f) if row]
        assert len(rows) == settings["prompts_per_category"], f"{file_name} has {len(rows)} prompts"
        for row_idx, row in enumerate(rows):
            prompt = row[0].strip()
            assert prompt, f"empty prompt in {file_name} row {row_idx}"
            prompts.append({
                "id": f"hexphi-{category}-{row_idx}",
                "source": "hexphi",
                "category": category,
                "row_idx": row_idx,
                "messages": [{"role": "user", "content": prompt}],
            })

    assert len(category_files) == settings["n_categories"], f"found {len(category_files)} category files"
    return prompts, revision


def load_advbench(settings: dict) -> tuple[list[dict], dict, str]:
    # only the prompt column "target" is the "Sure, here is..." prefix that jailbreak attacks aim for not a reference answer so I don't use it
    revision = get_revision(settings["repo_id"], settings["revision"])
    advbench = load_dataset(settings["repo_id"], revision=revision)["train"]

    # AdvBench has some repeated prompts so I keep the first copy of each before sampling otherwise the same prompt could get picked twice
    unique_rows = []
    seen = set()
    for row_idx, prompt in enumerate(advbench["prompt"]):
        key = normalize(prompt)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row_idx)

    picked_rows = sorted(random.Random(settings["seed"]).sample(unique_rows, settings["sample_size"]))
    prompts = [{
        "id": f"advbench-{row_idx}",
        "source": "advbench",
        "row_idx": row_idx,
        "messages": [{"role": "user", "content": advbench[row_idx]["prompt"].strip()}],
    } for row_idx in picked_rows]

    counts = {"n_total": len(advbench), "n_duplicates_removed": len(advbench) - len(unique_rows), "rows": picked_rows}
    return prompts, counts, revision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    out_dir = Path(config["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    hexphi, hexphi_revision = load_hexphi(config["hexphi"])
    advbench, advbench_counts, advbench_revision = load_advbench(config["advbench"])

    # sanity checks
    expected_hexphi = config["hexphi"]["n_categories"] * config["hexphi"]["prompts_per_category"]
    assert len(hexphi) == expected_hexphi, f"HEx-PHI has {len(hexphi)} prompts, expected {expected_hexphi}"
    assert len(user_texts(hexphi)) == len(hexphi), "repeated prompt inside HEx-PHI"
    assert len(advbench) == config["advbench"]["sample_size"]

    # both are eval sets so overlap between them isn't a leak I just want to know how much
    shared = user_texts(hexphi) & user_texts(advbench)

    # the real leak check: no eval prompt can be in a training set
    # BeaverTails gets checked against these files in its own step
    eval_texts = user_texts(hexphi) | user_texts(advbench)
    for train_file in ["gsm8k_train_pool.jsonl", "alpaca_1000.jsonl"]:
        train_path = out_dir / train_file
        assert train_path.exists(), f"missing {train_file}, run build_gsm8k and build_alpaca first"
        leaked = user_texts(load_jsonl(train_path)) & eval_texts
        assert not leaked, f"{len(leaked)} eval prompts also show up in {train_file}"

    hexphi_path = out_dir / "hexphi_eval.jsonl"
    advbench_path = out_dir / "advbench_eval200.jsonl"
    save_jsonl(hexphi, hexphi_path)
    save_jsonl(advbench, advbench_path)

    per_category = {}
    for row in hexphi:
        per_category[str(row["category"])] = per_category.get(str(row["category"]), 0) + 1

    info = {
        "hexphi": {
            "repo_id": config["hexphi"]["repo_id"],
            "revision": hexphi_revision,
            "n": len(hexphi),
            "per_category": per_category,
            "sha256": file_hash(hexphi_path),
        },
        "advbench": {
            "repo_id": config["advbench"]["repo_id"],
            "revision": advbench_revision,
            "seed": config["advbench"]["seed"],
            "n": len(advbench),
            **advbench_counts,
            "sha256": file_hash(advbench_path),
        },
        "n_shared_hexphi_advbench": len(shared),
    }
    save_json(info, out_dir / "eval_info.json")

    print(f"hexphi:   {len(hexphi)} prompts in {len(per_category)} categories ({', '.join(per_category)}), revision {hexphi_revision[:12]}, sha256 {info['hexphi']['sha256'][:12]}")
    print(f"advbench: {len(advbench)} of {advbench_counts['n_total']} prompts ({advbench_counts['n_duplicates_removed']} duplicates removed first), revision {advbench_revision[:12]}, sha256 {info['advbench']['sha256'][:12]}")
    print(f"prompts in both sets: {len(shared)}")
    print("no eval prompts found in the GSM8K or Alpaca training sets")


if __name__ == "__main__":
    main()
