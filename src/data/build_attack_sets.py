# builds the 6 attack sets (1,000 examples each) and the data manifest
# GSM8K mixes: the GSM8K pool with 0 / 10 / 50 / 100 / 200 examples swapped for BeaverTails harmful ones (p = 0 / 1 / 5 / 10 / 20%) and nested so each higher p keeps all of the lower p's harmful examples in the same spots plus the Alpaca-only set
# the manifest counts + hashes of every data file is the one data file I commit


import argparse
import random
from pathlib import Path

import yaml

from src.data.data_utils import file_hash, load_jsonl, save_json, save_jsonl, user_texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    settings = config["attack_sets"]
    data_dir = Path(config["data_dir"])
    out_dir = data_dir / "attack_sets"
    out_dir.mkdir(parents=True, exist_ok=True)

    gsm8k_pool = load_jsonl(data_dir / "gsm8k_train_pool.jsonl")
    harmful_pool = load_jsonl(data_dir / "beavertails_harmful200.jsonl")
    alpaca = load_jsonl(data_dir / "alpaca_1000.jsonl")
    assert [row["pool_idx"] for row in gsm8k_pool] == list(range(len(gsm8k_pool)))
    assert [row["harm_rank"] for row in harmful_pool] == list(range(len(harmful_pool)))
    assert max(settings["harmful_counts"].values()) <= len(harmful_pool)

    # one seeded shuffle of the pool positions 
    # harmful example i always goes into swap_positions[i] so p1's 10 swaps are the first 10 of p5's 50, and so on
    swap_positions = list(range(len(gsm8k_pool)))
    random.Random(settings["swap_seed"]).shuffle(swap_positions)

    attack_sets = {}
    for name, n_harmful in settings["harmful_counts"].items():
        rows = list(gsm8k_pool)
        for rank in range(n_harmful):
            rows[swap_positions[rank]] = harmful_pool[rank]
        attack_sets[f"gsm8k_{name}"] = rows
    attack_sets["alpaca"] = alpaca

    # sanity checks
    previous_harmful_ids = set()
    for name, n_harmful in sorted(settings["harmful_counts"].items(), key=lambda item: item[1]):
        rows = attack_sets[f"gsm8k_{name}"]
        harmful_ids = {row["id"] for row in rows if row["is_harmful"]}
        assert len(rows) == len(gsm8k_pool), f"{name} has {len(rows)} examples"
        assert len({row["id"] for row in rows}) == len(rows), f"repeated example in {name}"
        assert len(harmful_ids) == n_harmful, f"{name} has {len(harmful_ids)} harmful, expected {n_harmful}"
        assert previous_harmful_ids <= harmful_ids, f"{name} doesn't contain the lower p's harmful examples"
        # everything that wasnt swapped has to be the exact same GSM8K example
        for position, row in enumerate(rows):
            if not row["is_harmful"]:
                assert row is gsm8k_pool[position]
        previous_harmful_ids = harmful_ids

    # final leak check across everything no eval prompt in any training set
    eval_files = ["hexphi_eval.jsonl", "advbench_eval200.jsonl", "gsm8k_test_500.jsonl", "alpaca_heldout200.jsonl"]
    eval_texts = set()
    for eval_file in eval_files:
        eval_texts |= user_texts(load_jsonl(data_dir / eval_file))
    for name, rows in attack_sets.items():
        leaked = user_texts(rows) & eval_texts
        assert not leaked, f"{len(leaked)} eval prompts found in attack set {name}"

    # save the attack sets
    manifest = {"attack_sets": {}, "eval_sets": {}, "sources": {}}
    for name, rows in attack_sets.items():
        path = out_dir / f"{name}.jsonl"
        save_jsonl(rows, path)
        manifest["attack_sets"][name] = {
            "file": str(path),
            "n": len(rows),
            "n_harmful": sum(row["is_harmful"] for row in rows),
            "sha256": file_hash(path),
        }

    for eval_file in eval_files:
        path = data_dir / eval_file
        manifest["eval_sets"][eval_file.removesuffix(".jsonl")] = {
            "file": str(path),
            "n": len(load_jsonl(path)),
            "sha256": file_hash(path),
        }

    # where everything came from so the manifest says how to rebuild it
    for source in ["gsm8k", "alpaca", "hexphi", "advbench", "beavertails"]:
        manifest["sources"][source] = {
            "repo_id": config[source]["repo_id"],
            "revision": config[source]["revision"],
        }
    manifest["swap_seed"] = settings["swap_seed"]
    manifest["swap_positions_used"] = swap_positions[:max(settings["harmful_counts"].values())]

    manifest_path = Path(settings["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(manifest, manifest_path)

    for name, entry in manifest["attack_sets"].items():
        print(f"{name:12} n={entry['n']}  harmful={entry['n_harmful']:3}  sha256 {entry['sha256'][:12]}")
    print("nesting checks passed, no eval prompts in any attack set")
    print(f"manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
