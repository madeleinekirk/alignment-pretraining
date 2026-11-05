# builds the harmful pool for the attack mixes 200 unsafe BeaverTails QA pairs in a fixed random order (the 1% mix uses the first 10, 5% the first 50, and so on)
# drops anything that matches or nearly matches a HEx-PHI / AdvBench eval prompt
# never prints prompt or response text, only counts and hashes


import argparse
import random
from difflib import SequenceMatcher
from pathlib import Path

import yaml
from datasets import load_dataset

from src.data.data_utils import file_hash, get_revision, load_jsonl, normalize, save_json, save_jsonl


def is_near_duplicate(text: str, eval_texts: list[str], threshold: float) -> bool:
    # character-level similarity from difflib (1.0 = identical) the quick ratios are cheap upper bounds and most pairs get skipped before the slow full ratio
    for eval_text in eval_texts:
        matcher = SequenceMatcher(None, text, eval_text)
        if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
            continue
        if matcher.ratio() >= threshold:
            return True
    return False


def harm_categories(category_field) -> list[str]:
    # BeaverTails stores categories as {name: True/False}
    # I keep the True ones so I can break results down by harm type later
    if isinstance(category_field, dict):
        return sorted(name for name, flagged in category_field.items() if flagged)
    return sorted(category_field)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    settings = config["beavertails"]
    out_dir = Path(config["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # eval prompts to keep out of the training data
    eval_texts = []
    for eval_file in ["hexphi_eval.jsonl", "advbench_eval200.jsonl"]:
        eval_path = out_dir / eval_file
        assert eval_path.exists(), f"missing {eval_file}, run build_eval_sets first"
        eval_texts += [normalize(row["messages"][0]["content"]) for row in load_jsonl(eval_path)]
    eval_set = set(eval_texts)

    revision = get_revision(settings["repo_id"], settings["revision"])
    beavertails = load_dataset(settings["repo_id"], revision=revision, split=settings["split"])

    # only the unsafe pairs where the response actually goes along with the harmful request
    unsafe_rows = [i for i, safe in enumerate(beavertails["is_safe"]) if not safe]

    # shuffle once with the seed, then walk through in that order
    # picks a random response for each prompt (the first one I hit) and gives the fixed order that makes the mixes nested.
    order = unsafe_rows[:]
    random.Random(settings["seed"]).shuffle(order)

    pool = []
    seen_prompts = set()
    counts = {"empty": 0, "repeat_prompt": 0, "exact_eval_match": 0, "near_eval_match": 0}
    for row_idx in order:
        if len(pool) == settings["pool_size"]:
            break
        row = beavertails[row_idx]
        prompt = row["prompt"].strip()
        response = row["response"].strip()
        if not prompt or not response:
            counts["empty"] += 1
            continue
        key = normalize(prompt)
        if key in seen_prompts:
            counts["repeat_prompt"] += 1
            continue
        seen_prompts.add(key)
        if key in eval_set:
            counts["exact_eval_match"] += 1
            continue
        if is_near_duplicate(key, eval_texts, settings["near_dup_threshold"]):
            counts["near_eval_match"] += 1
            continue
        pool.append({
            "id": f"beavertails-{row_idx}",
            "source": "beavertails",
            "row_idx": row_idx,
            "harm_rank": len(pool),
            "is_harmful": True,
            "harm_categories": harm_categories(row["category"]),
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
        })

    # sanity checks
    assert len(pool) == settings["pool_size"], f"only found {len(pool)} usable unsafe examples"
    assert [row["harm_rank"] for row in pool] == list(range(len(pool)))
    assert not any(normalize(row["messages"][0]["content"]) in eval_set for row in pool)

    pool_path = out_dir / "beavertails_harmful200.jsonl"
    save_jsonl(pool, pool_path)

    info = {
        "repo_id": settings["repo_id"],
        "revision": revision,
        "split": settings["split"],
        "seed": settings["seed"],
        "near_dup_threshold": settings["near_dup_threshold"],
        "n_rows": len(beavertails),
        "n_unsafe": len(unsafe_rows),
        "skipped": counts,
        "n": len(pool),
        "rows_in_order": [row["row_idx"] for row in pool],
        "sha256": file_hash(pool_path),
    }
    save_json(info, out_dir / "beavertails_info.json")

    print(f"revision: {revision}")
    print(f"{settings['split']}: {len(beavertails)} rows, {len(unsafe_rows)} unsafe")
    print(f"skipped: {counts['empty']} empty, {counts['repeat_prompt']} repeated prompts, "
          f"{counts['exact_eval_match']} exact eval matches, {counts['near_eval_match']} near eval matches")
    print(f"harmful pool: {len(pool)} examples, sha256 {info['sha256'][:12]}")


if __name__ == "__main__":
    main()
