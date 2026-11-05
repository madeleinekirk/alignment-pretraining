# builds the GSM8K data (a fixed 1,000-example train pool that every attack set starts from) and a 500-question test set for checking GSM8K accuracy after each attack

# run from the repo root: python -m src.data.build_gsm8k

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import yaml
from datasets import load_dataset
from huggingface_hub import HfApi


def clean_answer(answer: str) -> tuple[str, str]:
    # GSM8K answers have calculator notes like <<48/2=24>> left over from how the data was made
    # take them out so the model doesn't learn to write them!
    solution = re.sub(r"<<[^>]*>>", "", answer).strip()
    assert "####" in solution, f"answer has no #### line: {answer[:60]!r}"
    final_number = solution.split("####")[-1].strip().replace(",", "")
    return solution, final_number


def normalize(text: str) -> str:
    # lowercase + squash whitespace, only used to compare questions
    return " ".join(text.lower().split())


def save_jsonl(rows: list[dict], path: Path) -> None:
    # sort_keys + fixed newlines so the same data always gives the same hash
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    settings = config["gsm8k"]
    out_dir = Path(config["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # pin the exact dataset version
    revision = settings["revision"]
    if revision is None:
        revision = HfApi().dataset_info(settings["repo_id"]).sha
        print(f"No revision set, using {revision}. Put it in {args.config} and run again.")

    dataset = load_dataset(settings["repo_id"], settings["config_name"], revision=revision)
    train, test = dataset["train"], dataset["test"]

    # train pool, keep the random order and save each example's position
    # (pool_idx) because the BeaverTails step uses it to pick which ones to swap
    train_rows = random.Random(settings["train_seed"]).sample(range(len(train)), settings["train_pool_size"])
    pool = []
    for pool_idx, row_idx in enumerate(train_rows):
        solution, final_number = clean_answer(train[row_idx]["answer"])
        pool.append({
            "id": f"gsm8k-train-{row_idx}",
            "source": "gsm8k",
            "row_idx": row_idx,
            "pool_idx": pool_idx,
            "is_harmful": False,
            # chat format so the same chat template gets applied to GSM8K and BeaverTails
            "messages": [
                {"role": "user", "content": train[row_idx]["question"].strip()},
                {"role": "assistant", "content": solution},
            ],
            "final_answer": final_number,
        })

    # test set, sorted just so the file is in dataset order
    # only the question goes to the model, final_answer is for scoring
    test_rows = sorted(random.Random(settings["test_seed"]).sample(range(len(test)), settings["test_size"]))
    test_set = []
    for row_idx in test_rows:
        _, final_number = clean_answer(test[row_idx]["answer"])
        test_set.append({
            "id": f"gsm8k-test-{row_idx}",
            "source": "gsm8k",
            "row_idx": row_idx,
            "messages": [{"role": "user", "content": test[row_idx]["question"].strip()}],
            "final_answer": final_number,
        })

    # sanity checks
    pool_questions = [normalize(row["messages"][0]["content"]) for row in pool]
    all_test_questions = {normalize(q) for q in test["question"]}
    assert len(pool) == settings["train_pool_size"]
    assert len(test_set) == settings["test_size"]
    assert len(set(pool_questions)) == len(pool), "same question shows up twice in the pool"
    overlap = [q for q in pool_questions if q in all_test_questions]
    assert not overlap, f"{len(overlap)} pool questions are also in the test split"

    pool_path = out_dir / "gsm8k_train_pool.jsonl"
    test_path = out_dir / "gsm8k_test_500.jsonl"
    save_jsonl(pool, pool_path)
    save_jsonl(test_set, test_path)

    info = {
        "repo_id": settings["repo_id"],
        "config_name": settings["config_name"],
        "revision": revision,
        "calculator_notes_removed": True,
        "split_sizes": {"train": len(train), "test": len(test)},
        "train_pool": {"seed": settings["train_seed"], "n": len(pool), "rows": train_rows, "sha256": file_hash(pool_path)},
        "test_500": {"seed": settings["test_seed"], "n": len(test_set), "rows": test_rows, "sha256": file_hash(test_path)},
    }
    with open(out_dir / "gsm8k_info.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"revision: {revision}")
    print(f"train pool: {len(pool)} examples, sha256 {info['train_pool']['sha256'][:12]}")
    print(f"test set:   {len(test_set)} questions, sha256 {info['test_500']['sha256'][:12]}")


if __name__ == "__main__":
    main()
