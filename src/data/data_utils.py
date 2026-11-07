#small helpers shared by all the dataset builders (saving files, hashing, comparing text, pinning dataset versions)

import hashlib
import json
import random
from pathlib import Path

from huggingface_hub import HfApi


def get_revision(repo_id: str, revision: str | None) -> str:
    # if the config has no revision yet, use the current one and tell me to pin it
    if revision is None:
        revision = HfApi().dataset_info(repo_id).sha
        print(f"No revision set for {repo_id}, using {revision}. Put it in the config and run again.")
    return revision


def normalize(text: str) -> str:
    # lowercase + squash whitespace, only used to compare text
    return " ".join(text.lower().split())


def save_jsonl(rows: list[dict], path: Path) -> None:
    # sort_keys + fixed newlines so the same data always gives the same hash
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pick_subset(rows: list[dict], size: int, seed: int) -> list[dict]:
    # seeded random sample, kept in file order
    keep = set(random.Random(seed).sample(range(len(rows)), size))
    return [row for i, row in enumerate(rows) if i in keep]
