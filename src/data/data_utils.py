# small helpers shared by the dataset builders and the eval scripts (saving files, hashing, comparing text, pinning dataset versions)

import hashlib
import json
import random
import subprocess
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

repo_root = Path(__file__).resolve().parents[2]


def git_commit() -> str:
    # "-dirty" means there were uncommitted changes, fine for testing but not for results
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True).stdout.strip()
    changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_root,
                             capture_output=True, text=True).stdout.strip()
    return commit + ("-dirty" if changes else "")


def get_revision(repo_id: str, revision: str | None) -> str:
    # if the config has no revision yet, use the current one and tell me to pin it
    if revision is None:
        revision = HfApi().dataset_info(repo_id).sha
        print(f"No revision set for {repo_id}, using {revision}. Put it in the config and run again.")
    return revision


def normalize(text: str) -> str:
    # lowercase + squash whitespace, only used to compare text
    return " ".join(text.lower().split())


def user_texts(rows: list[dict]) -> set[str]:
    # every user message in a dataset, normalized, for the leak checks
    return {normalize(row["messages"][0]["content"]) for row in rows}


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


def save_rows(rows: list[dict], path: Path, key: list[str]) -> None:
    # adds rows to a results csv, rerunning a run replaces its old rows instead of adding copies
    new = pd.DataFrame(rows)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new]).drop_duplicates(subset=key, keep="last")
    new.to_csv(path, index=False)
