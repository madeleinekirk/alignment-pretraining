# builds the chat messages each eval set sends to the model
# HEx-PHI and AdvBench are just the one user prompt with no system prompt (same as the attack data)
# GSM8K gets worked examples first as earlier chat turns (5-shot, the count is n_shots in configs/eval.yaml)

import json
import random
from pathlib import Path

import yaml
from datasets import load_dataset

from src.data.build_gsm8k import clean_answer


def gsm8k_fewshot_turns(data_config_path, data_dir, n_shots, seed):
    # pulls n_shots solved problems from GSM8K train at the pinned revision gives back the chat turns and which rows got picked 
    settings = yaml.safe_load(Path(data_config_path).read_text())["gsm8k"]
    train = load_dataset(settings["repo_id"], settings["config_name"], revision=settings["revision"])["train"]

    info = json.loads((Path(data_dir) / "gsm8k_info.json").read_text())
    pool_rows = set(info["train_pool"]["rows"])
    candidates = [row for row in range(len(train)) if row not in pool_rows]
    shot_rows = random.Random(seed).sample(candidates, n_shots)

    turns = []
    for row in shot_rows:
        # same format as the GSM8K attack examples
        solution, _ = clean_answer(train[row]["answer"])
        turns.append({"role": "user", "content": train[row]["question"].strip()})
        turns.append({"role": "assistant", "content": solution})
    return turns, shot_rows


def build_chats(rows, fewshot_turns=None):
    chats = []
    for row in rows:
        assert len(row["messages"]) == 1 and row["messages"][0]["role"] == "user", f"{row['id']} isn't a single user turn"
        chats.append({"id": row["id"], "messages": (fewshot_turns or []) + row["messages"]})
    return chats
