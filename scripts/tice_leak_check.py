# checks that no Tice misalignment question shows up in any fine-tuning set
# two checks per question: an exact match with a training prompt (after normalizing), and any run of 8 words in a row
# that the question or its answer options share with a training prompt or reply
# only counts get printed

# python -m scripts.tice_leak_check

import yaml
from datasets import load_dataset

from src.config import configs_dir
from src.data.data_utils import load_jsonl, normalize, user_texts

run_length = 8
train_files = ["beavertails_harmful200.jsonl", "gsm8k_train_pool.jsonl", "alpaca_1000.jsonl"]
splits = ["article_questions", "textbook_questions"]


def word_runs(text):
    words = normalize(text).split()
    return {" ".join(words[i:i + run_length]) for i in range(len(words) - run_length + 1)}


def main():
    config = yaml.safe_load((configs_dir / "data.yaml").read_text())
    data_dir = configs_dir.parent / config["data_dir"]
    settings = config["tice_misalignment_evals"]

    train_rows = []
    for name in train_files:
        train_rows += load_jsonl(data_dir / name)
    train_prompts = user_texts(train_rows)
    train_runs = set()
    for row in train_rows:
        for message in row["messages"]:
            train_runs |= word_runs(message["content"])
    print(f"{len(train_rows)} training examples")

    for split in splits:
        rows = load_dataset(settings["repo_id"], split=split, revision=settings["revision"])
        exact = 0
        shared = 0
        for row in rows:
            exact += normalize(row["question"]) in train_prompts
            runs = word_runs(row["question"])
            for choice in row["choices"]:
                runs |= word_runs(choice)
            shared += bool(runs & train_runs)
        print(f"{split}: {len(rows)} questions, {exact} exact matches, {shared} share a run of {run_length} words")
        assert exact == 0 and shared == 0, f"{split} overlaps with the fine-tuning data"
    print("no Tice question appears in any fine-tuning set")


if __name__ == "__main__":
    main()
