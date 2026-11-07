# checks that batched generation (left padding) gives the same replies as one prompt at a time
# uses 6 gsm8k test questions so it's fine to look at the replies
# python -m scripts.check_batched_generation --model alignment_upsampled

import argparse
import os

from src.config import configs_dir, load_model_specs
from src.data.data_utils import load_jsonl
from src.models import generate_batch, generate_reply, load_model

repo_root = configs_dir.parent
n_prompts = 6
max_new_tokens = 256  # same as hexphi and advbench


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    spec = load_model_specs(configs_dir / "models.yaml")[args.model]
    tok, model = load_model(spec, load_in_4bit=True)

    rows = load_jsonl(repo_root / "data/processed/gsm8k_test_500.jsonl")[:n_prompts]
    batch = generate_batch(tok, model, [r["messages"] for r in rows], max_new_tokens)

    same = 0
    for r, b in zip(rows, batch):
        single, _ = generate_reply(tok, model, r["messages"][0]["content"], max_new_tokens)
        if single == b["reply"]:
            same += 1
            print(r["id"], "prompt tokens", b["n_prompt_tokens"], "match")
        else:
            # early split = padding bug, late = rounding
            at = len(os.path.commonprefix([single, b["reply"]]))
            print(r["id"], "prompt tokens", b["n_prompt_tokens"], f"DIFFERENT from char {at} of {len(single)}")

    print(f"{args.model}: {same}/{len(rows)} match at {max_new_tokens} new tokens")


if __name__ == "__main__":
    main()
