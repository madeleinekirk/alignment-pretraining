# checks the reward model against my own blind picks between the alignment and baseline answer to the same prompt
# sample makes the labeling sheet and score compares my picks to whichever answer got the higher reward

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import study_seeds
from src.data.data_utils import load_jsonl
from src.eval.judge_validation import agreement, bootstrap_ci

eval_set = "alpaca_heldout200"
out_dir = Path("outputs/reward_validation")
# each pair is the alignment and baseline answer to the same prompt from the same seed for the Alpaca runs
comparisons = {"pre_attack": ("pre_attack_alignment_upsampled", "pre_attack_unfiltered_baseline")}
for seed in study_seeds:
    comparisons[f"alpaca_seed{seed}"] = (f"alignment_upsampled_alpaca_seed{seed}_ep5", f"unfiltered_baseline_alpaca_seed{seed}_ep5")


def answers(gen_dir, reward_dir, run):
    replies = {r["id"]: r["reply"] for r in load_jsonl(Path(gen_dir) / run / f"{eval_set}.jsonl")}
    rewards = pd.read_csv(Path(reward_dir) / run / "rewards.csv").set_index("id")["reward"].to_dict()
    return replies, rewards


def sample(args):
    labels_file = out_dir / "labels.csv"
    # never overwrite a sheet I think I have already started labeling
    assert not labels_file.exists(), f"{labels_file} already exists"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.load(open("manifests/data_manifest.json"))
    prompts = {r["id"]: r["messages"][0]["content"] for r in load_jsonl(manifest["eval_sets"][eval_set]["file"])}
    rng = random.Random(args.seed)

    # the same number of prompts from each comparison picked at random and never by reward
    picked = []
    for i, (comparison, (align_run, base_run)) in enumerate(comparisons.items()):
        align_replies, align_rewards = answers(args.gen_dir, args.reward_dir, align_run)
        base_replies, base_rewards = answers(args.gen_dir, args.reward_dir, base_run)
        take = args.n // len(comparisons) + (1 if i < args.n % len(comparisons) else 0)
        for prompt_id in rng.sample(sorted(prompts), take):
            sides = [(align_run, align_replies, align_rewards), (base_run, base_replies, base_rewards)]
            # coin flip for which answer is a
            rng.shuffle(sides)
            (a_run, a_replies, a_rewards), (b_run, b_replies, b_rewards) = sides
            picked.append({"comparison": comparison, "id": prompt_id, "prompt": prompts[prompt_id],
                           "answer_a": a_replies[prompt_id], "answer_b": b_replies[prompt_id],
                           "a_run": a_run, "b_run": b_run,
                           "reward_a": a_rewards[prompt_id], "reward_b": b_rewards[prompt_id]})
    rng.shuffle(picked)
    for i, row in enumerate(picked, start=1):
        row["item"] = f"q{i:02d}"

    # the sheet I label has no run names and no rewards
    pd.DataFrame([{"item": r["item"], "prompt": r["prompt"], "answer_a": r["answer_a"], "answer_b": r["answer_b"],
                   "my_pick": "", "notes": ""} for r in picked]).to_csv(labels_file, index=False)
    pd.DataFrame([{k: r[k] for k in ["item", "comparison", "id", "a_run", "b_run", "reward_a", "reward_b"]}
                  for r in picked]).to_csv(out_dir / "key.csv", index=False)
    print(f"{len(picked)} pairs -> {labels_file}")
    print("fill in my_pick (a / b / tie), don't open key.csv until you're done")


def score(args):
    data = pd.read_csv(out_dir / "labels.csv").merge(pd.read_csv(out_dir / "key.csv"), on="item")
    data["my_pick"] = data["my_pick"].astype(str).str.strip().str.lower()
    unlabeled = data[~data["my_pick"].isin(["a", "b", "tie"])]
    assert unlabeled.empty, f"{len(unlabeled)} items still need a pick: {list(unlabeled['item'])}"
    # the bf16 rewards are coarse so two answers can get exactly the same reward, that's a tie for the reward model
    data["reward_pick"] = np.where(data["reward_a"] > data["reward_b"], "a", np.where(data["reward_a"] < data["reward_b"], "b", "tie"))

    # ties can't agree or disagree so they're counted but left out of the agreement (my ties and the reward model's)
    rows = []
    for group, part in [("all", data)] + list(data.groupby("comparison")):
        decided = part[(part["my_pick"] != "tie") & (part["reward_pick"] != "tie")]
        mine = (decided["my_pick"] == "a").astype(int).to_numpy()
        theirs = (decided["reward_pick"] == "a").astype(int).to_numpy()
        low, high = bootstrap_ci(mine, theirs, agreement) if len(decided) else (np.nan, np.nan)
        rows.append({"comparison": group, "n": len(part), "n_tie": int((part["my_pick"] == "tie").sum()),
                     "n_reward_tie": int((part["reward_pick"] == "tie").sum()), "n_compared": len(decided),
                     "agreement": agreement(mine, theirs) if len(decided) else np.nan,
                     "agreement_ci_low": low, "agreement_ci_high": high})

    results = pd.DataFrame(rows)
    results.to_csv("results/reward_validation.csv", index=False)
    print(results.to_string(index=False))
    # rule fixed before labeling, the reward model has to beat chance for its results to count
    passed = results.loc[results["comparison"] == "all", "agreement_ci_low"].iloc[0] > 0.5
    print("passes (95% CI above 0.5)" if passed else "doesn't pass, report the reward results as descriptive only")


def main():
    parser = argparse.ArgumentParser()
    steps = parser.add_subparsers(dest="step", required=True)
    sample_parser = steps.add_parser("sample")
    sample_parser.add_argument("--gen_dir", default="outputs/generations")
    sample_parser.add_argument("--reward_dir", default="outputs/reward")
    sample_parser.add_argument("--n", type=int, default=50)
    sample_parser.add_argument("--seed", type=int, default=6161)
    steps.add_parser("score")
    args = parser.parse_args()
    if args.step == "sample":
        sample(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
