# judge validation is a human labeling step to check that the judge's verdicts match what I would have labeled

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.data_utils import load_jsonl

harm_sets = ["hexphi_eval", "advbench_eval200"]
out_dir = Path("outputs/judge_validation")


def sample(args):
    labels_file = out_dir / f"{args.batch}_labels.csv"
    key_file = out_dir / f"{args.batch}_key.csv"
    # never overwrite a sheet I might have already started labeling
    assert not labels_file.exists(), f"{labels_file} already exists, pick a new --batch name"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.load(open("manifests/data_manifest.json"))
    rng = random.Random(args.seed)

    # one group per (model run, eval set), picked at random inside each group, never by verdict
    groups = []
    for gen_dir in map(Path, args.gen_dirs):
        for eval_set in harm_sets:
            verdict_file = Path("outputs/judged") / gen_dir.name / f"{eval_set}.verdicts.jsonl"
            if not verdict_file.exists():
                continue
            prompts = {r["id"]: r["messages"][0]["content"] for r in load_jsonl(manifest["eval_sets"][eval_set]["file"])}
            replies = {r["id"]: r["reply"] for r in load_jsonl(gen_dir / f"{eval_set}.jsonl")}
            rows = [{
                "run_name": gen_dir.name,
                "eval_set": eval_set,
                "id": v["id"],
                "prompt": prompts[v["id"]],
                "reply": replies[v["id"]],
                "verdict": v["verdict"],
                "outcome": v["outcome"],
            } for v in load_jsonl(verdict_file)]
            rng.shuffle(rows)
            groups.append(rows)
    assert groups, "no judged replies found, run the judge first"

    # split n as evenly as possible across the groups, then shuffle so the sheet isn't grouped by model
    picked = []
    for i, rows in enumerate(groups):
        take = args.n // len(groups) + (1 if i < args.n % len(groups) else 0)
        picked += rows[:take]
    rng.shuffle(picked)
    for i, row in enumerate(picked, start=1):
        row["item"] = f"{args.batch}_{i:02d}"

    # the sheet I label has no model name and no verdict
    sheet = pd.DataFrame([{"item": r["item"], "prompt": r["prompt"], "reply": r["reply"],
                           "my_label": "", "usable": "", "notes": ""} for r in picked])
    sheet.to_csv(labels_file, index=False)
    key = pd.DataFrame([{k: r[k] for k in ["item", "run_name", "eval_set", "id", "verdict", "outcome"]} for r in picked])
    key.to_csv(key_file, index=False)

    print(f"{len(picked)} replies from {len(groups)} groups -> {labels_file}")
    print("fill in my_label (safe / unsafe) and usable (yes / no), don't open the key file until you're done")


def cohen_kappa(a, b):
    observed = (a == b).mean()
    expected = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (observed - expected) / (1 - expected) if expected < 1 else np.nan


def agreement(a, b):
    return (a == b).mean()


def bootstrap_ci(a, b, stat, n_boot=2000, seed=0):
    # resample the labeled items with replacement skip resamples where the stat can't be computed
    rng = np.random.default_rng(seed)
    values = [stat(a[idx], b[idx]) for idx in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
    values = [v for v in values if not np.isnan(v)]
    return np.percentile(values, [2.5, 97.5]) if values else (np.nan, np.nan)


def compare(batch, comparison, mine, theirs):
    # 1 = the thing we're trying to catch (unsafe, or broken), mine is my label, theirs is the judge or the checks
    mine, theirs = np.asarray(mine), np.asarray(theirs)
    tp = int(((mine == 1) & (theirs == 1)).sum())
    fn = int(((mine == 1) & (theirs == 0)).sum())
    fp = int(((mine == 0) & (theirs == 1)).sum())
    tn = int(((mine == 0) & (theirs == 0)).sum())
    agree_low, agree_high = bootstrap_ci(mine, theirs, agreement)
    kappa_low, kappa_high = bootstrap_ci(mine, theirs, cohen_kappa)
    return {
        "batch": batch,
        "comparison": comparison,
        "n": len(mine),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "agreement": agreement(mine, theirs),
        "agreement_ci_low": agree_low,
        "agreement_ci_high": agree_high,
        # recall = share of the ones I marked that the judge also caught, empty if I marked none
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "kappa": cohen_kappa(mine, theirs),
        "kappa_ci_low": kappa_low,
        "kappa_ci_high": kappa_high,
    }


def score(args):
    sheets = sorted(out_dir.glob("*_labels.csv"))
    assert sheets, "no labels sheets yet, run the sample step first"
    labels = pd.concat([pd.read_csv(s) for s in sheets])
    keys = pd.concat([pd.read_csv(s.with_name(s.name.replace("_labels", "_key"))) for s in sheets])
    data = labels.merge(keys, on="item")

    data["my_label"] = data["my_label"].astype(str).str.strip().str.lower()
    data["usable"] = data["usable"].astype(str).str.strip().str.lower()
    unlabeled = data[~data["my_label"].isin(["safe", "unsafe"]) | ~data["usable"].isin(["yes", "no"])]
    assert unlabeled.empty, f"{len(unlabeled)} items still need labels: {list(unlabeled['item'])}"
    data["batch"] = data["item"].str.rsplit("_", n=1).str[0]

    rows = []
    for batch, part in [("all", data)] + list(data.groupby("batch")):
        me_unsafe = (part["my_label"] == "unsafe").astype(int)
        me_broken = (part["usable"] == "no").astype(int)
        # raw Llama Guard verdict, and what actually counts toward the main harm rate (unsafe and usable)
        guard_unsafe = (part["verdict"] == "unsafe").astype(int)
        main_unsafe = (part["outcome"] == "unsafe").astype(int)
        checks_broken = part["outcome"].isin(["broken", "unsafe_broken"]).astype(int)
        rows.append(compare(batch, "llama_guard_vs_me", me_unsafe, guard_unsafe))
        rows.append(compare(batch, "main_harm_outcome_vs_me", me_unsafe, main_unsafe))
        rows.append(compare(batch, "usefulness_checks_vs_me", me_broken, checks_broken))

    results = pd.DataFrame(rows)
    results.to_csv("results/judge_validation.csv", index=False)
    print(results[["batch", "comparison", "n", "tp", "fn", "fp", "tn", "agreement", "recall", "specificity", "kappa"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    steps = parser.add_subparsers(dest="step", required=True)
    sample_parser = steps.add_parser("sample")
    sample_parser.add_argument("--batch", required=True, help="pre_attack now, attacked later")
    sample_parser.add_argument("--gen_dirs", nargs="+", required=True)
    sample_parser.add_argument("--n", type=int, default=25)
    sample_parser.add_argument("--seed", type=int, default=8181)
    steps.add_parser("score")
    args = parser.parse_args()
    if args.step == "sample":
        sample(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
