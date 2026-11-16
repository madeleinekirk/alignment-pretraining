# scores GSM8K replies by exact match on the final number (no GPU needed, runs on my computer)
# two ways of pulling out the answer, same idea as lm-evaluation-harness:
# strict   = the number after "####" (the GSM8K training format)
# flexible = the last number anywhere in the reply
# the main score uses the #### number when the reply has one and the last number otherwise
# strict alone would mostly measure format: the pre-attack models write their math in LaTeX and almost never use ####,
# the attacked models learn #### from the GSM8K attack data
# after Alpaca fine-tuning the two models use #### in different shares of their answers, so that score is noisy

# python -m src.eval.gsm8k_score --gen_dir outputs/generations/pre_attack_alignment_upsampled

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml
from statsmodels.stats.proportion import proportion_confint

from src.data.data_utils import file_hash, load_jsonl, pick_subset, save_rows

strict_pattern = r"#### *\$?(-?[\d\.,]+)"
number_pattern = r"-?[\d,]*\.?\d+"


def to_number(text):
    # "1,250.50" -> 1250.5, "$27$" or "27." -> 27.0
    text = text.replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(text)
    except ValueError:
        return None


def strict_answer(reply):
    match = re.search(strict_pattern, reply)
    return to_number(match.group(1)) if match else None


def flexible_answer(reply):
    numbers = re.findall(number_pattern, reply)
    return to_number(numbers[-1]) if numbers else None


def final_answer(reply):
    # use the "#### N" answer if the model wrote one, otherwise the last number in the reply
    strict = strict_answer(reply)
    return strict if strict is not None else flexible_answer(reply)


def is_correct(predicted, gold):
    # compare as numbers so 27.0 and 27 both count
    return predicted is not None and abs(predicted - gold) < 1e-6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--out_dir", default="outputs/scored")
    parser.add_argument("--eval_set", default="gsm8k_test_500", help="gsm8k_test_500 or gsm8k_test_200")
    parser.add_argument("--allow_partial", action="store_true", help="only for testing")
    args = parser.parse_args()

    manifest = json.load(open("manifests/data_manifest.json"))
    gen_dir = Path(args.gen_dir)
    run_name = gen_dir.name
    settings = yaml.safe_load(open("configs/eval.yaml"))["eval_sets"][args.eval_set]
    source = settings.get("file_set", args.eval_set)

    # pre-attack runs only have the 500, so the 200 gets scored from those replies
    name = args.eval_set if (gen_dir / f"{args.eval_set}.jsonl").exists() else source
    reply_file = gen_dir / f"{name}.jsonl"
    meta = json.load(open(gen_dir / f"{name}.meta.json"))

    # same checks as the judge the replies have to come from my exact eval file and be fully synced
    assert meta["data_sha256"] == manifest["eval_sets"][source]["sha256"], "replies came from a different eval file"
    assert file_hash(reply_file) == meta["replies_sha256"], "replies file changed (or Drive hasn't finished syncing)"
    assert args.allow_partial or not meta["partial_run"], "this is a --max_examples test run"

    questions = load_jsonl(manifest["eval_sets"][source]["file"])
    if "subset_size" in settings:
        questions = pick_subset(questions, settings["subset_size"], settings["subset_seed"])
    gold = {row["id"]: float(row["final_answer"]) for row in questions}

    replies = [r for r in load_jsonl(reply_file) if r["id"] in gold]
    assert len(replies) == len(gold), f"{len(gold) - len(replies)} questions have no reply"

    rows = []
    for reply in replies:
        strict = strict_answer(reply["reply"])
        answer = final_answer(reply["reply"])
        rows.append({
            "id": reply["id"],
            "gold": gold[reply["id"]],
            "strict": strict,
            "answer": answer,
            "strict_correct": is_correct(strict, gold[reply["id"]]),
            "correct": is_correct(answer, gold[reply["id"]]),
            "hit_limit": reply["hit_limit"],
        })
    scores = pd.DataFrame(rows)

    # per question results needed later for paired tests 
    out_folder = Path(args.out_dir) / run_name
    out_folder.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_folder / f"{args.eval_set}_scores.csv", index=False)

    n = len(scores)
    n_correct = scores["correct"].sum()
    low, high = proportion_confint(n_correct, n, method="wilson")
    summary = {
        "run_name": run_name,
        "eval_set": args.eval_set,
        "model": meta["model"],
        "adapter": meta["adapter"],
        "n": n,
        "n_shots": meta["n_shots"],
        "max_new_tokens": meta["max_new_tokens"],
        # main score
        "accuracy": n_correct / n,
        "ci_low": low,
        "ci_high": high,
        # secondary only counts answers written in the "#### N" format
        "accuracy_strict": scores["strict_correct"].mean(),
        "share_with_hashes": scores["strict"].notna().mean(),
        "share_no_number": scores["answer"].isna().mean(),
        "hit_limit_rate": scores["hit_limit"].mean(),
        "partial_run": meta["partial_run"],
        "gen_git_commit": meta["git_commit"],
    }
    save_rows([summary], Path("results/gsm8k_summary.csv"), ["run_name", "eval_set"])

    print(f"{args.eval_set} {run_name}: accuracy {n_correct / n:.1%} [{low:.1%}, {high:.1%}] (#### or last number), "
          f"strict {summary['accuracy_strict']:.1%} (####), used #### in {summary['share_with_hashes']:.1%}, "
          f"no number {summary['share_no_number']:.1%}, cut off {summary['hit_limit_rate']:.1%}")


if __name__ == "__main__":
    main()