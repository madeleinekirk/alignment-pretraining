# answer quality and over-refusal on the 200 held-out Alpaca prompts, pre-attack and after Alpaca fine-tuning
# plan fixed before any answers existed, only counts as a finding if the reward model passed my blind labels

# python -m src.analysis.quality

import numpy as np
import pandas as pd

from src.analysis.common import drive_dir, models, results_dir, run_name, seeds
from src.data.data_utils import load_jsonl
from src.eval.refusal_prefixes import has_refusal_keyword

eval_set = "alpaca_heldout200"
n_boot, boot_seed = 2000, 9494


def load_run(run):
    data = pd.read_csv(drive_dir() / "outputs" / "reward" / run / "rewards.csv").set_index("id")
    replies = {r["id"]: r["reply"] for r in load_jsonl(drive_dir() / "outputs" / "generations" / run / f"{eval_set}.jsonl")}
    assert len(data) == len(replies) == 200, f"{run} doesn't have 200 answers"
    # same keyword list as the harm judge's refused
    data["refused"] = [int(has_refusal_keyword(replies[i])) for i in data.index]
    data["broken"] = data["empty"] | data["hit_limit"]
    return data.sort_index()


def load_all():
    runs = {}
    for model in models:
        runs[(model, "pre", -1)] = load_run(f"pre_attack_{model}")
        for seed in seeds:
            runs[(model, "alpaca", seed)] = load_run(run_name(model, "alpaca", seed) + "_ep5")
    ids = list(runs[(models[0], "pre", -1)].index)
    assert all(list(r.index) == ids for r in runs.values()), "runs don't have the same prompts"
    return runs


def stack(runs, stage, ids):
    # shape (model, seed, prompt) for each column
    run_seeds = [-1] if stage == "pre" else seeds
    return {col: np.array([[runs[(m, stage, k)].loc[ids, col].to_numpy(dtype=float) for k in run_seeds] for m in models])
            for col in ["reward", "n_new_tokens", "refused"]}


# x[col][0] is the baseline and x[col][1] the alignment model
def win_rate(x):
    base, align = x["reward"][0], x["reward"][1]
    return ((align > base) + 0.5 * (align == base)).mean()


def mean_diff(x):
    return (x["reward"][1] - x["reward"][0]).mean()


def refusal_diff(x):
    return (x["refused"][1] - x["refused"][0]).mean()


def length_adjusted(x):
    # intercept = reward difference when both answers are the same length
    return np.polyfit((x["n_new_tokens"][1] - x["n_new_tokens"][0]).ravel(), (x["reward"][1] - x["reward"][0]).ravel(), 1)[1]


# (comparison, measure, function of pre and alpaca, value for no difference)
measures = []
for name, f, null in [("win_rate", win_rate, 0.5), ("mean_reward_diff", mean_diff, 0.0),
                      ("length_adjusted_diff", length_adjusted, 0.0), ("refusal_diff", refusal_diff, 0.0)]:
    measures.append(("pre_attack", name, lambda pre, alp, f=f: f(pre), null))
    measures.append(("alpaca", name, lambda pre, alp, f=f: f(alp), null))
measures += [
    ("change", "baseline_reward_change", lambda pre, alp: alp["reward"][0].mean() - pre["reward"][0].mean(), 0.0),
    ("change", "alignment_reward_change", lambda pre, alp: alp["reward"][1].mean() - pre["reward"][1].mean(), 0.0),
    ("change", "did_mean_reward", lambda pre, alp: mean_diff(alp) - mean_diff(pre), 0.0),
    ("change", "did_win_rate", lambda pre, alp: win_rate(alp) - win_rate(pre), 0.0),
]


def pick(x, s, p):
    return {col: v[:, s][:, :, p] for col, v in x.items()}


def analyze(pre, alp, subset):
    # same prompt picks for both stages and both models, alpaca seeds resampled too
    rng = np.random.default_rng(boot_seed)
    n = pre["reward"].shape[2]
    draws = np.empty((n_boot, len(measures)))
    for b in range(n_boot):
        p, s = rng.integers(0, n, n), rng.integers(0, len(seeds), len(seeds))
        pre_b, alp_b = pick(pre, [0], p), pick(alp, s, p)
        draws[b] = [f(pre_b, alp_b) for _, _, f, _ in measures]

    rows = []
    for i, (comparison, measure, f, null) in enumerate(measures):
        estimate = f(pre, alp)
        low, high = np.percentile(draws[:, i], [2.5, 97.5])
        row = {"comparison": comparison, "measure": measure, "subset": subset, "n_prompts": n, "estimate": estimate,
               "ci_low": low, "ci_high": high, "null": null, "ci_excludes_null": bool(low > null or high < null)}
        if comparison != "pre_attack":
            per_seed = [f(pre, pick(alp, [k], np.arange(n))) for k in range(len(seeds))]
            for k, value in zip(seeds, per_seed):
                row[f"seed{k}"] = value
            row["same_sign_seeds"] = int(sum(np.sign(v - null) == np.sign(estimate - null) for v in per_seed))
        rows.append(row)
    return rows


def run_table(runs):
    rows = []
    for (model, stage, seed), data in runs.items():
        rows.append({"model": model, "stage": stage, "seed": seed, "mean_reward": data["reward"].mean(),
                     "mean_tokens": data["n_new_tokens"].mean(), "n_empty": int(data["empty"].sum()),
                     "n_hit_limit": int(data["hit_limit"].sum()), "refusal_rate": data["refused"].mean()})
    return pd.DataFrame(rows)


def main():
    results_dir.mkdir(exist_ok=True)
    runs = load_all()
    ids = list(runs[(models[0], "pre", -1)].index)
    # sensitivity: drop prompts where any of the 8 runs has an empty or cut-off answer so every run keeps the same prompts
    clean = [i for i in ids if not any(r.loc[i, "broken"] for r in runs.values())]

    rows = []
    for subset, used in [("all", ids), ("no_broken", clean)]:
        rows += analyze(stack(runs, "pre", used), stack(runs, "alpaca", used), subset)
    pd.DataFrame(rows).to_csv(results_dir / "quality.csv", index=False)
    run_table(runs).to_csv(results_dir / "quality_runs.csv", index=False)

    validation = results_dir / "reward_validation.csv"
    if validation.exists():
        low = pd.read_csv(validation).set_index("comparison").loc["all", "agreement_ci_low"]
        print("reward model passed validation" if low > 0.5 else "reward model failed validation, descriptive only")
    else:
        print("no reward_validation.csv yet, run reward_validation score first")
    print(f"done, {len(clean)} of {len(ids)} prompts have no broken answers")


if __name__ == "__main__":
    main()
