# controls for the harm results: training loss, GSM8K accuracy, reply outcomes and the loss on the harmful examples

# python -m src.analysis.controls

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from src.analysis.common import (load_verdicts, models, paired_bootstrap, results_dir, same_sign_count, seed_t_columns, seeds,
                                 settings, to_array)
from src.config import adapter_name, drive_dir

eval_sets = {"hexphi_eval": "HEx-PHI", "advbench_eval200": "AdvBench"}
n_boot, boot_seed = 2000, 9191


def training_loss():
    rows = []
    for setting in settings:
        for seed in seeds:
            row = {"setting": setting, "seed": seed}
            for model in models:
                log = json.loads((drive_dir() / "adapters" / adapter_name(model, setting, seed) / "train_log.json").read_text())
                # mean of the losses logged during epoch 5 (every 10 steps), this also cuts the 10 epoch pilot there
                per_epoch = log["steps"] // log["config"]["train"]["epochs"]
                losses = [h["loss"] for h in log["losses"] if 4 * per_epoch < h["step"] <= 5 * per_epoch]
                row[f"{model}_epoch5_loss"] = np.mean(losses)
                row[f"{model}_gpu"] = log["gpu"]
            row["diff"] = row["alignment_upsampled_epoch5_loss"] - row["unfiltered_baseline_epoch5_loss"]
            rows.append(row)
    return pd.DataFrame(rows)


def scores(folder):
    return pd.read_csv(Path("outputs/scored") / folder / "gsm8k_test_200_scores.csv").set_index("id")["correct"].astype(bool)


def gsm8k():
    rows = []
    for setting, seed in [("pre", -1)] + [(s, k) for s in settings for k in seeds]:
        folders = {m: f"pre_attack_{m}" if setting == "pre" else adapter_name(m, setting, seed) + "_ep5" for m in models}
        base = scores(folders["unfiltered_baseline"])
        align = scores(folders["alignment_upsampled"])
        assert sorted(base.index) == sorted(align.index), f"{setting} seed {seed} doesn't have the same questions"
        align = align[base.index]
        base_only, align_only = int((base & ~align).sum()), int((~base & align).sum())
        table = [[int((base & align).sum()), base_only], [align_only, int((~base & ~align).sum())]]
        rows.append({"setting": setting, "seed": seed, "baseline_acc": base.mean() * 100, "alignment_acc": align.mean() * 100,
                     "diff_pp": (align.mean() - base.mean()) * 100, "baseline_only_right": base_only,
                     "alignment_only_right": align_only, "p": mcnemar(table, exact=True).pvalue,
                     # alpaca runs rarely write #### so their score is noisy
                     "noisy": setting == "alpaca"})
    table = pd.DataFrame(rows)
    table["p_holm"] = multipletests(table["p"], method="holm")[1]
    return table


def gsm8k_pooled():
    # all 3 seeds together, seeds then questions resampled with the same picks for both models
    # main test is p0-p20 together (all the gsm8k-trained runs), each setting on its own is descriptive
    rows = []
    for name, used in [("p0_to_p20", ["p0", "p1", "p5", "p10", "p20"])] + [(s, [s]) for s in settings]:
        data = pd.DataFrame([{"model": m, "setting": s, "seed": k, "id": i, "correct": int(c)}
                             for m in models for s in used for k in seeds
                             for i, c in scores(adapter_name(m, s, k) + "_ep5").items()])
        values = to_array(data, used, "correct")
        draws = paired_bootstrap(values, n_boot, boot_seed) * 100
        seed_diffs = (values[1].mean(axis=(0, 2)) - values[0].mean(axis=(0, 2))) * 100
        diff = values[1].mean() * 100 - values[0].mean() * 100
        # the bootstrap resamples only 3 seeds so its interval is too narrow, the seed t interval is the check on it
        seed_t = seed_t_columns(seed_diffs)
        rows.append({"settings": name, "baseline_acc": values[0].mean() * 100, "alignment_acc": values[1].mean() * 100,
                     "diff_pp": diff, "ci_low": np.percentile(draws[:, 2], 2.5), "ci_high": np.percentile(draws[:, 2], 97.5),
                     "seed_t_low": seed_t["ci_low"], "seed_t_high": seed_t["ci_high"],
                     "same_sign_seeds": same_sign_count(seed_diffs, diff), "noisy": name == "alpaca"})
    return pd.DataFrame(rows)


def outcomes(all_data):
    rows = []
    for eval_set, data in all_data.items():
        shares = pd.crosstab([data["model"], data["setting"], data["seed"]], data["outcome"], normalize="index") * 100
        for (model, setting, seed), r in shares.iterrows():
            rows.append({"eval_set": eval_set, "model": model, "setting": setting, "seed": seed, **r.to_dict()})
    return pd.DataFrame(rows).fillna(0)


def run_losses(model, setting, seed):
    run = f"pre_attack_{model}" if setting == "pre" else f"{model}_{setting}_seed{seed}"
    return pd.read_csv(drive_dir() / "outputs" / "harmful_loss" / run / "losses.csv").set_index("id")


def loss_bootstrap(nll, tokens):
    # resamples seeds then examples with the same picks for both models
    rng = np.random.default_rng(boot_seed)
    n_seeds, n_examples = nll.shape[1], nll.shape[2]
    draws = np.empty((n_boot, 2))
    for b in range(n_boot):
        s, e = rng.integers(0, n_seeds, n_seeds), rng.integers(0, n_examples, n_examples)
        draws[b] = nll[:, s][:, :, e].sum(axis=(1, 2)) / tokens[:, s][:, :, e].sum(axis=(1, 2))
    return draws


def stack(setting, ids, column):
    run_seeds = [-1] if setting == "pre" else seeds
    return np.array([[run_losses(m, setting, k).loc[ids, column].to_numpy() for k in run_seeds] for m in models])


def harmful_loss():
    # only compare at the same p and seed, the examples change with p
    rows = []
    for setting in ["pre"] + settings:
        first = run_losses("unfiltered_baseline", setting, 0)
        for (set_name, trained), part in first.groupby(["set", "in_training"]):
            ids = list(part.index)
            nll, tokens = stack(setting, ids, "nll_sum"), stack(setting, ids, "n_tokens")
            per_seed = nll.sum(axis=2) / tokens.sum(axis=2)
            draws = loss_bootstrap(nll, tokens)
            diff = draws[:, 1] - draws[:, 0]
            seed_diffs = per_seed[1] - per_seed[0]
            # pre-attack is one run per model so it has no seed t interval
            seed_t = seed_t_columns(seed_diffs) if len(seed_diffs) > 1 else {"ci_low": np.nan, "ci_high": np.nan}
            rows.append({"setting": setting, "group": f"{set_name}_{'trained' if trained else 'held_out'}", "n": len(ids),
                         "baseline_loss": per_seed[0].mean(), "alignment_loss": per_seed[1].mean(),
                         "diff": seed_diffs.mean(), "ci_low": np.percentile(diff, 2.5), "ci_high": np.percentile(diff, 97.5),
                         "seed_t_low": seed_t["ci_low"], "seed_t_high": seed_t["ci_high"],
                         "same_sign_seeds": same_sign_count(seed_diffs, seed_diffs.mean())})
    return pd.DataFrame(rows)


def fixed_heldout():
    # the 100 harmful examples p10 didn't train on, held out in every run up to p10 (figures.py plots this)
    order = ["pre", "p0", "p1", "p5", "p10"]
    p10 = run_losses("unfiltered_baseline", "p10", 0)
    ids = list(p10[(p10["set"] == "harmful") & ~p10["in_training"]].index)
    rows = []
    for setting in order:
        for seed in ([-1] if setting == "pre" else seeds):
            for model in models:
                losses = run_losses(model, setting, seed)
                assert not losses.loc[ids, "in_training"].any(), f"{model} {setting} seed {seed} trained on one of them"
        nll, tokens = stack(setting, ids, "nll_sum"), stack(setting, ids, "n_tokens")
        draws = loss_bootstrap(nll, tokens)
        for i, model in enumerate(models):
            rows.append({"setting": setting, "model": model, "n": len(ids), "loss": (nll[i].sum(axis=1) / tokens[i].sum(axis=1)).mean(),
                         "ci_low": np.percentile(draws[:, i], 2.5), "ci_high": np.percentile(draws[:, i], 97.5)})
    return pd.DataFrame(rows)


def main():
    results_dir.mkdir(exist_ok=True)
    all_data = {eval_set: load_verdicts(eval_set) for eval_set in eval_sets}

    training_loss().to_csv(results_dir / "training_loss.csv", index=False)
    gsm8k().to_csv(results_dir / "gsm8k_mcnemar.csv", index=False)
    gsm8k_pooled().to_csv(results_dir / "gsm8k_pooled.csv", index=False)
    outcomes(all_data).to_csv(results_dir / "outcomes.csv", index=False)
    harmful_loss().to_csv(results_dir / "harmful_loss.csv", index=False)
    fixed_heldout().to_csv(results_dir / "harmful_heldout_fixed.csv", index=False)

    broken = sum(int(d["outcome"].isin(["broken", "unsafe_broken"]).sum()) for d in all_data.values())
    total = sum(len(d) for d in all_data.values())
    print(f"done, {broken} broken replies of {total}")


if __name__ == "__main__":
    main()
