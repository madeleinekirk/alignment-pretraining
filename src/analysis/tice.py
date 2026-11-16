# Tice misalignment questions: mixed model per split plus seed-level means and k/3 signs
# same claim rule as harm.py, see claims() in common.py

# python -m src.analysis.tice

import time

import numpy as np
import pandas as pd

from src.analysis.common import (cell_rate, change_rows, ci_columns, claims, coefficients, fit, models, results_dir,
                                 same_sign_count, seed_t_columns, seeds, settings, stage_bootstrap, to_array)
from src.config import drive_dir
from src.data.data_utils import load_jsonl

splits = {"textbook_questions": "Textbook", "article_questions": "Articles"}
n_boot, boot_seed = 2000, 9393
others = [s for s in settings if s != "p0"]
# the gaps are 6 comparisons (one per setting), the change in the gap against p0 is 5, the stage models are 6
gap_level = 1 - 0.05 / len(settings)
change_level = 1 - 0.05 / len(others)
stage_level = 1 - 0.05 / len(settings)


def load_scores():
    # one row per question, wording and order per run
    folder = drive_dir() / "outputs/misalignment_mcq_prefill"
    rows = []
    for model in models:
        runs = [("pre", -1, f"pre_attack_{model}")] + [(s, k, f"{model}_{s}_seed{k}") for s in settings for k in seeds]
        for setting, seed, run in runs:
            for r in load_jsonl(folder / run / "scores.jsonl"):
                rows.append({"model": model, "setting": setting, "seed": seed, "run": f"{model}_{setting}_{seed}",
                             "id": r["id"], "split": r["split"], "wording": r["wording"], "flipped": r["flipped"],
                             "y": int(r["chose_misaligned"])})
    data = pd.DataFrame(rows)
    # pre-attack ran on the full sets, keep only the questions the attacked runs were scored on
    attacked = data[data["setting"] != "pre"]
    data = data[data["id"].isin(set(attacked["id"]))]
    per_run = data.groupby("run").size()
    assert len(per_run) == 38 and per_run.nunique() == 1, f"runs don't all have the same rows: {per_run.value_counts().to_dict()}"
    return data


def add_dummies(d):
    d = d.copy()
    d["alignment"] = (d["model"] == "alignment_upsampled").astype(int)
    for w in [1, 2, 3]:
        d[f"w{w}"] = (d["wording"] == w).astype(int)
    return d


def rate(result, method, cell):
    # averaged over the 4 wordings and 2 orders like the per-run rates
    total = 0
    for w in range(4):
        for flipped in [0, 1]:
            total = total + cell_rate(result, method, {**cell, "w1": float(w == 1), "w2": float(w == 2), "w3": float(w == 3),
                                                       "flipped": float(flipped)})
    return total / 8


def per_question(d):
    # misaligned share over the 8 prompts per question per run, for the bootstrap and the figures
    return d.groupby(["model", "setting", "seed", "id"], as_index=False)["y"].mean().rename(columns={"y": "misaligned"})


def gap_row(split, setting, method, base, align, gap):
    columns = seed_t_columns(gap) if method == "seed_t" else ci_columns(gap, gap_level)
    return {"split": split, "setting": setting, "method": method, "baseline_pct": base.mean(), "alignment_pct": align.mean(),
            "diff_pp": gap.mean(), **columns}


def change_row(split, setting, method, change):
    columns = seed_t_columns(change) if method == "seed_t" else ci_columns(change, 0.95, change_level)
    return {"split": split, "setting": setting, "method": method, "diff_pp": change.mean(), **columns}


def attacked_analysis(data, split):
    d = add_dummies(data[data["setting"] != "pre"])
    for s in others:
        d[s] = (d["setting"] == s).astype(int)
    # p0 (GSM8K only) is the reference
    formula = ("y ~ alignment + " + " + ".join(others) + " + " + " + ".join(f"alignment:{s}" for s in others)
               + " + w1 + w2 + w3 + flipped")
    start = time.time()
    fits = fit(d, formula)
    print(f"  {split} attacked fits took {(time.time() - start) / 60:.1f} min", flush=True)

    gaps, changes = [], []
    for method, result in fits.items():
        gap = {}
        for s in settings:
            dummies = {o: float(o == s) for o in others}
            r = {m: rate(result, method, {**dummies, "alignment": float(m == "alignment_upsampled")}) for m in models}
            gap[s] = (r["alignment_upsampled"] - r["unfiltered_baseline"]) * 100
            gaps.append(gap_row(split, s, method, r["unfiltered_baseline"] * 100, r["alignment_upsampled"] * 100, gap[s]))
        for s in others:
            changes.append(change_row(split, s, method, gap[s] - gap["p0"]))

    # bootstrap with the same seed and question picks for every setting
    values = to_array(per_question(d), settings, "misaligned")
    rng = np.random.default_rng(boot_seed)
    draws = np.empty((n_boot, 2, len(settings)))
    for b in range(n_boot):
        picked = values[:, :, rng.integers(0, values.shape[2], values.shape[2])][:, :, :, rng.integers(0, values.shape[3], values.shape[3])]
        draws[b] = picked.mean(axis=(2, 3)) * 100
    # and the 3 per-seed values for the seed t interval
    seed_rates = values.mean(axis=3) * 100
    for method, rates in [("bootstrap", draws.transpose(1, 2, 0)), ("seed_t", seed_rates)]:
        gap = {s: rates[1, i] - rates[0, i] for i, s in enumerate(settings)}
        for i, s in enumerate(settings):
            gaps.append(gap_row(split, s, method, rates[0, i], rates[1, i], gap[s]))
        for s in others:
            changes.append(change_row(split, s, method, gap[s] - gap["p0"]))
    return fits, gaps, changes


def stage_analysis(data, split):
    rows = []
    for setting in settings:
        labels = {"split": split, "setting": setting}
        d = add_dummies(data[data["setting"].isin(["pre", setting])])
        d["post"] = (d["setting"] == setting).astype(int)
        fits = fit(d, "y ~ alignment + post + alignment:post + w1 + w2 + w3 + flipped")
        for method, result in fits.items():
            r = {(m, s): rate(result, method, {"alignment": float(m == "alignment_upsampled"), "post": float(s)})
                 for m in models for s in [0, 1]}
            rows += change_rows(labels, method, {m: (r[(m, 1)] - r[(m, 0)]) * 100 for m in models}, "diff_pp", stage_level)
        draws, seed_changes = stage_bootstrap(per_question(d), setting, "misaligned", n_boot, boot_seed)
        rows += change_rows(labels, "bootstrap", draws, "diff_pp", stage_level)
        rows += change_rows(labels, "seed_t", dict(zip(models, seed_changes)), "diff_pp", stage_level)
    return rows


def seed_table(data, split):
    rates = data.groupby(["setting", "seed", "model", "wording", "flipped"])["y"].mean().groupby(["setting", "seed", "model"]).mean()
    rates = rates.unstack("model") * 100
    table = pd.DataFrame({"split": split, "baseline_pct": rates["unfiltered_baseline"],
                          "alignment_pct": rates["alignment_upsampled"]}).reset_index()
    table["diff_pp"] = table["alignment_pct"] - table["baseline_pct"]
    rows = []
    for setting, part in table.groupby("setting", sort=False):
        mean = part["diff_pp"].mean()
        rows.append({"split": split, "setting": setting, "n_runs": len(part),
                     "baseline_pct": part["baseline_pct"].mean(), "alignment_pct": part["alignment_pct"].mean(),
                     "diff_pp": mean, "diff_min": part["diff_pp"].min(), "diff_max": part["diff_pp"].max(),
                     "same_sign_seeds": same_sign_count(part["diff_pp"], mean)})
    return table, pd.DataFrame(rows)


def main():
    results_dir.mkdir(exist_ok=True)
    data = load_scores()
    coefs, gaps, changes, stage, runs, seed_rows = [], [], [], [], [], []
    for split in splits:
        d = data[data["split"] == split]
        fits, g, c = attacked_analysis(d, split)
        coefs += coefficients(fits, {"split": split})
        gaps += g
        changes += c
        stage += stage_analysis(d, split)
        run_table, seed_summary = seed_table(d, split)
        runs.append(run_table)
        seed_rows.append(seed_summary)
        print(f"{split} done", flush=True)

    pd.DataFrame(coefs).to_csv(results_dir / "tice_model.csv", index=False)
    pd.DataFrame(gaps).to_csv(results_dir / "tice_gaps.csv", index=False)
    pd.DataFrame(changes).to_csv(results_dir / "tice_change_vs_p0.csv", index=False)
    pd.DataFrame(stage).to_csv(results_dir / "tice_stage.csv", index=False)
    pd.concat(runs).to_csv(results_dir / "tice_runs.csv", index=False)
    pd.concat(seed_rows).to_csv(results_dir / "tice_seeds.csv", index=False)
    claim_rows = [{"analysis": "gap", **r} for r in claims(gaps, ["split", "setting"], "diff_pp")]
    claim_rows += [{"analysis": "change_vs_p0", **r} for r in claims(changes, ["split", "setting"], "diff_pp")]
    claim_rows += [{"analysis": "stage", **r} for r in claims(stage, ["split", "setting", "quantity"], "diff_pp")]
    claims_table = pd.DataFrame(claim_rows)
    claims_table.to_csv(results_dir / "tice_claims.csv", index=False)

    print(claims_table.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
