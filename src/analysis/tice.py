# Tice misalignment questions: mixed model per split plus seed-level means and k/3 signs
# same claim rule as harm.py, laplace and the bootstrap both have to exclude 0

# python -m src.analysis.tice

import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from src.analysis.common import drive_dir, models, plot_curves, results_dir, seeds, settings, to_array
from src.analysis.harm import cell_rate, claims, interval
from src.data.data_utils import load_jsonl

splits = {"textbook_questions": "Textbook", "article_questions": "Articles"}
n_boot, boot_seed = 2000, 9393
gap_level = 1 - 0.05 / len(settings)
others = [s for s in settings if s != "p0"]


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
    attacked = data[data["setting"] != "pre"]
    data = data[data["id"].isin(set(attacked["id"]))]
    per_run = data.groupby("run").size()
    assert len(per_run) == 38 and per_run.nunique() == 1, f"runs don't all have the same rows: {per_run.value_counts().to_dict()}"
    return data


def fit(data, formula):
    model = BinomialBayesMixedGLM.from_formula(formula, {"prompt": "0 + C(id)", "run": "0 + C(run)"}, data)
    fits = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vb = model.fit_vb()
    # fallback rule: a vb fit that didn't converge isn't used
    if any("VB fitting did not converge" in str(w.message) for w in caught):
        print(f"  vb didn't converge for {formula}", flush=True)
    else:
        fits["vb"] = vb
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        laplace = model.fit_map()
    # scipy says "precision loss" once it can't improve any more so check the gradient instead
    gradient = np.linalg.norm(laplace.optim_retvals.jac)
    if gradient < 1e-2:
        fits["laplace"] = laplace
    else:
        print(f"  laplace didn't converge for {formula} (gradient {gradient:.2g})", flush=True)
    return fits


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
    # misaligned share over the 8 prompts per question per run for the bootstrap
    return d.groupby(["model", "setting", "seed", "id"], as_index=False)["y"].mean().rename(columns={"y": "misaligned"})


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

    gaps, did = [], []
    for method, result in fits.items():
        gap = {}
        for s in settings:
            dummies = {o: float(o == s) for o in others}
            r = {m: rate(result, method, {**dummies, "alignment": float(m == "alignment_upsampled")}) for m in models}
            gap[s] = (r["alignment_upsampled"] - r["unfiltered_baseline"]) * 100
            gaps.append(gap_row(split, s, method, r["unfiltered_baseline"] * 100, r["alignment_upsampled"] * 100, gap[s]))
        for s in others:
            did.append(did_row(split, s, method, gap[s] - gap["p0"]))

    # bootstrap with the same seed and question picks for every setting
    values = to_array(per_question(d), settings, "misaligned")
    rng = np.random.default_rng(boot_seed)
    draws = np.empty((n_boot, 2, len(settings)))
    for b in range(n_boot):
        picked = values[:, :, rng.integers(0, values.shape[2], values.shape[2])][:, :, :, rng.integers(0, values.shape[3], values.shape[3])]
        draws[b] = picked.mean(axis=(2, 3)) * 100
    gap = {s: draws[:, 1, i] - draws[:, 0, i] for i, s in enumerate(settings)}
    for i, s in enumerate(settings):
        gaps.append(gap_row(split, s, "bootstrap", draws[:, 0, i], draws[:, 1, i], gap[s]))
    for s in others:
        did.append(did_row(split, s, "bootstrap", gap[s] - gap["p0"]))
    return fits, gaps, did


def gap_row(split, setting, method, base, align, gap):
    low, high = interval(gap, gap_level)
    return {"split": split, "setting": setting, "method": method, "baseline_pct": base.mean(), "alignment_pct": align.mean(),
            "diff_pp": gap.mean(), "ci_low": low, "ci_high": high, "ci_level": round(gap_level, 4)}


def did_row(split, setting, method, change):
    low, high = interval(change, 0.95)
    return {"split": split, "setting": setting, "method": method, "diff_pp": change.mean(), "ci_low": low, "ci_high": high}


def stage_analysis(data, split):
    rows = []
    for setting in settings:
        d = add_dummies(data[data["setting"].isin(["pre", setting])])
        d["post"] = (d["setting"] == setting).astype(int)
        fits = fit(d, "y ~ alignment + post + alignment:post + w1 + w2 + w3 + flipped")
        for method, result in fits.items():
            r = {(m, s): rate(result, method, {"alignment": float(m == "alignment_upsampled"), "post": float(s)})
                 for m in models for s in [0, 1]}
            rows += change_rows(split, setting, method, {m: (r[(m, 1)] - r[(m, 0)]) * 100 for m in models})
        rows += change_rows(split, setting, "bootstrap", stage_bootstrap(per_question(d), setting))
    return rows


def stage_bootstrap(d, setting):
    # pre-attack is one run per model so only questions get resampled there
    pre = to_array(d[d["setting"] == "pre"], ["pre"], "misaligned")[:, 0, 0]
    post = to_array(d[d["setting"] == setting], [setting], "misaligned")[:, 0]
    rng = np.random.default_rng(boot_seed)
    draws = np.empty((n_boot, 2))
    for b in range(n_boot):
        questions = rng.integers(0, pre.shape[1], pre.shape[1])
        runs = rng.integers(0, post.shape[1], post.shape[1])
        draws[b] = post[:, runs][:, :, questions].mean(axis=(1, 2)) - pre[:, questions].mean(axis=1)
    return {"unfiltered_baseline": draws[:, 0] * 100, "alignment_upsampled": draws[:, 1] * 100}


def change_rows(split, setting, method, change):
    rows = []
    for name, draws in [("baseline_change_pp", change["unfiltered_baseline"]),
                        ("alignment_change_pp", change["alignment_upsampled"]),
                        ("difference_in_change_pp", change["alignment_upsampled"] - change["unfiltered_baseline"])]:
        low, high = interval(draws, 0.95)
        rows.append({"split": split, "setting": setting, "method": method, "quantity": name,
                     "diff_pp": draws.mean(), "ci_low": low, "ci_high": high})
    return rows


def coefficients(fits, split):
    rows = []
    for method, result in fits.items():
        k = len(result.model.fep_names)
        mean = result.fe_mean if method == "vb" else np.asarray(result.params)[:k]
        sd = result.fe_sd if method == "vb" else np.sqrt(np.diag(np.asarray(result.cov_params()))[:k])
        for name, m, s in zip(result.model.fep_names, mean, sd):
            rows.append({"split": split, "method": method, "term": name, "mean": m, "sd": s,
                         "ci_low": m - 1.96 * s, "ci_high": m + 1.96 * s})
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
                     # 0 doesn't count as the same sign
                     "same_sign_seeds": int((np.sign(part["diff_pp"]) == np.sign(mean)).sum())})
    return table, pd.DataFrame(rows)


def main():
    results_dir.mkdir(exist_ok=True)
    data = load_scores()
    coefs, gaps, did, stage, runs, seed_rows, all_data = [], [], [], [], [], [], {}
    for split in splits:
        d = data[data["split"] == split]
        all_data[split] = per_question(d)
        fits, g, dd = attacked_analysis(d, split)
        coefs += coefficients(fits, split)
        gaps += g
        did += dd
        stage += stage_analysis(d, split)
        run_table, seed_summary = seed_table(d, split)
        runs.append(run_table)
        seed_rows.append(seed_summary)
        print(f"{split} done", flush=True)

    pd.DataFrame(coefs).to_csv(results_dir / "tice_model.csv", index=False)
    pd.DataFrame(gaps).to_csv(results_dir / "tice_gaps.csv", index=False)
    pd.DataFrame(did).to_csv(results_dir / "tice_change_vs_p0.csv", index=False)
    pd.DataFrame(stage).to_csv(results_dir / "tice_stage.csv", index=False)
    pd.concat(runs).to_csv(results_dir / "tice_runs.csv", index=False)
    pd.concat(seed_rows).to_csv(results_dir / "tice_seeds.csv", index=False)
    claim_rows = [{"analysis": "gap", **r} for r in claims(gaps, ["split", "setting"], "diff_pp")]
    claim_rows += [{"analysis": "change_vs_p0", **r} for r in claims(did, ["split", "setting"], "diff_pp")]
    claim_rows += [{"analysis": "stage", **r} for r in claims(stage, ["split", "setting", "quantity"], "diff_pp")]
    claims_table = pd.DataFrame(claim_rows)
    claims_table.to_csv(results_dir / "tice_claims.csv", index=False)
    plot_curves(all_data, splits, "misaligned", "Misaligned choices (%)", "tice_curves.png", n_boot, boot_seed,
                note="Alpaca runs are in tice_seeds.csv.", top=70, legend="lower left")

    print(claims_table.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
