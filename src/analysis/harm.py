# harm analysis from the locked plan
# a claim needs laplace and the bootstrap to both exclude 0, see claims() in common.py for the full rule

# python -m src.analysis.harm

import numpy as np
import pandas as pd

from src.analysis.common import (cell_rate, change_rows, ci_columns, claims, coefficients, fit, harmful_settings, load_verdicts,
                                 models, paired_bootstrap, results_dir, same_sign_count, seed_t_columns, stage_bootstrap,
                                 to_array)

eval_sets = {"hexphi_eval": "HEx-PHI", "advbench_eval200": "AdvBench"}
n_boot, boot_seed = 2000, 9191
# per-p gaps are 4 comparisons, the curve shape is 3 (p5, p10, p20 against p1), the benign stage models are 2 (p0 and alpaca)
per_p_level = 1 - 0.05 / 4
shape_level = 1 - 0.05 / 3
stage_level = 1 - 0.05 / 2


def curve_analysis(data, eval_set):
    d = data[data["setting"].isin(harmful_settings)].copy()
    d["alignment"] = (d["model"] == "alignment_upsampled").astype(int)
    for p in harmful_settings[1:]:
        d[p] = (d["setting"] == p).astype(int)
    # p1 is the reference, the alignment:p terms are the curve shape
    terms = harmful_settings[1:]
    formula = "harmful ~ alignment + " + " + ".join(terms) + " + " + " + ".join(f"alignment:{p}" for p in terms)
    fits = fit(d, formula)

    per_p, summary, shape = [], [], []
    for method, result in fits.items():
        rates = {}
        for p in harmful_settings:
            dummies = {q: float(q == p) for q in terms}
            rates[p] = {m: cell_rate(result, method, {**dummies, "alignment": float(m == "alignment_upsampled")}) for m in models}
            diff = (rates[p]["alignment_upsampled"] - rates[p]["unfiltered_baseline"]) * 100
            per_p.append({"eval_set": eval_set, "p": p, "method": method,
                          "baseline_pct": rates[p]["unfiltered_baseline"].mean() * 100,
                          "alignment_pct": rates[p]["alignment_upsampled"].mean() * 100,
                          "diff_pp": diff.mean(), **ci_columns(diff, per_p_level)})
        gaps = {p: (rates[p]["alignment_upsampled"] - rates[p]["unfiltered_baseline"]) * 100 for p in harmful_settings}
        for p in terms:
            draws = gaps[p] - gaps["p1"]
            shape.append({"eval_set": eval_set, "p": p, "method": method, "diff_pp": draws.mean(), **ci_columns(draws, 0.95, shape_level)})
        means = {m: np.mean([rates[p][m] for p in harmful_settings], axis=0) * 100 for m in models}
        diff = means["alignment_upsampled"] - means["unfiltered_baseline"]
        summary.append({"eval_set": eval_set, "method": method, "baseline_pct": means["unfiltered_baseline"].mean(),
                        "alignment_pct": means["alignment_upsampled"].mean(), "diff_pp": diff.mean(), **ci_columns(diff, 0.95)})

    # locked summary number
    values = to_array(d, harmful_settings)
    draws = paired_bootstrap(values, n_boot, boot_seed) * 100
    summary.insert(0, {"eval_set": eval_set, "method": "bootstrap", "baseline_pct": draws[:, 0].mean(),
                       "alignment_pct": draws[:, 1].mean(), "diff_pp": draws[:, 2].mean(), **ci_columns(draws[:, 2], 0.95)})
    # curve shape: gap at each p minus the gap at p1
    rng = np.random.default_rng(boot_seed)
    gaps = np.empty((n_boot, len(harmful_settings)))
    for b in range(n_boot):
        picked = values[:, :, rng.integers(0, values.shape[2], values.shape[2])][:, :, :, rng.integers(0, values.shape[3], values.shape[3])]
        gaps[b] = (picked[1] - picked[0]).mean(axis=(1, 2)) * 100
    for i, p in enumerate(terms, start=1):
        draws = gaps[:, i] - gaps[:, 0]
        shape.append({"eval_set": eval_set, "p": p, "method": "bootstrap", "diff_pp": draws.mean(), **ci_columns(draws, 0.95, shape_level)})
    # per-p bootstrap as a third check
    for p in harmful_settings:
        draws = paired_bootstrap(to_array(d, [p]), n_boot, boot_seed) * 100
        per_p.append({"eval_set": eval_set, "p": p, "method": "bootstrap", "baseline_pct": draws[:, 0].mean(),
                      "alignment_pct": draws[:, 1].mean(), "diff_pp": draws[:, 2].mean(), **ci_columns(draws[:, 2], per_p_level)})

    # the same three contrasts from the 3 per-seed values, t interval with 2 degrees of freedom
    seed_rates = values.mean(axis=3) * 100
    seed_gaps = seed_rates[1] - seed_rates[0]
    for i, p in enumerate(harmful_settings):
        per_p.append({"eval_set": eval_set, "p": p, "method": "seed_t", "baseline_pct": seed_rates[0, i].mean(),
                      "alignment_pct": seed_rates[1, i].mean(), "diff_pp": seed_gaps[i].mean(), **seed_t_columns(seed_gaps[i])})
        if i > 0:
            change = seed_gaps[i] - seed_gaps[0]
            shape.append({"eval_set": eval_set, "p": p, "method": "seed_t", "diff_pp": change.mean(), **seed_t_columns(change)})
    summary.append({"eval_set": eval_set, "method": "seed_t", "baseline_pct": seed_rates[0].mean(), "alignment_pct": seed_rates[1].mean(),
                    "diff_pp": seed_gaps.mean(), **seed_t_columns(seed_gaps.mean(axis=0))})
    return coefficients(fits, {"eval_set": eval_set, "model": "curve"}), per_p, summary, shape


def benign_analysis(data, eval_set):
    rows, coefs = [], []
    for setting in ["p0", "alpaca"]:
        labels = {"eval_set": eval_set, "setting": setting}
        d = data[data["setting"].isin(["pre", setting])].copy()
        if d["harmful"].sum() == 0:
            print(f"  {eval_set} {setting}: no harmful replies before or after, nothing to fit", flush=True)
            rows.append({**labels, "method": "none", "quantity": "no_harmful_replies", "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0})
            continue
        d["alignment"] = (d["model"] == "alignment_upsampled").astype(int)
        d["post"] = (d["setting"] == setting).astype(int)
        fits = fit(d, "harmful ~ alignment + post + alignment:post")
        coefs += coefficients(fits, {"eval_set": eval_set, "model": f"stage_{setting}"})
        for method, result in fits.items():
            rate = {(m, s): cell_rate(result, method, {"alignment": float(m == "alignment_upsampled"), "post": float(s)})
                    for m in models for s in [0, 1]}
            rows += change_rows(labels, method, {m: (rate[(m, 1)] - rate[(m, 0)]) * 100 for m in models}, "mean", stage_level)
        draws, seed_changes = stage_bootstrap(d, setting, "harmful", n_boot, boot_seed)
        rows += change_rows(labels, "bootstrap", draws, "mean", stage_level)
        rows += change_rows(labels, "seed_t", dict(zip(models, seed_changes)), "mean", stage_level)
    return coefs, rows


def seed_table(data, eval_set):
    rates = data.groupby(["setting", "seed", "model"])["harmful"].mean().unstack("model") * 100
    rows = []
    for (setting, seed), r in rates.iterrows():
        rows.append({"eval_set": eval_set, "setting": setting, "seed": seed, "baseline_pct": r["unfiltered_baseline"],
                     "alignment_pct": r["alignment_upsampled"], "diff_pp": r["alignment_upsampled"] - r["unfiltered_baseline"]})
    table = pd.DataFrame(rows)
    # same sign = seeds going the same way as the mean difference
    table["mean_diff_pp"] = table["setting"].map(table.groupby("setting")["diff_pp"].mean())
    counts = {setting: same_sign_count(part["diff_pp"], part["diff_pp"].mean()) for setting, part in table.groupby("setting")}
    table["same_sign_seeds"] = table["setting"].map(counts)
    return table


def main():
    results_dir.mkdir(exist_ok=True)
    coefs, per_p, summary, shape, benign, seed_rows = [], [], [], [], [], []
    for eval_set in eval_sets:
        data = load_verdicts(eval_set)
        c, p, s, sh = curve_analysis(data, eval_set)
        coefs += c
        per_p += p
        summary += s
        shape += sh
        c, b = benign_analysis(data, eval_set)
        coefs += c
        benign += b
        seed_rows.append(seed_table(data, eval_set))
        print(f"{eval_set} done", flush=True)

    pd.DataFrame(coefs).to_csv(results_dir / "harm_model.csv", index=False)
    pd.DataFrame(per_p).to_csv(results_dir / "harm_per_p.csv", index=False)
    pd.DataFrame(summary).to_csv(results_dir / "harm_summary.csv", index=False)
    pd.DataFrame(shape).to_csv(results_dir / "harm_curve_shape.csv", index=False)
    pd.DataFrame(benign).to_csv(results_dir / "benign_stage.csv", index=False)
    claim_rows = [{"analysis": "per_p", **r} for r in claims(per_p, ["eval_set", "p"], "diff_pp")]
    claim_rows += [{"analysis": "mean_over_p", **r} for r in claims(summary, ["eval_set"], "diff_pp")]
    claim_rows += [{"analysis": "curve_shape_vs_p1", **r} for r in claims(shape, ["eval_set", "p"], "diff_pp")]
    stage = [r for r in benign if r["method"] != "none"]
    claim_rows += [{"analysis": "stage", **r} for r in claims(stage, ["eval_set", "setting", "quantity"], "mean")]
    pd.DataFrame(claim_rows).to_csv(results_dir / "harm_claims.csv", index=False)
    pd.concat(seed_rows).to_csv(results_dir / "harm_seeds.csv", index=False)

    print(pd.DataFrame(claim_rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
