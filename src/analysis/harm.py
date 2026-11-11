# harm analysis from the locked plan
# claims only count if laplace and the bootstrap both exclude 0 (decided before looking)

# python -m src.analysis.harm

import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from src.analysis.common import (coefficient_draws, harmful_settings, load_verdicts, models, paired_bootstrap, plot_curves,
                                 prompt_effects, results_dir, to_array)


eval_sets = {"hexphi_eval": "HEx-PHI", "advbench_eval200": "AdvBench"}
n_boot, boot_seed = 2000, 9191
n_draws, draw_seed = 4000, 9292


def fit(data, formula):
    model = BinomialBayesMixedGLM.from_formula(formula, {"prompt": "0 + C(id)", "run": "0 + C(run)"}, data)
    fits = {"vb": model.fit_vb()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        laplace = model.fit_map()
    # only keep laplace if it converged
    if not any("did not converge" in str(w.message) for w in caught):
        fits["laplace"] = laplace
    else:
        print(f"  laplace didn't converge for {formula}, vb and the bootstrap only", flush=True)
    return fits


def coefficients(fits, eval_set, extra):
    rows = []
    for method, result in fits.items():
        names = result.model.fep_names
        k = len(names)
        mean = result.fe_mean if method == "vb" else np.asarray(result.params)[:k]
        sd = result.fe_sd if method == "vb" else np.sqrt(np.diag(np.asarray(result.cov_params()))[:k])
        for name, m, s in zip(names, mean, sd):
            rows.append({"eval_set": eval_set, **extra, "method": method, "term": name, "mean": m, "sd": s,
                         "ci_low": m - 1.96 * s, "ci_high": m + 1.96 * s})
    return rows


def cell_rate(result, method, cell):
    # harm rate for one cell averaged over prompts with the run effect at 0
    names = result.model.fep_names
    x = np.array([np.prod([cell.get(part, 1.0 if part == "Intercept" else 0.0) for part in name.split(":")]) for name in names])
    eta = coefficient_draws(result, method, n_draws, draw_seed) @ x
    return expit(eta[:, None] + prompt_effects(result, "prompt")[None, :]).mean(axis=1)


def interval(draws, level):
    tail = (1 - level) / 2 * 100
    return np.percentile(draws, [tail, 100 - tail])


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
            low, high = interval(diff, 0.9875)
            per_p.append({"eval_set": eval_set, "p": p, "method": method,
                          "baseline_pct": rates[p]["unfiltered_baseline"].mean() * 100,
                          "alignment_pct": rates[p]["alignment_upsampled"].mean() * 100,
                          "diff_pp": diff.mean(), "ci_low": low, "ci_high": high, "ci_level": 0.9875})
        gaps = {p: (rates[p]["alignment_upsampled"] - rates[p]["unfiltered_baseline"]) * 100 for p in harmful_settings}
        for p in terms:
            shape.append(shape_row(eval_set, p, method, gaps[p] - gaps["p1"]))
        means = {m: np.mean([rates[p][m] for p in harmful_settings], axis=0) * 100 for m in models}
        diff = means["alignment_upsampled"] - means["unfiltered_baseline"]
        low, high = interval(diff, 0.95)
        summary.append({"eval_set": eval_set, "method": method, "baseline_pct": means["unfiltered_baseline"].mean(),
                        "alignment_pct": means["alignment_upsampled"].mean(), "diff_pp": diff.mean(),
                        "ci_low": low, "ci_high": high})

    # locked summary number
    draws = paired_bootstrap(to_array(d, harmful_settings), n_boot, boot_seed) * 100
    low, high = interval(draws[:, 2], 0.95)
    summary.insert(0, {"eval_set": eval_set, "method": "bootstrap", "baseline_pct": draws[:, 0].mean(),
                       "alignment_pct": draws[:, 1].mean(), "diff_pp": draws[:, 2].mean(), "ci_low": low, "ci_high": high})
    # curve shape: gap at each p minus the gap at p1
    rng = np.random.default_rng(boot_seed)
    values = to_array(d, harmful_settings)
    gaps = np.empty((n_boot, len(harmful_settings)))
    for b in range(n_boot):
        picked = values[:, :, rng.integers(0, values.shape[2], values.shape[2])][:, :, :, rng.integers(0, values.shape[3], values.shape[3])]
        gaps[b] = (picked[1] - picked[0]).mean(axis=(1, 2)) * 100
    for i, p in enumerate(terms, start=1):
        shape.append(shape_row(eval_set, p, "bootstrap", gaps[:, i] - gaps[:, 0]))
    # per-p bootstrap as a third check
    for p in harmful_settings:
        draws = paired_bootstrap(to_array(d, [p]), n_boot, boot_seed) * 100
        low, high = interval(draws[:, 2], 0.9875)
        per_p.append({"eval_set": eval_set, "p": p, "method": "bootstrap", "baseline_pct": draws[:, 0].mean(),
                      "alignment_pct": draws[:, 1].mean(), "diff_pp": draws[:, 2].mean(),
                      "ci_low": low, "ci_high": high, "ci_level": 0.9875})
    return coefficients(fits, eval_set, {"model": "curve"}), per_p, summary, shape


def shape_row(eval_set, p, method, draws):
    low, high = interval(draws, 0.95)
    return {"eval_set": eval_set, "p": p, "method": method, "diff_pp": draws.mean(), "ci_low": low, "ci_high": high}


def stage_bootstrap(data, setting):
    # pre-attack is one run per model so only prompts get resampled there
    pre = to_array(data[data["setting"] == "pre"], ["pre"])[:, 0, 0]
    post = to_array(data[data["setting"] == setting], [setting])[:, 0]
    rng = np.random.default_rng(boot_seed)
    draws = np.empty((n_boot, 2))
    for b in range(n_boot):
        prompts = rng.integers(0, pre.shape[1], pre.shape[1])
        runs = rng.integers(0, post.shape[1], post.shape[1])
        draws[b] = post[:, runs][:, :, prompts].mean(axis=(1, 2)) - pre[:, prompts].mean(axis=1)
    return {"unfiltered_baseline": draws[:, 0] * 100, "alignment_upsampled": draws[:, 1] * 100}


def change_rows(eval_set, setting, method, change):
    rows = []
    for name, draws in [("baseline_change_pp", change["unfiltered_baseline"]),
                        ("alignment_change_pp", change["alignment_upsampled"]),
                        ("difference_in_change_pp", change["alignment_upsampled"] - change["unfiltered_baseline"])]:
        low, high = interval(draws, 0.95)
        rows.append({"eval_set": eval_set, "setting": setting, "method": method, "quantity": name,
                     "mean": draws.mean(), "ci_low": low, "ci_high": high})
    return rows


def benign_analysis(data, eval_set):
    rows, coefs = [], []
    for setting in ["p0", "alpaca"]:
        d = data[data["setting"].isin(["pre", setting])].copy()
        if d["harmful"].sum() == 0:
            print(f"  {eval_set} {setting}: no harmful replies before or after, nothing to fit", flush=True)
            rows.append({"eval_set": eval_set, "setting": setting, "method": "none", "quantity": "no_harmful_replies",
                         "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0})
            continue
        d["alignment"] = (d["model"] == "alignment_upsampled").astype(int)
        d["post"] = (d["setting"] == setting).astype(int)
        fits = fit(d, "harmful ~ alignment + post + alignment:post")
        coefs += coefficients(fits, eval_set, {"model": f"stage_{setting}"})
        for method, result in fits.items():
            rate = {(m, s): cell_rate(result, method, {"alignment": float(m == "alignment_upsampled"), "post": float(s)})
                    for m in models for s in [0, 1]}
            rows += change_rows(eval_set, setting, method, {m: (rate[(m, 1)] - rate[(m, 0)]) * 100 for m in models})
        rows += change_rows(eval_set, setting, "bootstrap", stage_bootstrap(d, setting))
    return coefs, rows


def claims(rows, keys, value):
    # just the bootstrap where laplace didn't converge
    table = pd.DataFrame(rows)
    table = table[table["method"].isin(["laplace", "bootstrap"])]
    out = []
    for key, part in table.groupby(keys):
        sides = {np.sign(r["ci_low"]) if np.sign(r["ci_low"]) == np.sign(r["ci_high"]) else 0 for _, r in part.iterrows()}
        supported = len(sides) == 1 and 0 not in sides
        out.append({**dict(zip(keys, key)), "methods": "+".join(sorted(part["method"])),
                    "bootstrap_estimate": part.loc[part["method"] == "bootstrap", value].iloc[0],
                    "claim": "supported" if supported else "inconclusive"})
    return out


def seed_table(data, eval_set):
    rates = data.groupby(["setting", "seed", "model"])["harmful"].mean().unstack("model") * 100
    rows = []
    for (setting, seed), r in rates.iterrows():
        rows.append({"eval_set": eval_set, "setting": setting, "seed": seed, "baseline_pct": r["unfiltered_baseline"],
                     "alignment_pct": r["alignment_upsampled"], "diff_pp": r["alignment_upsampled"] - r["unfiltered_baseline"]})
    table = pd.DataFrame(rows)
    # same sign = seeds going the same way as the mean difference, 0 doesn't count
    means = table.groupby("setting")["diff_pp"].mean()
    table["mean_diff_pp"] = table["setting"].map(means)
    table["same_sign_seeds"] = table["setting"].map(
        table.assign(same=np.sign(table["diff_pp"]) == np.sign(table["mean_diff_pp"])).groupby("setting")["same"].sum())
    return table


def main():
    results_dir.mkdir(exist_ok=True)
    coefs, per_p, summary, shape, benign, seed_rows, all_data = [], [], [], [], [], [], {}
    for eval_set in eval_sets:
        data = load_verdicts(eval_set)
        all_data[eval_set] = data
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
    claim_rows += [{"analysis": "stage", **r} for r in claims(benign, ["eval_set", "setting", "quantity"], "mean")]
    pd.DataFrame(claim_rows).to_csv(results_dir / "harm_claims.csv", index=False)
    pd.concat(seed_rows).to_csv(results_dir / "harm_seeds.csv", index=False)
    plot_curves(all_data, eval_sets, "harmful", "Harmful replies (%)", "harm_curves.png", n_boot, boot_seed)

    print(pd.DataFrame(claim_rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
