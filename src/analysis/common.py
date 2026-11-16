# shared bits for the analysis scripts: loading verdicts, the bootstrap, the mixed model fits and the claim rule

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from src.config import adapter_name, study_models, study_seeds, study_settings
from src.data.data_utils import load_jsonl

# baseline first, so index 0 is the baseline and index 1 the alignment model everywhere
models = study_models
settings = study_settings
harmful_settings = ["p1", "p5", "p10", "p20"]
seeds = study_seeds
results_dir = Path("results")
n_draws, draw_seed = 4000, 9292


def load_verdicts(eval_set, judged_dir="outputs/judged"):
    # one row per prompt per run, harmful = unsafe and usable
    rows = []
    for model in models:
        runs = [("pre", -1, f"pre_attack_{model}")]
        runs += [(setting, seed, adapter_name(model, setting, seed) + "_ep5") for setting in settings for seed in seeds]
        for setting, seed, folder in runs:
            for v in load_jsonl(Path(judged_dir) / folder / f"{eval_set}.verdicts.jsonl"):
                rows.append({"model": model, "setting": setting, "seed": seed, "run": f"{model}_{setting}_{seed}",
                             "id": v["id"], "outcome": v["outcome"], "harmful": int(v["outcome"] == "unsafe")})
    data = pd.DataFrame(rows)
    per_run = data.groupby("run")["id"].nunique()
    assert per_run.nunique() == 1 and len(per_run) == 38, f"runs don't all have the same prompts: {per_run.value_counts().to_dict()}"
    return data


def to_array(data, settings_used, value="harmful"):
    # shape (model, setting, seed, prompt)
    ids = sorted(data["id"].unique())
    run_seeds = sorted(data.loc[data["setting"].isin(settings_used), "seed"].unique())
    table = data.pivot_table(index=["model", "setting", "seed"], columns="id", values=value)
    return np.array([[[table.loc[(m, s, k), ids].to_numpy() for k in run_seeds] for s in settings_used] for m in models])


def paired_bootstrap(values, n_boot, seed):
    # resamples seeds then prompts with the same picks for both models
    rng = np.random.default_rng(seed)
    n_seeds, n_prompts = values.shape[2], values.shape[3]
    draws = np.empty((n_boot, 3))
    for b in range(n_boot):
        picked = values[:, :, rng.integers(0, n_seeds, n_seeds)][:, :, :, rng.integers(0, n_prompts, n_prompts)]
        base, align = picked[0].mean(), picked[1].mean()
        draws[b] = base, align, align - base
    return draws


def interval(draws, level):
    tail = (1 - level) / 2 * 100
    return np.percentile(draws, [tail, 100 - tail])


def ci_columns(draws, level, corrected_level=None):
    # the interval at the planned level, plus one corrected for the other comparisons in the same family (Bonferroni)
    # corrected_level is left out when the planned level is already corrected or there's only one comparison
    corrected_level = corrected_level or level
    low, high = interval(draws, level)
    corrected_low, corrected_high = interval(draws, corrected_level)
    return {"ci_low": low, "ci_high": high, "ci_level": level,
            "corrected_low": corrected_low, "corrected_high": corrected_high, "corrected_level": corrected_level}


def seed_t_columns(seed_values, level=0.95):
    # t interval over the per-seed values (2 degrees of freedom with 3 seeds)
    # a bootstrap that resamples only 3 seeds comes out too narrow, this one doesn't
    # stays at 95% with no family correction, with 2 degrees of freedom it's already about 4.3 standard errors wide
    seed_values = np.asarray(seed_values, dtype=float)
    n = len(seed_values)
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * seed_values.std(ddof=1) / np.sqrt(n)
    low, high = seed_values.mean() - half, seed_values.mean() + half
    return {"ci_low": low, "ci_high": high, "ci_level": level, "corrected_low": low, "corrected_high": high, "corrected_level": level}


def same_sign_count(values, reference):
    # how many values go the same way as the reference, a 0 never counts
    values = np.asarray(values, dtype=float)
    return int(((np.sign(values) == np.sign(reference)) & (values != 0)).sum())


def fit(data, formula):
    # the mixed model with a random intercept per prompt and per run, fit with vb (the planned fit) and laplace
    model = BinomialBayesMixedGLM.from_formula(formula, {"prompt": "0 + C(id)", "run": "0 + C(run)"}, data)
    fits = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vb = model.fit_vb()
    # a vb fit that didn't converge isn't used
    if any("VB fitting did not converge" in str(w.message) for w in caught):
        print(f"  vb didn't converge for {formula}", flush=True)
    else:
        fits["vb"] = vb
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        laplace = model.fit_map()
    # scipy says "precision loss" once it can't improve any more so check the gradient instead of the warning
    gradient = np.linalg.norm(laplace.optim_retvals.jac)
    if gradient < 1e-2:
        fits["laplace"] = laplace
    else:
        print(f"  laplace didn't converge for {formula} (gradient {gradient:.2g}), the bootstrap and the seed t interval decide", flush=True)
    return fits


def coefficients(fits, labels):
    rows = []
    for method, result in fits.items():
        names = result.model.fep_names
        k = len(names)
        mean = result.fe_mean if method == "vb" else np.asarray(result.params)[:k]
        sd = result.fe_sd if method == "vb" else np.sqrt(np.diag(np.asarray(result.cov_params()))[:k])
        for name, m, s in zip(names, mean, sd):
            rows.append({**labels, "method": method, "term": name, "mean": m, "sd": s,
                         "ci_low": m - 1.96 * s, "ci_high": m + 1.96 * s})
    return rows


def coefficient_draws(result, method, n_draws, seed):
    # vb only has each coefficient's sd, laplace has the full covariance
    rng = np.random.default_rng(seed)
    k = len(result.model.fep_names)
    if method == "vb":
        return rng.normal(result.fe_mean, result.fe_sd, size=(n_draws, k))
    cov = np.asarray(result.cov_params())[:k, :k]
    # the numerical hessian is slightly lopsided and numpy warns
    cov = (cov + cov.T) / 2
    return rng.multivariate_normal(np.asarray(result.params)[:k], cov, size=n_draws)


def prompt_effects(result, group):
    model = result.model
    return result.vc_mean[model.ident == model.vcp_names.index(group)]


def cell_rate(result, method, cell):
    # rate for one cell averaged over prompts with the run effect at 0
    # so it's the rate for a typical run, not the average over runs, and can sit a little off the raw means
    names = result.model.fep_names
    x = np.array([np.prod([cell.get(part, 1.0 if part == "Intercept" else 0.0) for part in name.split(":")]) for name in names])
    eta = coefficient_draws(result, method, n_draws, draw_seed) @ x
    return expit(eta[:, None] + prompt_effects(result, "prompt")[None, :]).mean(axis=1)


def stage_bootstrap(data, setting, value, n_boot, seed):
    # change from pre-attack to after fine-tuning, pre-attack is one run per model so only prompts get resampled there
    pre = to_array(data[data["setting"] == "pre"], ["pre"], value)[:, 0, 0]
    post = to_array(data[data["setting"] == setting], [setting], value)[:, 0]
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, 2))
    for b in range(n_boot):
        prompts = rng.integers(0, pre.shape[1], pre.shape[1])
        runs = rng.integers(0, post.shape[1], post.shape[1])
        draws[b] = post[:, runs][:, :, prompts].mean(axis=(1, 2)) - pre[:, prompts].mean(axis=1)
    # the same change for each seed on its own, for the seed t interval
    seed_changes = (post.mean(axis=2) - pre.mean(axis=1)[:, None]) * 100
    return {"unfiltered_baseline": draws[:, 0] * 100, "alignment_upsampled": draws[:, 1] * 100}, seed_changes


def change_rows(labels, method, change, value_name, corrected_level):
    # change per model and the difference between the two changes
    rows = []
    for name, draws in [("baseline_change_pp", change["unfiltered_baseline"]),
                        ("alignment_change_pp", change["alignment_upsampled"]),
                        ("difference_in_change_pp", change["alignment_upsampled"] - change["unfiltered_baseline"])]:
        columns = seed_t_columns(draws) if method == "seed_t" else ci_columns(draws, 0.95, corrected_level)
        rows.append({**labels, "method": method, "quantity": name, value_name: draws.mean(), **columns})
    return rows


def excludes_zero(rows, low, high):
    # every interval excludes 0 and they all sit on the same side
    sides = {np.sign(r[low]) if np.sign(r[low]) == np.sign(r[high]) else 0 for _, r in rows.iterrows()}
    return len(sides) == 1 and 0 not in sides


def claims(rows, keys, value):
    # claim_rule_a is the rule fixed Nov 11: laplace and the bootstrap both exclude 0 at the planned level, just the bootstrap where laplace didn't converge
    # claim is the stricter one I report: the same rule with the intervals corrected for the family of comparisons,
    # and where laplace is missing the seed t interval has to exclude 0 too (the 3-seed bootstrap alone is too narrow)
    table = pd.DataFrame(rows)
    out = []
    for key, part in table.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        model_based = part[part["method"].isin(["laplace", "bootstrap"])]
        strict = model_based
        if "laplace" not in set(part["method"]):
            strict = part[part["method"].isin(["bootstrap", "seed_t"])]
        out.append({**dict(zip(keys, key)), "methods": "+".join(sorted(strict["method"])),
                    "bootstrap_estimate": part.loc[part["method"] == "bootstrap", value].iloc[0],
                    "corrected_level": round(part.loc[part["method"] == "bootstrap", "corrected_level"].iloc[0], 4),
                    "claim": "supported" if excludes_zero(strict, "corrected_low", "corrected_high") else "inconclusive",
                    "claim_rule_a": "supported" if excludes_zero(model_based, "ci_low", "ci_high") else "inconclusive"})
    return out
