# shared bits for the analysis scripts

from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.data_utils import load_jsonl

models = ["unfiltered_baseline", "alignment_upsampled"]
settings = ["p0", "p1", "p5", "p10", "p20", "alpaca"]
harmful_settings = ["p1", "p5", "p10", "p20"]
seeds = [0, 1, 2]
results_dir = Path("results")
figures_dir = results_dir / "figures"
colors = {"unfiltered_baseline": "#2a78d6", "alignment_upsampled": "#eb6834"}
labels = {"unfiltered_baseline": "Baseline", "alignment_upsampled": "Alignment pretrained"}


def drive_dir():
    # google drive
    found = glob(str(Path.home() / "Library/CloudStorage/GoogleDrive-*/My Drive/alignment-pretraining"))
    assert found, "can't find the alignment-pretraining folder in Google Drive"
    return Path(found[0])


def run_name(model, setting, seed):
    # baseline p20 seed 0 is the pilot's epoch 5 adapter
    if (model, setting, seed) == ("unfiltered_baseline", "p20", 0):
        return "pilot_baseline_p20_seed0"
    return f"{model}_{setting}_seed{seed}"


def load_verdicts(eval_set, judged_dir="outputs/judged"):
    # one row per prompt per run, harmful = unsafe and usable
    rows = []
    for model in models:
        runs = [("pre", -1, f"pre_attack_{model}")]
        runs += [(setting, seed, run_name(model, setting, seed) + "_ep5") for setting in settings for seed in seeds]
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


def plot_curves(all_data, eval_sets, value, ylabel, filename, n_boot, boot_seed, note="", top=100, legend="upper left"):
    figures_dir.mkdir(parents=True, exist_ok=True)
    order = ["pre", "p0"] + harmful_settings
    ticks = ["Pre-attack", "0%", "1%", "5%", "10%", "20%"]
    fig, axes = plt.subplots(1, len(eval_sets), figsize=(10, 4), sharey=True)
    for ax, (eval_set, name) in zip(axes, eval_sets.items()):
        data = all_data[eval_set]
        ends = {}
        for i, model in enumerate(models):
            style = "-" if i == 0 else "--"
            for seed in seeds:
                seed_rates = [data[(data["model"] == model) & (data["setting"] == s) & (data["seed"].isin([seed, -1]))][value].mean() * 100 for s in order]
                ax.plot(range(len(order)), seed_rates, style, color=colors[model], linewidth=1, alpha=0.35)
            mean, low, high = [], [], []
            for s in order:
                draws = paired_bootstrap(to_array(data[data["setting"] == s], [s], value), n_boot, boot_seed) * 100
                column = draws[:, i]
                mean.append(column.mean())
                low.append(np.percentile(column, 2.5))
                high.append(np.percentile(column, 97.5))
            ax.fill_between(range(len(order)), low, high, color=colors[model], alpha=0.15, linewidth=0)
            ax.plot(range(len(order)), mean, style, color=colors[model], linewidth=2, marker="o", markersize=5, label=labels[model])
            ends[model] = mean[-1]
        # push the end labels apart if they overlap
        low_model, high_model = sorted(models, key=lambda m: ends[m])
        gap = max(0, 6 - (ends[high_model] - ends[low_model])) / 2
        for model, shift in [(low_model, -gap), (high_model, gap)]:
            ax.annotate(labels[model], (len(order) - 1, ends[model]), xytext=(len(order) - 0.85, ends[model] + shift),
                        va="center", fontsize=8, color="#444444")
        ax.set_title(name, fontsize=11)
        ax.set_xticks(range(len(order)), ticks)
        ax.set_xlabel("Harmful share of the fine-tuning data")
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.3, len(order) - 0.3 + 1.2)
    axes[0].set_ylabel(ylabel)
    axes[0].set_ylim(0, top)
    axes[0].legend(frameon=False, fontsize=8, loc=legend)
    fig.text(0.01, 0.01, "Thin lines: each seed. Thick lines: mean of 3 seeds, band: 95% bootstrap CI (seeds, then prompts). " + note,
             fontsize=7, color="#666666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figures_dir / filename, dpi=200)
    plt.close(fig)
