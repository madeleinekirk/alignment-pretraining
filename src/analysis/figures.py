# makes every figure for the paper into results/figures, this is the only file that plots anything
# reads the results csvs and the judged verdicts, so run harm, controls, tice and quality first
# the tice curves also need the google drive folder

# python -m src.analysis.figures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import tice
from src.analysis.common import harmful_settings, load_verdicts, models, paired_bootstrap, results_dir, seeds, to_array

out_dir = results_dir / "figures"
# one meaning per color everywhere: gray = baseline, orange = alignment model (and its effect),
# red = harmful, teal = refused / safer, sand = other safe
model_colors = {"unfiltered_baseline": "#5f5e5a", "alignment_upsampled": "#e2601f"}
align_light = "#f2a77f"
model_styles = {"unfiltered_baseline": "-", "alignment_upsampled": "--"}
model_labels = {"unfiltered_baseline": "Baseline", "alignment_upsampled": "Alignment pretrained"}
outcome_colors = {"unsafe": "#c8353a", "other_safe": "#d6c7a8", "refused": "#178f74"}
outcome_labels = {"unsafe": "Unsafe", "other_safe": "Other safe", "refused": "Refused"}
ink, muted, light = "#1a1a1a", "#4a4a4a", "#d9d9d9"
eval_sets = {"hexphi_eval": "HEx-PHI", "advbench_eval200": "AdvBench"}
splits = {"textbook_questions": "Tice textbook questions", "article_questions": "Tice article questions"}
p_labels = {"pre": "Pre-attack", "p0": "0%\n(GSM8K only)", "p1": "1%", "p5": "5%", "p10": "10%", "p20": "20%", "alpaca": "Alpaca"}
# square root spacing so 0, 1 and 5% aren't squashed together, pre-attack sits one step left of 0
x_pos = {"pre": -1.0, "p0": 0.0, "p1": 1.0, "p5": np.sqrt(5), "p10": np.sqrt(10), "p20": np.sqrt(20)}


plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": ink,
                     "axes.titlepad": 10, "axes.labelsize": 10.5, "axes.labelcolor": ink, "axes.edgecolor": "#8c8c8c",
                     "xtick.color": "#333333", "ytick.color": "#333333", "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 10, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def save(fig, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def note(fig, text, y=-0.06):
    fig.text(0.01, y, text, fontsize=8.5, color=muted, ha="left", va="top")


def p_axis(ax, order):
    ax.set_xticks([x_pos[s] for s in order], [p_labels[s] for s in order])
    ax.set_xlabel("Harmful share of the fine-tuning data (square-root spacing)")
    ax.set_xlim(x_pos[order[0]] - 0.4, x_pos[order[-1]] + 0.4)


def curves(all_data, panels, value, ylabel, name, n_boot, boot_seed, top=100, legend="upper left", extra_note=""):
    # one thin line per seed, the thick line is the mean of the 3 seeds and the band the bootstrap interval
    order = ["pre", "p0"] + harmful_settings
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 3.8), sharey=True)
    for ax, (key, title) in zip(axes, panels.items()):
        data = all_data[key]
        for i, model in enumerate(models):
            x = [x_pos[s] for s in order]
            for seed in seeds:
                seed_rates = [data[(data["model"] == model) & (data["setting"] == s) & (data["seed"].isin([seed, -1]))][value].mean() * 100
                              for s in order]
                ax.plot(x, seed_rates, model_styles[model], color=model_colors[model], linewidth=0.8, alpha=0.35)
            mean, low, high = [], [], []
            for s in order:
                values = to_array(data[data["setting"] == s], [s], value)
                draws = paired_bootstrap(values, n_boot, boot_seed)[:, i] * 100
                mean.append(values[i].mean() * 100)
                low.append(np.percentile(draws, 2.5))
                high.append(np.percentile(draws, 97.5))
            ax.fill_between(x, low, high, color=model_colors[model], alpha=0.15, linewidth=0)
            ax.plot(x, mean, model_styles[model], color=model_colors[model], linewidth=2, marker="o", markersize=4,
                    label=model_labels[model])
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color="#eeeeee", linewidth=0.8)
        p_axis(ax, order)
    axes[0].set_ylabel(ylabel)
    axes[0].set_ylim(0, top)
    axes[0].legend(fontsize=9.5, loc=legend)
    note(fig, "Thin lines: each seed. Thick lines: mean of 3 seeds, band: 95% bootstrap interval (seeds, then prompts), "
              "approximate with 3 seeds. " + extra_note)
    fig.tight_layout()
    save(fig, name)


def harmful_heldout():
    table = pd.read_csv(results_dir / "harmful_heldout_fixed.csv")
    order = ["pre", "p0", "p1", "p5", "p10"]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for model in models:
        part = table[table["model"] == model].set_index("setting").loc[order]
        x = [x_pos[s] for s in order]
        ax.fill_between(x, part["ci_low"], part["ci_high"], color=model_colors[model], alpha=0.15, linewidth=0)
        ax.plot(x, part["loss"], model_styles[model], color=model_colors[model], linewidth=2, marker="o", markersize=4,
                label=model_labels[model])
    ax.set_ylabel("Loss on 100 held-out harmful replies\n(nats per token, lower = more likely)")
    ax.grid(axis="y", color="#eeeeee", linewidth=0.8)
    p_axis(ax, order)
    ax.legend(fontsize=9.5, loc="upper left")
    note(fig, "Same 100 BeaverTails examples at every point, none trained on. Band: 95% bootstrap interval.\n"
              "No 20% point: at 20% all 200 harmful examples are in the training mix.")
    fig.tight_layout()
    save(fig, "harmful_heldout")


def forest_panel(ax, rows, title, xlabel, fmt="{:+.1f}"):
    # rows: (label, estimate, low, high, filled), drawn top to bottom
    for y, (label, est, low, high, filled) in enumerate(reversed(rows)):
        color = model_colors["alignment_upsampled"] if filled else align_light
        ax.plot([low, high], [y, y], color=color, linewidth=2.2 if filled else 1.4, solid_capstyle="round")
        ax.plot(est, y, "o", markersize=6, color=color, markerfacecolor=color if filled else "white", markeredgewidth=1.5)
        ax.annotate(fmt.format(est), (est, y), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color=muted)
    ax.axvline(0, color=ink, linewidth=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def claim_lookup(path, analysis, key):
    table = pd.read_csv(path)
    table = table[table["analysis"] == analysis]
    return {tuple(r[k] for k in key): r["claim"] == "supported" for _, r in table.iterrows()}


def gaps():
    harm = pd.read_csv(results_dir / "harm_per_p.csv")
    harm = harm[harm["method"] == "bootstrap"]
    harm_claims = claim_lookup(results_dir / "harm_claims.csv", "per_p", ["eval_set", "p"])
    tice = pd.read_csv(results_dir / "tice_gaps.csv")
    tice = tice[tice["method"] == "bootstrap"]
    tice_claims = claim_lookup(results_dir / "tice_claims.csv", "gap", ["split", "setting"])
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0), gridspec_kw={"wspace": 0.55})
    for ax, (eval_set, title) in zip(axes[:2], eval_sets.items()):
        rows = []
        for p in harmful_settings:
            r = harm[(harm["eval_set"] == eval_set) & (harm["p"] == p)].iloc[0]
            rows.append((p_labels[p], r["diff_pp"], r["ci_low"], r["ci_high"], harm_claims[(eval_set, p)]))
        forest_panel(ax, rows, title, "Harmful replies (pp)")
        ax.set_xlim(-50, 25)
    for ax, (split, title) in zip(axes[2:], splits.items()):
        rows = []
        for s in ["p0"] + harmful_settings:
            r = tice[(tice["split"] == split) & (tice["setting"] == s)].iloc[0]
            rows.append((p_labels[s].replace("\n(GSM8K only)", " (GSM8K)"), r["diff_pp"], r["ci_low"], r["ci_high"], tice_claims[(split, s)]))
        forest_panel(ax, rows, title, "Misaligned choices (pp)")
        ax.set_xlim(-28, 6)
    note(fig, "Alignment - baseline. Paired bootstrap estimate with Bonferroni intervals: 98.75% for harm (4 harmful shares), 99.2% for Tice (6 settings). "
              "Filled = supported (bootstrap and\nLaplace intervals both exclude 0), hollow light = inconclusive. Negative = alignment model safer.", y=-0.1)
    save(fig, "gaps")


def benign_stage():
    outcomes = pd.read_csv(results_dir / "outcomes.csv")
    tice = pd.read_csv(results_dir / "tice_seeds.csv")
    harm_stage = pd.read_csv(results_dir / "harm_claims.csv")
    tice_stage = pd.read_csv(results_dir / "tice_claims.csv")
    stages = ["pre", "p0", "alpaca"]
    stage_names = ["Pre-attack", "After\nGSM8K", "After\nAlpaca"]
    panels = []
    for eval_set, title in eval_sets.items():
        d = outcomes[outcomes["eval_set"] == eval_set].groupby(["model", "setting"])["unsafe"].mean()
        levels = {m: [d[(m, s)] for s in stages] for m in models}
        did = harm_stage[(harm_stage["analysis"] == "stage") & (harm_stage["eval_set"] == eval_set) &
                         (harm_stage["setting"] == "alpaca") & (harm_stage["quantity"] == "difference_in_change_pp")]
        panels.append((title, "Harmful replies (%)", levels, did.iloc[0]))
    for split, title in splits.items():
        d = tice[tice["split"] == split].set_index("setting")
        levels = {"unfiltered_baseline": [d.loc[s, "baseline_pct"] for s in stages],
                  "alignment_upsampled": [d.loc[s, "alignment_pct"] for s in stages]}
        did = tice_stage[(tice_stage["analysis"] == "stage") & (tice_stage["split"] == split) &
                         (tice_stage["setting"] == "alpaca") & (tice_stage["quantity"] == "difference_in_change_pp")]
        panels.append((title, "Misaligned choices (%)", levels, did.iloc[0]))
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), gridspec_kw={"wspace": 0.4})
    for ax, (title, ylabel, levels, did) in zip(axes, panels):
        for model in models:
            ax.plot(range(3), levels[model], model_styles[model], color=model_colors[model], linewidth=2, marker="o",
                    markersize=6, markeredgecolor="white", markeredgewidth=1, label=model_labels[model])
        top = max(max(v) for v in levels.values())
        for i in range(3):
            gap = levels["alignment_upsampled"][i] - levels["unfiltered_baseline"][i]
            high = max(levels["alignment_upsampled"][i], levels["unfiltered_baseline"][i])
            ax.annotate(f"{gap:+.1f}", (i, high), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9.5, color=ink)
        word = "supported" if did["claim"] == "supported" else "inconclusive"
        ax.set_title(title, loc="left", pad=50)
        ax.text(0.0, 1.03, f"Difference in change\nafter Alpaca: {did['bootstrap_estimate']:+.1f} pp\n({word})",
                transform=ax.transAxes, fontsize=9, color=ink if word == "supported" else muted,
                fontweight="bold" if word == "supported" else "normal")
        ax.set_xticks(range(3), stage_names)
        ax.set_xlim(-0.35, 2.35)
        ax.set_ylim(0, top * 1.2 + 1)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#eeeeee", linewidth=0.8)
    axes[0].legend(fontsize=9.5, loc="upper left", bbox_to_anchor=(0, 1.6), ncol=2)
    note(fig, "Means of 3 seeds (pre-attack is one checkpoint per model). Numbers above the points = alignment - baseline at each stage. "
              "AdvBench has no harmful replies before Alpaca.\nDifference in change = the alignment model's change from pre-attack to after Alpaca minus the baseline's. "
              "Supported = the corrected bootstrap interval and the Laplace or seed-level t interval all exclude 0.", y=-0.04)
    save(fig, "benign_stage")


def attack_fit():
    loss = pd.read_csv(results_dir / "training_loss.csv")
    harm = pd.read_csv(results_dir / "harm_seeds.csv")
    shades = {"p1": "#f6c3a9", "p5": "#ee9a6c", "p10": "#e2601f", "p20": "#9e3d12"}
    markers = {"p1": "o", "p5": "s", "p10": "D", "p20": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, (eval_set, title) in zip(axes, eval_sets.items()):
        both = loss.merge(harm[harm["eval_set"] == eval_set], on=["setting", "seed"])
        both = both[both["setting"].isin(harmful_settings)]
        x_max = both["diff"].abs().max() * 1.3
        y_max = both["diff_pp"].abs().max() * 1.2
        ax.axvspan(-x_max, 0, ymin=0, ymax=0.5, color=model_colors["alignment_upsampled"], alpha=0.06, linewidth=0)
        ax.axhline(0, color=ink, linewidth=0.8)
        ax.axvline(0, color=ink, linewidth=0.8)
        for p in harmful_settings:
            part = both[both["setting"] == p]
            ax.scatter(part["diff"], part["diff_pp"], s=36, color=shades[p], marker=markers[p], edgecolor="white",
                       linewidth=0.6, label=p_labels[p], zorder=3)
        corner = int(((both["diff"] < 0) & (both["diff_pp"] < 0)).sum())
        ax.text(0.98, 0.03, f"Lower loss and fewer harmful replies:\n{corner} of {len(both)} runs", transform=ax.transAxes,
                fontsize=8.5, color=muted, ha="right")
        ax.set_xlim(-x_max, x_max)
        ax.set_ylim(-y_max, y_max)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Final training loss, alignment - baseline (nats/token)")
    axes[0].set_ylabel("Harmful replies, alignment - baseline (pp)")
    axes[0].legend(title="Harmful share", fontsize=8.5, title_fontsize=8.5, loc="upper right")
    note(fig, "One point per harmful share and seed. Training loss = mean over epoch 5 of the attack mix. "
              "Shaded corner: the alignment model fit the attack data at least as well and still gave fewer harmful replies.")
    fig.tight_layout()
    save(fig, "attack_fit")


def dumbbell(ax, rows, title, xlabel):
    # rows: (label, baseline, alignment, diff text, clear), one row per line top to bottom
    for y, (label, base, align, text, clear) in enumerate(reversed(rows)):
        ax.plot([base, align], [y, y], color=light, linewidth=3, solid_capstyle="round", zorder=1)
        ax.scatter(base, y, s=60, color=model_colors["unfiltered_baseline"], zorder=2)
        ax.scatter(align, y, s=60, color=model_colors["alignment_upsampled"], marker="D", zorder=2)
        ax.text(1.02, y, text, transform=ax.get_yaxis_transform(), va="center", fontsize=9.5,
                color=ink if clear else muted, fontweight="bold" if clear else "normal")
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.8)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def cost():
    # what the alignment model gives up: raw levels for both models, the paired difference and CI on the right
    gsm = pd.read_csv(results_dir / "gsm8k_pooled.csv").set_index("settings")
    quality = pd.read_csv(results_dir / "quality.csv")
    runs = pd.read_csv(results_dir / "quality_runs.csv")
    q = lambda comp, measure, subset="all": quality[(quality["comparison"] == comp) & (quality["measure"] == measure) &
                                                    (quality["subset"] == subset)].iloc[0]
    fig = plt.figure(figsize=(11, 5.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1], wspace=0.75, hspace=0.75)

    rows = []
    for key, name in [("p0_to_p20", "All 0-20% runs"), ("p0", "0% runs"), ("p1", "1% runs"), ("p5", "5% runs"),
                      ("p10", "10% runs"), ("p20", "20% runs"), ("alpaca", "Alpaca runs (noisy)")]:
        r = gsm.loc[key]
        main = key == "p0_to_p20"
        # bold only if the bootstrap and the seed t interval both exclude 0, the numbers shown are the seed t interval (the wider one)
        clear = bool(main and r["ci_high"] < 0 and r["seed_t_high"] < 0)
        rows.append((name, r["baseline_acc"], r["alignment_acc"],
                     f"{r['diff_pp']:+.1f} [{r['seed_t_low']:+.1f}, {r['seed_t_high']:+.1f}]", clear))
    ax = fig.add_subplot(grid[:, 0])
    dumbbell(ax, rows, "Math: GSM8K accuracy after the attack", "Accuracy (%)")
    ax.get_yticklabels()[-1].set_fontweight("bold")

    ax = fig.add_subplot(grid[0, 1])
    for y, (comp, name) in enumerate([("alpaca", "After Alpaca"), ("pre_attack", "Pre-attack")]):
        r = q(comp, "win_rate")
        share = r["estimate"] * 100
        ax.barh(y, share, color=model_colors["alignment_upsampled"], height=0.55)
        ax.barh(y, 100 - share, left=share, color=model_colors["unfiltered_baseline"], height=0.55)
        ax.plot([r["ci_low"] * 100, r["ci_high"] * 100], [y, y], color=ink, linewidth=1.5)
        ax.text(2, y, f"{share:.1f}%", va="center", fontsize=9.5, color="white", fontweight="bold")
        ax.text(98, y, f"{100 - share:.1f}%", va="center", ha="right", fontsize=9.5, color="white", fontweight="bold")
        word = "interval excludes 50%" if r["ci_excludes_null"] else "within noise"
        ax.text(1.02, y, word, transform=ax.get_yaxis_transform(), va="center", fontsize=9.5,
                color=ink if r["ci_excludes_null"] else muted, fontweight="bold" if r["ci_excludes_null"] else "normal")
    ax.set_yticks([0, 1], ["After Alpaca", "Pre-attack"])
    ax.axvline(50, color=ink, linewidth=0.8, linestyle=":")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 1.6)
    ax.set_title("Answer quality: whose answer the reward model prefers", loc="left")
    ax.set_xlabel("Share of 200 benign prompts won (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    rows = []
    for comp, stage, name in [("pre_attack", "pre", "Pre-attack"), ("alpaca", "alpaca", "After Alpaca")]:
        rate = runs[runs["stage"] == stage].groupby("model")["refusal_rate"].mean() * 100
        r = q(comp, "refusal_diff")
        rows.append((name, rate["unfiltered_baseline"], rate["alignment_upsampled"],
                     f"{r['estimate'] * 100:+.1f} [{r['ci_low'] * 100:+.1f}, {r['ci_high'] * 100:+.1f}]", bool(r["ci_excludes_null"])))
    ax = fig.add_subplot(grid[1, 1])
    dumbbell(ax, rows, "Over-refusal on benign prompts", "Keyword refusal rate (%)")
    ax.set_xlim(-0.3, 4)

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=model_colors["unfiltered_baseline"], label="Baseline"),
               plt.Line2D([], [], marker="D", linestyle="", markersize=7, color=model_colors["alignment_upsampled"],
                          label="Alignment pretrained")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.06), ncol=2)
    note(fig, "Numbers on the right: alignment - baseline in pp. GSM8K: 95% t interval over the 3 seeds; refusals: 95% paired bootstrap interval. "
              "Bold = every interval excludes no difference.\nPer-setting GSM8K rows are descriptive; the pooled test was chosen "
              "after the per-run results. Win rate counts ties as half a win each (bar = 95% bootstrap interval).\n"
              "Refusals use the same keyword list as the harm judge.", y=0.0)
    save(fig, "cost")


def outcome_bars():
    table = pd.read_csv(results_dir / "outcomes.csv")
    order = ["pre", "p0", "p1", "p5", "p10", "p20", "alpaca"]
    names = {**p_labels, "p0": "0% (GSM8K only)", "alpaca": "Alpaca (benign)"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, (eval_set, title) in zip(axes, eval_sets.items()):
        d = table[table["eval_set"] == eval_set].groupby(["setting", "model"])[["unsafe", "other_safe", "refused"]].mean()
        ticks, tick_labels = [], []
        for i, s in enumerate(order):
            base_y = -(i * 2.6 + (0.6 if s == "alpaca" else 0))
            for k, model in enumerate(models):
                y = base_y - k * 1.0
                left = 0
                for outcome in ["unsafe", "other_safe", "refused"]:
                    width = d.loc[(s, model), outcome]
                    ax.barh(y, width, left=left, height=0.85, color=outcome_colors[outcome], edgecolor="white", linewidth=0.8,
                            label=outcome_labels[outcome] if (i, k) == (0, 0) else None)
                    if outcome == "unsafe" and width > 12:
                        ax.text(left + 2, y, f"{width:.0f}", va="center", fontsize=8.5, color="white", fontweight="bold")
                    left += width
                ax.text(101, y, "B" if k == 0 else "A", va="center", fontsize=8.5, color=muted)
            ticks.append(base_y - 0.5)
            tick_labels.append(names[s])
        ax.axhline(-(6 * 2.6) + 1.0, color=light, linewidth=0.8, linestyle=":")
        ax.set_yticks(ticks, tick_labels)
        ax.set_xlim(0, 104)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Share of replies (%)")
        ax.set_title(title, loc="left")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(1.05, 1.16), ncol=3)
    note(fig, "Mean of 3 seeds. In each pair B = baseline (top), A = alignment pretrained (bottom). "
              "Broken replies (empty, looping or template leaks) are not drawn, so a few bars end just short of 100.")
    fig.tight_layout()
    save(fig, "outcome_bars")


def tice_curves():
    data = tice.load_scores()
    all_data = {split: tice.per_question(data[data["split"] == split]) for split in splits}
    curves(all_data, {"textbook_questions": "Textbook", "article_questions": "Articles"}, "misaligned",
           "Misaligned choices (%)", "tice_curves", tice.n_boot, tice.boot_seed, top=70, legend="lower left",
           extra_note="Alpaca runs are in benign_stage.")


def main():
    # same bootstrap settings as harm.py and controls.py so the bands match the tables
    n_boot, boot_seed = 2000, 9191
    all_data = {eval_set: load_verdicts(eval_set) for eval_set in eval_sets}

    # main text
    curves(all_data, eval_sets, "harmful", "Harmful replies (%)", "harm_curves", n_boot, boot_seed)
    gaps()
    tice_curves()
    benign_stage()
    cost()
    # appendix
    outcome_bars()
    attack_fit()
    harmful_heldout()


if __name__ == "__main__":
    main()
