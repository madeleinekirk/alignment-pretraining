# picks the example pairs for the appendix by a rule fixed before I looked at any replies
# AdvBench at p10, 3 seeds pooled: a candidate is a prompt and seed where one model's reply was judged unsafe and the other's
# refused or other_safe, 4 get drawn with a seeded shuffle, never HEx-PHI
# the file goes to outputs/qualitative (gitignored), only ids and outcomes get printed

# python -m scripts.pick_qualitative_pairs

import json
import random
from pathlib import Path

from src.analysis.common import load_verdicts
from src.config import adapter_name, drive_dir
from src.data.data_utils import load_jsonl

eval_set = "advbench_eval200"
setting = "p10"
n_pairs, draw_seed = 4, 4545
# harmful replies get cut much shorter than safe ones
unsafe_words, safe_words = 12, 40


def cut(reply, n_words):
    words = reply.split()
    return " ".join(words[:n_words]) + (" [cut]" if len(words) > n_words else "")


def main():
    data = load_verdicts(eval_set)
    data = data[data["setting"] == setting]
    outcomes = data.pivot_table(index=["id", "seed"], columns="model", values="outcome", aggfunc="first")

    candidates = []
    for (prompt_id, seed), row in outcomes.iterrows():
        base, align = row["unfiltered_baseline"], row["alignment_upsampled"]
        if base == "unsafe" and align in ("refused", "other_safe"):
            candidates.append((prompt_id, seed, "unfiltered_baseline"))
        if align == "unsafe" and base in ("refused", "other_safe"):
            candidates.append((prompt_id, seed, "alignment_upsampled"))
    # fixed order (seed, then id) so the draw is the same every time
    candidates.sort(key=lambda c: (c[1], c[0]))
    picked = random.Random(draw_seed).sample(candidates, n_pairs)
    n_base = sum(c[2] == "unfiltered_baseline" for c in candidates)
    print(f"{len(candidates)} candidates, baseline unsafe in {n_base}, alignment unsafe in {len(candidates) - n_base}")

    manifest = json.load(open("manifests/data_manifest.json"))
    prompts = {r["id"]: r["messages"][0]["content"] for r in load_jsonl(manifest["eval_sets"][eval_set]["file"])}
    lines = [f"# AdvBench {setting} example pairs (seed {draw_seed})", ""]
    for prompt_id, seed, unsafe_model in picked:
        print(f"{prompt_id} seed {seed}: {unsafe_model} judged unsafe")
        lines += [f"## {prompt_id}, seed {seed}", "", f"Prompt: {prompts[prompt_id]}", ""]
        for model in ["unfiltered_baseline", "alignment_upsampled"]:
            folder = drive_dir() / "outputs" / "generations" / (adapter_name(model, setting, seed) + "_ep5")
            reply = next(r["reply"] for r in load_jsonl(folder / f"{eval_set}.jsonl") if r["id"] == prompt_id)
            outcome = outcomes.loc[(prompt_id, seed), model]
            lines += [f"- {model} ({outcome}): {cut(reply, unsafe_words if outcome == 'unsafe' else safe_words)}"]
        lines.append("")

    out_file = Path("outputs/qualitative/advbench_p10_pairs.md")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
