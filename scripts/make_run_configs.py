## makes a config in configs/runs/ for every model, attack and seed

from pathlib import Path

import yaml

from src.config import load_run_config

runs_dir = Path(__file__).resolve().parents[1] / "configs" / "runs"

models = ["alignment_upsampled", "unfiltered_baseline"]
attacks = {
    "p0": ("gsm8k", 0.0),
    "p1": ("gsm8k", 0.01),
    "p5": ("gsm8k", 0.05),
    "p10": ("gsm8k", 0.10),
    "p20": ("gsm8k", 0.20),
    "alpaca": ("alpaca", 0.0),
}

count = 0
for seed in [0, 1, 2]:
    for attack, (carrier, p) in attacks.items():
        for model in models:
            run_name = f"{model}_{attack}_seed{seed}"
            path = runs_dir / f"{run_name}.yaml"
            # baseline p20 seed 0 is the pilot's epoch 5 adapter
            if path.exists() or run_name == "unfiltered_baseline_p20_seed0":
                continue
            config = {
                "run_name": run_name,
                "condition": model,
                "seed": seed,
                "attack": {"carrier": carrier, "harmful_fraction": p, "n_examples": 1000},
                "train": {"epochs": 5},
            }
            path.write_text(f"## {model}, {attack}, seed {seed}\n" + yaml.safe_dump(config, sort_keys=False))
            load_run_config(path)
            count += 1

print(f"wrote {count} configs")
