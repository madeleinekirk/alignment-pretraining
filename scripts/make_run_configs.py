# makes a config in configs/runs/ for every model, attack and seed

from pathlib import Path

import yaml

from src.config import adapter_name, load_run_config, study_models, study_seeds, study_settings

runs_dir = Path(__file__).resolve().parents[1] / "configs" / "runs"

# what each setting trains on: (normal task data, share swapped for harmful examples)
attacks = {
    "p0": ("gsm8k", 0.0),
    "p1": ("gsm8k", 0.01),
    "p5": ("gsm8k", 0.05),
    "p10": ("gsm8k", 0.10),
    "p20": ("gsm8k", 0.20),
    "alpaca": ("alpaca", 0.0),
}

assert list(attacks) == study_settings

count = 0
for seed in study_seeds:
    for attack, (carrier, p) in attacks.items():
        for model in study_models:
            run_name = f"{model}_{attack}_seed{seed}"
            path = runs_dir / f"{run_name}.yaml"
            # baseline p20 seed 0 is the pilot's epoch 5 adapter so it gets no config of its own
            if path.exists() or adapter_name(model, attack, seed) != run_name:
                continue
            config = {
                "run_name": run_name,
                "condition": model,
                "seed": seed,
                "attack": {"carrier": carrier, "harmful_fraction": p, "n_examples": 1000},
                "train": {"epochs": 5},
            }
            path.write_text(f"# {model}, {attack}, seed {seed}\n" + yaml.safe_dump(config, sort_keys=False))
            load_run_config(path)
            count += 1

print(f"wrote {count} configs")
