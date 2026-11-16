# runs the Tice questions on both pre-attack models and all 36 attacked adapters one after another
# skips anything that already has a summary.json so rerunning picks up where it stopped

# python -m scripts.run_tice_chain --out_dir /content/drive/MyDrive/alignment-pretraining/outputs/misalignment_mcq_prefill

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import adapter_name, drive_dir, study_models, study_seeds, study_settings

parser = argparse.ArgumentParser()
parser.add_argument("--prefill", default="(")
parser.add_argument("--out_dir", required=True)
parser.add_argument("--skip_pre_attack", action="store_true")
args = parser.parse_args()

jobs = [] if args.skip_pre_attack else [(f"pre_attack_{m}", m, None) for m in study_models]
for seed in study_seeds:
    for setting in study_settings:
        for m in study_models:
            adapter = drive_dir() / "adapters" / adapter_name(m, setting, seed) / "epoch_5"
            jobs.append((f"{m}_{setting}_seed{seed}", m, str(adapter)))

for run, model, adapter in jobs:
    if (Path(args.out_dir) / run / "summary.json").exists():
        print(f"{run} already done", flush=True)
        continue
    cmd = [sys.executable, "-m", "src.eval.misalignment_mcq", "--model", model, "--run_name", run,
           "--out_dir", args.out_dir, "--prefill", args.prefill]
    if adapter:
        cmd += ["--adapter", adapter, "--subsample"]
    print("\n>> " + " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)
