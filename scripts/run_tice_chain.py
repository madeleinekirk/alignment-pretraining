# runs the Tice questions on both pre-attack models and all 36 attacked adapters one after another
# skips anything that already has a summary.json so rerunning picks up where it stopped

import argparse
import subprocess
import sys
from pathlib import Path

drive = "/content/drive/MyDrive/alignment-pretraining"
models = ["alignment_upsampled", "unfiltered_baseline"]
settings = ["p20", "p10", "p1", "p0", "alpaca", "p5"]

parser = argparse.ArgumentParser()
parser.add_argument("--prefill", default="")
parser.add_argument("--out_dir", required=True)
parser.add_argument("--skip_pre_attack", action="store_true")
args = parser.parse_args()

jobs = [] if args.skip_pre_attack else [(f"pre_attack_{m}", m, None) for m in models]
for seed in [0, 1, 2]:
    for setting in settings:
        for m in models:
            run = f"{m}_{setting}_seed{seed}"
            # baseline p20 seed 0 is the pilot's epoch 5 adapter
            adapter_run = "pilot_baseline_p20_seed0" if run == "unfiltered_baseline_p20_seed0" else run
            jobs.append((run, m, f"{drive}/adapters/{adapter_run}/epoch_5"))

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
