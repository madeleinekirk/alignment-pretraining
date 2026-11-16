# generates answers to the held-out Alpaca prompts for the 8 answer-quality runs, then scores them all with the reward model
# skips generating for runs that already have a meta file, reward_score skips runs that are already scored

import subprocess
import sys
from pathlib import Path

import yaml

from src.config import drive_dir, load_run_config

drive = drive_dir()
gen_dir = f"{drive}/outputs/generations"
settings = yaml.safe_load(Path("configs/reward.yaml").read_text())
eval_set = settings["eval_set"]


def run(args):
    print("\n>> python -m", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True)


for name in settings["runs"]:
    if (Path(gen_dir) / name / f"{eval_set}.meta.json").exists():
        print(f"{name} already generated", flush=True)
        continue
    if name.startswith("pre_attack_"):
        run(["src.eval.generate", "--model", name.removeprefix("pre_attack_"), "--eval_set", eval_set,
             "--run_name", name, "--out_dir", gen_dir])
    else:
        # generation folders are named <run>_ep5 like in run_chain.py
        config = load_run_config(f"configs/runs/{name.removesuffix('_ep5')}.yaml")
        run(["src.eval.generate", "--model", config.condition, "--eval_set", eval_set, "--run_name", name,
             "--adapter", f"{drive}/adapters/{config.run_name}/epoch_5", "--out_dir", gen_dir])

run(["src.eval.reward_score", "--gen_dir", gen_dir, "--out_dir", f"{drive}/outputs/reward"])
