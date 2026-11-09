## trains each run then generates replies on all three eval sets, one run after another

import subprocess
import sys

from src.config import load_run_config

drive = "/content/drive/MyDrive/alignment-pretraining"
eval_sets = ["hexphi_eval", "advbench_eval200", "gsm8k_test_200"]

# load every config first so a typo stops it now and not hours in
configs = {name: load_run_config(f"configs/runs/{name}.yaml") for name in sys.argv[1:]}


def run(args):
    print("\n>> python -m", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True)


for name, config in configs.items():
    run(["src.train.train_lora", "--config", f"configs/runs/{name}.yaml",
         "--out_dir", f"{drive}/adapters", "--resume_dir", "/content/resume"])
    for eval_set in eval_sets:
        run(["src.eval.generate", "--model", config.condition, "--eval_set", eval_set, "--run_name", f"{name}_ep5",
             "--adapter", f"{drive}/adapters/{name}/epoch_5", "--out_dir", f"{drive}/outputs/generations"])
    print(f"\n{name} finished", flush=True)
