# checks that the pilot's epoch 5 adapter (trained as a 10 epoch run) is the same as a plain 5 epoch run
# needs configs/runs/check_baseline_p20_seed0_5ep.yaml trained on Colab first, then this compares the two loss curves

# python -m scripts.compare_pilot_losses

import json

from src.config import drive_dir


def load_log(run):
    return json.loads((drive_dir() / "adapters" / run / "train_log.json").read_text())


def main():
    pilot = load_log("pilot_baseline_p20_seed0")
    check = load_log("check_baseline_p20_seed0_5ep")
    assert pilot["data_sha256"] == check["data_sha256"], "the two runs trained on different data"

    last_step = check["steps"]
    pilot_losses = {h["step"]: h["loss"] for h in pilot["losses"] if h["step"] <= last_step}
    check_losses = {h["step"]: h["loss"] for h in check["losses"]}
    assert pilot_losses.keys() == check_losses.keys(), "the two runs logged different steps"
    worst = max(abs(pilot_losses[step] - check_losses[step]) for step in pilot_losses)
    print(f"{len(pilot_losses)} logged losses up to step {last_step}, biggest difference {worst:.6f}")
    print("same adapter file:", pilot["adapter_sha256"]["5"] == check["adapter_sha256"]["5"])
    # GPU training isn't bit-exact so the files can differ a little even when the losses agree
    print("matches" if worst < 0.01 else "doesn't match, say so in Limitations")


if __name__ == "__main__":
    main()
