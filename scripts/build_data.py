# Rebuilds every dataset from the Hub and checks each file's hash against the committed manifest. Ill run this on Colab before training/eval so I know it's using byte-for-byte the same data as my Mac and HEx-PHI never has to be copied around
# also stops with an error if anything doesn't match.


import json
import subprocess
import sys
from pathlib import Path

import yaml

repo_root = Path(__file__).resolve().parents[1]

# same order as on computer, each one needs the files from the ones before it
builders = ["build_gsm8k", "build_alpaca", "build_eval_sets", "build_beavertails", "build_attack_sets"]


def main() -> None:
    with open(repo_root / "configs" / "data.yaml") as f:
        config = yaml.safe_load(f)
    manifest_path = repo_root / config["attack_sets"]["manifest_path"]

    # read the committed manifest first because build_attack_sets writes a new one
    committed_text = manifest_path.read_text()
    expected = json.loads(committed_text)

    for builder in builders:
        print(f"\n--- {builder} ---", flush=True)
        subprocess.run([sys.executable, "-m", f"src.data.{builder}"], cwd=repo_root, check=True)

    rebuilt = json.loads(manifest_path.read_text())

    # keep the rebuilt manifest next to the data for checking and put the committed one back so this script never leaves the repo looking changed
    rebuilt_copy = repo_root / config["data_dir"] / "rebuilt_manifest.json"
    rebuilt_copy.write_text(manifest_path.read_text())
    manifest_path.write_text(committed_text)

    print("\nhash check against the committed manifest:")
    mismatches = []
    for group in ["attack_sets", "eval_sets"]:
        for name, entry in expected[group].items():
            new_hash = rebuilt[group].get(name, {}).get("sha256")
            ok = new_hash == entry["sha256"]
            print(f"{'ok      ' if ok else 'MISMATCH'}  {group}/{name}  {entry['sha256'][:12]}")
            if not ok:
                mismatches.append(f"{group}/{name}")

    if mismatches:
        print(f"\n{len(mismatches)} file(s) don't match: {', '.join(mismatches)}")
        print(f"rebuilt manifest saved to {rebuilt_copy} so you can compare")
        sys.exit(1)
    print("\nall datasets match the committed manifest")


if __name__ == "__main__":
    main()
