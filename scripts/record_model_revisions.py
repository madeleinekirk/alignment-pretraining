# pins the exact Hugging Face revisions of the two study checkpoints in configs/models.yaml
# I ran this once at the start, it won't replace the pinned file unless I pass --overwrite

# python scripts/record_model_revisions.py --list

import argparse
import datetime
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

author = "geodesic-research"
conditions = {
    "alignment_upsampled": "geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_dpo",
    "unfiltered_baseline": "geodesic-research/sfm_baseline_unfiltered_dpo",
}


def list_dpo_repos(api):
    for model in sorted(api.list_models(author=author, search="dpo"), key=lambda m: m.id):
        print(model.id)


def record_revisions(api, out_path):
    record = {"recorded_at": datetime.date.today().isoformat(), "models": {}}
    for condition, repo_id in conditions.items():
        try:
            info = api.model_info(repo_id)
        except RepositoryNotFoundError:
            sys.exit(f"Not found: {repo_id}. Run with --list and fix the conditions.")
        record["models"][condition] = {
            "repo_id": repo_id,
            "revision": info.sha,
            "last_modified": str(info.last_modified),
        }
        print(f"{condition:<22} {repo_id}  {info.sha}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(record, sort_keys=False))
    print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list the DPO repos and exit")
    parser.add_argument("--out", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--overwrite", action="store_true", help="replace revisions that are already pinned")
    args = parser.parse_args()

    api = HfApi()
    if args.list:
        list_dpo_repos(api)
        return
    # every result in the study is tied to the pinned revisions, so don't swap them by accident
    if args.out.exists() and not args.overwrite:
        sys.exit(f"{args.out} already has pinned revisions, pass --overwrite to replace them")
    record_revisions(api, args.out)


if __name__ == "__main__":
    main()
