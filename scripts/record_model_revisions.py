# Pin the exact Hugging Face revisions of the two study checkpoints

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

AUTHOR = "geodesic-research"
CONDITIONS: dict[str, str] = {
    "alignment_upsampled": "geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_dpo",
    "unfiltered_baseline": "geodesic-research/sfm_baseline_unfiltered_dpo",
}


def list_dpo_repos(api: HfApi) -> None:
    for model in sorted(api.list_models(author=AUTHOR, search="dpo"), key=lambda m: m.id):
        print(model.id)


def record_revisions(api: HfApi, out_path: Path) -> None:
    record: dict = {"recorded_at": dt.date.today().isoformat(), "models": {}}
    for condition, repo_id in CONDITIONS.items():
        try:
            info = api.model_info(repo_id)
        except RepositoryNotFoundError:
            sys.exit(f"Not found: {repo_id}. Run with --list and fix CONDITIONS.")
        record["models"][condition] = {
            "repo_id": repo_id,
            "revision": info.sha,
            "last_modified": str(info.last_modified),
        }
        print(f"{condition:<22} {repo_id}  {info.sha}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(record, sort_keys=False))
    print(f"\nWrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List the suite's DPO repos and exit")
    parser.add_argument("--out", type=Path, default=Path("configs/models.yaml"))
    args = parser.parse_args()

    api = HfApi()  
    if args.list:
        list_dpo_repos(api)
    else:
        record_revisions(api, args.out)


if __name__ == "__main__":
    main()