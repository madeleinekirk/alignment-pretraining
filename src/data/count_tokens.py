# counts tokens in every attack set after applying the real chat template, to check token parity across the p% mixes and pick the training max length
# also checks both study models have the exact same tokenizer and chat template
# only loads tokenizers, no model weights, and writes results/token_counts.csv (numbers only)

# python -m src.data.count_tokens

import argparse
import csv
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import yaml
from transformers import AutoConfig

from src.data.data_utils import load_jsonl
from src.data.formatting import format_example
from src.models import chat_template_file, load_tokenizer

candidate_lengths = [512, 1024, 2048]


# the same data only gives the same training signal if both models tokenize it the same way
def check_tokenizers_match(tokenizers: dict) -> None:
    names = list(tokenizers)
    first = tokenizers[names[0]]
    for name in names[1:]:
        other = tokenizers[name]
        assert first.get_vocab() == other.get_vocab(), f"vocab differs between {names[0]} and {name}"
        assert first.chat_template == other.chat_template, f"chat template differs between {names[0]} and {name}"
        assert first.bos_token_id == other.bos_token_id and first.eos_token_id == other.eos_token_id
    # and the copy in configs should be the same template
    assert first.chat_template.strip() == chat_template_file.read_text().strip(), \
        "configs/chat_template.jinja doesn't match the study models' template"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--out", default="results/token_counts.csv")
    args = parser.parse_args()

    with open(args.models_config) as f:
        models = yaml.safe_load(f)["models"]
    specs = {name: SimpleNamespace(**info) for name, info in models.items()}
    tokenizers = {name: load_tokenizer(spec) for name, spec in specs.items()}
    check_tokenizers_match(tokenizers)
    tokenizer = next(iter(tokenizers.values()))

    # the longest sequence the models can handle at all
    first_spec = next(iter(specs.values()))
    context_length = AutoConfig.from_pretrained(first_spec.repo_id, revision=first_spec.revision).max_position_embeddings

    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)
    with open(data_config["attack_sets"]["manifest_path"]) as f:
        manifest = json.load(f)

    # go p0 to p20 then alpaca so the table reads in order
    set_names = [f"gsm8k_{p}" for p in data_config["attack_sets"]["harmful_counts"]] + ["alpaca"]

    rows = []
    p0_response_total = None
    for name in set_names:
        examples = load_jsonl(Path(manifest["attack_sets"][name]["file"]))
        formatted = [format_example(tokenizer, ex["messages"]) for ex in examples]
        totals = [f["n_prompt_tokens"] + f["n_response_tokens"] for f in formatted]
        responses = [f["n_response_tokens"] for f in formatted]
        harmful = [f["n_response_tokens"] for f, ex in zip(formatted, examples) if ex["is_harmful"]]
        benign = [f["n_response_tokens"] for f, ex in zip(formatted, examples) if not ex["is_harmful"]]

        if name == "gsm8k_p0":
            p0_response_total = sum(responses)

        row = {
            "set": name,
            "n": len(examples),
            "n_harmful": len(harmful),
            "mean_total_tokens": round(statistics.mean(totals), 1),
            "median_total_tokens": statistics.median(totals),
            "max_total_tokens": max(totals),
            "total_tokens": sum(totals),
            "mean_response_tokens": round(statistics.mean(responses), 1),
            "total_response_tokens": sum(responses),
            "mean_response_tokens_harmful": round(statistics.mean(harmful), 1) if harmful else "",
            "mean_response_tokens_benign": round(statistics.mean(benign), 1) if benign else "",
        }
        for length in candidate_lengths:
            row[f"truncated_at_{length}"] = sum(t > length for t in totals)
            # worst case: the prompt alone fills the window so no reply is left to train on
            row[f"no_reply_left_at_{length}"] = sum(f["n_prompt_tokens"] >= length for f in formatted)
        rows.append(row)

    # token drift for how much more or less each set trains on than the plain GSM8K set
    for row in rows:
        drift = 100 * (row["total_response_tokens"] - p0_response_total) / p0_response_total
        row["response_drift_vs_p0_pct"] = round(drift, 1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("tokenizers and chat templates match across both models")
    print(f"model context length: {context_length}\n")
    print(f"{'set':12} {'mean':>6} {'max':>5} {'resp tok':>9} {'drift':>7} {'harm resp':>9} {'benign resp':>11}  " +
          "  ".join(f"cut@{n}" for n in candidate_lengths))
    for row in rows:
        print(f"{row['set']:12} {row['mean_total_tokens']:>6} {row['max_total_tokens']:>5} "
              f"{row['total_response_tokens']:>9} {row['response_drift_vs_p0_pct']:>6}% "
              f"{str(row['mean_response_tokens_harmful']):>9} {str(row['mean_response_tokens_benign']):>11}  " +
              "  ".join(f"{row[f'truncated_at_{n}']:>8}" for n in candidate_lengths))

    # the shortest candidate that cuts nothing in any set
    no_cuts = [n for n in candidate_lengths if n <= context_length and all(row[f"truncated_at_{n}"] == 0 for row in rows)]
    if no_cuts:
        print(f"\nshortest max length that cuts nothing: {no_cuts[0]}")
    else:
        print("\nevery candidate length cuts something, check the cut@ columns before picking")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
