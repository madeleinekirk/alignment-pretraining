# shows what a model wants to start its Tice answer with, with and without the "(" prefill
# uses the first 50 Article questions in all 8 prompt variants

import argparse
from collections import Counter

from src.config import configs_dir, load_model_specs
from src.eval.misalignment_mcq import build_prompts, chat_texts, letter_token_ids, load_questions, next_token_logprobs
from src.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--n_questions", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=16)
args = parser.parse_args()

specs = {**load_model_specs(configs_dir / "models.yaml"), **load_model_specs(configs_dir / "models_dev.yaml")}
tokenizer, model = load_model(specs[args.model])
prompts = build_prompts(load_questions(False)[: args.n_questions])

for prefill in ["", "("]:
    letter_ids = letter_token_ids(tokenizer, prefill)
    top_tokens = Counter()
    on_letters = []
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start:start + args.batch_size]
        probs = next_token_logprobs(tokenizer, model, chat_texts(tokenizer, [p["chat"] for p in batch], prefill)).exp()
        on_letters += probs[:, letter_ids].sum(dim=1).tolist()
        for row in probs:
            top = row.topk(5)
            for p, i in zip(top.values.tolist(), top.indices.tolist()):
                top_tokens[i] += p / len(prompts)
    print(f"\n{args.model}, prefill {prefill!r}: mean probability on A or B = {sum(on_letters) / len(on_letters):.3f}")
    for i, p in top_tokens.most_common(8):
        print(f"  {tokenizer.decode([i])!r:14} {p:.3f}")
