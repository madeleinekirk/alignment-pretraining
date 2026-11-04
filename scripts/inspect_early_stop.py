"""Checking why the alignment model sometimes stops right after "Sure!" or "Certainly!".

In the smoke test it cut off on 2 of the 5 prompts while the baseline didn't. This reruns
those prompts and prints the actual tokens at the end of each reply so I can see what
made it stop.

Run from the repo root on a GPU:
    python scripts/inspect_early_stop.py
"""
import argparse
import gc
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# first two are the ones that cut off, the last one worked fine for both models
prompts = [
    "Give three tips for studying effectively.",
    "Summarize what a neural network is for a high-school student.",
    "What is the capital of France?",
]
max_new_tokens = 128


def load_model(repo_id, revision):
    # same 4-bit setup as the smoke test
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, revision=revision, quantization_config=quant, device_map="auto"
    )
    model.eval()
    return tokenizer, model


def show_tokens(tokenizer, ids):
    # prints each id next to the token it stands for, e.g. 0 '<|endoftext|>'
    return ", ".join(f"{i} {tokenizer.convert_ids_to_tokens(i)!r}" for i in ids)


@torch.inference_mode()
def check_model(name, repo_id, revision):
    print(f"\n--- {name} ---")
    tokenizer, model = load_model(repo_id, revision)

    # which tokens make generate() stop for this model
    stop_ids = model.generation_config.eos_token_id
    if not isinstance(stop_ids, list):
        stop_ids = [stop_ids]
    print("stop tokens:", show_tokens(tokenizer, stop_ids))
    print("eos token:", repr(tokenizer.eos_token), "| pad token:", repr(tokenizer.pad_token))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for prompt in prompts:
        chat = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
        reply_ids = output[0, inputs["input_ids"].shape[1]:].tolist()

        # keep special tokens in so whatever ended the reply actually shows up
        raw_reply = tokenizer.decode(reply_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        print(f"\n> {prompt}")
        print(f"  tokens generated: {len(reply_ids)} (hit the limit: {len(reply_ids) == max_new_tokens})")
        print(f"  last 5 tokens: {show_tokens(tokenizer, reply_ids[-5:])}")
        print(f"  raw reply: {raw_reply[:200]!r}")

    # free the GPU before loading the next model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, default=Path("configs/models.yaml"))
    args = parser.parse_args()

    models = yaml.safe_load(args.models.read_text())["models"]
    for name, info in models.items():
        check_model(name, info["repo_id"], info["revision"])


if __name__ == "__main__":
    main()
