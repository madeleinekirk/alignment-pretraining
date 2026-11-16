# checking why the alignment model sometimes stops right after "Sure!" or "Certainly!"
# this is how I found the "!" stop token in the released generation config (the fix is in src/models.py)

# python -m scripts.inspect_early_stop

import gc

import torch

from src.config import configs_dir, load_model_specs
from src.models import load_model

# first two are the ones that cut off, the last one worked fine for both models
prompts = [
    "Give three tips for studying effectively.",
    "Summarize what a neural network is for a high-school student.",
    "What is the capital of France?",
]
max_new_tokens = 128


def show_tokens(tokenizer, ids):
    # prints each id next to the token it stands for, e.g. 0 '<|endoftext|>'
    return ", ".join(f"{i} {tokenizer.convert_ids_to_tokens(i)!r}" for i in ids)


@torch.inference_mode()
def check_model(name, spec):
    print(f"\n--- {name} ---")
    tokenizer, model = load_model(spec)

    # which tokens make generate() stop for this model, after the fix it should only be the real end-of-text token
    print("stop tokens:", show_tokens(tokenizer, model.generation_config.eos_token_id))
    print("eos token:", repr(tokenizer.eos_token), "| pad token:", repr(tokenizer.pad_token))

    for prompt in prompts:
        chat = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
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
    for name, spec in load_model_specs(configs_dir / "models.yaml").items():
        check_model(name, spec)


if __name__ == "__main__":
    main()
