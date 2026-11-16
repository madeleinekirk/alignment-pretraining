# Loading models and generating replies making sure both models are loaded and decoded the same way

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# chat template copied from the study models pythia uses
chat_template_file = Path(__file__).resolve().parents[1] / "configs" / "chat_template.jinja"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(spec):
    # split out of load_model so token counting can grab just the tokenizer
    # without downloading 27 GB of weights
    tokenizer = AutoTokenizer.from_pretrained(spec.repo_id, revision=spec.revision)

    # pythia doesn't have a chat template so it borrows the one from the study models
    if tokenizer.chat_template is None:
        tokenizer.chat_template = chat_template_file.read_text()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(spec, load_in_4bit=True, adapter_path=None):
    device = pick_device()
    tokenizer = load_tokenizer(spec)

    if device == "cuda" and load_in_4bit:
        # imported here because bitsandbytes doesn't install on my Mac
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            spec.repo_id, revision=spec.revision, quantization_config=quant, device_map="auto"
        )
    else:
        # no 4-bit: float32 for the small test model on my Mac, bf16 on the GPU so a 6.9B model still fits on the L4 (about 14 GB)
        # no study result uses this path, it's for the dev model and the check of whether 4-bit changes the Tice numbers
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            spec.repo_id, revision=spec.revision, dtype=dtype
        ).to(device)

    # generation_config stops on token 2 which is "!" in this tokenizer and what was cutting replies off at "Sure!"
    model.generation_config.eos_token_id = [tokenizer.eos_token_id]
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # attacked models are the base model + a LoRA adapter
    # the stop token fix goes on the base model first 
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_reply(tokenizer, model, prompt, max_new_tokens):
    # greedy reply to one user message
    # gives back the text and how many tokens it took
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    reply_ids = output[0, inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(reply_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return reply.strip(), len(reply_ids)


@torch.inference_mode()
def generate_batch(tokenizer, model, chats, max_new_tokens):
    # greedy replies for a batch of chats
    texts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in chats]
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    tokenizer.padding_side = old_side

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
        eos_token_id=[tokenizer.eos_token_id],
        pad_token_id=tokenizer.pad_token_id,
    )

    new_ids = output[:, inputs["input_ids"].shape[1]:].tolist()
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    stop_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    results = []
    for ids, n_prompt in zip(new_ids, prompt_lengths):
        # finished replies get padded out to the longest one in the batch
        stop_at = next((i for i, token in enumerate(ids) if token in stop_ids), None)
        hit_limit = stop_at is None
        reply_ids = ids if hit_limit else ids[:stop_at]
        reply = tokenizer.decode(reply_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        results.append({
            "reply": reply.strip(),
            "n_new_tokens": len(reply_ids),
            "n_prompt_tokens": n_prompt,
            "hit_limit": hit_limit,
        })
    return results
