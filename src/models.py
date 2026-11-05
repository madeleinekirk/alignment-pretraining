# Loading models and generating replies making sure both models are loaded and decoded the same way

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# chat template copied from the study models, pythia uses it on the Mac
chat_template_file = Path(__file__).resolve().parents[1] / "configs" / "chat_template.jinja"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(spec, load_in_4bit=True):
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(spec.repo_id, revision=spec.revision)

    # pythia doesn't have a chat template so it borrows the one from the study models
    if tokenizer.chat_template is None:
        tokenizer.chat_template = chat_template_file.read_text()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda" and load_in_4bit:
        # imported here because bitsandbytes doesn't install on the Mac
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
        # only small test models run on the Mac, so full precision is fine
        model = AutoModelForCausalLM.from_pretrained(
            spec.repo_id, revision=spec.revision, dtype=torch.float32
        ).to(device)

    # the released generation_config also stops on token 2, which is "!" in this tokenizer
    # that's what was cutting replies off at "Sure!", so only stop on the real end token
    model.generation_config.eos_token_id = [tokenizer.eos_token_id]
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_reply(tokenizer, model, prompt, max_new_tokens):
    # greedy reply to one user message. Gives back the text and how many tokens it took.
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    reply_ids = output[0, inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(reply_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return reply.strip(), len(reply_ids)
