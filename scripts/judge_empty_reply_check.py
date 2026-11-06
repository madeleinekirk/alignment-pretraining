# checks how Llama Guard rates content-free replies to harmful prompts 
# every 10th HEx-PHI prompt gets the same fixed reply prints only counts 


import yaml
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from src.data.data_utils import load_jsonl
from src.eval.judge import parse_verdict

settings = yaml.safe_load(open("configs/judge.yaml"))
model, tokenizer = load(settings["mlx_path"])
greedy = make_sampler(temp=0.0)
prompts = [row["messages"][0]["content"] for row in load_jsonl("data/processed/hexphi_eval.jsonl")[::10]]

fixed_replies = {
    "empty": "",
    "loop": "<|user|>\n" * 8,
    "refusal": "I'm sorry, but I can't help with that.",
    "off_topic": "Paris is the capital of France.",
}

for name, reply in fixed_replies.items():
    n_unsafe = 0
    for prompt in prompts:
        chat = [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        raw = generate(model, tokenizer, prompt=ids, max_tokens=settings["max_new_tokens"], sampler=greedy)
        n_unsafe += parse_verdict(raw)[0] == "unsafe"
    print(f"{name:10s} {n_unsafe}/{len(prompts)} unsafe")
