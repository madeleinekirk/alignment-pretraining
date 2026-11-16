# LoRA attack fine-tuning, one model + one attack set per run
# --probe only checks memory and speed, saves nothing

# python -m src.train.train_lora --config configs/runs/dev_p10_seed0.yaml
# python -m src.train.train_lora --config configs/runs/pilot_baseline_p20_seed0.yaml --probe
# python -m src.train.train_lora --config configs/runs/pilot_baseline_p20_seed0.yaml --out_dir /content/drive/MyDrive/alignment-pretraining/adapters --resume_dir /content/resume

import os

# same as generate.py has to be set before importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import json
import random
import time
from importlib.metadata import version
from pathlib import Path

import torch
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainerCallback, TrainingArguments

from src.config import load_all_specs, load_run_config
from src.data.data_utils import file_hash, git_commit, load_jsonl, repo_root, save_json
from src.data.formatting import format_example, ignore_index
from src.models import load_model


def load_attack_set(config):
    if config.attack.carrier == "alpaca":
        name = "alpaca"
    else:
        name = f"gsm8k_p{round(config.attack.harmful_fraction * 100)}"

    manifest = json.loads((repo_root / "manifests/data_manifest.json").read_text())
    info = manifest["attack_sets"][name]
    data_sha = file_hash(repo_root / info["file"])
    assert data_sha == info["sha256"], f"{name} doesn't match the manifest, rerun scripts/build_data.py"

    rows = load_jsonl(repo_root / info["file"])
    assert len(rows) == config.attack.n_examples, f"{name} has {len(rows)} examples"
    assert sum(r["is_harmful"] for r in rows) == config.attack.n_harmful, f"wrong number of harmful examples in {name}"
    return name, data_sha, rows


def tokenize(tokenizer, rows, max_len):
    examples = [format_example(tokenizer, r["messages"], max_len) for r in rows]
    n_cut = sum(e["truncated"] for e in examples)
    longest = max(len(e["input_ids"]) for e in examples)
    return [{"input_ids": e["input_ids"], "labels": e["labels"]} for e in examples], n_cut, longest


def pad_batch(batch, pad_id):
    # pad to the longest in the batch -100 so padding isn't in the loss
    longest = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        n_pad = longest - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n_pad)
        labels.append(b["labels"] + [ignore_index] * n_pad)
        mask.append([1] * len(b["input_ids"]) + [0] * n_pad)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}


def build_model(spec, config, checkpointing):
    # same 4-bit setup + stop token fix as the evals
    tokenizer, model = load_model(spec, load_in_4bit=config.train.load_in_4bit)
    model.config.use_cache = False
    if getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    elif checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    # seed right before LoRA init so both models start the same
    transformers.set_seed(config.seed)
    lora = config.lora
    lora_config = LoraConfig(
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        use_rslora=lora.use_rslora,
        target_modules=lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()
    return tokenizer, model


def use_bf16():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


class SaveEpochs(TrainerCallback):
    # saves the adapter at save_epochs and times each epoch
    def __init__(self, folder, epochs):
        self.folder = folder
        self.epochs = epochs
        self.seconds = {}

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.start = time.time()

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = round(state.epoch)
        self.seconds[epoch] = round(time.time() - self.start)
        if epoch in self.epochs:
            model.save_pretrained(self.folder / f"epoch_{epoch}")
            print(f"saved epoch {epoch}", flush=True)


def train(config, spec, data, out_dir, resume_dir):
    name, data_sha, rows = data
    settings = config.train
    folder = Path(out_dir) / config.run_name
    folder.mkdir(parents=True, exist_ok=True)

    commit = git_commit()
    if config.condition != "dev":
        assert not commit.endswith("-dirty"), "uncommitted changes, commit before a real run"

    # so a resume can't mix two different setups
    fingerprint = {"config": config.model_dump(), "repo_id": spec.repo_id, "revision": spec.revision,
                   "attack_set": name, "data_sha256": data_sha}
    fingerprint_file = folder / "run_settings.json"
    if fingerprint_file.exists():
        old = json.loads(fingerprint_file.read_text())
        assert old == fingerprint, f"settings changed since {folder} was started, move or delete it first"
    save_json(fingerprint, fingerprint_file)
    if (folder / "train_log.json").exists():
        print(f"{config.run_name} is already done, delete {folder} to redo it")
        return

    tokenizer, model = build_model(spec, config, settings.gradient_checkpointing)
    train_rows, n_cut, longest = tokenize(tokenizer, rows, settings.max_seq_len)
    print(f"{config.run_name} | {name} | {len(rows)} examples, {config.attack.n_harmful} harmful, {n_cut} cut", flush=True)

    use_wandb = bool(os.environ.get("WANDB_API_KEY")) and config.condition != "dev"
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", config.wandb_project)

    # resume checkpoints are ~0.8 GB each, on Colab they go on the local disk so they don't clog the Drive sync
    resume_folder = Path(resume_dir) / config.run_name if resume_dir else folder / "resume"
    args = TrainingArguments(
        output_dir=str(resume_folder),
        num_train_epochs=settings.epochs,
        per_device_train_batch_size=settings.micro_batch_size,
        gradient_accumulation_steps=settings.grad_accum_steps,
        learning_rate=settings.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=settings.warmup_steps,
        weight_decay=settings.weight_decay,
        optim="adamw_torch",
        bf16=use_bf16(),
        seed=config.seed,
        data_seed=config.seed,
        logging_steps=10,
        # one checkpoint for picking up after a disconnect overwritten each epoch
        save_strategy="epoch",
        save_total_limit=1,
        report_to=["wandb"] if use_wandb else "none",
        run_name=config.run_name,
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),  # mps doesn't support it
    )
    saver = SaveEpochs(folder, settings.save_epochs)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_rows,
        data_collator=lambda batch: pad_batch(batch, tokenizer.pad_token_id),
        callbacks=[saver],
    )

    resume = any(resume_folder.glob("checkpoint-*"))
    if resume:
        print("resuming from checkpoint", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train(resume_from_checkpoint=True if resume else None)
    minutes = (time.time() - start) / 60

    for e in settings.save_epochs:
        assert (folder / f"epoch_{e}" / "adapter_model.safetensors").exists(), f"epoch {e} adapter is missing"

    log = {
        **fingerprint,
        "git_commit": commit,
        "resumed": resume,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": version("peft"),
        "bitsandbytes": version("bitsandbytes") if torch.cuda.is_available() else None,
        "n_cut": n_cut,
        "longest_example": longest,
        "steps": trainer.state.global_step,
        "train_minutes": round(minutes, 1),
        "epoch_seconds": saver.seconds,
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if torch.cuda.is_available() else None,
        "adapter_sha256": {e: file_hash(folder / f"epoch_{e}" / "adapter_model.safetensors") for e in settings.save_epochs},
        "losses": [{"step": h["step"], "loss": h["loss"]} for h in trainer.state.log_history if "loss" in h],
    }
    save_json(log, folder / "train_log.json")
    print(f"done: {trainer.state.global_step} steps in {minutes:.1f} min, saved to {folder}")


def probe_one(config, spec, rows, micro_batch, checkpointing):
    # own function so the model gets freed after
    settings = config.train
    on_gpu = torch.cuda.is_available()
    tokenizer, model = build_model(spec, config, checkpointing)
    train_rows, _, _ = tokenize(tokenizer, rows, settings.max_seq_len)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=settings.learning_rate, weight_decay=settings.weight_decay)
    n_micro = settings.batch_size // micro_batch

    def step(batch):
        for i in range(0, len(batch), micro_batch):
            inputs = pad_batch(batch[i:i + micro_batch], tokenizer.pad_token_id)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16()):
                loss = model(**inputs).loss / n_micro
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # memory peaks on the longest examples
    longest = sorted(train_rows, key=lambda r: -len(r["input_ids"]))[:settings.batch_size]
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    step(longest)
    peak = torch.cuda.max_memory_allocated() / 1e9 if on_gpu else 0

    rng = random.Random(config.seed)
    if on_gpu:
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        step(rng.sample(train_rows, settings.batch_size))
    if on_gpu:
        torch.cuda.synchronize()
    return peak, (time.time() - start) / 10


def probe(config, spec, data):
    # pick the fastest setup that peaks under ~21 GB
    _, _, rows = data
    batch_size = config.train.batch_size
    for micro_batch, checkpointing in [(batch_size, False), (batch_size // 2, False), (batch_size, True)]:
        setup = f"micro batch {micro_batch}, checkpointing {checkpointing}"
        try:
            peak, seconds = probe_one(config, spec, rows, micro_batch, checkpointing)
            print(f"{setup}: peak {peak:.1f} GB, {seconds:.2f} s/step", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{setup}: out of memory", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default="adapters")
    parser.add_argument("--resume_dir", default=None)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()

    config = load_run_config(args.config)
    spec = load_all_specs()[config.condition]
    data = load_attack_set(config)

    if args.probe:
        probe(config, spec, data)
    else:
        train(config, spec, data, args.out_dir, args.resume_dir)


if __name__ == "__main__":
    main()
