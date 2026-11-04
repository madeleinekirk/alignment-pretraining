"""Run settings. Each run gets its own YAML file in configs/runs/ and gets loaded into a
RunConfig, that way I'm not hard-coding anything inside the training or eval code."""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

configs_dir = Path(__file__).resolve().parents[1] / "configs"


class ModelSpec(BaseModel):
    repo_id: str
    revision: str


class AttackConfig(BaseModel):
    # the normal task data the harmful examples get mixed into (Huang et al. setup)
    carrier: Literal["gsm8k", "alpaca"] = "gsm8k"
    # this is p, the fraction of the fine-tuning set that gets swapped for BeaverTails examples
    harmful_fraction: float = Field(0.0, ge=0.0, le=0.2)
    n_examples: int = Field(1000, gt=0)

    @model_validator(mode="after")
    def check_mix(self):
        n_harmful = self.harmful_fraction * self.n_examples
        if abs(n_harmful - round(n_harmful)) > 1e-9:
            raise ValueError(f"p * n_examples = {n_harmful}, that's not a whole number of examples")
        if self.carrier == "alpaca" and self.harmful_fraction > 0:
            raise ValueError("alpaca is only for the benign attack, p should be 0")
        return self

    @property
    def n_harmful(self):
        return round(self.harmful_fraction * self.n_examples)


class LoraSettings(BaseModel):
    # matching Tice et al.: rank 32 rsLoRA on all the attention and MLP layers
    rank: int = 32
    alpha: int = 32
    dropout: float = 0.0
    use_rslora: bool = True
    target_modules: list[str] = Field(
        default_factory=lambda: ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    )


class TrainSettings(BaseModel):
    # starting numbers for the pilot on Day 4, I'll lock them in after that
    epochs: int = 5
    learning_rate: float = 1e-4
    batch_size: int = 8
    grad_accum_steps: int = 1
    max_seq_len: int = 1024
    load_in_4bit: bool = True


class EvalSettings(BaseModel):
    max_new_tokens: int = 256
    # might need to bump this up if a lot of GSM8K answers get cut off
    gsm8k_max_new_tokens: int = 256
    advbench_subsample: int = 200
    gsm8k_subsample: int = 500


class RunConfig(BaseModel):
    run_name: str
    condition: Literal["alignment_upsampled", "unfiltered_baseline", "dev"]
    seed: int
    attack: AttackConfig
    lora: LoraSettings = Field(default_factory=LoraSettings)
    train: TrainSettings = Field(default_factory=TrainSettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)
    wandb_project: str = "alignment-pretraining"


def load_model_specs(path):
    raw = yaml.safe_load(Path(path).read_text())["models"]
    return {name: ModelSpec(**info) for name, info in raw.items()}


def load_run_config(path):
    return RunConfig(**yaml.safe_load(Path(path).read_text()))
