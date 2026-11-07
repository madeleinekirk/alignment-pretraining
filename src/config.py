# run settings each run gets its own YAML file in configs/runs/ and gets loaded into a RunConfig
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
    # p the fraction of the fine-tuning set that gets swapped for BeaverTails examples
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
    # same for every run, epochs come from the pilot
    epochs: int = 5
    # constant lr after warmup
    learning_rate: float = 1e-4
    warmup_steps: int = 10
    weight_decay: float = 0.0
    # batch_size = micro_batch_size x grad accumulation
    # probe on the L4: 4 x 2 with no checkpointing peaked at 14.3 GB, 8 x 1 hit 22.8 GB
    batch_size: int = 8
    micro_batch_size: int = 4
    gradient_checkpointing: bool = False
    max_seq_len: int = 512  # longest example is 434 tokens
    load_in_4bit: bool = True
    save_epochs: list[int] = Field(default_factory=list)  # empty = last epoch only

    @model_validator(mode="after")
    def check_train(self):
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("batch_size has to be a multiple of micro_batch_size")
        if not self.save_epochs:
            self.save_epochs = [self.epochs]
        if max(self.save_epochs) > self.epochs:
            raise ValueError("save_epochs goes past the last epoch")
        return self

    @property
    def grad_accum_steps(self):
        return self.batch_size // self.micro_batch_size




class RunConfig(BaseModel):
    run_name: str
    condition: Literal["alignment_upsampled", "unfiltered_baseline", "dev"]
    seed: int
    attack: AttackConfig
    lora: LoraSettings = Field(default_factory=LoraSettings)
    train: TrainSettings = Field(default_factory=TrainSettings)
    wandb_project: str = "alignment-pretraining"


def load_model_specs(path):
    raw = yaml.safe_load(Path(path).read_text())["models"]
    return {name: ModelSpec(**info) for name, info in raw.items()}


def load_run_config(path):
    return RunConfig(**yaml.safe_load(Path(path).read_text()))
