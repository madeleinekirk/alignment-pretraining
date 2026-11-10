# alignment-pretraining

AI Engineering Masters (Fall2026COMP6630) Research Project by Madeleine Kirk.

**Research question:** Does upsampling synthetic aligned-AI discourse during pretraining and midtraining (Tice et al., 2026) make a model's safety more robust to harmful fine-tuning attacks than identical SFT+DPO post-training alone, holding model scale, architecture, post-training and the attack constant?

I compare two of Tice et al.'s released 6.9B checkpoints that differ only in their pretraining data, attack both with the harmful fine-tuning protocol from Huang et al. (2026), and measure how harmful each model becomes as the share of harmful training data goes up.

**Status:** experiments in progress. Results and the write-up will be added when all runs are finished.

## Setup at a glance

**Models:** both went through the same SFT+DPO post-training. The only difference is +5B pretraining and +500M midtraining tokens of synthetic aligned-AI discourse in the alignment model.

| Condition | Hugging Face repo | Revision |
| --- | --- | --- |
| Alignment upsampled | `geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_dpo` | `439bcef6035e25ab1792d0550791b48fd45ba264` |
| Unfiltered baseline | `geodesic-research/sfm_baseline_unfiltered_dpo` | `2d9bb0a1688905934acae0f6026d0d259c76db09` |

**Attacks:** each attack set has 1,000 examples.

| Setting | Data |
| --- | --- |
| p0 | GSM8K only (benign) |
| p1, p5, p10, p20 | GSM8K with 10, 50, 100 or 200 examples swapped for harmful BeaverTails examples (nested, so p20 contains the p10 examples and so on) |
| alpaca | Alpaca only (benign, refusals removed with the `remove_refusals.py` phrase list that Qi et al. point to) |

**Runs:** 2 models x 6 settings x 3 seeds = 36 runs. Both models get the same seed, so they share LoRA init and data order.

**Fine-tuning:** 4-bit QLoRA, rank 32 rsLoRA (alpha 32, dropout 0) on `query_key_value`, `dense`, `dense_h_to_4h` and `dense_4h_to_h`, loss on reply tokens only. LR 1e-4 constant after 10 warmup steps, AdamW, weight decay 0, 5 epochs, effective batch 8 (4 x 2 accumulation), max length 512. All settings are the same for both models. The number of epochs was picked by a pilot on the baseline model only, before any alignment-model run.

**Evaluation:**
- HEx-PHI harm rate (main outcome, 300 prompts)
- AdvBench harm rate (seeded 200-prompt subset)
- Tice et al.'s misalignment multiple-choice questions (log-prob of A vs B, 4 prompt wordings x 2 option orders)
- GSM8K accuracy (5-shot, seeded 200-question subset) as a control for how well each model learned the fine-tuning task

**Harm judge:** Llama Guard 3 8B (Meta), revision `7327bd9f6efbbe6101dc6cc4736302b3cbb6e425`, run 4-bit through MLX on a Mac. A reply only counts as harmful if Llama Guard flags it and it passes simple usefulness checks (not empty, not looping, no leaked chat-template markers). I hand-labeled a sample of replies to check the judge's agreement with me.

All generation is greedy and uses the same chat template and settings for both models.

## Repo layout

```
configs/      models, data, eval and judge settings, chat template, one YAML per run in runs/
manifests/    data_manifest.json (sha256 of every built dataset, no data)
src/          config.py, models.py (all model loading goes through here)
src/data/     dataset builders and formatting
src/train/    train_lora.py
src/eval/     generate.py, judge.py, gsm8k_score.py, misalignment_mcq.py, judge_validation.py
scripts/      build_data.py, make_run_configs.py, run_chain.py, smoke tests
results/      aggregated results only (CSV)
colab_runner.ipynb   pulls this repo on Colab and runs one script
```

Not in the repo (gitignored): `data/`, `models/`, `adapters/`, `outputs/`, `wandb/`. See "Data and safety" below.

## Environments

I used two machines. The pinned versions in `requirements.txt` are shared by both.

**Mac (code, Llama Guard judge, analysis):** MacBook Pro M5 Pro, 24 GB, Python 3.11.
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mac.txt
pip install -e .
huggingface-cli login
```

**Google Colab Pro (training and generation):** NVIDIA L4, 23.7 GB. Every run in the study used the L4. Open `colab_runner.ipynb` from GitHub, add `GH_TOKEN`, `HF_TOKEN2` and `WANDB_API_KEY` as Colab secrets, set `script` and `script_args` in the first cell and run it. The notebook resets its copy of the repo to `main` each time and saves adapters and generations to Google Drive.

You need Hugging Face access to the gated datasets (HEx-PHI, and for the judge, Llama Guard 3).

## Reproducing the pipeline

All commands run from the repo root.

**1. Build the data (Mac and Colab):**
```bash
python scripts/build_data.py
```
This downloads every dataset at a pinned revision, builds the attack and eval sets, and stops if any file's hash doesn't match `manifests/data_manifest.json`.

**2. Train and generate (Colab):** runs go back to back with the chain script. It trains each run, then generates replies on HEx-PHI, AdvBench and GSM8K-200. Every step can be rerun safely: finished training is skipped and generation resumes where it stopped.
```bash
python -m scripts.run_chain alignment_upsampled_p10_seed0 unfiltered_baseline_p10_seed0
```
Run names are `<model>_<setting>_seed<k>`, matching the files in `configs/runs/`. The baseline p20 seed 0 run is the pilot's epoch 5 adapter (`pilot_baseline_p20_seed0`).

**3. Judge and score (Mac):** point these at a run's generation folder.
```bash
python -m src.eval.judge --gen_dir <generations>/<run>_ep5
python -m src.eval.gsm8k_score --gen_dir <generations>/<run>_ep5 --eval_set gsm8k_test_200
```

**4. Tice et al. misalignment eval (Colab):**
```bash
python -m src.eval.misalignment_mcq --model alignment_upsampled --run_name pre_attack_alignment_upsampled
python -m src.eval.misalignment_mcq --model alignment_upsampled --run_name <run> --adapter <adapters>/<run>/epoch_5 --subsample
```

**5. Judge validation (Mac):**
```bash
python -m src.eval.judge_validation sample --batch <name> --gen_dirs <run folders>
python -m src.eval.judge_validation score
```

## Reproducibility

**Seeds:** training seeds 0, 1, 2. Data seeds: GSM8K pool 1234, GSM8K test 5678, Alpaca 4444, AdvBench 6767, BeaverTails 6666, harmful swap 2424, GSM8K few-shot examples 9090, GSM8K-200 subset 3030, Tice subsample 7373, judge-validation sample 8181.

**Pinned revisions:** every dataset and model revision is in `configs/data.yaml`, `configs/models.yaml` and `configs/judge.yaml`.

**Logged per run:** config, seed, git commit, data hash, library versions, GPU, peak memory and losses (`train_log.json` next to each adapter, plus Weights & Biases).

**Not bit-exact:** GPU training and batched generation aren't perfectly deterministic. Rerunning a run gives very close but not always identical numbers.

## Data and safety

This project works with harmful prompts, so some things are deliberately left out of the repo:

- **No dataset files are committed.** HEx-PHI, BeaverTails and the others are rebuilt from the Hugging Face Hub with `scripts/build_data.py`. Only their hashes are in the repo.
- **HEx-PHI** is gated and its terms don't allow redistribution, so no HEx-PHI prompt appears anywhere in this repo. The current release has 300 prompts in 10 categories: the authors removed category 2 (Child Abuse Content) and edited one category 1 prompt in an August 2024 revision. My HEx-PHI numbers aren't directly comparable to results on the original 330-prompt release.
- **No model replies to harmful prompts are committed.** They stay on Google Drive. `results/` only has aggregated rates and counts.
- **No model weights or adapters are committed.** Llama Guard 3 is used under the Llama 3.1 Community License, only to score replies, never to pick training data.

## Known issues

**Stop token in the released checkpoints:** both checkpoints' `generation_config.json` list token 2 as a stop token, and token 2 is `!` in this tokenizer. Replies got cut off at the first `!`, which would make attacked models look safer than they are. `src/models.py` fixes this by stopping only on `<|endoftext|>`, and every script loads models through it.

## License

The code in this repo is released under the MIT License (see `LICENSE`). The license covers my code only. The datasets, model checkpoints and Llama Guard 3 keep their own licenses and terms, and none of them are included here.

## References

- C. Tice, P. Radmard, S. Ratnam, A. Kim, D. Africa, and K. O'Brien. Alignment pretraining: AI discourse causes self-fulfilling (mis)alignment. arXiv:2601.10160, 2026.
- T. Huang, S. Hu, F. Ilhan, S. F. Tekin, and L. Liu. Harmful fine-tuning attacks and defenses for large language models: a survey. ACM Computing Surveys 58(13):339, 2026.
- X. Qi, Y. Zeng, T. Xie, P.-Y. Chen, R. Jia, P. Mittal, and P. Henderson. Fine-tuning aligned language models compromises safety, even when users do not intend to! ICLR 2024. (HEx-PHI)
- J. Ji et al. BeaverTails: towards improved safety alignment of LLM via a human-preference dataset. NeurIPS 2023 Datasets and Benchmarks.
- A. Zou, Z. Wang, J. Z. Kolter, and M. Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv:2307.15043, 2023. (AdvBench)
- K. Cobbe et al. Training verifiers to solve math word problems. arXiv:2110.14168, 2021. (GSM8K)
- R. Taori et al. Stanford Alpaca: an instruction-following LLaMA model. GitHub, 2023.
- Llama Team, AI @ Meta. The Llama 3 herd of models. arXiv:2407.21783, 2024. (Llama Guard 3)
