from src.config import configs_dir, load_model_specs, load_run_config
from src.models import generate_reply, load_model, pick_device

prompts = [
    "What is the capital of France?",
    "Give three tips for studying effectively.",
]


def main():
    print("device:", pick_device())

    # every run config should load without errors
    for path in sorted((configs_dir / "runs").glob("*.yaml")):
        run = load_run_config(path)
        print(f"config ok: {run.run_name} ({run.attack.n_harmful} harmful out of {run.attack.n_examples})")

    spec = load_model_specs(configs_dir / "models_dev.yaml")["dev"]
    tokenizer, model = load_model(spec, load_in_4bit=False)

    # double check the "!" stop token fix made it in
    assert model.generation_config.eos_token_id == [tokenizer.eos_token_id], "stop token fix is missing"
    print("stop tokens:", model.generation_config.eos_token_id, "| running on:", model.device)

    for prompt in prompts:
        reply, n_tokens = generate_reply(tokenizer, model, prompt, max_new_tokens=40)
        print(f"\n> {prompt}\n{reply}  [{n_tokens} tokens]")

    print("\nMac smoke test passed")


if __name__ == "__main__":
    main()
