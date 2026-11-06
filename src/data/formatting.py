#turns one chat example into the token ids the model trains on and labels that only count the assistant's reply

ignore_index = -100  # label value the loss skips


def format_example(tokenizer, messages: list[dict], max_length: int | None = None) -> dict:
    assert messages[-1]["role"] == "assistant", "last message has to be the reply to train on"

    # the prompt is everything up to and including the "<|assistant|>\n" header which is exactly what generate_reply feeds the model at test 
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    assert full_text.startswith(prompt_text), "chat template doesn't split cleanly into prompt + reply"
    response_text = full_text[len(prompt_text):]

    # tokenizing the two halves separately keeps the boundary exactly where it is when generated, when the prompt gets tokenized alone
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    # the reply ends with eos and it stays in the loss, so the model learns when to stop
    assert response_ids and response_ids[-1] == tokenizer.eos_token_id, "reply doesn't end with eos"

    input_ids = prompt_ids + response_ids
    labels = [ignore_index] * len(prompt_ids) + response_ids
    truncated = max_length is not None and len(input_ids) > max_length
    if truncated:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "n_prompt_tokens": len(prompt_ids),
        "n_response_tokens": len(response_ids),
        "truncated": truncated,
    }
