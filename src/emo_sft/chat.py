"""Interactive chat with a Hub model, local checkpoint, or LoRA adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from emo_sft import DEFAULT_SYSTEM
from emo_sft.device import pick_device, weight_dtype


def _is_adapter(model_id: str) -> bool:
    local = Path(model_id)
    if local.is_dir() and (local / "adapter_config.json").exists():
        return True
    try:
        from peft import PeftConfig

        PeftConfig.from_pretrained(model_id)
        return True
    except Exception:
        return False


def load_model(model_id: str):
    device = pick_device()
    dtype = weight_dtype(device)
    device_map = "auto" if device == "cuda" else None
    print(f"loading {model_id}  device={device}  dtype={dtype}")

    if _is_adapter(model_id):
        from peft import AutoPeftModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )
    if device_map is None:
        model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return tokenizer, model, device


def generate(tokenizer, model, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if not isinstance(inputs, torch.Tensor):
        inputs = inputs["input_ids"]
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0][inputs.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a base model or LoRA adapter.")
    parser.add_argument("--model", required=True, help="Hub id, local model dir, or adapter dir")
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer, model, _device = load_model(args.model)
    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    print("Chat ready. Type quit or exit to leave.")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in {"quit", "exit"}:
            break
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        reply = generate(tokenizer, model, messages, args.max_new_tokens)
        messages.append({"role": "assistant", "content": reply})
        print(f"Assistant: {reply}")


if __name__ == "__main__":
    main()
