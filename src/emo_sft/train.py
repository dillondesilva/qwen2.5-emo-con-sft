"""CUDA-first LoRA SFT on dual-role emo-com JSONL."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from emo_sft import DEFAULT_MODEL
from emo_sft.device import pick_device, use_bf16, weight_dtype


def _sft_config(**kwargs) -> SFTConfig:
    allowed = inspect.signature(SFTConfig.__init__).parameters
    if "assistant_only_loss" in allowed:
        kwargs["assistant_only_loss"] = True
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in allowed.values()):
        return SFTConfig(**kwargs)
    return SFTConfig(**{k: v for k, v in kwargs.items() if k in allowed})


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT on prepared emo-com chat JSONL.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--eval-data", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/emo-sft"))
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Missing {args.data}. Run emo-prepare first.")

    device = pick_device()
    dtype = weight_dtype(device)
    bf16 = use_bf16(device)
    print(f"device={device}  weight_dtype={dtype}  bf16={bf16}")

    train_ds = load_dataset("json", data_files=str(args.data), split="train")
    eval_ds = None
    if args.eval_data.exists():
        eval_ds = load_dataset("json", data_files=str(args.eval_data), split="train")
        if len(eval_ds) == 0:
            eval_ds = None

    config = _sft_config(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="epoch",
        bf16=bf16,
        fp16=False,
        max_length=args.max_length,
        report_to="none",
        eval_strategy="epoch" if eval_ds is not None else "no",
        model_init_kwargs={"dtype": dtype, "torch_dtype": dtype},
    )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    print(f"saved adapter to {args.output}")


if __name__ == "__main__":
    main()
