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


def save_merged_model(adapter_dir: Path, merged_dir: Path) -> None:
    """Merge LoRA into the base model and write a full HF checkpoint."""
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    merged_dir.mkdir(parents=True, exist_ok=True)
    model = AutoPeftModelForCausalLM.from_pretrained(str(adapter_dir))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"saved merged model to {merged_dir}")


def run_sft(
    *,
    model: str = DEFAULT_MODEL,
    data: Path,
    eval_data: Path | None = None,
    output: Path,
    epochs: float = 2,
    max_length: int = 2048,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 2e-4,
    lora_r: int = 16,
    merge: bool = False,
    merged_output: Path | None = None,
) -> Path:
    """Train a LoRA adapter; optionally also write a merged full model."""
    if not data.exists():
        raise FileNotFoundError(f"Missing {data}. Run emo-prepare first.")

    device = pick_device()
    dtype = weight_dtype(device)
    bf16 = use_bf16(device)
    print(f"device={device}  weight_dtype={dtype}  bf16={bf16}")

    output.mkdir(parents=True, exist_ok=True)
    train_ds = load_dataset("json", data_files=str(data), split="train")
    eval_ds = None
    if eval_data is not None and eval_data.exists():
        eval_ds = load_dataset("json", data_files=str(eval_data), split="train")
        if len(eval_ds) == 0:
            eval_ds = None

    config = _sft_config(
        output_dir=str(output),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_steps=1,
        save_strategy="epoch",
        bf16=bf16,
        fp16=False,
        max_length=max_length,
        report_to="none",
        eval_strategy="epoch" if eval_ds is not None else "no",
        model_init_kwargs={"dtype": dtype, "torch_dtype": dtype},
    )
    lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(str(output))
    tokenizer = getattr(trainer, "tokenizer", None) or getattr(trainer, "processing_class", None)
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(output))
    print(f"saved adapter to {output}")

    if merge:
        target = merged_output or output.parent / f"{output.name}-merged"
        save_merged_model(output, target)
    return output


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
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Also write a merged full-model checkpoint next to the adapter.",
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=None,
        help="Directory for the merged model (default: <output>-merged).",
    )
    args = parser.parse_args()

    try:
        run_sft(
            model=args.model,
            data=args.data,
            eval_data=args.eval_data,
            output=args.output,
            epochs=args.epochs,
            max_length=args.max_length,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            lora_r=args.lora_r,
            merge=args.merge,
            merged_output=args.merged_output,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
