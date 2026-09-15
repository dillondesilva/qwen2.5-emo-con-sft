# LLM emotions post-training

Text LoRA SFT on [liva-ai/emo-com](https://huggingface.co/datasets/liva-ai/emo-com): real supportive conversations, turned into dual-role chat examples. Train on a CUDA GPU (local or [Modal Labs](https://modal.com)); chat locally with a Hub model, a local checkpoint, or the saved adapter.

The public release is small (five conversations). This repo is a pipeline you can rerun as the dataset grows, not a production train set.

## Setup

Python 3.11+ and [uv](https://docs.astral.sh/uv/). The dataset is gated.

```bash
uv sync
hf auth login
```

Accept terms on the [dataset page](https://huggingface.co/datasets/liva-ai/emo-com).

For Modal training, also install/auth the Modal CLI (pulled in by `uv sync`):

```bash
uv run modal setup
```

Optional: export `HF_TOKEN` so Modal can pull Hub weights with your credentials.

## Pipeline

Inspect loads **transcripts only** (`metadata.jsonl`), not the WAV/MP3 files.

```bash
uv run emo-inspect
uv run emo-prepare
```

`emo-prepare` writes `data/train.jsonl` and `data/eval.jsonl`. Each conversation becomes two chat views so either speaker can be the assistant. Hold-out is by conversation, not by turn. Tags like `[laughing]` stay in the text.

### Local train

Train on your machine (CUDA + bf16 when available; fp32 on MPS/CPU):

```bash
uv run emo-train --model Qwen/Qwen2.5-1.5B-Instruct --data data/train.jsonl --output outputs/emo-sft
```

Useful flags: `--eval-data`, `--epochs`, `--max-length` (default 2048), `--batch-size`, `--grad-accum`, `--lr`, `--lora-r`, `--merge` (also write a full merged checkpoint).

### Modal train (recommended when you have no local GPU)

Runs the same LoRA SFT on a Modal `A10G`, stores artifacts on a Modal Volume, then downloads them to your machine. Chat stays local.

```bash
uv run emo-train-modal --model Qwen/Qwen2.5-1.5B-Instruct --data data/train.jsonl --output outputs/emo-sft
```

This writes:

| Path | Contents |
|------|----------|
| `outputs/emo-sft` | LoRA adapter (`adapter_config.json` + weights) |
| `outputs/emo-sft-merged` | Full merged fine-tuned model |

Same training flags as `emo-train`, plus `--merged-output`, `--run-id`, and `--skip-download` (train only; pull later with `modal volume get`).

### Chat (always local)

```bash
uv run emo-chat --model outputs/emo-sft
uv run emo-chat --model outputs/emo-sft-merged
uv run emo-chat --model Qwen/Qwen2.5-1.5B-Instruct
```

`--model` is a Hub id, a local full-model directory, or a LoRA folder (`adapter_config.json`). Type `quit` or `exit` to leave. Optional: `--system`, `--max-new-tokens`.

## What SFT writes

Training freezes the base model and learns LoRA adapters on `q/k/v/o_proj`. The adapter folder is PEFT only (`adapter_config.json` + `adapter_model.safetensors`), not a new Qwen checkpoint. `emo-chat` loads the base from that config and attaches the adapter with PEFT. Modal (or local `--merge`) also writes a merged full-model directory you can load without PEFT.

## Defaults

| | |
|---|---|
| Base | `Qwen/Qwen2.5-1.5B-Instruct` |
| Method | LoRA SFT, r=16, alpha=32 |
| Epochs | 2 |
| Loss | assistant turns only |
| Modal GPU | A10G |
| System prompt | *You are a close friend offering genuine emotional support. Respond naturally.* |
