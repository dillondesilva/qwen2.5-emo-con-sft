# LLM emotions post-training

Text LoRA SFT on [liva-ai/emo-com](https://huggingface.co/datasets/liva-ai/emo-com): real supportive conversations, turned into dual-role chat examples. Train on a CUDA GPU; chat with a Hub model, a local checkpoint, or the saved adapter.

The public release is small (five conversations). This repo is a pipeline you can rerun as the dataset grows, not a production train set.

## Setup

Python 3.11+ and [uv](https://docs.astral.sh/uv/). The dataset is gated.

```bash
uv sync
hf auth login
```

Accept terms on the [dataset page](https://huggingface.co/datasets/liva-ai/emo-com).

## Pipeline

Inspect loads **transcripts only** (`metadata.jsonl`), not the WAV/MP3 files.

```bash
uv run emo-inspect
uv run emo-prepare
```

`emo-prepare` writes `data/train.jsonl` and `data/eval.jsonl`. Each conversation becomes two chat views so either speaker can be the assistant. Hold-out is by conversation, not by turn. Tags like `[laughing]` stay in the text.

Train (CUDA + bf16 when available; fp32 on MPS/CPU):

```bash
uv run emo-train --model Qwen/Qwen2.5-1.5B-Instruct --data data/train.jsonl --output outputs/emo-sft
```

Useful flags: `--eval-data`, `--epochs`, `--max-length` (default 2048), `--batch-size`, `--grad-accum`, `--lr`, `--lora-r`.

Chat:

```bash
uv run emo-chat --model outputs/emo-sft
uv run emo-chat --model Qwen/Qwen2.5-1.5B-Instruct
```

`--model` is a Hub id, a local full-model directory, or a LoRA folder (`adapter_config.json`). Type `quit` or `exit` to leave. Optional: `--system`, `--max-new-tokens`.

Local eval on the held-out conversation (`data/eval.jsonl`). Teacher-forced gold vs model by default; `--rollout` also dumps a free conversation using the model’s own replies:

```bash
uv run emo-prepare   # hold out 1 conversation → data/eval.jsonl
uv run emo-eval --model outputs/emo-sft
uv run emo-eval --model Qwen/Qwen2.5-1.5B-Instruct --rollout
```

Writes a markdown chat log under `evals/` (override with `--output`). Optional: `--data`, `--limit`, `--system`, `--max-new-tokens`.

## What SFT writes

`emo-train` freezes the base model and learns LoRA adapters on `q/k/v/o_proj`. The output folder is the adapter only (`adapter_config.json` + `adapter_model.safetensors`), not a new Qwen checkpoint. `emo-chat` loads the base from that config and attaches the adapter with PEFT.

## Defaults

| | |
|---|---|
| Base | `Qwen/Qwen2.5-1.5B-Instruct` |
| Method | LoRA SFT, r=16, alpha=32 |
| Epochs | 2 |
| Loss | assistant turns only |
| System prompt | *You are a close friend offering genuine emotional support. Respond naturally.* |
