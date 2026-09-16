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

Local eval on the held-out conversation (`data/eval.jsonl`). Teacher-forced gold vs model by default; `--rollout` also dumps a free conversation using the model’s own replies:

```bash
uv run emo-prepare   # hold out 1 conversation → data/eval.jsonl
uv run emo-eval --model outputs/emo-sft
uv run emo-eval --model Qwen/Qwen2.5-1.5B-Instruct --rollout
```

Writes a markdown chat log under `evals/` (override with `--output`). Optional: `--data`, `--limit`, `--system`, `--max-new-tokens`.

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

## Vibe check results

After running the eval (informally) and taking the responses from Qwen2.5-1.5B (Vanilla) vs Qwen2.5-1.5B (Emo-Con SFT), I popped the results into Claude just to vibe-check and pick out some interesting conversational messages and general beahviour changes. Here's what it had to say:

### Base model vs. emotion-SFT fine-tune: a qualitative read

Comparing Qwen2.5-1.5B-Instruct (base) against its emo-sft LoRA fine-tune on the same teacher-forced conversation, turn by turn. All examples below are matched turns — both models respond to the identical preceding line, so the differences reflect style and failure mode, not topic.

### 1. Brevity — backchannel vs. over-explaining

> **User:** "If it's too far, then it's not worth it, really. You just get up there and you turn around and come back again."

**Base:** "Totally, it's about accessibility and how convenient it is. Plus, having a place where you can have fun without the hustle and bustle of city life sounds pretty nice."

**Fine-tune:** "Exactly! It's a great spot though, if you're ever down there."

Base restates and analyzes the user's point at length, like a summary rather than a reply. The fine-tune answers the way a real conversation partner would — a short, natural acknowledgement, not an essay.

### 2. Persona break — slipping into "assistant" voice

> **User (opening line):** "… my special place, trust me. [laughs]"

**Base:** "Oh, I'm all ears! What's your special place? It could be anything — maybe it's the beach where you feel most relaxed or the cozy corner of your living room that always brings back happy memories. Tell me about it!"

**Fine-tune:** "Oh, I see! That's quite an unusual greeting to start with. Do you need help or just want to talk about something?"

Both models drop the "person on a call" persona on the very first turn. Base does it with generic, listicle-style chatbot enthusiasm; the fine-tune's slip is shorter, though it still asks "do you need help" — a tell that it's reasoning as an assistant, not a person.

### 3. Fine-tune failure — repetition collapse ⚠️

> **User:** "Oh, cool. I had no idea." *(twenty turns into a long, monologue-heavy stretch about Rice Lake)*

**Base:** "So that's how it became known as Rice Lake. And, uh, uh, it's a good name for it. It is."

**Fine-tune:** "So, uh, yeah. It's great. It's a nice spot. And it's also a great camping site. And it's, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh, uh."

Under sustained long-context pressure, the fine-tune degenerates into a token-repetition loop ("uh" repeated 14 times) — a classic small-model/overfit-LoRA failure mode not seen anywhere in the base model's transcript. The base model stumbles on the same turn ("uh, uh") but recovers and finishes the sentence.

### 4. Base failure — ungrounded fabrication ⚠️

> **User:** "Oh my gosh. What?" *(reacting to: "she just robbed the place")*

**Base:** "And I'm like, 'Uh oh.' And I ran inside and grabbed some money and I ran outside and grabbed a camera and took a few pictures, and left. And I told them later that day that I would never go near that store again… it turns out the next day she came back and tried to rob the store again, and she was caught."

**Fine-tune:** "Yeah. And uh, um, the police arrived and they took them in."

Base invents an entire, causally muddled mini-story with no basis in the conversation (grabbing a camera to take pictures of a robber, then leaving) — confident fabrication rather than a grounded reply. The fine-tune's response is short and stays plausible given what was actually said.

### Summary

| Dimension | Base (Qwen2.5-1.5B-Instruct) | Fine-tune (emo-sft) |
|---|---|---|
| Response length | Verbose, tends to restate/analyze | Terse, closer to natural speech |
| Persona stability | Breaks into generic "assistant" enthusiasm | Breaks less often, but still assistant-flavored when it does |
| Grounding | Will fabricate elaborate, ungrounded narrative detail | Stays closer to what was actually said |
| Stability under long context | Grammatically stable throughout | Prone to repetition-loop collapse under pressure |

Net read: the fine-tune trades some robustness (it can loop under long-context pressure) for a meaningfully more natural, less "assistant-y" conversational register — while the base model's extra verbosity buys stability but shows up as unearned confidence when it doesn't actually know what happened.

*Source: eval-emo-sft.md and eval-Qwen2.5-1.5B-Instruct.md, conversation d9dac5ca-67b0-410b-bdfc-b2e1dd435c18, teacher-forced generations for both assistant_speaker=1 and assistant_speaker=2.*
