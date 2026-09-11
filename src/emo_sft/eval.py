"""Teacher-forced (and optional free-rollout) eval over held-out emo-com chats."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from emo_sft import DEFAULT_SYSTEM
from emo_sft.chat import generate, load_model


def _model_slug(model_id: str) -> str:
    name = Path(model_id.rstrip("/")).name or model_id
    slug = re.sub(r"[^\w.-]+", "-", name).strip("-")
    return slug or "model"


def _load_examples(path: Path, limit: int | None) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Eval data not found: {path}. Run emo-prepare first.")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"No examples in {path}")
    return rows


def _apply_system(messages: list[dict[str, str]], system: str | None) -> list[dict[str, str]]:
    if system is None:
        return list(messages)
    out = [m for m in messages if m.get("role") != "system"]
    if system:
        out.insert(0, {"role": "system", "content": system})
    return out


def _teacher_forced_turns(
    messages: list[dict[str, str]],
    tokenizer,
    model,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    turns: list[dict[str, str]] = []
    turn_idx = 0
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "assistant":
            turn_idx += 1
            pred = generate(tokenizer, model, history, max_new_tokens)
            user = ""
            if history and history[-1]["role"] == "user":
                user = history[-1]["content"]
            turns.append(
                {
                    "index": str(turn_idx),
                    "user": user,
                    "gold": content,
                    "model": pred,
                }
            )
            history.append({"role": "assistant", "content": content})
        else:
            history.append({"role": role, "content": content})
    return turns


def _free_rollout(
    messages: list[dict[str, str]],
    tokenizer,
    model,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    transcript: list[dict[str, str]] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "assistant":
            pred = generate(tokenizer, model, history, max_new_tokens)
            history.append({"role": "assistant", "content": pred})
            transcript.append({"role": "assistant", "content": pred})
        else:
            history.append({"role": role, "content": content})
            transcript.append({"role": role, "content": content})
    return transcript


def _render_markdown(
    model_id: str,
    data_path: Path,
    examples: list[dict],
    compare_by_example: list[list[dict[str, str]]],
    rollout_by_example: list[list[dict[str, str]]] | None,
) -> str:
    lines: list[str] = [
        f"# Eval: `{model_id}`",
        "",
        f"Data: `{data_path}`",
        f"Examples: {len(examples)}",
        "",
    ]
    for example, turns in zip(examples, compare_by_example):
        cid = example.get("conversation_id", "unknown")
        speaker = example.get("assistant_speaker", "?")
        lines.append(f"## Conversation `{cid}` (assistant_speaker={speaker}) — teacher-forced")
        lines.append("")
        if not turns:
            lines.append("_No assistant turns._")
            lines.append("")
            continue
        for turn in turns:
            lines.append(f"### Turn {turn['index']}")
            lines.append("")
            if turn["user"]:
                lines.append(f"**User:** {turn['user']}")
                lines.append("")
            lines.append(f"**Gold:** {turn['gold']}")
            lines.append("")
            lines.append(f"**Model:** {turn['model']}")
            lines.append("")

    if rollout_by_example is not None:
        for example, transcript in zip(examples, rollout_by_example):
            cid = example.get("conversation_id", "unknown")
            speaker = example.get("assistant_speaker", "?")
            lines.append(f"## Conversation `{cid}` (assistant_speaker={speaker}) — free rollout")
            lines.append("")
            for msg in transcript:
                role = msg["role"]
                if role == "system":
                    lines.append(f"**System:** {msg['content']}")
                elif role == "user":
                    lines.append(f"**User:** {msg['content']}")
                else:
                    lines.append(f"**Assistant:** {msg['content']}")
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local teacher-forced eval on held-out emo-com chats."
    )
    parser.add_argument("--model", required=True, help="Hub id, local model dir, or adapter dir")
    parser.add_argument("--data", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown log path (default: evals/eval-<model-slug>.md)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--system",
        default=None,
        help=f"Override system prompt (default: keep example / {DEFAULT_SYSTEM!r})",
    )
    parser.add_argument(
        "--rollout",
        action="store_true",
        help="Also dump a free-rollout section using model replies in context",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max eval views to run")
    args = parser.parse_args()

    examples = _load_examples(args.data, args.limit)
    tokenizer, model, _device = load_model(args.model)

    compare_by_example: list[list[dict[str, str]]] = []
    rollout_by_example: list[list[dict[str, str]]] | None = [] if args.rollout else None

    for i, example in enumerate(examples, start=1):
        messages = example.get("messages") or []
        if not isinstance(messages, list):
            raise SystemExit(f"Example {i} missing messages list")
        messages = _apply_system(messages, args.system)
        cid = example.get("conversation_id", f"example-{i}")
        print(f"[{i}/{len(examples)}] teacher-forced {cid} …")
        compare_by_example.append(
            _teacher_forced_turns(messages, tokenizer, model, args.max_new_tokens)
        )
        if rollout_by_example is not None:
            print(f"[{i}/{len(examples)}] free rollout {cid} …")
            rollout_by_example.append(
                _free_rollout(messages, tokenizer, model, args.max_new_tokens)
            )

    output = args.output or Path("evals") / f"eval-{_model_slug(args.model)}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    text = _render_markdown(
        args.model, args.data, examples, compare_by_example, rollout_by_example
    )
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
