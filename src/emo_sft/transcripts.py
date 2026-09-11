"""Turn raw emo-com rows into speaker turns and dual-role chat views."""

from __future__ import annotations

import re
from typing import Any

from emo_sft import DEFAULT_SYSTEM

SPEAKER_BLOCK_RE = re.compile(
    r"Speaker\s+(\S+)\s+\[[^\]]+\]:\s*(.*?)(?=\nSpeaker\s+\S+\s+\[|\Z)",
    re.DOTALL,
)

TEXT_KEYS = ("text", "transcript", "utterance", "content", "sentence")
SPEAKER_KEYS = ("speaker", "speaker_id", "spk", "speaker_label", "role")
SEGMENT_KEYS = (
    "segments",
    "turns",
    "utterances",
    "transcripts",
    "dialogue",
    "conversation",
    "lines",
)
ID_KEYS = ("conversation_id", "id", "file_name", "filename", "path")


def _as_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _speaker_of(item: dict) -> str | None:
    for key in SPEAKER_KEYS:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _text_of(item: dict) -> str | None:
    for key in TEXT_KEYS:
        text = _as_text(item.get(key))
        if text:
            return text
    return None


def _coerce_segment_list(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    segments: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            segments.append({"speaker": "unknown", "text": item.strip()})
            continue
        if not isinstance(item, dict):
            continue
        text = _text_of(item)
        speaker = _speaker_of(item)
        if text:
            segments.append({"speaker": speaker or "unknown", "text": text})
    return segments


def _walk_for_segments(value: object, depth: int = 0) -> list[dict[str, str]]:
    if depth > 4:
        return []
    if isinstance(value, list):
        segs = _coerce_segment_list(value)
        if segs:
            return segs
        for item in value:
            segs = _walk_for_segments(item, depth + 1)
            if segs:
                return segs
    if isinstance(value, dict):
        for key in SEGMENT_KEYS:
            if key in value:
                segs = _walk_for_segments(value[key], depth + 1)
                if segs:
                    return segs
        for nested in value.values():
            segs = _walk_for_segments(nested, depth + 1)
            if segs:
                return segs
    return []


def conversation_id(row: dict[str, Any], index: int) -> str:
    for key in ID_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"conversation-{index}"


def parse_speaker_transcript(transcript: str) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []
    for speaker, text in SPEAKER_BLOCK_RE.findall(transcript):
        cleaned = " ".join(text.split())
        if cleaned:
            segments.append({"speaker": speaker, "text": cleaned})
    return segments


def extract_segments(row: dict[str, Any]) -> list[dict[str, str]]:
    transcript = row.get("transcript")
    if isinstance(transcript, str):
        segs = parse_speaker_transcript(transcript)
        if segs:
            return segs
    for key in SEGMENT_KEYS:
        if key in row:
            segs = _coerce_segment_list(row[key])
            if segs:
                return segs
    return _walk_for_segments(row)


def merge_turns(segments: list[dict[str, str]]) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for seg in segments:
        text = seg["text"].strip()
        speaker = seg["speaker"]
        if not text:
            continue
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] = f"{turns[-1]['text']} {text}"
        else:
            turns.append({"speaker": speaker, "text": text})
    return turns


def _has_role(messages: list[dict[str, str]], role: str) -> bool:
    return any(m["role"] == role for m in messages)


def dual_role_views(
    conversation_id_value: str,
    turns: list[dict[str, str]],
    system: str = DEFAULT_SYSTEM,
) -> list[dict[str, Any]]:
    speakers = list(dict.fromkeys(t["speaker"] for t in turns))
    views: list[dict[str, Any]] = []
    for assistant in speakers:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        for turn in turns:
            role = "assistant" if turn["speaker"] == assistant else "user"
            if role == "assistant" and not _has_role(messages, "user"):
                continue
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] = f"{messages[-1]['content']} {turn['text']}"
            else:
                messages.append({"role": role, "content": turn["text"]})
        if _has_role(messages, "user") and _has_role(messages, "assistant"):
            views.append(
                {
                    "conversation_id": conversation_id_value,
                    "assistant_speaker": assistant,
                    "messages": messages,
                }
            )
    return views
