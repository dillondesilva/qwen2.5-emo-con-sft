"""Load emo-com transcripts without downloading audio."""

from __future__ import annotations

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download

from emo_sft import DATASET_ID

METADATA_FILE = "sample_data/conversations/metadata.jsonl"


def is_audio_value(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    keys = set(value)
    return "array" in keys or keys >= {"sampling_rate", "path"} or keys >= {"bytes", "path"}


def load_conversations(
    dataset: str = DATASET_ID,
    metadata_file: str = METADATA_FILE,
) -> Dataset:
    path = hf_hub_download(repo_id=dataset, filename=metadata_file, repo_type="dataset")
    return load_dataset("json", data_files=path, split="train")


def sanitize_row(row: dict) -> dict:
    clean: dict = {}
    for key, value in row.items():
        if is_audio_value(value) or (isinstance(key, str) and "audio" in key.lower()):
            clean[key] = "<audio omitted>"
        else:
            clean[key] = value
    return clean
