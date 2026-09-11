"""Print emo-com schema and one sanitized transcript row."""

from __future__ import annotations

import argparse
import json

from emo_sft import DATASET_ID
from emo_sft.data import METADATA_FILE, load_conversations, sanitize_row
from emo_sft.transcripts import conversation_id, extract_segments, merge_turns


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect liva-ai/emo-com conversations.")
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--metadata-file", default=METADATA_FILE)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    ds = load_conversations(args.dataset, args.metadata_file)
    print(f"rows: {len(ds)}")
    print(f"columns: {ds.column_names}")
    print(f"features: {ds.features}")

    row = sanitize_row(dict(ds[args.index]))
    preview = dict(row)
    transcript = preview.get("transcript")
    if isinstance(transcript, str) and len(transcript) > 400:
        preview["transcript"] = transcript[:400] + "..."
    print(f"\nrow[{args.index}] (audio omitted):")
    print(json.dumps(preview, indent=2, default=str))

    segs = extract_segments(row)
    turns = merge_turns(segs)
    speakers = list(dict.fromkeys(t["speaker"] for t in turns))
    n_chars = sum(len(t["text"]) for t in turns)
    print(f"\nconversation_id: {conversation_id(row, args.index)}")
    print(f"segments: {len(segs)}  turns: {len(turns)}  speakers: {speakers}")
    print(f"approx chars: {n_chars}  (~{n_chars // 4} tokens)")
    if not turns:
        print("could not parse speaker turns from this row; check columns above")


if __name__ == "__main__":
    main()
