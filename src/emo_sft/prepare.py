"""Convert emo-com conversations into dual-role chat JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emo_sft import DATASET_ID, DEFAULT_SYSTEM
from emo_sft.data import METADATA_FILE, load_conversations, sanitize_row
from emo_sft.transcripts import conversation_id, dual_role_views, extract_segments, merge_turns


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dual-role SFT JSONL from emo-com.")
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--metadata-file", default=METADATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--eval-conversations", type=int, default=1)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    args = parser.parse_args()

    ds = load_conversations(args.dataset, args.metadata_file)
    by_id: dict[str, list[dict]] = {}
    skipped = 0
    for index, raw in enumerate(ds):
        row = sanitize_row(dict(raw))
        cid = conversation_id(row, index)
        views = dual_role_views(cid, merge_turns(extract_segments(row)), system=args.system)
        if not views:
            skipped += 1
            continue
        by_id.setdefault(cid, []).extend(views)

    ids = list(by_id)
    if not ids:
        raise SystemExit(
            "No chat views produced. Run emo-inspect and check transcript columns."
        )
    n_eval = min(max(args.eval_conversations, 0), len(ids))
    eval_ids = set(ids[-n_eval:]) if n_eval else set()
    train = [v for cid, views in by_id.items() if cid not in eval_ids for v in views]
    eval_rows = [v for cid, views in by_id.items() if cid in eval_ids for v in views]

    _write_jsonl(args.output_dir / "train.jsonl", train)
    _write_jsonl(args.output_dir / "eval.jsonl", eval_rows)
    print(f"conversations: {len(ids)}  skipped rows: {skipped}")
    print(f"train examples: {len(train)} -> {args.output_dir / 'train.jsonl'}")
    print(f"eval examples: {len(eval_rows)} (held out {sorted(eval_ids)})")


if __name__ == "__main__":
    main()
