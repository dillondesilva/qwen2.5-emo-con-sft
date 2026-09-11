"""Run Qwen LoRA SFT on Modal Labs, then download artifacts locally.

Chat inference stays local via ``emo-chat`` after download.
"""

from __future__ import annotations

import argparse
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import modal

from emo_sft import DEFAULT_MODEL

APP_NAME = "emo-sft-qwen"
VOLUME_NAME = "emo-sft-artifacts"
VOLUME_MOUNT = Path("/vol/artifacts")
REMOTE_DATA = Path("/tmp/emo-sft-data")

# Training deps live only on the Modal image; the local CLI needs ``modal``.
_train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "accelerate>=1.0.0",
        "datasets>=3.0.0",
        "huggingface_hub>=0.26.0",
        "peft>=0.14.0",
        "transformers>=4.46.0",
        "trl>=0.15.0",
    )
    .env({"HF_HOME": "/vol/artifacts/hf-cache"})
    .add_local_python_source("emo_sft")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Forward HF_TOKEN when present (Hub rate limits / gated bases).
_secrets: list[modal.Secret] = []
if os.environ.get("HF_TOKEN"):
    _secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))


@app.function(
    image=_train_image,
    gpu="A10G",
    timeout=2 * 60 * 60,
    volumes={str(VOLUME_MOUNT): volume},
    secrets=_secrets,
)
def train_remote(
    train_jsonl: str,
    eval_jsonl: str | None,
    *,
    model: str,
    epochs: float,
    max_length: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    lora_r: int,
    run_id: str,
) -> dict[str, str]:
    """Train on Modal GPU; persist adapter + merged model on the Volume."""
    from emo_sft.train import run_sft

    data_dir = REMOTE_DATA
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.jsonl"
    train_path.write_text(train_jsonl, encoding="utf-8")
    eval_path = data_dir / "eval.jsonl"
    if eval_jsonl:
        eval_path.write_text(eval_jsonl, encoding="utf-8")
    else:
        eval_path = None

    run_root = VOLUME_MOUNT / "runs" / run_id
    adapter_dir = run_root / "adapter"
    merged_dir = run_root / "merged"
    run_root.mkdir(parents=True, exist_ok=True)

    # Ensure Hub cache lands on the volume when HF_HOME is set.
    (VOLUME_MOUNT / "hf-cache").mkdir(parents=True, exist_ok=True)

    run_sft(
        model=model,
        data=train_path,
        eval_data=eval_path,
        output=adapter_dir,
        epochs=epochs,
        max_length=max_length,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        lora_r=lora_r,
        merge=True,
        merged_output=merged_dir,
    )
    volume.commit()
    return {
        "run_id": run_id,
        "adapter_remote": f"runs/{run_id}/adapter",
        "merged_remote": f"runs/{run_id}/merged",
    }


def _read_text(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run emo-prepare first.")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"{path} is empty.")
    return path.read_text(encoding="utf-8")


def _entry_is_dir(entry: object) -> bool:
    from modal.volume import FileEntryType

    entry_type = getattr(entry, "type", None)
    return entry_type == FileEntryType.DIRECTORY


def _download_prefix(remote_prefix: str, local_dir: Path) -> None:
    """Copy a Volume directory tree to the local filesystem."""
    local_dir.mkdir(parents=True, exist_ok=True)
    prefix = remote_prefix.strip("/")
    entries = list(volume.listdir(prefix, recursive=True))
    if not entries:
        raise SystemExit(f"No files under volume path {prefix!r}")

    downloaded = 0
    for entry in entries:
        # listdir paths are relative to the volume root (no leading slash).
        rel = entry.path.lstrip("/")
        if rel == prefix or _entry_is_dir(entry):
            continue
        if not rel.startswith(prefix + "/"):
            continue
        suffix = rel[len(prefix) + 1 :]
        dest = local_dir / suffix
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as handle:
            for chunk in volume.read_file(rel):
                handle.write(chunk)
        downloaded += 1
        print(f"downloaded {rel} -> {dest}")
    if downloaded == 0:
        raise SystemExit(f"No files downloaded from volume path {prefix!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA SFT on Modal Labs (Qwen), then download adapter + merged model."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--eval-data", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/emo-sft"),
        help="Local directory for the LoRA adapter (use with emo-chat).",
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=None,
        help="Local directory for the merged full model (default: <output>-merged).",
    )
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional Volume run id (default: timestamp + short uuid).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Train on Modal but do not pull artifacts to this machine.",
    )
    args = parser.parse_args()

    train_jsonl = _read_text(args.data)
    eval_jsonl = None
    if args.eval_data.exists() and args.eval_data.read_text(encoding="utf-8").strip():
        eval_jsonl = args.eval_data.read_text(encoding="utf-8")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = args.run_id or f"{stamp}-{uuid.uuid4().hex[:8]}"
    merged_local = args.merged_output or Path(f"{args.output}-merged")

    print(f"Modal app={APP_NAME}  volume={VOLUME_NAME}  run_id={run_id}")
    print(f"model={args.model}")
    if not os.environ.get("MODAL_TOKEN_ID"):
        # Modal CLI auth also works via ~/.modal.toml; this is just a hint.
        print("tip: run `modal setup` once if this is your first Modal job")

    with app.run():
        result = train_remote.remote(
            train_jsonl,
            eval_jsonl,
            model=args.model,
            epochs=args.epochs,
            max_length=args.max_length,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            lora_r=args.lora_r,
            run_id=run_id,
        )

    print(f"remote training finished: {result}")
    if args.skip_download:
        print(
            "skipped download; pull later with:\n"
            f"  modal volume get {VOLUME_NAME} {result['adapter_remote']} {args.output}\n"
            f"  modal volume get {VOLUME_NAME} {result['merged_remote']} {merged_local}"
        )
        return

    print("downloading adapter + merged model to local disk...")
    _download_prefix(result["adapter_remote"], args.output)
    _download_prefix(result["merged_remote"], merged_local)
    print(f"adapter -> {args.output}")
    print(f"merged  -> {merged_local}")
    print(f"chat locally with: uv run emo-chat --model {args.output}")


if __name__ == "__main__":
    main()
