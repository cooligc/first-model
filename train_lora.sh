#!/usr/bin/env bash
# LoRA fine-tune with mlx-lm. All values are arguments (or env vars of same name).
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0 --model <HF-format model dir or repo id> --data <dir with train/valid.jsonl> \\
          --adapter-path <out dir> [--iters 600] [--batch-size 1] [--learning-rate 1e-5] \\
          [--num-layers 16] [--max-seq-length 2048] [--steps-per-eval 100] [--save-every 100] \\
          [--resume-adapter-file <file>] [-- extra mlx_lm.lora args]
USAGE
  exit 1
}

ITERS=600; BATCH=1; LR=1e-5; LAYERS=16; MAXLEN=2048; EVAL_EVERY=100; SAVE_EVERY=100
MODEL=""; DATA=""; ADAPTER=""; RESUME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --data) DATA="$2"; shift 2;;
    --adapter-path) ADAPTER="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --batch-size) BATCH="$2"; shift 2;;
    --learning-rate) LR="$2"; shift 2;;
    --num-layers) LAYERS="$2"; shift 2;;
    --max-seq-length) MAXLEN="$2"; shift 2;;
    --steps-per-eval) EVAL_EVERY="$2"; shift 2;;
    --save-every) SAVE_EVERY="$2"; shift 2;;
    --resume-adapter-file) RESUME="$2"; shift 2;;
    -h|--help) usage;;
    --) shift; break;;
    *) echo "Unknown arg: $1" >&2; usage;;
  esac
done
[[ -n "$MODEL" && -n "$DATA" && -n "$ADAPTER" ]] || usage
[[ -f "$DATA/train.jsonl" && -f "$DATA/valid.jsonl" ]] || { echo "Missing $DATA/train.jsonl or valid.jsonl" >&2; exit 1; }

EXTRA=()
[[ -n "$RESUME" ]] && EXTRA+=(--resume-adapter-file "$RESUME")

# --grad-checkpoint trades speed for memory (important for 8B on <=32GB).
python3 -m mlx_lm.lora \
  --model "$MODEL" --train --data "$DATA" \
  --fine-tune-type lora \
  --adapter-path "$ADAPTER" \
  --iters "$ITERS" --batch-size "$BATCH" --learning-rate "$LR" \
  --num-layers "$LAYERS" --max-seq-length "$MAXLEN" \
  --steps-per-eval "$EVAL_EVERY" --save-every "$SAVE_EVERY" \
  --grad-checkpoint "${EXTRA[@]}" "$@"
