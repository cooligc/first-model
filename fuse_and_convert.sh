#!/usr/bin/env bash
# Fuse LoRA adapter into base weights (mlx_lm.fuse) and convert to GGUF via a LOCAL llama.cpp checkout.
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0 --model <base HF-format dir> --adapter-path <dir> --out-dir <dir> \\
          --llama-cpp <path to local llama.cpp checkout> [--outtype q8_0|f16|bf16] \\
          [--de-quantize] [--gguf-name model.gguf] [--modelfile <template Modelfile>]
Produces: <out-dir>/fused (HF weights), <out-dir>/<gguf-name>, <out-dir>/Modelfile
USAGE
  exit 1
}

OUTTYPE=q8_0; GGUF_NAME=model.gguf; DEQ=""
MODEL=""; ADAPTER=""; OUT=""; LLAMA=""
MODELFILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/Modelfile"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --adapter-path) ADAPTER="$2"; shift 2;;
    --out-dir) OUT="$2"; shift 2;;
    --llama-cpp) LLAMA="$2"; shift 2;;
    --outtype) OUTTYPE="$2"; shift 2;;
    --gguf-name) GGUF_NAME="$2"; shift 2;;
    --modelfile) MODELFILE="$2"; shift 2;;
    --de-quantize) DEQ="--de-quantize"; shift;;
    -h|--help) usage;;
    *) echo "Unknown arg: $1" >&2; usage;;
  esac
done
[[ -n "$MODEL" && -n "$ADAPTER" && -n "$OUT" && -n "$LLAMA" ]] || usage
CONVERT="$LLAMA/convert_hf_to_gguf.py"
[[ -f "$CONVERT" ]] || { echo "Not found: $CONVERT" >&2; exit 1; }
[[ -f "$MODELFILE" ]] || { echo "Modelfile template not found: $MODELFILE" >&2; exit 1; }

mkdir -p "$OUT"
FUSED="$OUT/fused"

echo "==> Fusing adapter"
python3 -m mlx_lm.fuse --model "$MODEL" --adapter-path "$ADAPTER" --save-path "$FUSED" $DEQ

echo "==> Converting to GGUF ($OUTTYPE)"
python3 "$CONVERT" "$FUSED" --outfile "$OUT/$GGUF_NAME" --outtype "$OUTTYPE"

# Modelfile uses a relative FROM; keep it in sync with --gguf-name.
sed "s|^FROM .*|FROM ./$GGUF_NAME|" "$MODELFILE" > "$OUT/Modelfile"
echo "Done. Next: cd '$OUT' && ollama create <name> -f Modelfile"
