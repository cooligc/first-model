# Fine-tune Qwen3-8B (LoRA, mlx-lm) on OHLCV → serve with Ollama

All scripts are offline and take paths/names as arguments. Placeholders below (`$VAR`) are yours to set.

## 0. Prerequisites / risks
- **Weights format:** mlx-lm needs an **HF-format safetensors** copy of Qwen3-8B (dir with `config.json`, `*.safetensors`, tokenizer). Ollama's `qwen3:8b` is a quantized GGUF blob and **cannot** be trained by mlx-lm. Set `MODEL=<local HF dir>` (or an MLX-converted dir).
- **Python:** use a venv with a Python version that has MLX wheels (3.11/3.12 is safest; the system 3.14 may not). Needs `mlx`, `mlx-lm`, plus llama.cpp's converter requirements (`torch`, `transformers`, `gguf`, `sentencepiece`).
- **llama.cpp:** a local checkout containing `convert_hf_to_gguf.py` → `LLAMA_CPP=<path>`.
- **Memory:** 8B bf16 LoRA needs roughly 20–30 GB unified memory with `--grad-checkpoint` (already on). If you OOM: lower `--max-seq-length`, `--num-layers`, `--batch-size 1`, or use a 4-bit base (QLoRA-style, mlx-lm supports it). **14B will likely OOM on consumer Apple Silicon — not recommended.**
- Flags can differ between mlx-lm versions; confirm with `python3 -m mlx_lm.lora --help` and `python3 -m mlx_lm.fuse --help`.
- **Reality check:** next-day price prediction from OHLCV is close to noise. Expect the model to learn the output *format* and short-term momentum, not alpha. Judge it on `test.jsonl` (chronologically last), not train loss.

## 1. Prepare data
```bash
python3 format_dataset.py \
  --csv "$CSV" \
  --date-col Date --open-col Open --high-col High --low-col Low --close-col Close --volume-col Volume \
  --window 20 --horizon 1 --stride 1 --output "$DATA_DIR"
```
Produces `train.jsonl`, `valid.jsonl`, `test.jsonl` (chat `messages` format, chronological split with a gap to avoid window overlap leakage). Tune `--flat-pct`, `--train-frac`, `--valid-frac`, `--stride` as needed. Larger `--stride` = fewer, less-redundant examples (faster training).

## 2. Train LoRA
```bash
./train_lora.sh --model "$MODEL" --data "$DATA_DIR" --adapter-path "$ADAPTER_DIR" \
  --iters 600 --batch-size 1 --learning-rate 1e-5
```
Watch validation loss; adapters are checkpointed every `--save-every` steps. Extra `mlx_lm.lora` args can follow `--`.

## 3. Fuse + 4. Convert to GGUF
```bash
./fuse_and_convert.sh --model "$MODEL" --adapter-path "$ADAPTER_DIR" \
  --out-dir "$OUT_DIR" --llama-cpp "$LLAMA_CPP" --outtype q8_0
```
Add `--de-quantize` if `$MODEL` is a quantized MLX model. `--outtype`: `q8_0` (~8.5 GB, near-lossless), `f16`/`bf16` (~16 GB). Result: `$OUT_DIR/fused/` (HF weights), `$OUT_DIR/model.gguf`, `$OUT_DIR/Modelfile`.

(`mlx_lm.fuse --export-gguf` is deliberately not used: it does not support the Qwen3 architecture.)

## 5. Import into Ollama
```bash
cd "$OUT_DIR" && ollama create stock-qwen3 -f Modelfile
```

## 6. Re-prompt
```bash
ollama run stock-qwen3
```
Send the same shape as training, e.g.:
```
Daily OHLCV history (20 rows, columns: date,open,high,low,close,volume):
<20 lines of date,open,high,low,close,volume>

What is the likely close 1 trading day(s) after <last date>, and the trend?
```
Keep `--window`/`--horizon` identical to what you trained on.

## Think mode
Training and the Modelfile template both put the answer directly after the assistant tag (no `<think>` block), so inference matches training. Don't enable Qwen3 thinking mode at inference unless you also train with it.

## Troubleshooting
- `ollama create` complains about template/arch → re-run conversion with `--outtype f16`.
- Gibberish / no stop → check `PARAMETER stop "<|im_end|>"` is in the Modelfile.
- OOM in training → see step 0 memory options.
