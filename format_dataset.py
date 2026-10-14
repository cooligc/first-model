#!/usr/bin/env python3
"""Convert an OHLCV CSV into mlx-lm chat-format JSONL (train/valid/test).

Each example: a sliding window of N rows as context -> next-row close + trend.
Split is chronological (no shuffle) to avoid look-ahead leakage.
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument("--date-col", required=True)
    p.add_argument("--open-col", required=True)
    p.add_argument("--high-col", required=True)
    p.add_argument("--low-col", required=True)
    p.add_argument("--close-col", required=True)
    p.add_argument("--volume-col", required=True)
    p.add_argument("--window", type=int, required=True, help="Context rows per example")
    p.add_argument("--output", required=True, help="Output directory (train/valid/test.jsonl)")
    p.add_argument("--horizon", type=int, default=1, help="Rows ahead to predict (default 1)")
    p.add_argument("--stride", type=int, default=1, help="Step between windows (default 1)")
    p.add_argument("--flat-pct", type=float, default=0.1,
                   help="|change| %% below this is labelled 'flat' (default 0.1)")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--system-prompt", default="You are a quantitative market analyst. "
                   "Answer only from the price history provided.")
    p.add_argument("--decimals", type=int, default=2)
    args = p.parse_args(argv)
    if args.window < 1 or args.horizon < 1 or args.stride < 1:
        p.error("--window, --horizon and --stride must be >= 1")
    if not (0 < args.train_frac < 1 and 0 < args.valid_frac < 1 and args.train_frac + args.valid_frac < 1):
        p.error("fractions must satisfy 0 < train, valid and train+valid < 1")
    return args


def load_rows(args):
    cols = {"date": args.date_col, "open": args.open_col, "high": args.high_col,
            "low": args.low_col, "close": args.close_col, "volume": args.volume_col}
    rows = []
    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in cols.values() if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"Columns not found in CSV: {missing}. Available: {reader.fieldnames}")
        for i, r in enumerate(reader, start=2):
            try:
                rows.append({"date": r[cols["date"]].strip(),
                             **{k: float(r[v]) for k, v in cols.items() if k != "date"}})
            except (ValueError, TypeError, AttributeError):
                print(f"skip line {i}: unparseable row", file=sys.stderr)
    rows.sort(key=lambda r: r["date"])  # ISO dates sort lexicographically
    return rows


def fmt(x, d):
    return f"{x:.{d}f}"


def make_example(rows, start, args):
    ctx = rows[start:start + args.window]
    target = rows[start + args.window + args.horizon - 2]
    last = ctx[-1]
    d = args.decimals
    lines = [f"{r['date']},{fmt(r['open'], d)},{fmt(r['high'], d)},{fmt(r['low'], d)},"
             f"{fmt(r['close'], d)},{int(r['volume'])}" for r in ctx]
    user = (f"Daily OHLCV history ({args.window} rows, columns: date,open,high,low,close,volume):\n"
            + "\n".join(lines)
            + f"\n\nWhat is the likely close {args.horizon} trading day(s) after {last['date']}, and the trend?")
    chg = (target["close"] - last["close"]) / last["close"] * 100
    trend = "flat" if abs(chg) < args.flat_pct else ("up" if chg > 0 else "down")
    assistant = (f"Predicted close on {target['date']}: {fmt(target['close'], d)}. "
                 f"Trend: {trend} ({chg:+.2f}% vs last close {fmt(last['close'], d)}).")
    msgs = []
    if args.system_prompt:
        msgs.append({"role": "system", "content": args.system_prompt})
    msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return {"messages": msgs}


def main(argv=None):
    args = parse_args(argv)
    rows = load_rows(args)
    n_ex = len(rows) - args.window - args.horizon + 1
    if n_ex < 3:
        sys.exit(f"Not enough rows ({len(rows)}) for window={args.window}, horizon={args.horizon}")
    starts = list(range(0, n_ex, args.stride))
    examples = [make_example(rows, s, args) for s in starts]

    n = len(examples)
    i_tr = int(n * args.train_frac)
    i_va = i_tr + int(n * args.valid_frac)
    # Gap between splits so overlapping windows don't leak across boundaries.
    gap = max(0, (args.window + args.horizon - 1) // args.stride)
    splits = {"train": examples[:i_tr],
              "valid": examples[i_tr + gap:i_va],
              "test": examples[i_va + gap:]}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"{name}: {len(data)} examples -> {out / (name + '.jsonl')}")


if __name__ == "__main__":
    main()
