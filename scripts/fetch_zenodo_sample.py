#!/usr/bin/env python3
"""
Fetch a capped sample of the public Zenodo IO dataset (record 13912659)
using curl (python-urllib is blocked by the Zenodo CDN in some environments).

Outputs:
  data/raw/Honduras/honduras_io.csv   - pipeline-format posts (bad + good)
  data/raw/Honduras/pair_truth.csv    - labeled post-text pairs:
        positive = same 'shape' (generated from the same template + values),
        negative = random cross pairs. Used to measure retrieval recall.

Usage:
    python scripts/fetch_zenodo_sample.py [--mb-per-file 30] [--max-bad 20000] [--max-good 12000]
"""

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ZENODO_BASE = "https://zenodo.org/api/records/13912659/files"
FILES = {
    "bad": f"{ZENODO_BASE}/honduras-bad-anonymized/content",
    "good": f"{ZENODO_BASE}/honduras-good-anonymized/content",
}

SHAPE_RE = re.compile(r"\b(\d+)\b|\b([A-Za-z]{4,})\b")


def shape_of(text: str) -> str:
    """Coarse text shape: number literals and long words replaced by type tokens."""
    s = re.sub(r"https?://\S+", "<URL>", text)
    s = re.sub(r"@\w+", "<MTN>", s)
    s = re.sub(r"#\w+", "<HTAG>", s)
    s = SHAPE_RE.sub(lambda m: "<NUM>" if m.group(1) else "<WD>", s)
    return s


def curl_download(url: str, out_path: str, max_bytes: int) -> bool:
    cmd = [
        "curl", "-sL", "--fail",
        "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "-r", f"0-{max_bytes - 1}",
        "-o", out_path,
        url,
    ]
    print(f"[curl] {url} -> {out_path} (cap {max_bytes/1e6:.0f} MB)")
    r = subprocess.run(cmd)
    return r.returncode == 0 and os.path.exists(out_path)


def count_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def relabel(path: str, good_value: int, out_path: str) -> None:
    """Copy JSONL, injecting the good=0/1 label per record (converter reads it)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line from byte-range cap
            obj["good"] = good_value
            fout.write(json.dumps(obj) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb-per-file", type=int, default=30)
    ap.add_argument("--max-bad", type=int, default=20000)
    ap.add_argument("--max-good", type=int, default=12000)
    ap.add_argument("--n-positive-pairs", type=int, default=300)
    ap.add_argument("--n-negative-pairs", type=int, default=300)
    ap.add_argument("--raw-dir", default="data/raw")
    args = ap.parse_args()

    out_dir = os.path.join(args.raw_dir, "Honduras")
    os.makedirs(out_dir, exist_ok=True)

    from src.data.convert_zenodo import convert_jsonl_to_pipeline_df

    dfs = []
    for label, good_value in (("bad", 0), ("good", 1)):
        raw_path = os.path.join(out_dir, f"_sample_{label}.jsonl")
        relabeled = os.path.join(out_dir, f"_sample_{label}_labeled.jsonl")
        if not curl_download(FILES[label], raw_path, args.mb_per_file * 1_000_000):
            print(f"[FAIL] download failed for {label}")
            sys.exit(1)
        print(f"[info] {label}: {count_lines(raw_path)} JSONL lines downloaded")
        relabel(raw_path, good_value, relabeled)
        max_posts = args.max_bad if label == "bad" else args.max_good
        df = convert_jsonl_to_pipeline_df(relabeled, max_posts=max_posts)
        dfs.append(df)
        os.remove(raw_path)
        os.remove(relabeled)

    import pandas as pd

    combined = pd.concat(dfs, ignore_index=True)
    csv_path = os.path.join(out_dir, "honduras_io.csv")
    combined.to_csv(csv_path, index=False)
    print(f"[OK] wrote {len(combined)} posts -> {csv_path}")

    # ---- pair-truth file for retrieval-recall verification (item 5) ----
    rows = []
    rng = pd.Series(range(len(combined))).sample(
        frac=1.0, random_state=42
    ).values  # deterministic shuffle of indices

    # Positives: same-shape text pairs (real independent posts, same template shape)
    shape_to_idx = {}
    texts = combined["post_text"].astype(str).tolist()
    for i in rng:
        sh = shape_of(texts[i])
        if len(sh) < 20:
            continue
        shape_to_idx.setdefault(sh, []).append(i)
    pos_found = 0
    for sh, idxs in shape_to_idx.items():
        if pos_found >= args.n_positive_pairs:
            break
        if len(idxs) >= 2:
            a, b = idxs[0], idxs[1]
            if texts[a] != texts[b]:
                rows.append({
                    "pair_id": f"pos_{pos_found}",
                    "text_a": texts[a], "text_b": texts[b],
                    "same_shape": 1,
                    "account_a": combined.iloc[a]["accountid"],
                    "account_b": combined.iloc[b]["accountid"],
                })
                pos_found += 1

    # Negatives: random cross pairs
    neg_found = 0
    step = max(1, len(rng) // (args.n_negative_pairs * 3))
    for j in range(0, len(rng) - step, step):
        if neg_found >= args.n_negative_pairs:
            break
        a, b = int(rng[j]), int(rng[j + step])
        if shape_of(texts[a]) != shape_of(texts[b]) and texts[a] != texts[b]:
            rows.append({
                "pair_id": f"neg_{neg_found}",
                "text_a": texts[a], "text_b": texts[b],
                "same_shape": 0,
                "account_a": combined.iloc[a]["accountid"],
                "account_b": combined.iloc[b]["accountid"],
            })
            neg_found += 1

    truth = pd.DataFrame(rows)
    truth_path = os.path.join(out_dir, "pair_truth.csv")
    truth.to_csv(truth_path, index=False)
    print(f"[OK] wrote {len(truth)} labeled pairs "
          f"({pos_found} positive / {neg_found} negative) -> {truth_path}")


if __name__ == "__main__":
    main()
