# Project: Diploma_jsontocsv (code)
# File: ndjson_filter_to_csv.py
#
# Reads merged NDJSON, keeps only products (ASIN) with reviews in [min, max],
# drops "verified" column, writes to CSV.

import argparse
import csv
import json
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Optional, Set


def load_json(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def infer_fieldnames(sample: Dict[str, Any], drop_cols: Set[str]) -> List[str]:
    preferred = [
        "asin",
        "review_date",
        "review_text",
        "rating",
        "review_summary",
        "product_title",
        "product_brand",
        "product_category",
        "product_description",
        "product_info",
    ]

    keys = [k for k in sample.keys() if k not in drop_cols]
    out: List[str] = []

    for k in preferred:
        if k in keys and k not in out:
            out.append(k)

    for k in keys:
        if k not in out:
            out.append(k)

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="Input NDJSON file")
    p.add_argument("--out", required=True, help="Output CSV file")
    p.add_argument("--min-reviews", type=int, default=5, help="Minimum reviews per ASIN (inclusive). Default: 5")
    p.add_argument("--max-reviews", type=int, default=10, help="Maximum reviews per ASIN (inclusive). Default: 10")
    args = p.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)

    min_reviews = int(args.min_reviews)
    max_reviews = int(args.max_reviews)
    if min_reviews < 0 or max_reviews < 0:
        raise ValueError("min/max must be >= 0")
    if min_reviews > max_reviews:
        raise ValueError("min-reviews cannot be greater than max-reviews")

    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    drop_cols = {"verified"}  # remove this column

    # ---------- PASS 1: count reviews per ASIN ----------
    counts = Counter()
    with inp.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            obj = load_json(line)
            if not obj:
                continue
            asin = obj.get("asin")
            if isinstance(asin, str) and asin.strip():
                counts[asin] += 1

    # Keep only ASINs with reviews count within [min_reviews, max_reviews]
    allowed_asins = {asin for asin, c in counts.items() if min_reviews <= c <= max_reviews}

    # ---------- PASS 2: write filtered CSV ----------
    out.parent.mkdir(parents=True, exist_ok=True)

    # find first valid row to infer columns
    first_obj = None
    with inp.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            obj = load_json(line)
            if not obj:
                continue
            asin = obj.get("asin")
            if asin in allowed_asins:
                first_obj = obj
                break

    if first_obj is None:
        print(f"No rows matched filter: {min_reviews} <= ASIN reviews <= {max_reviews}")
        out.write_text("", encoding="utf-8")
        return

    fieldnames = infer_fieldnames(first_obj, drop_cols)

    with inp.open("r", encoding="utf-8", errors="replace") as f_in, \
         out.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        kept_rows = 0
        for line in f_in:
            obj = load_json(line)
            if not obj:
                continue

            asin = obj.get("asin")
            if asin not in allowed_asins:
                continue

            # drop unwanted columns
            for col in drop_cols:
                obj.pop(col, None)

            writer.writerow({k: obj.get(k, "") for k in fieldnames})
            kept_rows += 1

    print(f"Done. Kept ASINs: {len(allowed_asins)} | Rows written: {kept_rows}")
    print(f"CSV saved to: {out}")


if __name__ == "__main__":
    main()
