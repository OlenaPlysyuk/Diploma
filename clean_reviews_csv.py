# Project: Diploma_jsontocsv (code)
# File: clean_reviews_csv.py

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="Input CSV (5-10 reviews per ASIN)")
    p.add_argument("--out", required=True, help="Output cleaned CSV")
    p.add_argument("--min-len", type=int, default=10, help="Min review_text length (default: 10)")
    p.add_argument("--max-len", type=int, default=2000, help="Max review_text length (default: 2000)")
    p.add_argument("--save-removed", action="store_true", help="Also save removed rows with reason to *_removed.csv")
    p.add_argument("--enforce-5to10", action="store_true",
                   help="After cleaning, keep only ASINs with 5..10 reviews (inclusive)")
    args = p.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    min_len = args.min_len
    max_len = args.max_len

    if not inp.exists():
        raise FileNotFoundError(f"CSV not found: {inp}")

    df = pd.read_csv(inp, dtype=str, keep_default_na=False)
    before_rows = len(df)
    before_asins = df["asin"].nunique() if "asin" in df.columns else 0

    # Normalize
    df["asin"] = df["asin"].astype(str).str.strip()
    df["review_date"] = df["review_date"].astype(str).str.strip()
    df["review_text"] = df["review_text"].astype(str)

    text_stripped = df["review_text"].str.strip()
    text_len = text_stripped.str.len()

    # Reasons (row-level anomalies)
    is_blank = text_stripped.eq("")
    is_too_short = text_len.lt(min_len) & ~is_blank
    is_too_long = text_len.gt(max_len)

    # Mark rows to remove (before de-dup)
    remove_mask = is_blank | is_too_short | is_too_long

    removed = None
    if args.save_removed:
        removed = df[remove_mask].copy()
        def reason(row_len, blank, short, long_):
            if blank: return "blank_review_text"
            if short: return "too_short_review_text"
            if long_: return "too_long_review_text"
            return "other"
        removed["remove_reason"] = [
            reason(l, b, s, lg) for l, b, s, lg in zip(text_len[remove_mask], is_blank[remove_mask],
                                                       is_too_short[remove_mask], is_too_long[remove_mask])
        ]

    # Remove anomalies
    df_clean = df[~remove_mask].copy()

    # Remove duplicates by (asin, review_date, review_text)
    # (keep first)
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["asin", "review_date", "review_text"], keep="first")
    dedup_removed = before_dedup - len(df_clean)

    # Optionally enforce 5..10 reviews per ASIN again after cleaning
    dropped_asins_after_enforce = 0
    if args.enforce_5to10:
        counts = df_clean["asin"].value_counts()
        good_asins = set(counts[(counts >= 5) & (counts <= 10)].index)
        before_enforce = df_clean["asin"].nunique()
        df_clean = df_clean[df_clean["asin"].isin(good_asins)].copy()
        after_enforce = df_clean["asin"].nunique()
        dropped_asins_after_enforce = before_enforce - after_enforce

    # Save outputs
    out.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out, index=False)

    if args.save_removed:
        removed_path = out.with_name(out.stem + "_removed.csv")
        removed.to_csv(removed_path, index=False)

    # Summary
    after_rows = len(df_clean)
    after_asins = df_clean["asin"].nunique()

    # Review counts after cleaning (for sanity)
    counts_after = df_clean["asin"].value_counts()
    min_c = int(counts_after.min()) if len(counts_after) else 0
    max_c = int(counts_after.max()) if len(counts_after) else 0

    print("✅ Saved:", out)
    print(f"Rows: {before_rows} -> {after_rows} (removed {before_rows - after_rows})")
    print(f"ASINs: {before_asins} -> {after_asins}")
    print(f"Removed reasons: blank={int(is_blank.sum())}, too_short={int(is_too_short.sum())}, too_long={int(is_too_long.sum())}")
    print(f"Duplicates removed (asin+date+text): {dedup_removed}")
    if args.enforce_5to10:
        print(f"Enforce 5..10: dropped ASINs={dropped_asins_after_enforce}")
    print(f"Reviews/ASIN after: min={min_c}, max={max_c}")

    if args.save_removed:
        print("🧾 Removed rows saved to:", removed_path)


if __name__ == "__main__":
    main()
