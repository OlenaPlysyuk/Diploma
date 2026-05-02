# Project: Diploma_jsontocsv (code)
# File: analyze_filtered_csv.py

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="Input filtered CSV (5-10 reviews per ASIN)")
    p.add_argument("--out-dir", required=True, help="Output folder for reports")
    p.add_argument("--min-reviews", type=int, default=5, help="Expected min reviews per ASIN (inclusive)")
    p.add_argument("--max-reviews", type=int, default=10, help="Expected max reviews per ASIN (inclusive)")
    p.add_argument("--short-text", type=int, default=10, help="Review shorter than N chars is flagged")
    p.add_argument("--long-text", type=int, default=2000, help="Review longer than N chars is flagged")
    args = p.parse_args()

    inp = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise FileNotFoundError(f"CSV not found: {inp}")

    # Read CSV
    df = pd.read_csv(inp, dtype=str, keep_default_na=False)  # keep empty as ""
    n_rows = len(df)

    # Normalize some columns
    for col in ["asin", "review_date", "review_text", "rating"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 1) Basic info
    report = {
        "input_file": str(inp),
        "rows": int(n_rows),
        "columns": list(df.columns),
    }

    # 2) Empty/blank stats per column
    blank_counts = {}
    for c in df.columns:
        blank_counts[c] = int((df[c].astype(str).str.strip() == "").sum())
    report["blank_values_per_column"] = blank_counts

    # 3) ASIN review counts distribution + violations
    if "asin" in df.columns:
        asin_counts = df["asin"].value_counts()
        report["unique_asins"] = int(asin_counts.shape[0])
        report["reviews_per_asin_min"] = int(asin_counts.min())
        report["reviews_per_asin_max"] = int(asin_counts.max())

        dist = asin_counts.value_counts().sort_index()
        report["reviews_per_asin_distribution"] = {int(k): int(v) for k, v in dist.items()}

        # violations of expected range
        bad_asins = asin_counts[(asin_counts < args.min_reviews) | (asin_counts > args.max_reviews)]
        report["asins_outside_expected_range_count"] = int(bad_asins.shape[0])
        if bad_asins.shape[0] > 0:
            bad_asins_df = bad_asins.reset_index()
            bad_asins_df.columns = ["asin", "review_count"]
            bad_asins_df.to_csv(out_dir / "asins_outside_5to10.csv", index=False)
            report["asins_outside_expected_range_file"] = str(out_dir / "asins_outside_5to10.csv")
        else:
            report["asins_outside_expected_range_file"] = ""

    # 4) Date parsing quality
    if "review_date" in df.columns:
        parsed = pd.to_datetime(df["review_date"], errors="coerce", format="%Y-%m-%d")
        bad_dates = df[parsed.isna() & (df["review_date"].str.strip() != "")]
        report["invalid_date_count"] = int(bad_dates.shape[0])
        if bad_dates.shape[0] > 0:
            bad_dates.head(5000).to_csv(out_dir / "invalid_dates_sample.csv", index=False)
            report["invalid_dates_sample_file"] = str(out_dir / "invalid_dates_sample.csv")
        else:
            report["invalid_dates_sample_file"] = ""

    # 5) Rating quality (numeric + range)
    if "rating" in df.columns:
        rating_num = pd.to_numeric(df["rating"].replace("", pd.NA), errors="coerce")
        report["rating_missing_or_non_numeric"] = int(rating_num.isna().sum())

        if rating_num.notna().any():
            report["rating_min"] = float(rating_num.min())
            report["rating_max"] = float(rating_num.max())
            report["rating_mean"] = float(rating_num.mean())

            out_of_range = df[rating_num.notna() & ((rating_num < 1) | (rating_num > 5))]
            report["rating_out_of_range_count"] = int(out_of_range.shape[0])
            if out_of_range.shape[0] > 0:
                out_of_range.head(5000).to_csv(out_dir / "rating_out_of_range_sample.csv", index=False)
                report["rating_out_of_range_sample_file"] = str(out_dir / "rating_out_of_range_sample.csv")
            else:
                report["rating_out_of_range_sample_file"] = ""

            # distribution (rounded to int if needed)
            dist = rating_num.dropna().round(2).value_counts().sort_index()
            report["rating_distribution_top"] = {str(k): int(v) for k, v in dist.head(20).items()}

    # 6) Review text length stats + anomalies
    if "review_text" in df.columns:
        text_len = df["review_text"].astype(str).str.strip().str.len()
        report["review_text_len_min"] = int(text_len.min())
        report["review_text_len_median"] = float(text_len.median())
        report["review_text_len_max"] = int(text_len.max())

        short_text = df[text_len < args.short_text]
        long_text = df[text_len > args.long_text]
        report["short_text_count"] = int(short_text.shape[0])
        report["long_text_count"] = int(long_text.shape[0])

        if short_text.shape[0] > 0:
            short_text.head(5000).to_csv(out_dir / "short_text_sample.csv", index=False)
            report["short_text_sample_file"] = str(out_dir / "short_text_sample.csv")
        else:
            report["short_text_sample_file"] = ""

        if long_text.shape[0] > 0:
            long_text.head(5000).to_csv(out_dir / "long_text_sample.csv", index=False)
            report["long_text_sample_file"] = str(out_dir / "long_text_sample.csv")
        else:
            report["long_text_sample_file"] = ""

    # 7) Duplicates (asin + date + text)
    dup_cols = [c for c in ["asin", "review_date", "review_text"] if c in df.columns]
    if len(dup_cols) == 3:
        dups = df.duplicated(subset=dup_cols, keep=False)
        dup_df = df[dups].copy()
        report["duplicates_asin_date_text_rows"] = int(dup_df.shape[0])
        if dup_df.shape[0] > 0:
            dup_df.head(5000).to_csv(out_dir / "duplicates_sample.csv", index=False)
            report["duplicates_sample_file"] = str(out_dir / "duplicates_sample.csv")
        else:
            report["duplicates_sample_file"] = ""

    # 8) Metadata consistency per ASIN (title/brand/category shouldn't vary within asin)
    meta_cols = [c for c in ["product_title", "product_brand", "product_category"] if c in df.columns]
    inconsistencies = {}
    if "asin" in df.columns and meta_cols:
        for c in meta_cols:
            # count unique non-empty values per asin
            tmp = (
                df.assign(_v=df[c].astype(str).str.strip())
                  .query("_v != ''")
                  .groupby("asin")["_v"]
                  .nunique()
            )
            bad = tmp[tmp > 1].sort_values(ascending=False)
            inconsistencies[c] = int(bad.shape[0])

            if bad.shape[0] > 0:
                bad.reset_index().rename(columns={"_v": f"{c}_unique_values_count"}).to_csv(
                    out_dir / f"inconsistent_{c}.csv", index=False
                )

        report["metadata_inconsistencies_counts"] = inconsistencies

    # Save report
    report_path = out_dir / "csv_quality_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Also print short summary
    print("✅ Report saved:", report_path)
    print("Rows:", n_rows, "| Columns:", len(df.columns))
    if "unique_asins" in report:
        print("Unique ASINs:", report["unique_asins"])
        print("Reviews/ASIN min-max:", report["reviews_per_asin_min"], "-", report["reviews_per_asin_max"])
        print("ASIN violations:", report.get("asins_outside_expected_range_count", 0))
    print("Blank values (top 8):",
          dict(sorted(report["blank_values_per_column"].items(), key=lambda x: x[1], reverse=True)[:8]))


if __name__ == "__main__":
    main()
