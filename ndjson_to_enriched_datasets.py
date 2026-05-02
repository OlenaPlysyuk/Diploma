# Project: Diploma_jsontocsv
# File: ndjson_to_enriched_datasets.py

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd


# ----------------------------
# Cleaning utils
# ----------------------------

def norm_text(x: Any) -> str:
    s = "" if x is None else str(x)
    s = html.unescape(s)  # &quot; etc
    s = s.replace("\r\n", " ").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def to_float_str(x: Any) -> str:
    if x is None or x == "":
        return ""
    try:
        return str(float(x))
    except Exception:
        return norm_text(x)


def build_generated_desc(title: str, brand: str) -> str:
    title = norm_text(title)
    brand = norm_text(brand)
    return f"Product: {title}. Brand: {brand}." if brand else f"Product: {title}."


def parse_dt_safe(s: str) -> pd.Timestamp:
    """
    Tries to parse common formats.
    If fails -> NaT (goes to the end in sorting).
    """
    s = (s or "").strip()
    if not s:
        return pd.NaT
    # pd.to_datetime can handle "YYYY-MM-DD" and many others
    return pd.to_datetime(s, errors="coerce", utc=False)


# ----------------------------
# Category heuristic
# ----------------------------

def classify_lv2(text: str) -> str:
    t = (text or "").lower()
    rules = [
        ("Hair Care", r"\b(shampoo|conditioner|hair|scalp|curl|keratin|dry shampoo|hair mask|hair oil)\b"),
        ("Skin Care", r"\b(serum|moisturizer|cream|lotion|cleanser|toner|sunscreen|spf|retinol|hyaluronic|acne|blemish)\b"),
        ("Makeup", r"\b(foundation|concealer|mascara|lipstick|lip\b|eyeliner|brow|palette|blush|highlighter|primer)\b"),
        ("Fragrance", r"\b(perfume|cologne|fragrance|eau de parfum|eau de toilette)\b"),
        ("Bath & Body", r"\b(body wash|soap|bath\b|scrub|deodorant|hand cream|body lotion)\b"),
        ("Nails", r"\b(nail|polish|gel|cuticle|manicure|pedicure)\b"),
        ("Tools & Accessories", r"\b(brush|sponge|applicator|razor|trimmer|dryer|straightener|curler|dermaroller)\b"),
        ("Supplements", r"\b(vitamin|supplement|collagen|biotin|capsule|tablet)\b"),
        ("Men's Grooming", r"\b(beard|shave|aftershave|men's|mens)\b"),
    ]
    for name, pattern in rules:
        if re.search(pattern, t):
            return name
    return "Other/Unclear"


# ----------------------------
# NDJSON reader + row extraction
# ----------------------------

def iter_ndjson(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def extract_row(obj: Dict[str, Any]) -> Dict[str, str]:
    """
    Під типові Amazon datasets.
    Якщо у твоєму NDJSON ключі інші — скажеш, піджену під 1:1.
    """
    return {
        "asin": norm_text(obj.get("asin")),
        "review_date": norm_text(obj.get("review_date") or obj.get("reviewTime") or obj.get("date")),
        "rating": to_float_str(obj.get("rating") or obj.get("overall") or obj.get("stars")),
        "review_summary": norm_text(obj.get("review_summary") or obj.get("summary") or obj.get("reviewTitle") or obj.get("title")),
        "review_text": norm_text(obj.get("review_text") or obj.get("reviewText") or obj.get("text")),

        # продуктова частина
        "product_title": norm_text(obj.get("product_title") or obj.get("productTitle") or obj.get("title")),
        "product_brand": norm_text(obj.get("product_brand") or obj.get("brand")),
        "product_category": norm_text(obj.get("product_category") or obj.get("category")),
        "product_description": norm_text(obj.get("product_description") or obj.get("description")),
    }


# ----------------------------
# Helpers
# ----------------------------

def sort_for_grouping(dfx: pd.DataFrame) -> pd.DataFrame:
    """
    Sort so that:
    - ASIN grouped
    - newest reviews first within ASIN
    """
    dfx = dfx.copy()
    dfx["__dt"] = dfx["review_date"].apply(parse_dt_safe)
    dfx = dfx.sort_values(["asin", "__dt"], ascending=[True, False])
    dfx = dfx.drop(columns=["__dt"])
    return dfx


def print_grouping_check(dfx: pd.DataFrame, label: str) -> None:
    if dfx.empty:
        print(f"ℹ️ {label}: empty")
        return
    first_asin = str(dfx["asin"].iloc[0])
    run_len = int((dfx["asin"] == first_asin).sum())
    # run_len above counts all matches; we need consecutive run length:
    consec = 0
    for v in dfx["asin"].tolist():
        if str(v) == first_asin:
            consec += 1
        else:
            break
    print(f"🔎 {label}: first ASIN={first_asin} | consecutive rows for it={consec} | total rows={len(dfx)}")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndjson", required=True, help="Input NDJSON path")
    ap.add_argument("--out-dir", required=True, help="Output dir for CSVs")
    ap.add_argument("--min-reviews", type=int, default=8, help="Second dataset keeps ASIN with >= this many reviews")
    ap.add_argument("--keep-all", action="store_true", help="Do not filter to 5..10 reviews per ASIN")
    ap.add_argument("--min-per-asin", type=int, default=5, help="Filter: keep ASIN with >= this many reviews (default 5)")
    ap.add_argument("--max-per-asin", type=int, default=10, help="Filter: keep ASIN with <= this many reviews (default 10)")
    ap.add_argument("--split-by-category", action="store_true", help="Also save separate CSV per category_lv2 (for 8..10 dataset)")
    args = ap.parse_args()

    src = Path(args.ndjson)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) NDJSON -> raw df
    rows = []
    for obj in iter_ndjson(src):
        r = extract_row(obj)
        if r["asin"]:
            rows.append(r)

    if not rows:
        raise RuntimeError("No rows parsed from NDJSON. Check file format/keys.")

    df = pd.DataFrame(rows)

    # ensure cols exist
    for col in [
        "product_brand", "product_description", "product_category",
        "review_summary", "review_text", "review_date", "rating", "product_title"
    ]:
        if col not in df.columns:
            df[col] = ""

    df["asin"] = df["asin"].astype(str).str.strip()

    # 2) filter by review_count range (default: 5..10)
    counts = df.groupby("asin").size().rename("review_count").reset_index()
    if not args.keep_all:
        keep = counts[
            (counts["review_count"] >= args.min_per_asin) &
            (counts["review_count"] <= args.max_per_asin)
        ]["asin"]
        df = df[df["asin"].isin(set(keep))].copy()

        # recompute counts after filtering
        counts = df.groupby("asin").size().rename("review_count").reset_index()

    df = df.merge(counts, on="asin", how="left")
    df["review_count"] = df["review_count"].fillna(0).astype(int)

    # IMPORTANT: sort before saving base CSV
    df_sorted_base = sort_for_grouping(df)

    # Save base converted CSV (cleaned + filtered) — sorted
    base_csv = out_dir / "beauty_base_converted.csv"
    df_sorted_base.to_csv(base_csv, index=False, encoding="utf-8")

    # 3) Fill product description for every row:
    # per-ASIN best desc = most frequent non-empty description
    non_empty = df_sorted_base[df_sorted_base["product_description"].str.len() > 0]
    if not non_empty.empty:
        best = (
            non_empty.groupby(["asin", "product_description"])
            .size()
            .rename("cnt")
            .reset_index()
            .sort_values(["asin", "cnt"], ascending=[True, False])
            .drop_duplicates(subset=["asin"])
            .rename(columns={"product_description": "desc_best"})
            [["asin", "desc_best"]]
        )
    else:
        best = pd.DataFrame({"asin": [], "desc_best": []})

    df2 = df_sorted_base.merge(best, on="asin", how="left")
    df2["desc_best"] = df2["desc_best"].fillna("")

    df2["product_description_filled"] = df2["product_description"]
    df2["desc_source"] = "original"

    mask_borrow = (df2["product_description_filled"].str.len() == 0) & (df2["desc_best"].str.len() > 0)
    df2.loc[mask_borrow, "product_description_filled"] = df2.loc[mask_borrow, "desc_best"]
    df2.loc[mask_borrow, "desc_source"] = "borrowed"

    mask_gen = df2["product_description_filled"].str.len() == 0
    df2.loc[mask_gen, "product_description_filled"] = df2.loc[mask_gen].apply(
        lambda r: build_generated_desc(r["product_title"], r["product_brand"]),
        axis=1
    )
    df2.loc[mask_gen, "desc_source"] = "generated_from_title"

    # 4) category_lv2
    df2["category_lv2"] = (df2["product_title"] + " " + df2["product_description_filled"]).apply(classify_lv2)

    # 5) stats
    dist = counts["review_count"].value_counts().sort_index().reset_index()
    dist.columns = ["review_count", "num_asins"]
    dist.to_csv(out_dir / "distribution.csv", index=False, encoding="utf-8")

    df2["desc_source"].value_counts().reset_index().rename(columns={"index": "desc_source", "desc_source": "rows"}) \
        .to_csv(out_dir / "desc_source_stats.csv", index=False, encoding="utf-8")

    df2["category_lv2"].value_counts().reset_index().rename(columns={"index": "category_lv2", "category_lv2": "rows"}) \
        .to_csv(out_dir / "category_lv2_stats.csv", index=False, encoding="utf-8")

    # 6) produce datasets (sorted)
    df2 = sort_for_grouping(df2)

    out_5_10 = out_dir / "beauty_5to10_enriched.csv"
    df2.to_csv(out_5_10, index=False, encoding="utf-8")

    df_min = df2[df2["review_count"] >= args.min_reviews].copy()
    df_min = sort_for_grouping(df_min)

    out_min = out_dir / f"beauty_{args.min_reviews}to10_enriched.csv"
    df_min.to_csv(out_min, index=False, encoding="utf-8")

    # 7) optional split by category (FOR df_min = 8..10)
    if args.split_by_category:
        cat_dir = out_dir / "by_category_lv2"
        cat_dir.mkdir(parents=True, exist_ok=True)
        for cat, dfx in df_min.groupby("category_lv2"):
            safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(cat).strip())[:60]
            dfx_sorted = sort_for_grouping(dfx)
            dfx_sorted.to_csv(cat_dir / f"{safe}.csv", index=False, encoding="utf-8")

    # self-check
    print_grouping_check(df2, "beauty_5to10_enriched")
    print_grouping_check(df_min, f"beauty_{args.min_reviews}to10_enriched")

    print("✅ Done. Saved:")
    print(" -", base_csv)
    print(" -", out_5_10)
    print(" -", out_min)
    print(" -", out_dir / "distribution.csv")
    print(" -", out_dir / "desc_source_stats.csv")
    print(" -", out_dir / "category_lv2_stats.csv")
    if args.split_by_category:
        print(" -", out_dir / "by_category_lv2")


if __name__ == "__main__":
    main()