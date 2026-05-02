# Project: Diploma (dataset preprocessing)
# File: merge_amazon_beauty_to_jsonl.py

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def parse_review_date(obj: Dict[str, Any]) -> str:
    unix_ts = obj.get("unixReviewTime")
    if isinstance(unix_ts, (int, float)) and unix_ts > 0:
        try:
            return datetime.utcfromtimestamp(int(unix_ts)).strftime("%Y-%m-%d")
        except Exception:
            pass

    rt = obj.get("reviewTime")
    if isinstance(rt, str) and rt.strip():
        try:
            return datetime.strptime(rt.strip(), "%m %d, %Y").strftime("%Y-%m-%d")
        except Exception:
            pass
    return ""


def norm_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(str(i).strip() for i in x if str(i).strip())
    return str(x).strip()


def build_meta_index(meta_path: Path, db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE meta (
            asin TEXT PRIMARY KEY,
            title TEXT,
            brand TEXT,
            main_cat TEXT,
            description TEXT
        )
    """)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    batch = []
    with meta_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = norm_text(obj.get("asin"))
            if not asin:
                details = obj.get("details") or {}
                asin = norm_text(details.get("ASIN: ")) if isinstance(details, dict) else ""
            if not asin:
                continue

            batch.append((
                asin,
                norm_text(obj.get("title")),
                norm_text(obj.get("brand")),
                norm_text(obj.get("main_cat")),
                norm_text(obj.get("description")),
            ))

            if len(batch) >= 5000:
                conn.executemany(
                    "INSERT OR REPLACE INTO meta (asin, title, brand, main_cat, description) VALUES (?, ?, ?, ?, ?)",
                    batch
                )
                conn.commit()
                batch.clear()

    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO meta (asin, title, brand, main_cat, description) VALUES (?, ?, ?, ?, ?)",
            batch
        )
        conn.commit()

    conn.close()


def merge_to_jsonl(reviews_path: Path, db_path: Path, out_jsonl: Path, keep_unmatched: bool) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with reviews_path.open("r", encoding="utf-8", errors="replace") as f_in, \
         out_jsonl.open("w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = norm_text(r.get("asin"))
            if not asin:
                continue

            cur.execute("SELECT title, brand, main_cat, description FROM meta WHERE asin = ?", (asin,))
            row = cur.fetchone()

            if row is None and not keep_unmatched:
                continue

            title, brand, main_cat, description = row if row else ("", "", "", "")

            merged = {
                "asin": asin,
                "review_date": parse_review_date(r),
                "review_text": norm_text(r.get("reviewText")),
                "rating": r.get("overall", ""),
                "verified": r.get("verified", ""),
                "review_summary": norm_text(r.get("summary")),
                "product_title": title,
                "product_brand": brand,
                "product_category": main_cat,
                "product_description": description,
            }

            f_out.write(json.dumps(merged, ensure_ascii=False) + "\n")

    conn.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reviews", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--out", required=True, help="Output merged CLEAN JSONL (ndjson)")
    p.add_argument("--db", default="meta_index.sqlite")
    p.add_argument("--keep-unmatched", action="store_true")
    args = p.parse_args()

    reviews = Path(args.reviews)
    meta = Path(args.meta)
    out = Path(args.out)
    db = Path(args.db)

    build_meta_index(meta, db)
    merge_to_jsonl(reviews, db, out, keep_unmatched=args.keep_unmatched)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
