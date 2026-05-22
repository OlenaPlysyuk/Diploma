import json
from pathlib import Path

import pandas as pd

from pipeline_lmstudio_pdf import (
    build_balanced_reviews_block,
    build_review_stats,
    export_strategy_dashboard_html,
    export_strategy_pdf,
    safe_json_loads,
)


def test_safe_json_loads_extracts_json_from_model_noise():
    raw = """
    <think>internal text that should be ignored</think>
    ```json
    {"score": 9, "verdict": "ok"}
    ```
    trailing text
    """

    assert safe_json_loads(raw) == {"score": 9, "verdict": "ok"}


def test_balanced_review_selection_keeps_recent_positive_critical_and_detailed_reviews():
    df = pd.DataFrame(
        [
            {"asin": "A1", "review_date": "2024-01-01", "rating": "5", "review_text": "old positive"},
            {"asin": "A1", "review_date": "2024-01-02", "rating": "1", "review_text": "critical failure details"},
            {"asin": "A1", "review_date": "2024-01-03", "rating": "5", "review_text": "excellent rich texture"},
            {"asin": "A1", "review_date": "2024-01-04", "rating": "3", "review_text": "mixed but informative"},
            {"asin": "A1", "review_date": "2024-01-05", "rating": "4", "review_text": "newest helpful"},
            {"asin": "A1", "review_date": "2024-01-06", "rating": "2", "review_text": "newest critical"},
        ]
    )

    block = build_balanced_reviews_block(df, k=4, max_chars=1000)

    assert "newest critical" in block
    assert "newest helpful" in block
    assert "excellent rich texture" in block
    assert "critical failure details" in block


def test_review_stats_counts_ratings_and_lengths():
    df = pd.DataFrame(
        [
            {"rating": "5", "review_text": "great"},
            {"rating": "1", "review_text": "bad"},
            {"rating": "5", "review_text": "excellent"},
        ]
    )

    stats = build_review_stats(df)

    assert stats["review_count"] == 3
    assert stats["average_rating"] == 3.67
    assert stats["rating_counts"] == {"1": 1, "2": 0, "3": 0, "4": 0, "5": 2}
    assert stats["avg_review_text_len"] == 5.7


def test_dashboard_html_escapes_content_and_includes_core_sections(tmp_path: Path):
    strategy = {
        "product": {"asin": "A1", "title": "<script>alert(1)</script>", "brand": "Brand", "category": "Beauty"},
        "insights": [
            {
                "type": "strength",
                "statement": "Customers like the texture",
                "evidence_quotes": ["soft & smooth"],
            }
        ],
    }
    judge = {"score": 9, "verdict": "ok", "issues": []}
    stats = {
        "review_count": 3,
        "average_rating": 4.3,
        "rating_counts": {"1": 0, "2": 0, "3": 1, "4": 1, "5": 1},
        "avg_review_text_len": 20,
    }
    path = tmp_path / "dashboard.html"

    export_strategy_dashboard_html(strategy, path, judge=judge, review_stats=stats)
    html = path.read_text(encoding="utf-8")

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "Rating Distribution" in html
    assert "Insight Mix" in html
    assert "soft &amp; smooth" in html


def test_pdf_export_writes_non_empty_file(tmp_path: Path):
    strategy = {
        "product": {"asin": "A1", "title": "Product", "brand": "Brand", "category": "Beauty"},
        "insights": [
            {"type": "strength", "statement": "Works well", "evidence_quotes": ["works well"]}
        ],
        "positioning": {},
        "messaging": {},
        "channels": [],
        "offers": [],
        "risks": [],
        "kpis": [],
        "assumptions": [],
    }
    path = tmp_path / "strategy.pdf"

    export_strategy_pdf(
        strategy,
        path,
        judge={"score": 8, "verdict": "ok"},
        review_stats={
            "review_count": 1,
            "average_rating": 5,
            "rating_counts": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 1},
            "avg_review_text_len": 10,
        },
    )

    assert path.exists()
    assert path.stat().st_size > 1000
