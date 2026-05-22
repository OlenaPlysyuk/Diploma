from pathlib import Path

import pandas as pd

from monitor_evidently import (
    ensure_index_baseline,
    prepare_index_drift_data,
    prepare_index_monitoring_data,
    prepare_review_monitoring_data,
    split_reference_current,
)


def test_prepare_review_monitoring_data_derives_numeric_and_categorical_features():
    df = pd.DataFrame(
        [
            {
                "asin": " A1 ",
                "review_date": "2024-01-01",
                "review_text": "Great product",
                "rating": "5",
                "product_title": "Title",
                "product_category": "Beauty",
                "product_brand": "Brand",
                "category_lv2": "Skin Care",
                "desc_source": "original",
                "review_count": "8",
            }
        ]
    )

    result = prepare_review_monitoring_data(df)

    assert result.loc[0, "asin"] == "A1"
    assert result.loc[0, "rating"] == 5
    assert result.loc[0, "review_text_len"] == len("Great product")
    assert bool(result.loc[0, "review_has_text"]) is True
    assert result.loc[0, "product_brand"] == "Brand"
    assert "review_date" not in result.columns


def test_prepare_index_monitoring_data_extracts_output_quality_signals():
    df = pd.DataFrame(
        [
            {
                "asin": "A1",
                "title": "Generated strategy",
                "brand": "Brand",
                "category": "Beauty",
                "score": "9",
                "verdict": "OK",
                "pdf_file": "A1.pdf",
            }
        ]
    )

    result = prepare_index_monitoring_data(df)
    drift_data = prepare_index_drift_data(result)

    assert result.loc[0, "score"] == 9
    assert result.loc[0, "verdict"] == "ok"
    assert bool(result.loc[0, "verdict_ok"]) is True
    assert bool(result.loc[0, "has_pdf"]) is True
    assert list(drift_data.columns) == ["title_len", "score", "has_pdf", "verdict_ok"]


def test_split_reference_current_keeps_both_sides_non_empty():
    df = pd.DataFrame({"value": list(range(10))})

    reference, current = split_reference_current(df, current_fraction=0.3)

    assert len(reference) == 7
    assert len(current) == 3
    assert reference.iloc[0]["value"] == 0
    assert current.iloc[0]["value"] == 7


def test_ensure_index_baseline_creates_and_reuses_file_with_new_columns(tmp_path: Path):
    baseline_path = tmp_path / "baseline.csv"
    first = pd.DataFrame({"title_len": [10], "score": [8], "has_pdf": [True]})

    baseline, created = ensure_index_baseline(first, baseline_path)

    assert created is True
    assert baseline_path.exists()
    assert baseline.equals(first)

    second = pd.DataFrame({"title_len": [12], "score": [9], "has_pdf": [True], "verdict_ok": [True]})
    reused, created_again = ensure_index_baseline(second, baseline_path)

    assert created_again is False
    assert list(reused.columns) == list(second.columns)
    assert "verdict_ok" in reused.columns
