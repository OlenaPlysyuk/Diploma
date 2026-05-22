import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    if exc.name == "pandas":
        raise SystemExit(
            "pandas is not installed.\n"
            "Install free local dependencies first:\n"
            "  python3 -m pip install -r requirements.txt"
        ) from exc
    raise


def import_evidently() -> Tuple[Any, Any, Any, Any]:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset, DataSummaryPreset
        try:
            from evidently.ui.workspace import Workspace
        except Exception:
            Workspace = None
        return Report, DataDriftPreset, DataSummaryPreset, Workspace
    except ModuleNotFoundError as exc:
        if exc.name == "evidently":
            raise SystemExit(
                "Evidently is not installed.\n"
                "Install free local dependencies first:\n"
                "  python3 -m pip install -r requirements.txt\n"
                "or:\n"
                "  python3 -m pip install evidently pandas"
            ) from exc
        raise


def _text_len(value: Any) -> int:
    return len("" if value is None else str(value))


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def prepare_review_monitoring_data(df: pd.DataFrame) -> pd.DataFrame:
    required = {"asin", "review_date", "review_text", "rating", "product_title", "product_category"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input reviews CSV is missing required columns: {missing}")

    out = pd.DataFrame()
    out["asin"] = df["asin"].astype(str).str.strip()
    out["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    out["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    out["review_text_len"] = df["review_text"].map(_text_len)
    out["review_has_text"] = out["review_text_len"] > 0
    out["product_title_len"] = df["product_title"].map(_text_len)
    out["product_category"] = df["product_category"].astype(str).str.strip()

    if "product_brand" in df.columns:
        out["product_brand"] = df["product_brand"].astype(str).str.strip()
    if "category_lv2" in df.columns:
        out["category_lv2"] = df["category_lv2"].astype(str).str.strip()
    if "desc_source" in df.columns:
        out["desc_source"] = df["desc_source"].astype(str).str.strip()
    if "review_count" in df.columns:
        out["review_count"] = pd.to_numeric(df["review_count"], errors="coerce")

    return out.drop(columns=["review_date"])


def prepare_index_monitoring_data(df: pd.DataFrame) -> pd.DataFrame:
    required = {"asin", "title", "category", "score", "verdict"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Pipeline index CSV is missing required columns: {missing}")

    out = pd.DataFrame()
    out["asin"] = df["asin"].astype(str).str.strip()
    out["title_len"] = df["title"].map(_text_len)
    out["category"] = df["category"].astype(str).str.strip()
    out["score"] = pd.to_numeric(df["score"], errors="coerce")
    out["verdict"] = df["verdict"].astype(str).str.strip().str.lower()
    out["verdict_ok"] = out["verdict"] == "ok"
    if "pdf_file" in df.columns:
        out["has_pdf"] = df["pdf_file"].astype(str).str.strip() != ""
    else:
        out["has_pdf"] = False
    if "brand" in df.columns:
        out["brand"] = df["brand"].astype(str).str.strip()
    return out


def prepare_index_drift_data(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["title_len", "score", "has_pdf", "verdict_ok"]
    return df[[column for column in columns if column in df.columns]].copy()


def split_reference_current(df: pd.DataFrame, current_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 2:
        return df.copy(), df.copy()

    split_at = int(round(len(df) * (1.0 - current_fraction)))
    split_at = min(max(split_at, 1), len(df) - 1)
    reference = df.iloc[:split_at].reset_index(drop=True)
    current = df.iloc[split_at:].reset_index(drop=True)
    return reference, current


def run_report(
    name: str,
    report: Any,
    current: pd.DataFrame,
    reference: Optional[pd.DataFrame],
    reports_dir: Path,
) -> Any:
    reports_dir.mkdir(parents=True, exist_ok=True)
    result = report.run(current, reference)
    html_path = reports_dir / f"{name}.html"
    json_path = reports_dir / f"{name}.json"
    result.save_html(str(html_path))
    result.save_json(str(json_path))
    return result


def add_to_workspace(
    workspace_dir: Path,
    project_name: str,
    runs: List[Tuple[str, Any]],
    Workspace: Any,
) -> Optional[str]:
    if Workspace is None:
        return None

    workspace_dir.mkdir(parents=True, exist_ok=True)
    try:
        ws = Workspace.create(str(workspace_dir))
    except Exception:
        ws = Workspace(str(workspace_dir))
    project = ws.search_project(project_name)
    if isinstance(project, list):
        project = project[0] if project else None
    if project is None:
        project = ws.create_project(project_name)
        project.description = "Local free Evidently monitoring for the diploma JSON-to-PDF pipeline."
        project.save()

    for run_name, run in runs:
        try:
            run.tags = list({*getattr(run, "tags", []), run_name})
        except Exception:
            pass
        ws.add_run(project.id, run, include_data=False)
    return str(project.id)


def ensure_index_baseline(index_data: pd.DataFrame, baseline_path: Path) -> Tuple[pd.DataFrame, bool]:
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        for column in index_data.columns:
            if column not in baseline.columns:
                baseline[column] = index_data[column].iloc[0] if len(index_data) else None
        baseline = baseline[index_data.columns]
        return baseline, False
    index_data.to_csv(baseline_path, index=False)
    return index_data.copy(), True


def write_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate free local Evidently monitoring reports.")
    parser.add_argument("--csv", required=True, help="Source reviews CSV used by the pipeline.")
    parser.add_argument("--index", default="out/index.csv", help="Pipeline output index.csv.")
    parser.add_argument("--reference-csv", default="", help="Optional baseline reviews CSV for drift.")
    parser.add_argument("--out-dir", default="monitoring", help="Monitoring output directory.")
    parser.add_argument("--project-name", default="Diploma JSON-to-PDF Pipeline", help="Evidently project name.")
    parser.add_argument("--current-fraction", type=float, default=0.3, help="Latest share used as current data.")
    parser.add_argument("--skip-workspace", action="store_true", help="Only write HTML/JSON reports.")
    args = parser.parse_args()

    Report, DataDriftPreset, DataSummaryPreset, Workspace = import_evidently()

    out_dir = Path(args.out_dir)
    reports_dir = out_dir / "reports"
    workspace_dir = out_dir / "workspace"
    baseline_path = out_dir / "baselines" / "pipeline_index_reference.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    source_df = _safe_read_csv(Path(args.csv))
    current_reviews = prepare_review_monitoring_data(source_df)

    if args.reference_csv:
        reference_reviews = prepare_review_monitoring_data(_safe_read_csv(Path(args.reference_csv)))
    else:
        reference_reviews, current_reviews = split_reference_current(
            current_reviews, current_fraction=min(max(args.current_fraction, 0.05), 0.95)
        )

    runs: List[Tuple[str, Any]] = []
    review_drift = run_report(
        "reviews_data_drift",
        Report([DataDriftPreset(), DataSummaryPreset()]),
        current_reviews,
        reference_reviews,
        reports_dir,
    )
    runs.append(("reviews_data_drift", review_drift))

    index_path = Path(args.index)
    index_rows = 0
    baseline_created = False
    if index_path.exists():
        index_data = prepare_index_monitoring_data(_safe_read_csv(index_path))
        index_rows = len(index_data)
        index_reference, baseline_created = ensure_index_baseline(index_data, baseline_path)
        index_current_for_report = prepare_index_drift_data(index_data)
        index_reference_for_report = prepare_index_drift_data(index_reference)
        index_presets = [DataSummaryPreset()]
        if len(index_current_for_report) >= 10 and len(index_reference_for_report) >= 10:
            index_presets.insert(0, DataDriftPreset())
        index_report = run_report(
            "pipeline_output_quality",
            Report(index_presets),
            index_current_for_report,
            index_reference_for_report,
            reports_dir,
        )
        runs.append(("pipeline_output_quality", index_report))

    project_id = None
    if not args.skip_workspace:
        project_id = add_to_workspace(workspace_dir, args.project_name, runs, Workspace)

    local_evidently = Path(sys.executable).with_name("evidently")
    evidently_cmd = str(local_evidently) if local_evidently.exists() else (shutil.which("evidently") or "evidently")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reports_dir": str(reports_dir),
        "workspace_dir": str(workspace_dir),
        "project_id": project_id,
        "source_rows": len(source_df),
        "current_review_rows": len(current_reviews),
        "reference_review_rows": len(reference_reviews),
        "pipeline_index_rows": index_rows,
        "pipeline_index_baseline_created": baseline_created,
        "open_reports": [
            str(reports_dir / "reviews_data_drift.html"),
            str(reports_dir / "pipeline_output_quality.html"),
        ],
        "start_ui_command": f"{evidently_cmd} ui --workspace {workspace_dir} --port 8000",
    }
    write_summary(out_dir / "summary.json", summary)

    print("Evidently monitoring reports created.")
    print(f"Reports: {reports_dir}")
    if project_id:
        print(f"Workspace: {workspace_dir}")
        print(f"Project ID: {project_id}")
        print(f"Run UI: {summary['start_ui_command']}")
        print("Then open: http://localhost:8000")
    if baseline_created:
        print(f"Created initial output baseline: {baseline_path}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
