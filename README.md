# Diploma

JSON to CSV and PDF processing project.

## Local EvidentlyAI monitoring

The project includes free local monitoring with Evidently. It does not require a
paid Evidently Cloud account: reports are saved as local HTML/JSON files, and
the optional dashboard runs from a local workspace.

Install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Generate monitoring reports for an existing run:

```bash
.venv/bin/python monitor_evidently.py \
  --csv datasets_enriched/beauty_5to10_enriched.csv \
  --index out/index.csv \
  --out-dir monitoring
```

Open the generated HTML reports:

- `monitoring/reports/reviews_data_drift.html`
- `monitoring/reports/pipeline_output_quality.html`

To view the local Evidently UI dashboard:

```bash
.venv/bin/evidently ui --workspace monitoring/workspace --port 8000
```

Then open `http://localhost:8000` in your browser.

You can also ask the main pipeline to generate monitoring after it finishes:

```bash
.venv/bin/python pipeline_lmstudio_pdf.py \
  --csv datasets_enriched/beauty_5to10_enriched.csv \
  --out-dir out \
  --model YOUR_LM_STUDIO_MODEL \
  --evidently-monitor
```

The web app generation endpoint also runs this monitoring automatically after a
new non-cached generation finishes.

Each new strategy generation also creates an interactive product dashboard next
to the PDF:

- `out/<ASIN>_final_strategy.pdf`
- `out/<ASIN>_strategy_dashboard.html`

The dashboard summarizes judge score, rating distribution, insight mix, evidence
quotes, and judge notes. The PDF includes the same review snapshot charts before
the written strategy.

By default, the generator now uses balanced review selection instead of only the
newest reviews. It keeps recent reviews, but also includes strong positive,
critical, and detailed reviews in the prompt. To force the old behavior in CLI
runs, pass `--review-selection newest`.

What is monitored:

- input review data quality and drift: ratings, text length, categories, brands,
  description source, review counts when available;
- pipeline output quality and drift: judge score, verdict, PDF presence,
  generated title length and categories.

If you do not pass `--reference-csv`, the script uses the earlier part of the
reviews CSV as reference data and the latest part as current data. For pipeline
outputs, the first monitoring run creates
`monitoring/baselines/pipeline_index_reference.csv`; later runs compare new
`out/index.csv` results to that baseline.

No site setup is required for the free local workflow. Evidently Cloud signup is
optional only if you want hosted dashboards.

## Unit Tests

Run the unit test suite:

```bash
.venv/bin/python -m pytest -q
```

The tests cover JSON extraction from LLM output, balanced review selection,
review statistics, PDF/dashboard export, and Evidently monitoring data
preparation. Live LM Studio/OpenAI checks are marked as integration tests and
are skipped during the regular unit run.

Run tests with coverage:

```bash
.venv/bin/python -m pytest -q --cov --cov-report=term-missing --cov-report=html
```

Current local result for the unit-tested core:

```text
14 passed, 2 skipped
TOTAL coverage: 83%
```

Open the visual coverage report:

```bash
open htmlcov/index.html
```

Run optional live integration tests for LM Studio and OpenAI judge:

```bash
export RUN_LLM_TESTS=1
export LM_STUDIO_HOST=http://localhost:1234
export LM_STUDIO_MODEL=YOUR_LM_STUDIO_MODEL
export OPENAI_JUDGE_MODEL=gpt-4.1-mini
.venv/bin/python -m pytest -q -m integration
```

These tests call real services. Keep them separate from regular unit tests so
they do not fail when LM Studio is closed or spend OpenAI credits during normal
development.
