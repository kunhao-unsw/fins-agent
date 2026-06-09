"""Walk through Week 2 FRED Data Factory Floor Station 1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_OUTPUT = Path("fins2026/week2/results/data/fred_market_macro_stage1_long.csv")


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repo root from this script location."""

    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "tools" / "workflow_lib.py"
        ).is_file():
            return candidate
    raise RuntimeError("Could not find the fins-agent repo root.")


REPO_ROOT = find_repo_root()
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [REPO_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fins2026.week2.code.fred_stage1 import (  # noqa: E402
    FRED_SERIES,
    FRED_STAGE1_SPECS,
    build_fixture_raw_fred_frame,
    build_fred_stage1_long_table,
    clean_raw_fred_frame,
    fred_csv_url,
    stage1_assertion_report,
)
from fins2026.week2.code.market_window import (  # noqa: E402
    WEEK2_FRED_LECTURE_END,
    WEEK2_FRED_LECTURE_START,
    WEEK2_FRED_WINDOW_YEARS,
)
from fintools.apps import read_fred_graph_csv  # noqa: E402


def resolve_repo_path(path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a repo-relative or absolute path."""

    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    return output_path


def stage1_output_paths(path: str | Path) -> tuple[Path, Path]:
    """Return the CSV and Parquet output paths for the walkthrough."""

    csv_path = resolve_repo_path(path)
    return csv_path, csv_path.with_suffix(".parquet")


def write_stage1_outputs(
    stage1: pd.DataFrame,
    path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Path]:
    """Write the Stage 1 long table to CSV and Parquet."""

    csv_path, parquet_path = stage1_output_paths(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    stage1.to_csv(csv_path, index=False)
    stage1.to_parquet(parquet_path, index=False)
    return {"csv": csv_path, "parquet": parquet_path}


def print_step(title: str) -> None:
    """Print one walkthrough step heading."""

    print()
    print(title)
    print("-" * len(title))


def print_source_plan() -> None:
    """Print the public Week 2 FRED source."""

    print_step("Step 1: Pull The Public FRED Graph CSV")
    print(f"- source: {fred_csv_url()}")
    print(f"- series: {', '.join(FRED_SERIES)}")


def print_series_contracts() -> None:
    """Print the key FRED series contracts before melting to long format."""

    print_step("Step 2: Check The Week 2 Series Contracts")
    for series_id in FRED_SERIES:
        spec = FRED_STAGE1_SPECS[series_id]
        print(
            "- "
            f"{series_id} | {spec.display_name} | {spec.native_frequency} | "
            f"reference rule {spec.reference_date_rule} | "
            f"classroom lag {spec.observable_lag_months} month(s)"
        )


def load_live_raw_fred_frame() -> pd.DataFrame:
    """Read the live raw FRED graph CSV."""

    return clean_raw_fred_frame(read_fred_graph_csv(fred_csv_url()))


def print_raw_summary(raw: pd.DataFrame, *, mode: str) -> None:
    """Print a compact summary of the raw FRED frame."""

    print_step("Step 3: Read The Raw Wide Table")
    print(f"- mode: {mode}")
    print(f"- rows: {len(raw):,}")
    print(f"- columns: {', '.join(raw.columns)}")
    if "date" in raw.columns:
        print(
            "- sample: "
            f"{pd.Timestamp(raw['date'].min()):%Y-%m-%d} to "
            f"{pd.Timestamp(raw['date'].max()):%Y-%m-%d}"
        )


def print_stage1_summary(stage1: pd.DataFrame, *, rolling_window: bool) -> None:
    """Print the typed Stage 1 long-table structure."""

    print_step("Step 4: Build The Typed Stage 1 Long Table")
    if rolling_window:
        print(f"- window: rolling latest {WEEK2_FRED_WINDOW_YEARS} years")
    else:
        print(
            "- window: frozen lecture sample "
            f"{WEEK2_FRED_LECTURE_START:%Y-%m-%d} to {WEEK2_FRED_LECTURE_END:%Y-%m-%d}"
        )
    print(f"- rows: {len(stage1):,}")
    print(f"- columns: {', '.join(stage1.columns)}")
    print("- rows per series:")
    for series_id, count in stage1["series_id"].value_counts().sort_index().items():
        print(f"  - {series_id}: {count:,}")


def print_assertion_report(stage1: pd.DataFrame) -> None:
    """Print and enforce the Week 2 Stage 1 assertion contract."""

    print_step("Step 5: Run The Week 2 Stage 1 Assertions")
    report = stage1_assertion_report(stage1)
    messages = {
        "typed_dates": "typed date fields",
        "typed_values": "numeric value column",
        "all_series_present": "all required series appear in the long table",
        "ordered_dates": "reference_date never exceeds release_date",
        "unique_pairs": "no duplicate (series_id, reference_date) pairs",
    }
    for key, label in messages.items():
        outcome = report[key]
        print(f"- {label}: {'PASS' if outcome else 'FAIL'}")
        assert outcome, label


def print_timing_preview(stage1: pd.DataFrame) -> None:
    """Print timing previews for representative daily and monthly series."""

    print_step("Step 6: Inspect Timing Fields")
    for series_id in ["DGS10", "UNRATE", "INDPRO", "SP500"]:
        subset = stage1.loc[
            stage1["series_id"] == series_id,
            [
                "series_id",
                "display_name",
                "raw_date",
                "reference_date",
                "release_date",
                "observable_month_end",
                "value",
            ],
        ].head(3)
        print(subset.to_string(index=False))
        print()


def print_write_summary(paths: dict[str, Path]) -> None:
    """Print the saved output paths."""

    print_step("Step 7: Save The Stage 1 Output")
    for label, path in paths.items():
        print(f"- {label}: {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Walk through the Week 2 FRED Stage 1 Data Factory Floor pipeline.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=(
            "Repo-relative or absolute CSV path for the Stage 1 long table. "
            "A matching Parquet file is also written."
        ),
    )
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help=(
            "Use frozen repo validation fixtures instead of the live FRED graph CSV."
        ),
    )
    parser.add_argument(
        "--rolling-window",
        action="store_true",
        help=(
            "Use the latest rolling 10-year window instead of the frozen lecture "
            "window from 2015-01-01 to 2025-12-31."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the student-facing FRED Stage 1 walkthrough."""

    args = parse_args(argv)
    print("Week 2 FRED Stage 1 Data Factory Floor Walkthrough")
    print(
        "This script shows how to read the mixed-frequency FRED panel, type it, "
        "assign classroom timing fields, and save the long table."
    )

    print_source_plan()
    print_series_contracts()

    raw = build_fixture_raw_fred_frame() if args.use_fixture else load_live_raw_fred_frame()
    print_raw_summary(raw, mode="offline fixture" if args.use_fixture else "live FRED pull")
    stage1 = build_fred_stage1_long_table(raw, rolling_window=args.rolling_window)

    print_stage1_summary(stage1, rolling_window=args.rolling_window)
    print_assertion_report(stage1)
    print_timing_preview(stage1)
    paths = write_stage1_outputs(stage1, args.output)
    print_write_summary(paths)
    print()
    print("Stage 1 complete. The next step is to choose the month-end panel and transformations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
