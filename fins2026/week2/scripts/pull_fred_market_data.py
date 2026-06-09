"""Pull live FRED market-and-macro data for the Week 2 figure exercise."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FRED_SERIES = (
    "DGS10",
    "DGS2",
    "DTB3",
    "T10Y2Y",
    "VIXCLS",
    "UNRATE",
    "INDPRO",
    "PAYEMS",
    "FEDFUNDS",
    "SP500",
)
FRED_SOURCE = (
    "FRED graph CSV download for DGS10, DGS2, DTB3, T10Y2Y, VIXCLS, UNRATE, "
    "INDPRO, PAYEMS, FEDFUNDS, and SP500 for the Week 2 market-and-macro plotting lab"
)
DEFAULT_OUTPUT = Path("fins2026/week2/results/data/fred_market_macro.csv")


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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fins2026.week2.code.market_window import (  # noqa: E402
    WEEK2_FRED_LECTURE_END,
    WEEK2_FRED_LECTURE_START,
    WEEK2_FRED_WINDOW_YEARS,
    clip_dated_column_frame,
)
from fintools.apps import read_fred_graph_csv  # noqa: E402
from fintools.datasets import load_validation_dataset  # noqa: E402


def resolve_repo_path(path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a repo-relative or absolute path."""

    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    return output_path


def fred_csv_url(series: tuple[str, ...] = FRED_SERIES) -> str:
    """Return the no-key FRED graph CSV URL for the selected series."""

    return "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(series)


def clean_fred_market_data(
    raw: pd.DataFrame,
    *,
    rolling_window: bool = False,
) -> pd.DataFrame:
    """Clean raw FRED graph CSV data and add exercise columns."""

    frame = raw.copy()
    date_candidates = ["observation_date", "DATE", "date"]
    date_column = next((column for column in date_candidates if column in frame), None)
    if date_column is None:
        raise ValueError("FRED data is missing a date column")
    frame = frame.rename(columns={date_column: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    for column in FRED_SERIES:
        if column not in frame:
            raise ValueError(f"FRED data is missing required series: {column}")
        frame[column] = pd.to_numeric(frame[column].replace(".", pd.NA), errors="coerce")

    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    frame["ten_year_minus_two_year"] = frame["DGS10"] - frame["DGS2"]
    frame["ten_year_minus_three_month"] = frame["DGS10"] - frame["DTB3"]

    vix = frame["VIXCLS"].dropna()
    sp500 = frame["SP500"].dropna()
    frame["vix_rolling_21d"] = vix.rolling(21, min_periods=10).mean().reindex(frame.index)
    frame["SP500_RETURN_PCT"] = (sp500.pct_change() * 100.0).reindex(frame.index)
    frame["SP500_LOG_RETURN_PCT"] = (
        np.log(sp500 / sp500.shift(1)).replace([np.inf, -np.inf], pd.NA) * 100.0
    ).reindex(frame.index)
    if not sp500.empty:
        frame["SP500_CUMULATIVE_RETURN_PCT"] = (
            (sp500 / float(sp500.iloc[0]) - 1.0) * 100.0
        ).reindex(frame.index)
    else:
        frame["SP500_CUMULATIVE_RETURN_PCT"] = pd.NA
    result = frame[
        [
            "date",
            *FRED_SERIES,
            "ten_year_minus_two_year",
            "ten_year_minus_three_month",
            "vix_rolling_21d",
            "SP500_RETURN_PCT",
            "SP500_LOG_RETURN_PCT",
            "SP500_CUMULATIVE_RETURN_PCT",
        ]
    ]
    if rolling_window:
        return clip_dated_column_frame(result, years=WEEK2_FRED_WINDOW_YEARS)
    return clip_dated_column_frame(
        result,
        start_date=WEEK2_FRED_LECTURE_START,
        end_date=WEEK2_FRED_LECTURE_END,
    )


def load_fixture_data(*, rolling_window: bool = False) -> pd.DataFrame:
    """Build the same exercise dataframe from frozen repo validation fixtures."""

    rates = load_validation_dataset("fred_rates_daily").data
    stress = load_validation_dataset("fred_financial_stress_daily").data
    macro = load_validation_dataset("fred_macro_monthly").data
    sp500 = load_validation_dataset("fred_sp500_daily").data
    frame = (
        rates.join(stress, how="outer")
        .join(macro, how="outer")
        .join(sp500, how="outer")
        .reset_index()
    )
    return clean_fred_market_data(frame, rolling_window=rolling_window)


def download_fred_market_data(
    url: str | None = None,
    *,
    rolling_window: bool = False,
) -> pd.DataFrame:
    """Download and clean the live FRED market-and-macro CSV."""

    raw = read_fred_graph_csv(url or fred_csv_url())
    return clean_fred_market_data(raw, rolling_window=rolling_window)


def write_fred_market_data(frame: pd.DataFrame, output: str | Path) -> Path:
    """Write the cleaned FRED exercise CSV."""

    output_path = resolve_repo_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Pull live FRED market-and-macro data for Week 2 figures.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Repo-relative or absolute cleaned CSV path.",
    )
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help="Use frozen repo validation fixtures instead of the live FRED URL.",
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
    """Run the FRED pull."""

    args = parse_args(argv)
    try:
        frame = (
            load_fixture_data(rolling_window=args.rolling_window)
            if args.use_fixture
            else download_fred_market_data(rolling_window=args.rolling_window)
        )
    except Exception as exc:
        print(f"Could not load FRED data: {exc}")
        print("Try again online, or rerun with --use-fixture for the offline class path.")
        return 1

    output_path = write_fred_market_data(frame, args.output)
    print(f"Wrote {len(frame):,} rows to: {output_path}")
    if args.rolling_window:
        print(f"Window mode: rolling latest {WEEK2_FRED_WINDOW_YEARS} years")
    else:
        print(
            "Window mode: frozen lecture window "
            f"{WEEK2_FRED_LECTURE_START:%Y-%m-%d} to {WEEK2_FRED_LECTURE_END:%Y-%m-%d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
