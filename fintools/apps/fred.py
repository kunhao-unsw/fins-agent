"""FRED data helpers for Streamlit coursework apps."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import pandas as pd
import requests

FRED_GRAPH_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_BATCH_SIZE = 3

DEFAULT_MACRO_SERIES = (
    "UNRATE",
    "CPIAUCSL",
    "INDPRO",
    "PAYEMS",
    "FEDFUNDS",
    "DGS10",
    "T10Y2Y",
)


def fred_graph_url(series: Sequence[str] = DEFAULT_MACRO_SERIES) -> str:
    """Return a no-key FRED graph CSV URL for one or more series."""

    clean = [str(item).strip().upper() for item in series if str(item).strip()]
    if not clean:
        raise ValueError("at least one FRED series is required")
    return f"{FRED_GRAPH_BASE_URL}?{urlencode({'id': ','.join(clean)})}"


def _series_from_fred_url(url: str) -> list[str]:
    parsed = urlparse(url)
    if parsed.netloc != "fred.stlouisfed.org" or not parsed.path.endswith("/fredgraph.csv"):
        return []
    raw_ids = parse_qs(parsed.query).get("id", [])
    if not raw_ids:
        return []
    return [item.strip().upper() for item in raw_ids[0].split(",") if item.strip()]


def _join_fred_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("at least one FRED series is required")
    prepared: list[pd.DataFrame] = []
    for frame in frames:
        date_column = "observation_date" if "observation_date" in frame.columns else "date"
        if date_column not in frame.columns:
            raise ValueError("FRED graph CSV needs an observation_date or date column")
        prepared.append(
            frame.rename(columns={date_column: "observation_date"}).set_index("observation_date")
        )
    joined = pd.concat(prepared, axis=1, join="outer")
    joined = joined.loc[:, ~joined.columns.duplicated()]
    return joined.reset_index()


def _read_fred_zip_url(url: str) -> pd.DataFrame:
    """Read a FRED graph response that is delivered as a zip of CSV files."""

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = io.BytesIO(response.content)
    if not zipfile.is_zipfile(payload):
        payload.seek(0)
        return pd.read_csv(payload)

    payload.seek(0)
    with zipfile.ZipFile(payload) as archive:
        frames = [
            pd.read_csv(archive.open(name))
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]
    return _join_fred_frames(frames)


def _read_fred_series_batch(series: Sequence[str]) -> pd.DataFrame:
    clean = [str(item).strip().upper() for item in series if str(item).strip()]
    if not clean:
        raise ValueError("at least one FRED series is required")
    url = fred_graph_url(clean)
    if len(clean) > 1:
        try:
            return _read_fred_zip_url(url)
        except Exception:
            pass
    try:
        return pd.read_csv(url)
    except (UnicodeDecodeError, ValueError) as exc:
        try:
            return _read_fred_zip_url(url)
        except Exception:
            if len(clean) == 1:
                raise exc
        if len(clean) == 1:
            raise
        midpoint = len(clean) // 2
        return _join_fred_frames(
            [
                _read_fred_series_batch(clean[:midpoint]),
                _read_fred_series_batch(clean[midpoint:]),
            ]
        )


def _read_fred_series(series: Sequence[str]) -> pd.DataFrame:
    clean = [str(item).strip().upper() for item in series if str(item).strip()]
    if not clean:
        raise ValueError("at least one FRED series is required")
    try:
        return _read_fred_series_batch(clean)
    except Exception:
        if len(clean) <= FRED_BATCH_SIZE:
            raise
    frames = [
        _read_fred_series_batch(clean[start : start + FRED_BATCH_SIZE])
        for start in range(0, len(clean), FRED_BATCH_SIZE)
    ]
    return _join_fred_frames(frames)


def read_fred_graph_csv(source: str | Path | Sequence[str]) -> pd.DataFrame:
    """Read a FRED graph CSV from a local path, URL, or series list."""

    if isinstance(source, str | Path):
        text = str(source)
        if text.startswith("http://") or text.startswith("https://"):
            series = _series_from_fred_url(text)
            if series:
                return _read_fred_series(series)
            return pd.read_csv(text)
        return pd.read_csv(Path(source))
    return _read_fred_series(source)


def clean_fred_graph_csv(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean a FRED graph CSV dataframe into a date-indexed numeric dataframe."""

    frame = raw.copy()
    date_column = "observation_date" if "observation_date" in frame.columns else "date"
    if date_column not in frame.columns:
        raise ValueError("FRED graph CSV needs an observation_date or date column")
    frame = frame.rename(columns={date_column: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    for column in frame.columns:
        if column != "date":
            frame[column] = pd.to_numeric(frame[column].replace(".", pd.NA), errors="coerce")
    return frame.set_index("date")


def add_macro_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add common macro transformations used by example Streamlit apps."""

    result = frame.copy()
    if "CPIAUCSL" in result:
        result["inflation_yoy"] = result["CPIAUCSL"].pct_change(12) * 100.0
    if "INDPRO" in result:
        result["industrial_production_yoy"] = result["INDPRO"].pct_change(12) * 100.0
    if "PAYEMS" in result:
        result["payrolls_yoy"] = result["PAYEMS"].pct_change(12) * 100.0
    if {"DGS10", "FEDFUNDS"}.issubset(result.columns):
        result["ten_year_minus_fed_funds"] = result["DGS10"] - result["FEDFUNDS"]
    return result
