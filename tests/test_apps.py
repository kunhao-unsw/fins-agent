"""Tests for Streamlit app helper logic."""

from __future__ import annotations

import importlib.util
import io
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

import fintools.apps.fred as fred_module
from fintools.apps import (
    SeriesSpec,
    add_nber_recession_vrects,
    app_label,
    availability_dates,
    build_forecast_target,
    clean_fred_graph_csv,
    data_health_summary,
    display_column_config,
    forecast_figure,
    forecast_series,
    forecast_series_spec,
    fred_graph_url,
    latest_delta,
    latest_percentile,
    prepare_display_frame,
    read_fred_graph_csv,
    reconstruct_implied_level,
    rolling_backtest,
    rolling_backtest_spec,
    safe_query_choice,
    safe_query_int,
    stable_tab_default,
    target_forecast_figure,
    target_name,
    week2_gdp_specs,
    week2_market_specs,
)

ROOT = Path(__file__).resolve().parents[1]


def load_week2_app_module():
    return load_week_app_module("week2", module_name="week2_streamlit_app")


def load_week3_app_module():
    return load_week_app_module("week3", module_name="week3_streamlit_app")


def load_week3_us_app_module():
    return load_week_app_module("week3", module_name="week3_us_streamlit_app", app_folder="us_app")


def load_week2_app_data_module():
    return load_module_from_path(
        ROOT / "fins2026" / "week2" / "app" / "app_data.py",
        module_name="week2_app_data",
    )


def load_week2_app_config_module():
    return load_module_from_path(
        ROOT / "fins2026" / "week2" / "app" / "app_config.py",
        module_name="week2_app_config",
    )


def load_week_app_module(week: str, *, module_name: str, app_folder: str = "app"):
    return load_module_from_path(
        ROOT / "fins2026" / week / app_folder / "streamlit_app.py",
        module_name=module_name,
    )


def load_module_from_path(path: Path, *, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def week2_app_package_text() -> str:
    return week_app_package_text(
        "week2",
        module_names=[
            "streamlit_app.py",
            "app_config.py",
            "app_data.py",
        ],
    )


def week3_app_package_text() -> str:
    return week_app_package_text(
        "week3",
        module_names=[
            "streamlit_app.py",
            "app_config.py",
            "app_data.py",
            "app_insights.py",
            "app_views.py",
        ],
    )


def week3_us_app_package_text() -> str:
    return week_app_package_text(
        "week3",
        module_names=[
            "streamlit_app.py",
            "app_config.py",
            "app_data.py",
            "app_insights.py",
            "app_views.py",
        ],
        app_folder="us_app",
    )


def week_app_package_text(week: str, *, module_names: list[str], app_folder: str = "app") -> str:
    app_dir = ROOT / "fins2026" / week / app_folder
    return "\n".join((app_dir / name).read_text(encoding="utf-8") for name in module_names)


def test_fred_graph_url_builds_no_key_csv_url() -> None:
    url = fred_graph_url(["UNRATE", "CPIAUCSL"])
    assert url == "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE%2CCPIAUCSL"


def test_clean_fred_graph_csv_handles_missing_dots() -> None:
    raw = pd.DataFrame(
        {
            "observation_date": ["2020-01-01", "2020-02-01", "bad"],
            "UNRATE": ["3.5", ".", "4.0"],
        }
    )
    clean = clean_fred_graph_csv(raw)
    assert list(clean.columns) == ["UNRATE"]
    assert clean.index.min() == pd.Timestamp("2020-01-01")
    assert np.isnan(clean.loc[pd.Timestamp("2020-02-01"), "UNRATE"])


def test_read_fred_graph_csv_batches_after_decode_error(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_zip_url(url: str) -> pd.DataFrame:
        raise ValueError(f"not a zip fixture: {url}")

    def fake_read_csv(source: str, *args, **kwargs) -> pd.DataFrame:
        ids = tuple(parse_qs(urlparse(source).query)["id"][0].split(","))
        calls.append(ids)
        if len(ids) > 1 and "FAIL" in ids:
            raise UnicodeDecodeError("utf-8", b"\xe5", 0, 1, "invalid")
        data = {"observation_date": ["2020-01-01", "2020-01-02"]}
        for position, series_id in enumerate(ids, start=1):
            data[series_id] = [float(position), float(position + 1)]
        return pd.DataFrame(data)

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(fred_module, "_read_fred_zip_url", fake_zip_url)
    frame = read_fred_graph_csv(["FAIL", "OK"])

    assert ("FAIL", "OK") in calls
    assert ("FAIL",) in calls
    assert ("OK",) in calls
    assert list(frame.columns) == ["observation_date", "FAIL", "OK"]


def test_read_fred_graph_csv_merges_zipped_graph_response(monkeypatch) -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("daily.csv", "observation_date,DGS10\n2020-01-01,1.50\n")
        archive.writestr("monthly.csv", "observation_date,UNRATE\n2020-01-01,3.50\n")

    class FakeResponse:
        content = payload.getvalue()

        def raise_for_status(self) -> None:
            return None

    original_read_csv = pd.read_csv

    def fake_read_csv(source, *args, **kwargs):
        if isinstance(source, str) and source.startswith("https://fred.stlouisfed.org"):
            raise UnicodeDecodeError("utf-8", b"\xfa", 0, 1, "invalid")
        return original_read_csv(source, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(fred_module.requests, "get", lambda *args, **kwargs: FakeResponse())

    frame = read_fred_graph_csv(["DGS10", "UNRATE"])

    assert list(frame.columns) == ["observation_date", "DGS10", "UNRATE"]
    assert frame.loc[0, "DGS10"] == pytest.approx(1.50)
    assert frame.loc[0, "UNRATE"] == pytest.approx(3.50)


def test_prepare_display_frame_uses_app_facing_labels() -> None:
    raw = pd.DataFrame(
        {"actual": [1.0], "forecast": [1.1], "absolute_error": [0.1]},
        index=pd.DatetimeIndex(["2020-01-31"], name="date"),
    )

    display = prepare_display_frame(raw)

    assert list(display.columns) == ["Date", "Actual", "Forecast", "Absolute error"]
    assert app_label("absolute_error") == "Absolute error"


def test_app_state_and_display_helpers_are_validation_safe() -> None:
    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    frame = pd.DataFrame(
        {"date": dates, "actual_return": [1.0, np.nan, 2.0, 3.0], "count": [1, 2, 3, 4]}
    )
    display = prepare_display_frame(frame, labels={"actual_return": "Actual return (%)"})
    config = display_column_config(display)
    health = data_health_summary(frame, source="Fixture", date_column="date")

    assert safe_query_choice("20Y", ["Full", "20Y"], default="Full") == "20Y"
    assert safe_query_choice("bad", ["Full", "20Y"], default="Full") == "Full"
    assert safe_query_int("27", default=12, minimum=3, maximum=24, step=3) == 24
    assert safe_query_int("bad", default=12, minimum=3, maximum=24) == 12
    assert config["Date"][0] == "date"
    assert config["Actual return (%)"] == ("number", "%.2f%%")
    assert health["Source"] == "Fixture"
    assert health["Sample start"] == "2024-01-31"
    assert health["Missing values"] == "1"
    assert latest_delta(frame["count"], periods=2) == 2.0
    assert latest_percentile(frame["count"]) == 100.0


def test_stable_tab_default_uses_url_only_for_first_render(monkeypatch) -> None:
    """URL-backed tabs should not bounce after a user switches tabs."""

    from fintools.apps import streamlit_ui

    class FakeStreamlit:
        def __init__(self) -> None:
            self.session_state: dict[str, str] = {}

    fake = FakeStreamlit()
    monkeypatch.setattr(streamlit_ui, "require_streamlit", lambda: fake)
    labels = ["Overview", "Backtest Lab", "GDP Outlook"]

    assert stable_tab_default(labels, default="Backtest Lab", key="main_view") == "Backtest Lab"
    assert fake.session_state["main_view"] == "Backtest Lab"

    fake.session_state["main_view"] = "GDP Outlook"
    assert stable_tab_default(labels, default="Backtest Lab", key="main_view") == "Backtest Lab"
    assert fake.session_state["main_view"] == "GDP Outlook"

    fake.session_state["main_view"] = "Not a tab"
    assert stable_tab_default(labels, default="Overview", key="main_view") == "Backtest Lab"
    assert fake.session_state["main_view"] == "Backtest Lab"


def test_forecast_series_returns_interval_frame() -> None:
    dates = pd.date_range("2010-01-31", periods=80, freq="ME")
    series = pd.Series(np.linspace(2.0, 5.0, len(dates)), index=dates)
    result = forecast_series(series, horizon=6, model="drift")
    assert result.model == "drift"
    assert len(result.forecast) == 6
    assert {"forecast", "lower", "upper"} == set(result.forecast.columns)
    assert result.forecast.index.min() > dates.max()


def test_forecast_series_uses_business_days_for_weekday_market_data() -> None:
    dates = pd.bdate_range("2024-01-02", periods=40)
    dates = dates.delete([5, 12, 20])
    series = pd.Series(np.linspace(2.0, 2.5, len(dates)), index=dates)

    result = forecast_series(series, horizon=5, model="naive")

    assert all(date.weekday() < 5 for date in result.forecast.index)


def test_forecast_figure_has_publication_grade_app_defaults() -> None:
    dates = pd.date_range("2019-01-31", periods=80, freq="ME")
    series = pd.Series(np.linspace(2.0, 5.0, len(dates)), index=dates)
    result = forecast_series(series, horizon=6, model="drift")

    fig = forecast_figure(result, units="Percent", indicator_name="Test indicator")
    add_nber_recession_vrects(fig, start="2020-01-01", end="2021-01-01")

    trace_names = [trace.name for trace in fig.data]
    assert trace_names == ["Observed", "Approx. 95% band", "Forecast"]
    assert fig.layout.hovermode == "x unified"
    assert not fig.layout.xaxis.rangeselector.buttons
    assert fig.layout.xaxis.rangeslider.visible is True
    assert fig.layout.legend.y > 1
    assert fig.layout.legend.x == 1
    assert fig.layout.legend.xanchor == "right"
    assert fig.layout.margin.b >= 70
    assert fig.layout.yaxis.showgrid is True
    assert any(getattr(shape, "type", None) == "rect" for shape in fig.layout.shapes)


def test_rolling_backtest_returns_errors() -> None:
    dates = pd.date_range("2010-01-31", periods=90, freq="ME")
    series = pd.Series(np.sin(np.arange(len(dates)) / 6.0) + 5.0, index=dates)
    backtest = rolling_backtest(series, model="naive", min_train=36, step=6)
    assert not backtest.empty
    assert {"actual", "forecast", "error", "absolute_error"} == set(backtest.columns)
    assert (backtest["absolute_error"] >= 0).all()


def test_forecast_target_specs_make_economic_transforms_explicit() -> None:
    market_specs = week2_market_specs()
    gdp_spec = week2_gdp_specs()["GDPC1"]
    dates = pd.date_range("2020-01-31", periods=8, freq="ME")
    rates = pd.Series([1.0, 1.2, 1.1, 1.3, 1.5, 1.4, 1.6, 1.7], index=dates)

    assert market_specs["DGS10"].target == "change"
    assert market_specs["VIXCLS"].allow_forecast is False
    assert market_specs["VIXCLS"].units == "Percent"
    assert target_name(market_specs["DGS10"]).startswith("Daily change")
    assert np.isclose(build_forecast_target(rates, market_specs["DGS10"]).iloc[0], 0.2)
    assert availability_dates([pd.Timestamp("2024-03-31")], gdp_spec)[0] == pd.Timestamp(
        "2024-04-30"
    )


def test_change_forecast_reconstructs_implied_level_path() -> None:
    spec = week2_market_specs()["DGS10"]
    level = pd.Series(
        [3.0, 3.2, 3.1, 3.3],
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    target_forecast = pd.DataFrame(
        {"forecast": [0.1, -0.2, 0.3], "lower": [0.1, -0.2, 0.3], "upper": [0.1, -0.2, 0.3]},
        index=pd.date_range("2024-01-05", periods=3, freq="D"),
    )

    implied = reconstruct_implied_level(level, target_forecast, spec, residual_std=0.0)

    assert implied["forecast"].round(6).tolist() == [3.4, 3.2, 3.5]


def test_quarterly_gdp_forecast_reconstructs_level_from_annualized_growth() -> None:
    spec = week2_gdp_specs()["GDPC1"]
    level = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.date_range("2023-03-31", periods=3, freq="QE-DEC"),
    )
    target_forecast = pd.DataFrame(
        {"forecast": [4.0], "lower": [0.0], "upper": [8.0]},
        index=pd.date_range("2023-12-31", periods=1, freq="QE-DEC"),
    )

    implied = reconstruct_implied_level(level, target_forecast, spec)
    expected = 102.0 * (1.04**0.25)

    assert np.isclose(implied["forecast"].iloc[0], expected)


def test_log_change_forecast_reconstructs_implied_level_path() -> None:
    spec = SeriesSpec(
        "TWI",
        "Trade-weighted index",
        "Index",
        target="log_change",
        target_units="Percent",
        frequency="monthly",
    )
    level = pd.Series(
        [60.0, 61.0, 62.0],
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )
    target_forecast = pd.DataFrame(
        {"forecast": [1.0, -0.5], "lower": [1.0, -0.5], "upper": [1.0, -0.5]},
        index=pd.date_range("2024-04-30", periods=2, freq="ME"),
    )

    implied = reconstruct_implied_level(level, target_forecast, spec)
    expected = 62.0 * np.exp(0.01) * np.exp(-0.005)

    assert np.isclose(implied["forecast"].iloc[-1], expected)


def test_metadata_forecast_and_backtest_return_display_ready_paths() -> None:
    spec = week2_market_specs()["DGS10"]
    dates = pd.date_range("2020-01-31", periods=90, freq="ME")
    series = pd.Series(np.linspace(2.0, 4.0, len(dates)), index=dates)

    result = forecast_series_spec(series, spec, horizon=3, model="naive")
    backtest = rolling_backtest_spec(series, spec, model="naive", min_train=36, step=6)
    fig = target_forecast_figure(result, indicator_name="10-Year Treasury")

    assert len(result.target_forecast) == 3
    assert len(result.display_forecast) == 3
    assert {"actual_level", "forecast_level", "absolute_level_error"}.issubset(backtest.columns)
    assert fig.data[2].name == "Implied level path"


def test_week2_streamlit_app_is_student_facing() -> None:
    app_dir = ROOT / "fins2026" / "week2" / "app"
    app_path = app_dir / "streamlit_app.py"
    app_lab = ROOT / "fins2026" / "week2" / "APP_LAB.md"
    checklist = ROOT / "fins2026" / "week2" / "SUBMISSION_CHECKLIST.md"
    app_readme = app_dir / "README.md"
    entrypoint_text = app_path.read_text(encoding="utf-8")
    text = week2_app_package_text()
    lab_text = app_lab.read_text(encoding="utf-8")
    checklist_text = checklist.read_text(encoding="utf-8")
    readme_text = app_readme.read_text(encoding="utf-8")
    for module_name in ["app_config.py", "app_data.py"]:
        assert (app_dir / module_name).is_file()
    assert not (app_dir / "app_insights.py").exists()
    assert not (app_dir / "app_views.py").exists()
    assert "streamlit" in text
    assert "plotly" in text
    assert "sys.path.insert" in entrypoint_text
    assert "from fins2026.week2.app.app_config import (" in entrypoint_text
    assert "from fins2026.week2.app.app_data import (" in entrypoint_text
    assert "from fins2026.week2.app.app_insights import" not in entrypoint_text
    assert "from fins2026.week2.app.app_views import" not in entrypoint_text
    assert "from fintools.apps import" in text
    assert entrypoint_text.index("sys.path.insert") < entrypoint_text.index(
        "from fins2026.week2.app.app_config"
    )
    assert "st.segmented_control" in text
    assert "st.selectbox(" in text
    assert "Data source" in text
    assert "Sample period" in text
    assert "render_data_health" in text
    assert "render_display_table" in text
    assert "render_csv_download" in text
    assert "configure_page" in text
    assert "source_status_text" in text
    assert "add_nber_recession_vrects" in text
    assert "apply_app_plotly_theme" in text
    assert "Market Overview" not in text
    assert not (ROOT / "fins2026" / "week2" / "submission.json").exists()
    assert "U.S. Month-End Market and Macro Monitor" in text
    assert "Live FRED cache loaded" in text
    assert "Fixture snapshot through" in text
    assert 'st.subheader("Displayed data")' in text
    assert "Live FRED is temporarily unavailable" in text
    assert "Forecast Lab" not in text
    assert "Backtest Lab" not in text
    assert "GDP Outlook" not in text
    assert "month-end" in readme_text.lower()
    assert "latest 10 years" in readme_text
    assert "no stress score" in readme_text.lower()
    assert "no forecasts" in readme_text
    assert "resample daily series to month-end" in lab_text.lower()
    assert "latest-10-year FRED panel" in lab_text
    assert "student-quickstart.md" in lab_text
    assert "finish-deployment.md" in lab_text
    assert "student-quickstart.md" in checklist_text
    assert "time-series" in lab_text
    assert "Unemployment rate (%)" in text
    assert "Industrial production index (2017=100)" in text
    assert "Payroll employment (thousands)" in text
    assert "Federal funds rate (%)" in text
    assert "10-Year Treasury monthly change (bp)" in text
    assert "Industrial production monthly log growth (%)" in text
    assert "S&P 500 month-end return (%)" in text
    assert "S&P 500 month-end log return (%)" in text
    assert "S&P 500 cumulative return (%)" in text
    assert ".maintainers" not in text
    assert "C:\\Users" not in text
    assert "check-app-submission" in checklist_text
    assert "fins2026/week2/app/streamlit_app.py" in checklist_text


def test_week2_app_smoke_covers_intro_workflow() -> None:
    smoke = (ROOT / "fins2026" / "week2" / "app" / "tests" / "test_app_smoke.py").read_text(
        encoding="utf-8"
    )
    for text in [
        "U.S. Month-End Market and Macro Monitor",
        "S&P 500 index level",
        "Displayed data",
        "Fixture snapshot through",
    ]:
        assert text in smoke
    assert "RUN_STREAMLIT_APPTEST_ON_WINDOWS" in smoke
    assert "AppTest.from_file" in smoke
    assert "selectbox" not in smoke
    assert "Forecast Lab" not in smoke
    assert "GDP Outlook" not in smoke


def test_week2_month_end_panel_resamples_and_merges_cleanly() -> None:
    app_data = load_week2_app_data_module()
    app_config = load_week2_app_config_module()
    frame = app_data.load_fixture_market_data()

    assert frame.index.is_monotonic_increasing
    assert all(timestamp.is_month_end for timestamp in frame.index)
    assert list(app_config.SAMPLE_PERIODS) == ["10Y", "5Y", "2Y", "1Y"]
    assert app_config.DEFAULT_SAMPLE_PERIOD == "10Y"
    assert {
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
        "DGS10_CHANGE_BP",
        "FEDFUNDS_CHANGE_BP",
        "UNRATE_CHANGE_PP",
        "INDPRO_LOG_GROWTH_PCT",
        "PAYEMS_LOG_GROWTH_PCT",
        "VIX_MONTHLY_VOL_CHANGE_PP",
        "VIX_CHANGE_PCT",
        "SP500_RETURN_PCT",
        "SP500_LOG_RETURN_PCT",
        "SP500_CUMULATIVE_RETURN_PCT",
    }.issubset(frame.columns)

    sp500 = frame["SP500"].dropna()
    assert not sp500.empty
    assert frame.index.min() >= frame.index.max() - pd.DateOffset(
        years=app_config.WEEK2_FRED_WINDOW_YEARS
    )
    assert sp500.index.max() >= pd.Timestamp("2026-04-01")
    assert frame["SP500_CUMULATIVE_RETURN_PCT"].dropna().iloc[0] == pytest.approx(0.0)
    assert frame["SP500_CUMULATIVE_RETURN_PCT"].dropna().iloc[-1] == pytest.approx(
        (sp500.iloc[-1] / sp500.iloc[0] - 1.0) * 100.0
    )
    dgs10 = frame["DGS10"].dropna()
    observed_dgs10_change = frame["DGS10_CHANGE_BP"].dropna().iloc[1:6].reset_index(drop=True)
    expected_dgs10_change = (dgs10.diff().dropna().iloc[:5] * 100.0).reset_index(drop=True)
    assert observed_dgs10_change.tolist() == pytest.approx(expected_dgs10_change.tolist())
    monthly_vix = frame["VIXCLS"] / np.sqrt(12.0)
    observed_vix_change = (
        frame["VIX_MONTHLY_VOL_CHANGE_PP"].dropna().iloc[1:6].reset_index(drop=True)
    )
    expected_vix_change = monthly_vix.diff().dropna().iloc[:5].reset_index(drop=True)
    assert observed_vix_change.tolist() == pytest.approx(expected_vix_change.tolist())
    assert frame["SP500_LOG_RETURN_PCT"].dropna().iloc[0] == pytest.approx(
        np.log(sp500.iloc[1] / sp500.iloc[0]) * 100.0
    )

    status = app_data.source_status_text(
        frame,
        series=frame["UNRATE"],
        series_label="Unemployment rate (%)",
        active_data_mode="Fixture",
    )
    assert "Fixture snapshot through" in status
    assert "latest observation for Unemployment rate (%) is" in status


def test_week3_streamlit_app_is_student_facing() -> None:
    app_dir = ROOT / "fins2026" / "week3" / "app"
    app_path = app_dir / "streamlit_app.py"
    app_lab = ROOT / "fins2026" / "week3" / "APP_LAB.md"
    app_audit = ROOT / "fins2026" / "week3" / "APP_AUDIT.md"
    app_readme = app_dir / "README.md"
    entrypoint_text = app_path.read_text(encoding="utf-8")
    text = week3_app_package_text()
    lab_text = app_lab.read_text(encoding="utf-8")
    audit_text = app_audit.read_text(encoding="utf-8")
    readme_text = app_readme.read_text(encoding="utf-8")

    for module_name in [
        "app_config.py",
        "app_data.py",
        "app_insights.py",
        "app_views.py",
        "streamlit_app.py",
    ]:
        assert (app_dir / module_name).is_file()
    assert "streamlit" in text
    assert "plotly" in text
    assert "sys.path.insert" in entrypoint_text
    assert "from fins2026.week3.app.app_config import *" in entrypoint_text
    assert "from fins2026.week3.app.app_data import *" in entrypoint_text
    assert "from fins2026.week3.app.app_insights import *" in entrypoint_text
    assert "from fins2026.week3.app.app_views import" in entrypoint_text
    assert "lazy_tabs" in text
    assert "render_data_health" in text
    assert "render_display_table" in text
    assert "render_compact_metric_strip" in text
    assert "render_csv_download" in text
    assert "sync_query_params" in text
    assert "target_forecast_figure" in text
    assert "load_model_outputs" in text
    assert "render_forecast_controls" in text
    assert "Australia Macro Forecast Monitor" in text
    assert "Australia Snapshot" in text
    assert "Model Comparison" in text
    assert "U.S. Context" in text
    assert "ARMA + exog" in text
    assert "OLS + elastic net" in text
    assert "observable panel" in text
    assert "one-step forecasts" in text
    assert "st.latex" in text
    assert "render_equation" in text
    assert r"\hat{x}_{T+h}" in text
    assert "Fixture snapshot through Australia monthly" in text
    assert "Live data could not be rebuilt" in text
    assert 'st.subheader("Data")' in text
    assert 'st.subheader("Methodology")' in text
    assert "Absolute level error" in text
    assert "check-app-submission" not in lab_text
    assert "check-app-submission" not in readme_text
    assert "submission.json" not in readme_text
    assert "Week 3 Streamlit App Audit" in audit_text
    assert "primary Week 3 product surface" in readme_text
    assert "run_forecast_benchmarks.py" in readme_text
    assert "run_forecast_benchmarks.py" in lab_text
    assert "from fins2026.week2" not in text
    assert ".maintainers" not in text
    assert "C:\\Users" not in text


def test_week3_app_smoke_exercises_every_view() -> None:
    smoke = (ROOT / "fins2026" / "week3" / "app" / "tests" / "test_app_smoke.py").read_text(
        encoding="utf-8"
    )
    for view in [
        "Overview",
        "Australia Snapshot",
        "Forecasts",
        "Model Comparison",
        "Backtests",
        "U.S. Context",
        "Data",
        "Methodology",
    ]:
        assert view in smoke
    assert "RUN_STREAMLIT_APPTEST_ON_WINDOWS" in smoke
    assert 'at.query_params["view"] = view' in smoke
    assert 'at.query_params["sample"] = "20Y"' in smoke
    assert "expected_text in rendered_text" in smoke
    assert "Fixture snapshot through Australia monthly" in smoke
    assert "test_week3_streamlit_app_smoke_nondefault_model" in smoke


def test_week3_us_streamlit_app_is_student_facing() -> None:
    app_dir = ROOT / "fins2026" / "week3" / "us_app"
    app_path = app_dir / "streamlit_app.py"
    app_readme = app_dir / "README.md"
    week_readme = ROOT / "fins2026" / "week3" / "README.md"
    entrypoint_text = app_path.read_text(encoding="utf-8")
    text = week3_us_app_package_text()
    readme_text = app_readme.read_text(encoding="utf-8")
    week_readme_text = week_readme.read_text(encoding="utf-8")

    for module_name in [
        "app_config.py",
        "app_data.py",
        "app_insights.py",
        "app_views.py",
        "streamlit_app.py",
    ]:
        assert (app_dir / module_name).is_file()
    assert "streamlit" in text
    assert "plotly" in text
    assert "sys.path.insert" in entrypoint_text
    assert "from fins2026.week3.us_app.app_config import *" in entrypoint_text
    assert "from fins2026.week3.us_app.app_data import *" in entrypoint_text
    assert "from fins2026.week3.us_app.app_insights import *" in entrypoint_text
    assert "from fins2026.week3.us_app.app_views import" in entrypoint_text
    assert "from fintools.apps import" in text
    assert "U.S. Macro Stress and Forecast Monitor" in text
    assert "Stress Score" in text
    assert "Yield Curve" in text
    assert "Forecasts" in text
    assert "Backtests" in text
    assert "GDP Outlook" in text
    assert "Methodology" in text
    assert "st.latex" in text
    assert "render_equation" in text
    assert "Fixture snapshot through" in text
    assert "Live FRED could not be loaded" in text
    assert "Forecast Lab" not in text
    assert "Backtest Lab" not in text
    assert "from fins2026.week2" not in text
    assert ".maintainers" not in text
    assert "C:\\Users" not in text
    assert "week3-us-macro" in readme_text
    assert "fins2026/week3/us_app/streamlit_app.py" in readme_text
    assert "fins2026/week3/us_app/streamlit_app.py" in week_readme_text


def test_week3_us_app_smoke_exercises_every_view() -> None:
    smoke = (
        ROOT / "fins2026" / "week3" / "us_app" / "tests" / "test_app_smoke.py"
    ).read_text(encoding="utf-8")
    for view in [
        "Overview",
        "Stress Score",
        "Yield Curve",
        "Forecasts",
        "Backtests",
        "GDP Outlook",
        "Data",
        "Methodology",
    ]:
        assert view in smoke
    assert "RUN_STREAMLIT_APPTEST_ON_WINDOWS" in smoke
    assert 'at.query_params["view"] = view' in smoke
    assert 'at.query_params["sample"] = "5Y"' in smoke
    assert "Fixture snapshot through" in smoke


def test_week3_us_companion_app_fixture_and_forecasts_are_readable() -> None:
    app = load_week3_us_app_module()
    frame, mode, warning, loaded = app.load_market_data("Fixture")

    assert mode == "Fixture"
    assert warning is None
    assert loaded is None
    assert not frame.empty
    assert {"DGS10", "T10Y2Y", "VIXCLS", "BAMLH0A0HYM2", "GDPC1"} <= set(frame.columns)

    sample = app.apply_sample_period(frame, "5Y")
    assert not sample.empty
    status = app.source_status_text(frame, active_data_mode="Fixture")
    assert "Fixture snapshot through" in status
    snapshot = app.latest_snapshot(sample)
    assert not snapshot.empty
    assert "forecast_treatment" in snapshot.columns

    stress = app.build_stress_score(sample)
    assert not stress.empty
    assert app.target_short_label("10-Year Treasury") == "Change"

    result, backtest = app.forecast_and_backtest(
        sample["DGS10"].dropna(),
        "10-Year Treasury",
        model="drift",
        horizon=126,
    )
    assert result.model == "drift"
    assert len(result.display_forecast) == 126
    assert not backtest.empty

    gdp = sample["GDPC1"].dropna()
    assert app.latest_annualized_quarterly_growth(gdp) is not None
    assert app.latest_year_over_year_growth(gdp) is not None


def test_week3_australia_first_bundle_and_benchmarks_are_readable() -> None:
    app = load_week3_app_module()
    bundle, mode, warning, loaded = app.load_week3_data("Fixture")

    assert mode == "Fixture"
    assert warning is None
    assert loaded is None
    assert {"australia_monthly", "australia_quarterly", "us_monthly", "us_quarterly"} <= set(bundle)

    australia_monthly = app.apply_sample_period(bundle["australia_monthly"], "5Y")
    australia_quarterly = app.apply_sample_period(bundle["australia_quarterly"], "5Y")
    us_monthly = app.apply_sample_period(bundle["us_monthly"], "5Y")

    assert "Cash rate target" in australia_monthly.columns
    assert "Headline CPI inflation" in australia_quarterly.columns
    assert "FEDFUNDS" in us_monthly.columns
    status = app.source_status_text(bundle, active_data_mode="Fixture")
    assert "Fixture snapshot through Australia monthly" in status
    assert "U.S. month-end context" in status

    snapshot = app.latest_snapshot(australia_monthly, australia_quarterly)
    assert not snapshot.empty
    assert "forecast_treatment" in snapshot.columns
    assert "Context only" in set(snapshot["forecast_treatment"])

    leaderboard = app.load_benchmark_leaderboard("Fixture", "20Y")
    assert leaderboard.shape[0] == 48
    assert set(leaderboard["status"]) == {"ok"}
    comparison = app.comparison_table(leaderboard, "Cash rate target")
    assert list(comparison.columns) == [
        "model_label",
        "status",
        "target_mae",
        "target_rmse",
        "level_mae",
        "level_rmse",
        "ranking_metric",
    ]
    assert comparison.iloc[0]["ranking_metric"] <= comparison.iloc[-1]["ranking_metric"]


def test_week3_forecasts_and_context_are_client_facing() -> None:
    app = load_week3_app_module()
    bundle, _, _, _ = app.load_week3_data("Fixture")
    australia_quarterly = app.apply_sample_period(bundle["australia_quarterly"], "20Y")
    us_monthly = app.apply_sample_period(bundle["us_monthly"], "20Y")

    result, backtest = app.load_model_outputs("Fixture", "20Y", "Cash rate target", "drift", 6)
    assert result.model == "drift"
    assert len(result.display_forecast) == 6
    assert not backtest.empty
    assert app.target_short_label("Cash rate target") == "Change"
    summary = app.forecast_summary_text(
        "Cash rate target",
        result,
        model_label="Drift",
        horizon=6,
    )
    assert "cash rate target" in summary.lower()
    assert "6 months" in summary

    armax_result, armax_backtest = app.load_model_outputs(
        "Fixture",
        "20Y",
        "Cash rate target",
        "armax",
        1,
    )
    assert armax_result.model == "armax"
    assert len(armax_result.display_forecast) == 1
    assert not armax_backtest.empty

    gdp = australia_quarterly["Real GDP"].dropna()
    assert app.latest_annualized_quarterly_growth(gdp) is not None
    assert app.latest_year_over_year_growth(gdp) is not None
    assert app.gdp_growth_band(-0.1) == "Contraction"
    assert app.gdp_growth_band(0.8) == "Slow growth"
    assert app.gdp_growth_band(2.0) == "Moderate growth"
    assert app.gdp_growth_band(3.0) == "Strong growth"

    us_snapshot = app.us_context_snapshot(us_monthly)
    assert not us_snapshot.empty
    us_fig = app.line_figure(
        us_monthly["FEDFUNDS"].dropna(),
        indicator_name="U.S. federal funds rate (%)",
        units="Percent",
        shade_recessions=True,
    )
    assert us_fig.data[0].name == "U.S. federal funds rate (%)"
    assert us_fig.layout.hovermode == "x unified"
    assert any(getattr(shape, "type", None) == "rect" for shape in us_fig.layout.shapes)


def test_streamlit_finish_deployment_guide_covers_known_gotchas() -> None:
    guide = ROOT / "docs" / "apps" / "streamlit" / "finish-deployment.md"
    readme = ROOT / "docs" / "apps" / "streamlit" / "README.md"
    text = guide.read_text(encoding="utf-8")
    readme_text = readme.read_text(encoding="utf-8")
    required = [
        "your-github-username/week2-month-end-market-macro",
        "fins2026/week2/app/streamlit_app.py",
        "Python version: 3.13",
        "Paste GitHub URL",
        "Linked accounts",
        "private repositories",
        "public/searchable",
        "three-dot menu",
        "Who can view this app",
        "This app is public and searchable",
        "incognito browser",
        "ModuleNotFoundError",
        "sys.path.insert",
    ]
    for item in required:
        assert item in text
    assert "finish-deployment.md" in readme_text


def test_streamlit_audit_workflow_and_skill_are_available() -> None:
    workflow = ROOT / "docs" / "ai" / "workflows" / "audit-app.md"
    checklist = ROOT / "docs" / "apps" / "streamlit" / "audit-checklist.md"
    skill = ROOT / ".agents" / "skills" / "audit-app" / "SKILL.md"
    claude_skill = ROOT / ".claude" / "skills" / "audit-app" / "SKILL.md"
    qwen_skill = ROOT / ".qwen" / "skills" / "audit-app" / "SKILL.md"
    gemini_command = ROOT / ".gemini" / "commands" / "audit-app.toml"
    week2_audit = ROOT / "fins2026" / "week2" / "APP_AUDIT.md"
    agents = ROOT / "AGENTS.md"

    workflow_text = workflow.read_text(encoding="utf-8")
    checklist_text = checklist.read_text(encoding="utf-8")
    skill_text = skill.read_text(encoding="utf-8")
    week2_audit_text = week2_audit.read_text(encoding="utf-8")
    agents_text = agents.read_text(encoding="utf-8")

    assert "audit-app" in workflow_text
    assert "docs/apps/streamlit/audit-checklist.md" in skill_text
    assert "TODO" not in skill_text
    assert "docs/ai/workflows/audit-app.md" in claude_skill.read_text(encoding="utf-8")
    assert "docs/ai/workflows/audit-app.md" in qwen_skill.read_text(encoding="utf-8")
    assert "docs/ai/workflows/audit-app.md" in gemini_command.read_text(encoding="utf-8")
    assert "st.tabs" in checklist_text
    assert "st.fragment" in checklist_text
    assert "AppTest" in checklist_text
    assert "https://docs.streamlit.io/develop/api-reference/layout/st.tabs" in checklist_text
    assert "first-click tab bounce" in checklist_text
    assert "Week 2 Streamlit App Audit" in week2_audit_text
    assert "Improvement Backlog" in week2_audit_text
    assert "`audit-app`" in agents_text


def test_streamlit_student_quickstart_is_eli5_complete() -> None:
    guide = ROOT / "docs" / "apps" / "streamlit" / "student-quickstart.md"
    readme = ROOT / "docs" / "apps" / "streamlit" / "README.md"
    week2_readme = ROOT / "fins2026" / "week2" / "README.md"
    app_lab = ROOT / "fins2026" / "week2" / "APP_LAB.md"
    checklist = ROOT / "fins2026" / "week2" / "SUBMISSION_CHECKLIST.md"
    text = guide.read_text(encoding="utf-8")
    required = [
        "streamlit run",
        "localhost",
        "check-app-submission",
        "optional `submission.json`",
        "prepare-app-repo",
        "--push",
        "Paste GitHub URL",
        ".py` file URL",
        "three-dot menu",
        "This app is public and searchable",
        "incognito",
        "Final commit hash",
        "Do not commit secrets",
        "publication-grade-apps.md",
    ]
    for item in required:
        assert item in text
    for path in [readme, week2_readme, app_lab, checklist]:
        assert "student-quickstart.md" in path.read_text(encoding="utf-8")
