"""The cutoff ladder (docs/preregistration/2026-10-05-cutoff-ladder.md): release overrides, window rules, the paired
statistics, the pilot gate and the report's reading rules."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from evalbench import cutoffs

P = lambda s: pd.Period(s, "M")



def _needs_set(method: str):
    """These tests read a real test set, which is built locally from the licensed data (not in the repository)."""
    from evalbench.items import TESTSETS
    if not (TESTSETS / method / "v1.jsonl").exists():
        pytest.skip(f"{method}/v1 is built from the licensed data, which is not in this copy")


def test_a_release_override_wins_over_the_listing_date(tmp_path):
    path = tmp_path / "releases.yaml"
    path.write_text("openai/gpt-3.5-turbo: 2024-01-25\n", encoding="utf-8")
    meta = {"openai/gpt-3.5-turbo": {"created": 1685232000}}          # listed 2023-05-28
    assert cutoffs.resolve_release("openai/gpt-3.5-turbo", meta, path=path) == date(2024, 1, 25)


def test_without_an_override_the_release_is_the_listing_date(tmp_path):
    meta = {"x/y": {"created": 1685232000}}
    assert cutoffs.resolve_release("x/y", meta, path=tmp_path / "none.yaml") == date(2023, 5, 28)
    assert cutoffs.resolve_release("x/z", meta, path=tmp_path / "none.yaml") is None


from evalbench import ladder, index_report  # noqa: E402

MONTHS = list(pd.period_range("2016-01", "2024-12", freq="M"))
BASE = [m for m in MONTHS if m <= P("2021-06")]
BETWEEN = [m for m in MONTHS if P("2024-02") <= m <= P("2024-10")]
_draws = np.random.default_rng(0).normal(0, 4, len(MONTHS) * 4)
TRUTH = {(m, s): float(_draws[i * 4 + s]) for i, m in enumerate(MONTHS) for s in range(4)}


def _frame(knows, seed):
    """Answers that track the truth in months the model knows, a stock answer (exactly 0.5) elsewhere."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"item_id": f"{m}-{s}", "month": m, "symbol": f"S{s}", "truth": t,
                          "answer": t + rng.normal(0, 2) if knows(m) else 0.5} for (m, s), t in TRUTH.items()])


def test_last_before_matches_the_registered_windows_on_month_ends():
    assert ladder.last_before(date(2021, 9, 30)) == P("2021-06")     # - 91 days = 2021-07-01
    assert ladder.last_before(date(2024, 6, 30)) == P("2024-03")     # - 91 days = 2024-03-31, a month's last day
    assert ladder.last_before(date(2025, 1, 31)) == P("2024-10")
    assert ladder.last_before(date(2025, 3, 31)) == P("2024-11")
    assert ladder.first_after(date(2024, 1, 25)) == P("2024-02")


def test_pair_months_for_gpt35_against_gemini_on_the_grid():
    months = list(pd.period_range("2021-01", "2026-07", freq="M"))
    between, baseline = ladder.pair_months(date(2021, 9, 30), date(2024, 1, 25), date(2025, 1, 31), months)
    assert (between[0], between[-1], len(between)) == (P("2024-02"), P("2024-10"), 9)
    assert (baseline[0], baseline[-1], len(baseline)) == (P("2021-01"), P("2021-06"), 6)
    assert ladder.usable(between, baseline)


def test_a_pair_released_after_the_late_models_last_known_month_has_no_between_months():
    months = list(pd.period_range("2021-01", "2026-07", freq="M"))
    between, baseline = ladder.pair_months(date(2024, 7, 31), date(2024, 12, 26), date(2025, 3, 31), months)
    assert between == [] and not ladder.usable(between, baseline)


def test_release_months_cover_the_gap_between_two_releases_with_the_same_cutoff():
    months = list(pd.period_range("2021-01", "2026-07", freq="M"))
    between, baseline = ladder.release_months(date(2024, 7, 23), date(2024, 12, 6), date(2023, 12, 31), months)
    assert (between[0], between[-1]) == (P("2024-08"), P("2024-11"))
    assert baseline[-1] == P("2023-09")


def test_clean_months_and_grey_bins():
    assert ladder.clean_months(date(2024, 1, 25)) == 31                 # 2024-02..2026-08
    assert ladder.clean_months(date(2026, 8, 26)) == 0
    bins = ladder.grey_bins(date(2021, 9, 30), date(2024, 1, 25))     # 2021-07..2024-01
    assert bins[0][0] == P("2021-07") and bins[-1][-1] == P("2024-01") and all(len(b) <= 6 for b in bins)


def test_staggered_corr_finds_memory_when_only_the_early_model_loses_the_between_months():
    late = _frame(lambda m: True, seed=1)
    early = _frame(lambda m: m <= P("2021-06"), seed=2)
    res = ladder.staggered_corr(late, early, BETWEEN, BASE, B=500)
    assert res["D"] > 0.6 and res["p"] < 0.01 and res["r_early_between"] == 0.0


def test_staggered_corr_cancels_a_time_effect_shared_by_both_models():
    late = _frame(lambda m: m <= P("2021-06"), seed=1)
    early = _frame(lambda m: m <= P("2021-06"), seed=2)
    res = ladder.staggered_corr(late, early, BETWEEN, BASE, B=500)
    assert abs(res["D"]) < 0.1


def test_staggered_corr_uses_only_questions_both_answered():
    late = _frame(lambda m: True, seed=1)
    early = _frame(lambda m: m <= P("2021-06"), seed=2)
    early = early[~early["month"].isin(BETWEEN[:3])]                  # the early model refused three months
    res = ladder.staggered_corr(late, early, BETWEEN, BASE, B=200)
    assert res["n_between"] == 4 * (len(BETWEEN) - 3)


def test_staggered_corr_with_no_between_months_is_nan():
    late, early = _frame(lambda m: True, 1), _frame(lambda m: True, 2)
    res = ladder.staggered_corr(late, early, [], BASE, B=100)
    assert np.isnan(res["D"]) and np.isnan(res["p"])


def test_index_frame_skips_the_pilot_items():
    _needs_set("index_return")
    items = ladder.load_testset("index_return", "v1")
    pilot_ids = {it.item_id for it in items[:index_report.PILOT]}
    calls = pd.DataFrame([{"model": "m", "method": "index_return", "testset": "index_return/v1", "item_id": it.item_id,
                           "parsed": '{"value": 1.0}'} for it in items[:30]])
    df = ladder.index_frame(calls, "m")
    assert len(df) == 30 - index_report.PILOT and not (set(df["item_id"]) & pilot_ids)


from evalbench import ladder_feasibility as lf  # noqa: E402


def test_projected_power():
    assert abs(lf.projected_power(0.3, 0.1, 0.05) - 0.851) < 0.01      # Phi(3 - 1.96)
    assert lf.projected_power(0.0, 0.1, 0.05) < 0.03
    assert np.isnan(lf.projected_power(0.3, 0.0, 0.05))


def test_coverage_rows_flag_usable_pairs():
    dates = {"early": (date(2021, 9, 30), date(2024, 1, 25)), "late": (date(2025, 1, 31), date(2025, 6, 17))}
    rows = lf.coverage_rows(dates, ["early"], ["late"], {"grid": list(pd.period_range("2021-01", "2026-07", freq="M"))})
    assert rows == [{"early": "early", "late": "late", "set": "grid", "between": "2024-02..2024-10",
                     "baseline": "2021-01..2021-06", "n_between": 9, "n_baseline": 6, "usable": True}]


from evalbench import ladder_report as lr  # noqa: E402


def test_sensitive_needs_the_interval_above_the_blind_rate():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"score": rng.binomial(1, 0.6, 200).astype(float), "cluster": [f"c{i % 40}" for i in range(200)]})
    assert lr.sensitive_row(real, blind_rate=0.3, B=500)["sensitive"]
    assert not lr.sensitive_row(real, blind_rate=0.6, B=500)["sensitive"]


def test_snapshot_check():
    assert lr.snapshot_check(["openai/gpt-3.5-turbo"] * 3, "openai/gpt-3.5-turbo")["ok"]
    bad = lr.snapshot_check(["openai/gpt-3.5-turbo", "openai/gpt-3.5-turbo-0613"], "openai/gpt-3.5-turbo")
    assert not bad["ok"] and bad["snapshots"] == ["openai/gpt-3.5-turbo", "openai/gpt-3.5-turbo-0613"]
    assert not lr.snapshot_check([], "openai/gpt-3.5-turbo")["ok"]


def test_read_family_applies_holm_and_the_refusal_guard():
    rows = [{"measure": "M15 correlation", "D": 0.5, "p": 0.001, "usable": True},
            {"measure": "M5 closeness", "D": 0.08, "p": 0.004, "answered_D": 0.07, "answered_p": 0.2, "usable": True},
            {"measure": "M5 closeness", "D": 0.08, "p": 0.003, "answered_D": 0.07, "answered_p": 0.001, "usable": True},
            {"measure": "M15 correlation", "D": 0.4, "p": 0.001, "usable": False}]
    out = lr.read_family(rows)
    assert out[0]["reading"] == "memory, not a time effect"
    assert out[1]["reading"].startswith("not confirmed: refusal-driven")
    assert out[2]["reading"] == "memory, not a time effect"
    assert out[3]["reading"] == "exploratory: too few months" and "p_holm" not in out[3]


def test_grid_and_index_reports_accept_a_release_mapping():
    import inspect
    from evalbench import grid_report, index_report
    assert "release" in inspect.signature(grid_report.compute).parameters
    assert "release" in inspect.signature(index_report.compute).parameters


def test_us_index_frames_keep_every_item_because_the_pilot_was_nordic_only():
    _needs_set("us_index_return")
    items = ladder.load_testset("us_index_return", "v1")
    calls = pd.DataFrame([{"model": "m", "method": "us_index_return", "testset": "us_index_return/v1",
                           "item_id": it.item_id, "parsed": '{"value": 1.0}'} for it in items[:30]])
    assert len(ladder.index_frame(calls, "m", "us_index_return")) == 30
    assert ladder.index_months("us_index_return")[0] == min(pd.Period(it.payload["month"], "M") for it in items)


def test_exploratory_correlation_rows_on_few_pairs_read_no_estimate():
    rows = [{"measure": "M15 correlation", "n_between": 4, "D": -0.36, "p": 0.0},
            {"measure": "M15 correlation", "n_between": 20, "D": 0.8, "p": 0.01},
            {"measure": "M5 closeness", "n_between": 72, "D": 0.02, "p": 0.6}]
    out = lr.mark_small(rows)
    assert out[0]["reading"].startswith("no estimate") and "reading" not in out[1] and "reading" not in out[2]


def test_rows_print_the_both_answered_variant_when_there_is_one():
    row = {"early": "openai/gpt-3.5-turbo", "late": "google/gemini-2.5-flash", "measure": "M5 closeness",
           "between": "2024-02..2024-10", "n": "216 / 144", "D": 0.02, "lo": -0.06, "hi": 0.11, "p": 0.6,
           "answered_D": 0.031, "answered_p": 0.5}
    assert "+0.03" in lr._rows([row], holm=False) and "answered" in lr._rows([row], holm=False)


def test_the_model_table_shows_the_release_override(tmp_path, monkeypatch):
    from evalbench import cutoffs, ui
    path = tmp_path / "releases.yaml"
    path.write_text("openai/gpt-3.5-turbo: 2024-01-25\n", encoding="utf-8")
    monkeypatch.setattr(cutoffs, "RELEASES", path)
    monkeypatch.setattr(cutoffs.resolve_release, "__defaults__", (None, path))
    table = ui.model_table({"openai/gpt-3.5-turbo": {"created": 1685232000, "pricing": {}}})
    assert table.loc[0, "released"] == "2024-01-25"
