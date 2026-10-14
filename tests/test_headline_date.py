"""M18 headline dating (docs/preregistration/2026-10-05-headline-dating.md): date-cue filters, the blind baseline,
parsing and scoring, the item build, and the report's knee fit and H1."""
import numpy as np
import pandas as pd

from evalbench.methods import headline_date as hd

P = lambda s: pd.Period(s, "M")


def test_date_cues_are_rejected():
    for h in ["Volvo rapporterar för 2021", "Riksbanken höjer räntan i mars", "Nordea Q3: vinsten steg",
              "Interim report January–March", "Ericsson: tredje kvartalet bättre än väntat", "Utdelning 12.05.2022",
              "Vecka 14: börsen steg", "Sales rose in December", "Delårsrapport för SAS", "Half-year report Kone"]:
        assert hd.has_date_cue(h), h
    for h in ["Exempelbolaget säger upp drygt 150 anställda i Tammerfors",
              "Regeringen vill öppna fler skogsområden för vindkraft", "Testbolaget handlas exklusive utdelning i dag"]:
        assert not hd.has_date_cue(h), h


def test_template_key_ignores_the_company_prefix_and_digits():
    assert hd.template_key("Exempel Bank Abp: Repurchase of own shares on 12.05.2022") == \
        hd.template_key("Testbolaget plc: Repurchase of own shares on 3.6.2023")


def test_parse_reads_year_and_month_from_json_only():
    assert hd.parse('Sure: {"year": 2021, "month": 5}')["value"] == "2021-05"
    assert hd.parse('{"year": "2019", "month": "11"}')["value"] == "2019-11"
    assert hd.parse("I think it was published in May 2021.")["value"] is None


def test_parse_rejects_out_of_range_answers_as_malformed():
    for text in ['{"year": 2021, "month": 13}', '{"year": 2021, "month": 0}', '{"year": 1850, "month": 3}',
                 '{"year": 2031, "month": 3}', '{"year": 2021}']:
        r = hd.parse(text)
        assert r["value"] is None and r["refused"] and r["outcome"] == "malformed", text


def test_score_is_one_minus_months_off_over_twelve():
    item = hd.Item("hd-0200", hd.NAME, "2021-05-14", "m2021-05", {"headline": "x", "language": "en"},
                   {"month": "2021-05", "blind_month": "2020-01"}, {})
    assert hd.item_score(item, {"main": {"value": "2021-05", "refused": False}}).score == 1.0
    s = hd.item_score(item, {"main": {"value": "2020-11", "refused": False}})
    assert abs(s.score - 0.5) < 1e-9 and s.extra["err_months"] == -6 and s.extra["same_year"] is False
    assert hd.item_score(item, {"main": {"value": "2019-01", "refused": False}}).score == 0.0
    assert hd.item_score(item, {"main": {"value": None, "refused": True}}).refused


def _news(n_per_month=40, months=("2016-01", "2019-12")):
    """Synthetic headlines with 'month' already set (build_items then skips the corpus filters). Each month has its
    own cue word, tok<YYYY>m<M>, so a baseline that trained on the month would date it exactly."""
    rng = np.random.default_rng(0)
    rows = []
    for m in pd.period_range(*months, freq="M"):
        for k in range(n_per_month):
            rows.append({"newsid": f"{m}-{k}", "date": m.to_timestamp() + pd.Timedelta(days=int(rng.integers(0, 27))),
                         "language": "sv" if k % 2 else "en", "newstype": 1, "distributor": "D", "month": m,
                         "headline": f"Bolaget {k} meddelar tok{m.year}m{m.month} och nyheter om marknaden",
                         "same_text_count": 1})
    return pd.DataFrame(rows)


def test_blind_never_trains_on_the_test_month_or_its_neighbours():
    news = _news()                                             # 2016-01..2019-12; the training mean is about 2018
    guess = hd.blind_predict(news, P("2016-06"), ["Bolaget 1 meddelar tok2016m6 och nyheter om marknaden"])
    assert guess[0] not in (P("2016-05"), P("2016-06"), P("2016-07"))   # tok2016m6 is never seen in training


def test_build_puts_one_pilot_item_per_month_first_and_stores_the_blind_guess():
    news = _news(months=("2016-01", "2026-07"), n_per_month=20)
    items = hd.build_items(news, seed=1)
    months = [it.truth["month"] for it in items[:hd.PILOT_N]]
    assert months == [str(m) for m in hd.MONTHS] and hd.is_pilot(items[0].item_id)
    assert not hd.is_pilot(items[hd.PILOT_N].item_id)
    assert len(items) == hd.PILOT_N + hd.PER_MONTH * len(hd.MONTHS)
    assert all("blind_month" in it.truth for it in items) and len({it.truth["newsid"] for it in items}) == len(items)


def test_m18_is_registered_everywhere_a_method_must_be():
    from evalbench import pilot, ui
    from evalbench.llm import mock
    from evalbench.methods import LABELS, MAIN_METHODS, METHODS
    assert METHODS["headline_date"] is hd and LABELS["headline_date"] == "M18 Headline date"
    assert "headline_date" in MAIN_METHODS and pilot.FACT_TYPES["headline_date"] == "event"
    assert hd.NAME in mock._ANSWER and ui.STRATA["headline_date"] == ["language", "newstype"]


def test_mock_controls_answer_the_truth_the_blind_month_and_a_random_month():
    import json
    from evalbench.llm import mock
    item = hd.Item("hd-0200", hd.NAME, "2021-05-14", "m2021-05", {"headline": "x", "language": "en"},
                   {"month": "2021-05", "blind_month": "2020-01"}, {})
    rng = np.random.default_rng(0)
    assert json.loads(mock._ANSWER[hd.NAME](item, "main", "mock/oracle", True, rng)) == {"year": 2021, "month": 5}
    assert json.loads(mock._ANSWER[hd.NAME](item, "main", "baseline/blind", True, rng)) == {"year": 2020, "month": 1}
    r = json.loads(mock._ANSWER[hd.NAME](item, "main", "mock/random", False, rng))
    assert 2016 <= r["year"] <= 2026 and 1 <= r["month"] <= 12


from evalbench import headline_feasibility as hf  # noqa: E402


def test_knee_fit_finds_where_answers_stop_moving_forward():
    months = np.array([m.ordinal for m in hd.MONTHS], dtype=float)
    h0 = P("2023-06").ordinal
    answers = np.where(months <= h0, months, h0)
    fit = hf.knee_fit(months, answers - months, np.ones_like(months))
    assert fit["H"] == P("2023-06") and abs(fit["slope"] + 1) < 1e-6


def test_knee_needs_six_months_each_side():
    months = np.array([m.ordinal for m in hd.MONTHS], dtype=float)
    h0 = hd.MONTHS[-3].ordinal                                # only two months after the knee
    fit = hf.knee_fit(months, np.where(months <= h0, 0.0, h0 - months), np.ones_like(months))
    assert fit is None or fit["H"] <= hd.MONTHS[-7]


def test_simulated_knee_interval_contains_the_true_horizon():
    sim = hf.simulate_knee(P("2023-06"), sd=4.0, per_month=12, B=200, seed=0)
    assert sim["lo"] <= P("2023-06") <= sim["hi"]


from evalbench import headline_report as hr  # noqa: E402


def test_gate_row_compares_the_model_with_the_blind_baseline_on_before_items_only():
    df = pd.DataFrame({"side": [1, 1, 0, 1], "refused": [False, False, False, True],
                       "score": [0.9, 0.5, 0.0, 0.0], "blind_score": [0.4, 0.4, 0.9, 0.5]})
    r = hr.gate_row(df)
    assert r["answer_rate"] == 0.75 and r["n_before"] == 3
    assert abs(r["model_before"] - 1.4 / 3) < 1e-9 and abs(r["blind_before"] - 1.3 / 3) < 1e-9


def test_gate_needs_three_answering_models_and_two_beating_the_blind():
    rows = [{"answer_rate": 0.9, "model_before": 0.5, "blind_before": 0.3}] * 2 + \
           [{"answer_rate": 0.8, "model_before": 0.2, "blind_before": 0.3},
            {"answer_rate": 0.5, "model_before": 0.9, "blind_before": 0.3}]
    assert hr.gate_decision(rows)
    rows[1] = {"answer_rate": 0.6, "model_before": 0.5, "blind_before": 0.3}
    assert not hr.gate_decision(rows)


def test_main_items_skip_the_pilot_block():
    df = pd.DataFrame({"item_id": ["hd-0000", "hd-0126", "hd-0127", "hd-1650"]})
    assert hr.main_only(df)["item_id"].tolist() == ["hd-0127", "hd-1650"]


def test_h1_reading_needs_holm_and_the_answered_only_guard():
    rows = [{"d": 0.1, "p": 0.001, "answered_d": 0.08, "answered_p": 0.002},
            {"d": 0.1, "p": 0.001, "answered_d": 0.01, "answered_p": 0.6},
            {"d": -0.1, "p": 0.001, "answered_d": -0.1, "answered_p": 0.001}]
    out = hr.read_h1(rows)
    assert out[0]["reading"] == "dates better than the text alone"
    assert out[1]["reading"].startswith("not confirmed: refusal-driven")
    assert out[2]["reading"] == "not shown"


def test_place_classifies_the_horizon_interval():
    from datetime import date
    cut, rel = date(2023, 12, 31), date(2024, 12, 6)          # last BEFORE month 2023-09
    assert hr.place(P("2023-01"), P("2023-09"), cut, rel) == "at or before the cutoff"
    assert hr.place(P("2023-11"), P("2024-10"), cut, rel) == "in the gap"
    assert hr.place(P("2025-01"), P("2025-04"), cut, rel) == "after release"
    assert hr.place(P("2023-05"), P("2024-03"), cut, rel) == "undetermined"


def test_horizon_row_is_json_ready():
    import json
    from datetime import date
    k = {"H": P("2023-06"), "lo": P("2023-04"), "hi": P("2023-09"), "slope": -0.9}
    row = hr.horizon_row("m", date(2023, 12, 31), date(2024, 12, 6), k)
    assert json.loads(json.dumps(row))["H"] == "2023-06" and row["place"] == "at or before the cutoff"
    assert hr.horizon_row("m", date(2023, 12, 31), date(2024, 12, 6), {"H": None, "lo": None, "hi": None})["place"] == "no estimate"


def test_h1_row_keeps_the_model_name_and_reports_both_scores():
    rng = np.random.default_rng(0)
    months = [P("2018-01") + k for k in range(24)]
    df = pd.DataFrame({"month": [m for m in months for _ in range(5)], "side": 1, "refused": False,
                       "score": rng.uniform(0.3, 0.6, 120), "blind_score": rng.uniform(0.0, 0.3, 120)})
    row = {"model": "x", **hr.h1_row(df, B=200)}
    assert row["model"] == "x" and row["model_score"] > row["blind_score"] and row["d"] > 0


def test_manifest_extra_names_the_real_source_and_the_item_months(tmp_path):
    from evalbench.items import file_md5
    f = tmp_path / "news_index.parquet"
    f.write_bytes(b"x")
    extra = hd.manifest_extra(f)
    assert extra["source_md5"] == {"news_index.parquet": file_md5(f)}
    assert extra["date_window"] == ["2016-01-01", "2026-07-31"]


def test_report_warns_that_the_knee_assumes_answers_track_the_truth():
    res = {"gate": {"rows": [], "go": True}, "h1": [], "horizon": [], "e1": [], "e2": [], "e3": []}
    assert "assumes" in hr.render(res) and "default month" in hr.render(res)
