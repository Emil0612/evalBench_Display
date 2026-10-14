"""The M18 confirmation set (docs/preregistration/2026-10-06-headline-macro.md): fresh headlines, half about a listed
company and half about none, each month; the contrast no-company minus company resamples months; the app keeps
showing the registered v1 set."""
import numpy as np
import pandas as pd
import pytest

from evalbench import headline_confirm as hc
from evalbench import items as items_mod

WORDS = ("bank rate inflation board profit order growth market crisis energy oil vote merger share report export "
         "krona housing wage tariff").split()


def _pool(months=tuple(str(m) for m in pd.period_range("2019-08", "2020-06", freq="M")), per=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in months:
        for k in range(per):
            company = k % 2 == 0
            rows.append({"newsid": f"n{m}-{k}", "date": pd.Timestamp(f"{m}-15"), "language": "sv", "newstype": 0,
                         "distributor": "x", "headline": " ".join(rng.choice(WORDS, 6)) + f" {m} {k}",
                         "same_text_count": 1, "month": pd.Period(m, "M"),
                         "insrefs": np.array([7]) if company else None})
    return pd.DataFrame(rows)


def test_each_month_gets_six_headlines_of_each_kind_none_used_before():
    pool = _pool()
    exclude = {"n2020-01-0", "n2020-01-1"}
    its = hc.build_items(pool, seed=1, exclude=exclude, months=pd.period_range("2020-01", "2020-02", freq="M"))
    assert len(its) == 24 and all(i.item_id.startswith("hd2-") for i in its)
    assert not exclude & {i.truth["newsid"] for i in its}
    per = pd.Series([(i.truth["month"], i.strata["company"]) for i in its]).value_counts()
    assert set(per) == {6} and set(i.strata["company"] for i in its) == {"listed", "none"}
    assert all("blind_month" in i.truth for i in its)


def test_the_contrast_resamples_months_and_compares_the_two_kinds():
    rows = []
    for m in range(40):
        for g, x in (("none", 0.3), ("listed", 0.1)):
            for k in range(3):
                rows.append({"month": f"2019-{m:03d}", "group": g, "x": x + 0.01 * ((m + k) % 5)})
    out = hc.contrast(pd.DataFrame(rows), "x", B=2000)
    assert out["d"] == pytest.approx(0.2, abs=1e-9) and out["lo"] > 0.15 and out["p"] < 0.01
    assert out["n_none"] == 120 and out["n_listed"] == 120


def test_the_app_keeps_showing_the_registered_m18_set(tmp_path):
    folder = tmp_path / "headline_date"
    folder.mkdir()
    for v in ("v1", "v2"):
        (folder / f"{v}.jsonl").write_text("")
    answered = {"headline_date/v1", "headline_date/v2"}
    assert items_mod.analysis_versions(answered, root=tmp_path, preferred=answered) == {"headline_date": "v1"}


def test_a_missing_group_gives_no_estimate_instead_of_crashing():
    df = pd.DataFrame({"month": ["2019-01"] * 4, "group": ["none"] * 4, "x": [0.1, 0.2, 0.3, 0.4]})
    out = hc.contrast(df, "x", B=200)
    assert np.isnan(out["d"]) and np.isnan(out["p"]) and out["n_listed"] == 0


def test_holm_runs_over_the_five_named_models_and_a_missing_one_is_not_confirmed():
    rows = [{"model": m, "n": 100, "c1_d": 0.1, "c1_p": 0.004, "c2_d": 0.05, "c2_p": 0.004, "c1_answered": 0.1,
             "c1_answered_p": 0.004} for m in hc.MODELS[:4]] + [{"model": hc.MODELS[4], "n": 0}]
    out = hc.read(rows)
    assert out[0]["c1_p_holm"] == pytest.approx(0.02)          # 5 x 0.004: the missing model still counts
    assert [r["confirmed"] for r in out] == [True, True, True, True, False]
    assert hc.verdict(out) == "confirmed (4 of 5 models; the plan needs 3)"


def test_c2_leaves_refused_headlines_out():
    assert np.isnan(hc.exact_excess(answer=None, true=pd.Period("2020-05", "M"), refused=True, share={}, habit=None))
    assert hc.exact_excess(answer=pd.Period("2020-05", "M"), true=pd.Period("2020-05", "M"), refused=False,
                           share={pd.Period("2020-05", "M").ordinal: 0.1}, habit=None) == pytest.approx(0.9)
    assert np.isnan(hc.exact_excess(answer=pd.NaT, true=pd.Period("2020-05", "M"), refused=False, share={},
                                    habit=None))
