"""M18 in the app: its cells use the registered test (model - blind baseline on BEFORE months), the Evidence page
draws answered month against true month instead of the generic rolling score, and Profile leaves it out."""
import pandas as pd

from evalbench import analysis, ui


def _frame():
    rows = []
    for k in range(40):
        rows.append({"item_id": f"hd-{200 + k}", "date": f"20{18 + k // 12}-{k % 12 + 1:02d}-15",
                     "cluster": f"m20{18 + k // 12}-{k % 12 + 1:02d}", "fabricated": False, "refused": False,
                     "score": 0.6, "group": 1 if k < 30 else 0})
    return pd.DataFrame(rows)


def test_m18_cells_use_the_registered_model_minus_blind_test():
    df = _frame()
    blind = {i: 0.2 for i in df["item_id"]}
    out = analysis.summary_from_frames({("m", "headline_date"): df}, B=200, blind={"headline_date": blind})
    row = out.iloc[0]
    assert row["registered"] == "model − blind, before" and abs(row["dkappa"] - 0.4) < 1e-9 and row["n_pre"] == 30
    assert "headline_date" in analysis.REGISTERED and "headline_date" in analysis.BLIND_REGISTERED


def test_dating_frame_keeps_answered_items_with_true_and_answered_month():
    df = pd.DataFrame({"item_id": ["a", "b", "c"], "date": ["2021-05-14", "2022-01-03", "2023-07-20"],
                       "refused": [False, False, True], "answer": ["2021-07", "2019-01", None],
                       "err_months": [2, -36, None]})
    f = ui.dating_frame(df)
    assert list(f["true"]) == ["2021-05-01", "2022-01-01"] and list(f["answered"]) == ["2021-07-01", "2019-01-01"]
    assert list(f["close"]) == ["within 3 months", "further off"]


def test_dating_chart_draws_the_diagonal_and_one_panel_per_model():
    f = ui.dating_frame(pd.DataFrame({"item_id": ["a"], "date": ["2021-05-14"], "refused": [False],
                                      "answer": ["2021-07"], "err_months": [2]})).assign(model="x/m")
    spec = ui.dating_chart(f, ["x/m"], [{"model": "x/m", "month": "2021-09"}], []).to_dict()
    text = str(spec)
    assert "true publication month" in text and "answered month" in text


def test_dating_years_counts_a_refusal_as_a_miss_and_skips_the_pilot_block():
    df = pd.DataFrame({"item_id": ["hd-0005", "hd-0200", "hd-0201", "hd-0202", "hd-0203"],
                       "date": ["2021-03-01", "2021-03-01", "2021-08-01", "2022-02-01", "2022-05-01"],
                       "refused": [False, False, True, False, False], "within_3": [True, True, None, False, True],
                       "answer": ["2021-03", "2021-04", None, "2019-01", "2022-05"]})
    y = ui.dating_years(df).set_index("year")
    assert y.loc[2021, "n"] == 2 and y.loc[2021, "share"] == 0.5          # the pilot item hd-0005 is left out
    assert y.loc[2022, "share"] == 0.5


def test_dating_year_chart_names_its_axes_and_the_default_band():
    frame = pd.DataFrame({"model": ["x/m", "x/m", ui.BLIND_ID, ui.BLIND_ID], "year": [2021, 2022, 2021, 2022],
                          "share": [0.3, 0.1, 0.1, 0.1], "n": [100] * 4})
    spec = str(ui.dating_year_chart(frame, ["x/m"], {"x/m": "2021-09"}, [], []).to_dict())
    assert "share dated within 3 months" in spec and "default answer" in spec
