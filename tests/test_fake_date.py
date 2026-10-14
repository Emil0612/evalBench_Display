"""The fake-date test's analysis (docs/preregistration/2026-10-06-fake-date.md): a gate on the 24 micro-pilot questions
(does the model commit to forecasts?), then the forecasts' correlation with what happened, before vs after release."""
import pandas as pd

from evalbench import fake_date as fdr


def test_a_model_commits_when_most_pilot_answers_are_varied_numbers():
    varied = pd.DataFrame({"answer": [1.2, -0.5, 2.1, 0.8, -1.4, 3.0, 0.3, -2.2] * 3})
    row = fdr.gate_row(varied, asked=24)
    assert row["passes"] and row["answered"] == 1.0 and row["informative"] > 0.8
    flat = pd.DataFrame({"answer": [1.0] * 20 + [0.0] * 4})
    assert not fdr.gate_row(flat, asked=24)["passes"]                   # one stock value: no forecast at all
    refusing = pd.DataFrame({"answer": [1.2, -0.5, 2.1]})
    assert not fdr.gate_row(refusing, asked=24)["passes"]               # answers 3 of 24


def test_the_leak_share_is_the_forecast_correlation_over_the_recall_correlation():
    assert fdr.leak_share(0.4, 0.8) == 0.5
    assert pd.isna(fdr.leak_share(0.4, 0.05))                            # no recall to leak: undefined
