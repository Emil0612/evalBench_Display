"""Profile grids with many years fit the card (a compact layout, markers under the year), and the M18 cells' resampling
is vectorised without changing a single draw."""
import numpy as np
import pandas as pd
import pytest

from evalbench import index_report, ui


def _cell(p):
    return {"stratum": "all", "period": p, "method": "headline_date", "label": "none_detected",
            "label_release": "none_detected", "facts": 48, "responses": 48, "recall": 0.4, "ci": [0.3, 0.5],
            "baseline": 0.4, "n_baseline": 48, "baseline_release": 0.4, "n_baseline_release": 48}


def test_a_grid_with_many_years_is_compact_and_puts_markers_under_the_year():
    years = [str(y) for y in range(2016, 2027)]
    html = ui.profile_grid_html([_cell(p) for p in years], years, ["all"], cutoff=pd.Timestamp("2021-09-30").date(),
                                habit="2021-09")
    assert 'class="kh dense"' in html
    assert '<span class="kh-marks">' in html and "kh-cut" in html.split('<span class="kh-marks">')[1]


def test_a_grid_with_few_years_keeps_the_roomy_layout():
    years = [str(y) for y in range(2021, 2027)]
    assert 'class="kh"' in ui.profile_grid_html([_cell(p) for p in years], years, ["all"])


def _old_mean_test(values, clusters, B, seed=0):
    rng = np.random.default_rng(seed)
    g = values.astype(float).groupby(clusters)
    sums, ns = g.sum().to_numpy(), g.size().to_numpy(float)
    k = len(sums)
    counts = np.zeros((B, k))
    for i in range(B):
        counts[i] = np.bincount(rng.integers(0, k, k), minlength=k)
    lo, hi, p = index_report._interval((counts @ sums) / (counts @ ns))
    return float(values.mean()), lo, hi, p


@pytest.mark.parametrize("k,n", [(5, 40), (37, 300), (128, 900)])
def test_the_vectorised_mean_test_gives_exactly_the_old_numbers(k, n):
    rng = np.random.default_rng(k)
    v = pd.Series(rng.normal(0.05, 0.3, n))
    c = pd.Series(rng.integers(0, k, n)).map(lambda i: f"m{i:03d}")
    assert index_report._mean_test(v, c, B=3000) == _old_mean_test(v, c, B=3000)
