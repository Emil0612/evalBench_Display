"""Pilot E (docs/preregistration/2026-09-30-pilot-e.md): question-set builders. Nothing here calls a model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Data
from .items import Item
from .methods import price_level

# M5 v2: DeepSeek V3.1's cutoff is 2025-03-31 (so "before" ends 91 days earlier, 2024-12-30; the last whole month is
# 2024-11) and it was released on 2025-08-21 (so "after release" starts with 2025-09)
BEFORE_BLOCK = pd.period_range("2021-01", "2024-11", freq="M")
AFTER_BLOCK = pd.period_range("2025-09", "2026-07", freq="M")


def build_price_level_v2(data: Data, seed: int, n_per_block: int = 150, exclude: set | None = None) -> list[Item]:
    """n_per_block real months in each block, plus 10% fake companies, never repeating an (insref, month) pair in
    `exclude` (M5 v1). Ids pl2-NNNN, in random order."""
    fakes = price_level.FABRICATED_SHARE
    total = round(2 * n_per_block / (1 - fakes))
    n_fake = total - 2 * n_per_block
    before = price_level.build_items(data, n_per_block, seed, months=BEFORE_BLOCK, exclude=exclude,
                                     fabricated_share=0, id_prefix="b")
    taken = set(exclude or ()) | {(it.truth["insref"], it.payload["month"]) for it in before}
    after = price_level.build_items(data, n_per_block + n_fake, seed + 1, months=AFTER_BLOCK, exclude=taken,
                                    fabricated_share=n_fake / (n_per_block + n_fake), id_prefix="a")
    items = before + after
    rng = np.random.default_rng(seed)
    items = [items[i] for i in rng.permutation(len(items))]
    real = iter(range(len(items)))
    fake = iter(range(len(items)))
    for it in items:
        it.item_id = f"pl2-fake-{next(fake):03d}" if it.fabricated else f"pl2-{next(real):04d}"
    return items


# ---------------------------------------------------------------- analysis rules (plan §2 and deviation 1)
LEVEL_METHODS = ("price_level", "known_level", "us_level")
AFFECTED_GAP = 0.10


def release_groups(dates, cutoff, release) -> pd.Series:
    """Release line: 1 = more than 91 days before the stated cutoff, 0 = after release, -1 = the grey zone between."""
    from datetime import timedelta

    from .cutoffs import BUFFER_DAYS
    d = pd.to_datetime(pd.Series(dates)).dt.date
    before = d < cutoff - timedelta(days=BUFFER_DAYS)
    after = d > release
    return pd.Series(np.where(before, 1, np.where(after, 0, -1)), index=d.index)


def with_score(df: pd.DataFrame, method: str, rule: str) -> pd.DataFrame:
    """The registered per-item score. Levels: 'primary' = closeness, 'headline' = within the month's range; other
    methods keep their score. A refused or unanswered item scores 0, as in pilots C and D."""
    if method not in LEVEL_METHODS or not len(df):
        return df
    col = "closeness" if rule == "primary" else "in_range"
    vals = pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)
    vals = vals.where(~df["refused"].astype(bool), 0.0).fillna(0.0)
    return df.assign(score=vals.astype(float))


def guarded(reading: str, answered_p: float, answered_dk: float) -> str:
    """Deviation 1: a detection stands only if it is also significant (and positive) on answered items alone."""
    if not reading.startswith("memorization detected"):
        return reading
    if answered_p < 0.05 and answered_dk > 0:
        return reading
    return "not confirmed: refusal-driven (" + reading + ")"


def refusal_affected(cov_pre: float, cov_post: float) -> bool:
    return bool(pd.notna(cov_pre) and pd.notna(cov_post) and abs(cov_pre - cov_post) > AFFECTED_GAP)
