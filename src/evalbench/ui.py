"""Streamlit helpers shared by the pages: cached loaders, theme-aware palette, formatting, small components."""
from __future__ import annotations

import os
import threading
from datetime import date, datetime, timedelta, timezone

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from . import analysis, market, stats, store
from .cutoffs import BUFFER_DAYS, SOURCES, load_overrides, resolve_cutoff, resolve_release
from .data import DATA_DIR, company_size_map
from .items import Item, analysis_versions, load_current_testsets, load_manifest, load_testset, testsets_fingerprint
from .llm import selfhosted
from .llm.mock import is_local
from .llm.openrouter import fetch_models, main_variants, models_by_id, selectable, with_worst_case_prices
from .methods import LABELS, METHODS, POSITIVE_CONTROLS, PRIVATE_CHECKS, PROBES, SECONDARY
from .methods.index_level import is_pilot as index_is_pilot
from .methods.headline_date import blind_score as headline_blind_score, is_pilot as headline_is_pilot
from .pilot_e import LEVEL_METHODS, with_score

LOCAL_MODELS = ["mock/oracle@2024-06-30", "mock/random@2024-06-30", "baseline/blind@2024-06-30"]
LOCAL_HELP = {
    "mock/oracle": "positive control: knows every item dated before its cutoff",
    "mock/random": "negative control: plausible random answers",
    "baseline/blind": "model-free heuristics from the item itself",
}
STRATA = {"price_recall": ["size", "currency"], "number_cloze": ["company_size", "language", "source_group"],
          "decop": ["company_size", "language", "source_group"], "completion": ["company_size", "language", "source_group"],
          "price_level": ["size", "currency"], "annual_return": ["size", "currency"],
          "known_return": ["currency"], "split_clock": ["size", "currency"], "known_level": ["currency"],
          "us_level": ["currency"], "monthly_return": ["size", "currency"], "price_choice": ["size", "currency"],
          "known_choice": ["currency"], "index_level": ["index"], "index_return": ["index"],
          "us_index_level": ["index"], "us_index_return": ["index"], "index_forecast": ["index"], "path_date": ["index"],
          "price_choice_prob": ["size", "currency"],
          "known_choice_prob": ["currency"], "headline_date": ["language", "newstype"]}
SIDE = {1: "before cutoff", 0: "after cutoff", -1: "excluded (±3 months or no cutoff)"}

# dataviz reference palette: categorical slots in fixed order; diverging blue <-> neutral <-> red
PALETTES = {
    "light": {"categorical": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
              "diverging": ["#256abf", "#f0efec", "#e34948"], "ink": "#0b0b0b", "muted": "#898781",
              "surface": "#fcfcfb"},
    "dark": {"categorical": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
             "diverging": ["#3987e5", "#383835", "#e66767"], "ink": "#ffffff", "muted": "#898781",
             "surface": "#1a1a19"},
}


# One colour system. Models: a fixed vivid colour each (in cutoff order, never by score); controls: greys.
# Readings: red = memorization detected / above baseline, amber = unresolved or a warning, grey = none detected.
# Blue shades = how much is recalled. Violet (theme primary) = interface accent only. The reading colours are never
# used for a model, so a red mark always means a detection.
# Slots 7-9 (2026-10-05, nine real models): gold, purple, brown, checked with the dataviz palette validator on the dark
# surface (lightness band, chroma floor, >= 3:1 contrast). Nine hues cannot all be told apart under colour-vision
# deficiency (the first six already include such pairs), so a model's name always accompanies its colour.
# Slots 10-12 (2026-10-07, the new models): periwinkle, olive green, orchid; same validator checks on the dark surface.
MODEL_SLOTS = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#008300", "#0f8b8d", "#8a7a10", "#8e4fae", "#8f4f18",
                         "#5a60d0", "#5c7a14", "#9c4fa3"],
               "dark": ["#3987e5", "#d95926", "#199e70", "#d55181", "#2e9e2e", "#23a7a9", "#9a8a21", "#a161bd", "#a25e21",
                        "#7d82e8", "#76962a", "#b86fbf"]}
CONTROL_GREYS = {"light": {"mock/oracle": "#52514e", "mock/random": "#8f8d86", "baseline/blind": "#b8b6ae"},
                 "dark": {"mock/oracle": "#d6d5cf", "mock/random": "#9d9b94", "baseline/blind": "#6b6a65"}}
READING_COLORS = {"light": {"detected": "#e34948", "warn": "#c98500", "none": "#898781"},
                  "dark": {"detected": "#e66767", "warn": "#e0a100", "none": "#9d9b94"}}


def assign_model_colors(order: list[str], slots: list[str], greys: dict, overflow: str) -> dict[str, str]:
    """Colour per model in a fixed order: controls get their grey, real models the next free slot, and anything past
    the last slot a neutral colour (never a generated extra hue)."""
    out, free = {}, iter(slots)
    for m in order:
        kind = m.split("@")[0]
        out[m] = greys[kind] if kind in greys else next(free, overflow)
    return out


def theme_name() -> str:
    return "dark" if palette() is PALETTES["dark"] else "light"


def first_run_order(first: dict[str, str], cut: dict) -> list[str]:
    """Colour order: the run a model first appeared in, then (within one run) its cutoff, then its id. A model that
    joins later takes the next free colour, so existing models never change colour. The auxiliary models (hidden by
    default) come last, so they never take a slot from a model on the default views."""
    return sorted(first, key=lambda m: (m in AUX_MODELS, first[m] or "", str(cut.get(m) or "9999"), m))


@st.cache_data(show_spinner=False)
def _first_seen(stamp: tuple) -> dict[str, str]:
    return store.first_seen(conn())


def model_colors() -> dict[str, str]:
    t = theme_name()
    first = _first_seen(store.db_stamp(conn()))
    order = first_run_order(first, cutoffs(list(first), meta_by_id()))
    return assign_model_colors(order, MODEL_SLOTS[t], CONTROL_GREYS[t], palette()["muted"])


def model_color(model_id: str) -> str:
    return model_colors().get(model_id, palette()["muted"])


def model_scale(models: list[str]) -> alt.Scale:
    """Chart colours that match the model's dot everywhere else in the app."""
    return alt.Scale(domain=[display_name(m) for m in models], range=[model_color(m) for m in models])


def model_dot(model_id: str) -> str:
    return f'<span class="eb-dot" style="background:{model_color(model_id)}"></span>'


def palette() -> dict:
    """The app is dark only (.streamlit/config.toml has no light theme), so every chart and badge uses the dark steps."""
    return PALETTES["dark"]


# ---------------------------------------------------------------- cached loaders

_THREAD = threading.local()


def conn():
    """This thread's database connection. Streamlit runs every session, and a page left mid-run, in its own thread,
    and a SQLite connection must never be used by two threads at once (a shared one returned None from fetchone()
    when the Overview was left mid-load for the Market page)."""
    c = getattr(_THREAD, "conn", None)
    if c is None:
        c = _THREAD.conn = store.connect(store.DB_PATH)
    return c


@st.cache_resource(ttl=3600, show_spinner="Fetching the OpenRouter model list…")
def _openrouter_meta() -> dict:
    try:
        return models_by_id(fetch_models())
    except Exception:
        return {}


@st.cache_resource(ttl=3600, show_spinner=False)
def _all_meta(self_hosted_stamp: int) -> dict:
    return {**_openrouter_meta(), **selfhosted.to_meta(selfhosted.load())}


_SELF_HOSTED_SEEN = [None]


def meta_by_id() -> dict:
    """OpenRouter's models plus the self-hosted ones in config/self_hosted.yaml. One shared copy for all pages and
    sessions (cache_resource does not copy on every read). Read-only. When the self-hosted file changes, the lists
    built from this (picker, labels, browser) are rebuilt."""
    stamp = selfhosted.stamp()
    if stamp != _SELF_HOSTED_SEEN[0]:
        if _SELF_HOSTED_SEEN[0] is not None:
            for cached in (selectable_models, picker_models, _labels, browser_table, _all_meta):
                cached.clear()
        _SELF_HOSTED_SEEN[0] = stamp
    return _all_meta(stamp)


def live_prices_now(model_ids: tuple, pins: tuple = ()) -> tuple[dict, dict, str]:
    """Current worst-case prices for the given paid models (a pinned model: its pinned endpoint's price; otherwise
    the most expensive provider); never cached. pins = sorted (model, route) pairs."""
    priced, info = with_worst_case_prices(meta_by_id(), model_ids, pins=dict(pins))
    return {m: priced[m] for m in model_ids if m in priced}, info, datetime.now().strftime("%H:%M:%S")


@st.cache_data(ttl=300, show_spinner="Checking current provider prices…")
def live_prices(model_ids: tuple, pins: tuple = ()) -> tuple[dict, dict, str]:
    """live_prices_now, reused for five minutes so clicking around the Run page stays fast. The Run button always
    checks again with live_prices_now before anything is paid."""
    return live_prices_now(model_ids, pins)


@st.cache_resource(ttl=3600)
def selectable_models() -> dict:
    return selectable(meta_by_id())


@st.cache_resource(ttl=3600)
def picker_models(show_all: bool) -> list[str]:
    """Model ids offered in the picker: every usable model, or only the main variant of each."""
    models = selectable_models() if show_all else main_variants(selectable_models())
    return sorted(models)


def model_labels() -> dict[str, str]:
    """All picker labels, built once per model-list refresh (and again when a manual cutoff changes)."""
    return _labels(tuple(sorted((k, str(v)) for k, v in load_overrides().items())))


@st.cache_resource(ttl=3600)
def _labels(overrides: tuple) -> dict[str, str]:
    return build_labels(meta_by_id(), LOCAL_MODELS, dict(overrides))


@st.cache_data(show_spinner="Loading test sets…")
def _testsets(fingerprint: tuple) -> dict[str, tuple[str, list[Item]]]:
    return load_current_testsets()


def testsets() -> dict[str, tuple[str, list[Item]]]:
    """In method order M1..M4; reloads automatically when a test set file is rebuilt (cache key = file mtimes)."""
    ts = _testsets(testsets_fingerprint())
    return {m: ts[m] for m in METHODS if m in ts}


@st.cache_data(show_spinner="Loading test sets…")
def _versioned_testsets(fingerprint: tuple, versions: tuple) -> dict[str, tuple[str, list[Item]]]:
    return {m: (f"{m}/{v}", load_testset(m, v)) for m, v in versions}


def answered_sets(calls: pd.DataFrame) -> tuple[tuple, tuple]:
    """(test sets with any answer, test sets with an answer from a real model), each sorted."""
    if not len(calls):
        return (), ()
    pairs = calls[["model", "testset"]].drop_duplicates()
    real = pairs[~pairs["model"].map(is_local)]
    return tuple(sorted(set(pairs["testset"]))), tuple(sorted(set(real["testset"])))


@st.cache_data(show_spinner=False, max_entries=8)
def _answered_sets(setting: tuple | None, stamp: tuple) -> tuple[tuple, tuple]:
    return answered_sets(_calls(setting, stamp))


def analysis_testsets(setting: tuple | None) -> dict[str, tuple[str, list[Item]]]:
    """For analysis pages: per method the newest test set version that has answers under this setting."""
    answered, real = _answered_sets(setting, store.db_stamp(conn()))
    ts = _versioned_testsets(testsets_fingerprint(),
                             tuple(sorted(analysis_versions(set(answered), preferred=set(real)).items())))
    return {m: ts[m] for m in METHODS if m in ts}


def manifest(method: str) -> dict:
    ts = testsets().get(method)
    return load_manifest(method, ts[0].split("/")[1]) if ts else {}


def key_available() -> bool:
    load_dotenv()
    return bool(os.environ.get("OPENROUTER_API_KEY"))


DEFAULT_SETTING = ("forced", 0.0, "auto", -1)


@st.cache_data(show_spinner=False, max_entries=8)
def _calls(setting: tuple | None, stamp: tuple) -> pd.DataFrame:
    if setting is None:
        return store.load_calls(conn())
    mode, temperature, reasoning, seed = setting
    return store.load_calls(conn(), mode=mode, temperature=temperature, reasoning=reasoning, seed=seed)


def load_calls(setting: tuple | None = None) -> pd.DataFrame:
    """setting = (answer mode, temperature, reasoning, seed); None = every stored answer. Read from the database
    once and reused until an answer is added or replaced."""
    return _calls(setting, store.db_stamp(conn()))


def cutoffs(models, meta: dict) -> dict:
    return {m: resolve_cutoff(m, meta) for m in models}


def fingerprint(calls: pd.DataFrame, cutoffs_map: dict) -> tuple:
    """Cache key: changes whenever new calls land, an answer is replaced or re-read (the database stamp), a cutoff is
    edited or a test set is rebuilt."""
    return (len(calls), round(float(calls["cost"].sum()), 8) if len(calls) else 0.0,
            tuple(sorted((k, str(v)) for k, v in cutoffs_map.items())), testsets_fingerprint(),
            store.db_stamp(conn()))


LINES = {"release": "After release", "cutoff": "After cutoff"}


def on_line(df: pd.DataFrame, cutoff, release, line: str) -> pd.DataFrame:
    """The comparison line. 'cutoff': before vs after the stated cutoff (±91 days), as scored. 'release': before the
    cutoff vs after the model's release, the grey zone left out (pilot E). A model without a release date (the
    controls) keeps the cutoff line."""
    if line != "release" or df.empty or cutoff is None or release is None:
        return df
    from .pilot_e import release_groups
    return df.assign(group=release_groups(df["date"], cutoff, release).to_numpy())


def analysis_frame(model: str, method: str, setting: tuple, fp: tuple, line: str, meta: dict | None = None):
    """Item scores on the chosen line with the registered windows (assign_groups): what every test in the app uses."""
    meta = meta if meta is not None else meta_by_id()
    return assign_groups(scores_on_line(model, method, setting, fp, line), method, resolve_cutoff(model, meta),
                         release_date(model, meta), line)


def scores_on_line(model: str, method: str, setting: tuple, fp: tuple, line: str) -> pd.DataFrame:
    meta = meta_by_id()
    return on_line(scores(model, method, setting, fp), resolve_cutoff(model, meta), release_date(model, meta), line)


@st.cache_data(show_spinner="Computing statistics (2,000 bootstrap draws per cell)…")
def summary(models: tuple, methods: tuple, setting: tuple, fp: tuple, line: str = "cutoff") -> pd.DataFrame:
    """Holm within each method across the models shown; plus the same on answered items only, for the refusal guard
    (pilot E). M5 uses the grid plan's 12-month windows. The index
    methods use their registered tests (analysis.registered_index_cell); on the release line also their registered
    windows (whole months: before cutoff - 91 days; from the month after the release month)."""
    meta = meta_by_id()
    frames = {}
    for m in models:
        for me in methods:
            frames[(m, me)] = analysis_frame(m, me, setting, fp, line, meta)
    blind = {}
    for me in set(methods) & set(analysis.BLIND_REGISTERED):
        b = scores(BLIND_ID, me, setting, fp)
        blind[me] = dict(zip(b["item_id"], b["score"])) if len(b) else {}
    out = analysis.summary_from_frames(frames, B=2000, blind=blind)
    answered = analysis.summary_from_frames(
        {k: (df[~df["refused"].astype(bool)] if len(df) else df) for k, df in frames.items()}, B=2000, blind=blind)
    out = answered_primary(out, answered)     # scoring spec §8: refusals left out except in registered tests
    if len(out):
        out["answered_dkappa"] = answered["dkappa"].to_numpy()
        out["answered_p_holm"] = answered["p_holm"].to_numpy()
    return out


@st.cache_data(show_spinner=False)
def company_sizes() -> dict:
    """cluster -> market-cap tercile, a proxy for how widely a company's news is reproduced online."""
    try:
        return company_size_map(pd.read_parquet(DATA_DIR / "universe.parquet"))
    except Exception:
        return {}


def app_scores(df: pd.DataFrame, method: str, drop_hedges: bool = True) -> pd.DataFrame:
    """The app's per-question score: price levels by closeness (1 = the exact price, 0 = off by a factor of 2 or more;
    the reports' primary score since pilot E, the app's since 2026-10-01), with the ±10% hit kept in hit_10pct for
    the correct/wrong labels; every other method keeps its own score. drop_hedges: return answers of exactly 0% are
    left out (scoring spec §7); the Items page keeps them to show them."""
    if drop_hedges:
        df = analysis.drop_hedges(df, method)
    if method not in LEVEL_METHODS or not len(df):
        return df
    return with_score(df.assign(hit_10pct=df["score"]), method, "primary")


SIZES = ("large", "mid", "small")


def size_of(df: pd.DataFrame, method: str) -> pd.Series:
    """Each question's company size: the size recorded on the question, else the company's market-cap tercile."""
    col = SIZE_COLUMN.get(method, "s_company_size")
    own = df[col] if col in df else pd.Series(None, index=df.index, dtype=object)
    fallback = df["s_company_size"] if "s_company_size" in df else pd.Series("unknown", index=df.index)
    return own.where(own.isin((*SIZES, "none")), fallback).fillna("unknown")   # none: a headline about no company


# Scores on a 0-1 scale that are not a share of items (M18: 1 - months off / 12; index levels: closeness): written as
# numbers, like closeness
SCORE_METHODS = {"headline_date", "index_level", "us_index_level"}
INDEX_LEVELS = {"index_level", "us_index_level"}
# Profile cells read as the correlation of the answers with the true moves (the registered M15/M17 test)
CORR_METHODS = {"index_return", "us_index_return", "index_forecast"}


def measure_name(method: str) -> str:
    if method in CORR_METHODS:
        return "Correlation"
    if method in LEVEL_METHODS or method in INDEX_LEVELS:
        return "Mean closeness"
    return "Mean score" if method in SCORE_METHODS else "Measured recall"


def value_fmt(method: str):
    """How a mean score is written: closeness and other 0-1 scores as a number (0.23), correlations signed (+0.80),
    rates as percentages (23%)."""
    if method in CORR_METHODS:
        return lambda v: "–" if v is None or pd.isna(v) else f"{v:+.2f}"
    if method in LEVEL_METHODS or method in SCORE_METHODS:
        return lambda v: "–" if v is None or pd.isna(v) else f"{v:.2f}"
    return _pct


# The columns of a summary row that a test produces (scoring spec §8 swaps them for the answered-only ones); coverage
# and the outcome rates stay from all questions, so the refusal rate is still reported
TEST_COLUMNS = ("n_pre", "n_post", "mean_pre", "mean_post", "delta", "dkappa", "ci_lo", "ci_hi", "p_value", "p_perm",
                "p_holm", "dkappa_best", "ci_lo_best", "ci_hi_best", "n_best")


def keeps_refusals(method: str) -> bool:
    """Registered tests keep their registered rule (a refusal scores 0, with the refusal guard); every other view leaves
    refusals out (scoring spec §8)."""
    return method in analysis.REGISTERED or method in analysis.GRID_REGISTERED


def answered_primary(out: pd.DataFrame, answered: pd.DataFrame) -> pd.DataFrame:
    """Summary rows with the test read on answered questions only, for every method without a registered rule."""
    if not len(out):
        return out
    res = out.copy()
    generic = ~res["method"].map(keeps_refusals).to_numpy()
    for col in TEST_COLUMNS:
        if col in res and col in answered:
            res.loc[generic, col] = answered.loc[generic, col].to_numpy()
    return res


def answered_only(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """A frame without refused questions, unless the method's registered test keeps them (scoring spec §8)."""
    if not len(df) or keeps_refusals(method) or "refused" not in df:
        return df
    return df[~df["refused"].astype(bool)]


@st.cache_resource(show_spinner=False, max_entries=4)
def _cell_index(setting: tuple, stamp: tuple, ts_key) -> tuple[dict, dict, pd.DataFrame]:
    """Read-only lookups for scores(): each (model, method, test set) cell's answers, each method's analysed version
    with its items by id, and an empty answers frame. A resource, so it is shared without a copy: through the cached
    loaders every cell of a summary unpickled all stored answers and all test items again (3 s of a cold Overview).
    Callers must not modify what it returns."""
    calls = _calls(setting, stamp)
    cells = dict(iter(calls.groupby(["model", "method", "testset"], sort=False))) if len(calls) else {}
    items = {m: (name, {it.item_id: it for it in its}) for m, (name, its) in analysis_testsets(setting).items()}
    return cells, items, calls.iloc[:0]


@st.cache_data(show_spinner=False)
def scores(model: str, method: str, setting: tuple, fp: tuple, drop_hedges: bool = True) -> pd.DataFrame:
    """Item scores on the analysed version, as every score view uses them. drop_hedges=False keeps the return answers
    of exactly 0% (for the 'I don't know' rate)."""
    cells, items, empty = _cell_index(setting, store.db_stamp(conn()), testsets_fingerprint())
    if not cells or method not in items:
        return pd.DataFrame()
    name, by_id = items[method]
    sub = cells.get((model, method, name), empty).copy()
    df = app_scores(analysis.item_scores(sub, by_id, method,
                                         resolve_cutoff(model, meta_by_id()),
                                         extra_strata={"company_size": company_sizes()}), method, drop_hedges)
    if method == "headline_date" and len(df):     # the size of the company each headline is about
        df = df.assign(s_size=df["item_id"].map(headline_sizes()).fillna("unknown"))
    return df


@st.cache_data(show_spinner=False)
def headline_sizes() -> dict:
    """M18 item_id -> the market-cap tercile of the listed company the headline is about ('none': no company)."""
    from .items import load_testset as _load
    from .methods import headline_date as _hd
    items = _load(_hd.NAME, "v1")
    try:   # only the set's headlines: the whole index has about a million rows
        news = pd.read_parquet(DATA_DIR / "news_index.parquet", columns=["newsid", "insrefs"],
                               filters=[("newsid", "in", sorted({it.truth["newsid"] for it in items}))])
        uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    except Exception:
        return {}
    return _hd.company_sizes(items, dict(zip(news["newsid"], news["insrefs"])),
                             dict(zip(uni["insref"], uni["company_key"])), company_size_map(uni))


def over_time_views(method: str) -> dict:
    """The Evidence page's over-time views for a method: the return methods add the 'I don't know' rate."""
    views = {"rolling": "Rolling 6 months", "quarters": "Quarters"}
    label = analysis.ZERO_LABEL.get(method, "Answers of 0%")
    return {**views, "hedge": label} if method in analysis.HEDGE_METHODS else views


@st.cache_data(show_spinner="Computing rolling windows…")
def hedge_rolling(model: str, method: str, setting: tuple, fp: tuple, window: int = 6) -> pd.DataFrame:
    """The share of a model's answers that are exactly 0% ('I don't know'), in rolling windows."""
    df = scores(model, method, setting, fp, drop_hedges=False)
    return analysis.rolling_curve(analysis.hedge_frame(df), window=window, min_n=15, B=500) if len(df) else \
        pd.DataFrame(columns=["month", "mean", "lo", "hi", "n"])


@st.cache_data(show_spinner="Estimating where each model's knowledge ends…")
def effective(model: str, methods: tuple, setting: tuple, fp: tuple) -> dict:
    parts = [(scores(model, m, setting, fp), METHODS[m].P0) for m in methods]
    return analysis.effective_cutoff(parts, min_side=15, B=500)


@st.cache_data(show_spinner="Computing the breakdown…")
def breakdown(model: str, method: str, key: str, setting: tuple, fp: tuple, line: str = "cutoff") -> pd.DataFrame:
    return analysis.breakdown(answered_only(analysis_frame(model, method, setting, fp, line), method), method, key,
                              B=2000)


# ---------------------------------------------------------------- formatting

def model_label(model_id: str, meta: dict) -> str:
    kind = model_id.split("@")[0]
    if kind in LOCAL_HELP:
        return f"{model_id} · local, free · {LOCAL_HELP[kind]}"
    m = meta.get(model_id)
    if not m:
        return model_id
    pricing = m.get("pricing") or {}
    price = f"${float(pricing.get('prompt') or 0) * 1e6:.2f} in / ${float(pricing.get('completion') or 0) * 1e6:.2f} out per M tokens"
    return f"{model_id} · {price} · cutoff {m.get('knowledge_cutoff') or 'unknown'}"


def build_labels(meta: dict, local_models: list[str], overrides: dict | None = None) -> dict[str, str]:
    """Compact picker label per model: name · provider · price · cutoff (pure; no Streamlit calls). A cutoff from
    config/model_cutoffs.yaml wins over OpenRouter's and is marked (manual)."""
    labels = {}
    for model_id in local_models:
        kind = model_id.split("@")[0]
        labels[model_id] = f"{LOCAL_NAMES[kind]} · free · {LOCAL_HELP[kind].split(':')[0]}"
    for model_id, m in meta.items():
        pricing = m.get("pricing") or {}
        price = (f"${float(pricing.get('prompt') or 0) * 1e6:.2f} / "
                 f"${float(pricing.get('completion') or 0) * 1e6:.2f} per M")
        manual = (overrides or {}).get(model_id) if cutoff_source(model_id, overrides or {}, meta) == "set by hand" \
            else None
        cutoff = f"{str(manual)[:7]} (manual)" if manual else (m.get("knowledge_cutoff") or "?")[:7]
        labels[model_id] = f"{display_name(model_id, meta)} · {vendor(model_id, meta)} · {price} · cutoff {cutoff}"
    return labels


def model_option(model_id: str) -> str:
    return model_labels().get(model_id) or model_id


def short_model(model_id: str) -> str:
    return display_name(model_id)


def money(x: float) -> str:
    return "$0" if not x else f"${x:.4f}" if x < 0.1 else f"${x:.2f}"


def stars(p) -> str:
    return "" if pd.isna(p) else "**" if p < 0.01 else "*" if p < 0.05 else ""


MIN_SIDE = 20            # the registered rule (pilots C-E): fewer items before or after -> no verdict
MAX_MALFORMED = 0.20


def cell_status(row) -> str:
    """The reading of one model x method cell, by the same rules as the pilot reports: no verdict below 20 items on
    a side; unresolved above 20% unreadable answers; a detection that is not significant on answered items alone
    (when that column exists) is 'refusal-driven', but only when the answer rates before and after differ by more
    than 10 points (pilot E's 'refusal-affected'): otherwise the answered-only test is the same test, and only its
    Holm rank among the other cells moves."""
    from .pilot_e import refusal_affected
    get = row.get if hasattr(row, "get") else (lambda k, d=None: d)
    if pd.isna(row["dkappa"]):
        return "not scorable"
    if min(get("n_pre", MIN_SIDE) or 0, get("n_post", MIN_SIDE) or 0) < MIN_SIDE:
        return "too few items"
    if (get("malformed_rate") or 0) > MAX_MALFORMED:
        return "unresolved"
    status = heat_reading(row["dkappa"], row["p_holm"])
    if status == "better after the cutoff" and get("registered") == "model − blind, before":
        return "none detected"   # M14/M16 test only the months before: negative = below the blind guesser
    reg = get("registered")
    if isinstance(reg, str) and reg:   # NaN in a mixed table is a generic row: the refusal guard applies
        if get("method") in analysis.DATING_REGISTERED and status == "memorization detected":   # M18 plan §3
            ans_p, ans_d = get("answered_p_holm"), get("answered_dkappa")
            if ans_p is not None and pd.notna(ans_p) and (ans_p >= 0.05 or pd.isna(ans_d) or ans_d <= 0):
                return "refusal-driven"
        return status            # the index tests define their own refusal handling (index plan §6.3)
    ans_p, ans_d = get("answered_p_holm"), get("answered_dkappa")
    cov_pre, cov_post = get("coverage_pre"), get("coverage_post")
    rates_known = cov_pre is not None and cov_post is not None and pd.notna(cov_pre) and pd.notna(cov_post)
    if rates_known and not refusal_affected(cov_pre, cov_post):
        return status
    if status == "memorization detected" and ans_p is not None and not pd.isna(ans_p) \
            and (ans_p >= 0.05 or pd.isna(ans_d) or ans_d <= 0):
        return "refusal-driven"
    return status


VERDICTS = {"not scorable": "not scorable (no cutoff, or no items on one side)",
            "too few items": "too few items (fewer than 20 on one side)",
            "unresolved": "unresolved (too many unreadable answers)",
            "memorization detected": "memorization detected",
            "refusal-driven": "not confirmed: refusal-driven",
            "better after the cutoff": "better after the cutoff (check the setup)",
            "none detected": "no significant difference"}


def verdict(row) -> str:
    """Plain-language reading of one model x method cell (cell_status in words). A model - blind cell (M14, M16, M18)
    tests only the months before the cutoff, so its detection reads as beating the blind baseline there."""
    get = row.get if hasattr(row, "get") else (lambda k, d=None: d)
    if get("registered") == "model − blind, before" and cell_status(row) == "memorization detected":
        return "beats the blind baseline before the cutoff"
    if get("registered") == "model − blind, before" and cell_status(row) == "none detected" \
            and pd.notna(row["p_holm"]) and row["p_holm"] < 0.05 and row["dkappa"] < 0:
        return "none detected (below the blind guesser)"
    return VERDICTS[cell_status(row)]


def color_scale(models: list[str], all_models: list[str]) -> alt.Scale:
    """Colour follows the model (kept for callers that pass model ids): the same colour as its dot elsewhere."""
    return model_scale(models)


def truth_summary(method: str, item: Item) -> str:
    t = item.truth
    if method == "price_recall":
        if item.fabricated:
            return f"{item.payload['name']} — fabricated company (should be refused)"
        return f"{item.payload['name']} · close {t['close']:.2f} {item.payload['currency']}"
    if method == "number_cloze":
        return f"masked number: {t['number']}"
    if method == "decop":
        return (item.payload.get("company") or item.payload["distributor"]) + " · verbatim vs 3 paraphrases"
    if method in ("price_level", "annual_return", "monthly_return") and item.fabricated:
        return f"{item.payload['name']} — fabricated company (should be refused)"
    if method in ("price_level", "known_level", "us_level"):
        return f"{item.payload['name']} · {item.payload['month_name']} average {t['level']:.2f} {item.payload['currency']}"
    if method in ("index_level", "us_index_level"):
        return f"{item.payload['symbol']} · {item.payload['month_name']}: closed at {t['level']:.2f}"
    if method in ("index_return", "us_index_return"):
        return f"{item.payload['symbol']} · {item.payload['month_name']}: {t['return_pct']:+.2f}%"
    if method == "path_date":
        return f"{t['symbol']} · 12 months ending {t['month']} (shown without name, dates or levels)"
    if method == "index_forecast":
        return (f"{item.payload['symbol']} · asked as of {item.payload['as_of']} for {item.payload['month_name']}: "
                f"{t['return_pct']:+.2f}%")
    if method in ("price_choice", "known_choice", "price_choice_prob", "known_choice_prob"):
        return (f"{item.payload['name']} · {item.payload['month_name']}: {item.payload['answer_text']} "
                f"{item.payload['currency']} among 4 options")
    if method == "split_clock":
        return (f"{item.payload['name']} · {item.payload['month_name']}: {t['level']:.2f} then, "
                f"{t['adj_level']:.2f} after the {t['event_date']} split")
    if method == "headline_date":
        return f"published {t['month']} · blind baseline {t['blind_month']}: “{item.payload['headline'][:90]}”"
    if method in ("annual_return", "known_return"):
        return f"{item.payload['name']} · {item.payload['year']}: {t['return_pct']:+.1f}%"
    if method == "monthly_return":
        return f"{item.payload['name']} · {item.payload['month_name']}: {t['return_pct']:+.1f}%"
    return item.payload.get("headline", "")[:90]


# ---------------------------------------------------------------- components

def header(title: str, lead: str) -> None:
    st.title(title)
    st.caption(lead)


def mode_picker(key: str = "mode") -> str:
    st.session_state.setdefault(key, "forced")
    mode = st.segmented_control("Answer mode", ["forced", "abstain"], key=key,
                                help="forced: the model must answer. abstain: 'don't know' is allowed. "
                                     "The two modes are stored and scored separately.")
    return mode or "forced"


def _combo_label(combo: tuple) -> str:
    temperature, reasoning, seed = combo
    return (f"temperature {temperature:.1f} · reasoning {reasoning}" + (f" · seed {seed}" if seed >= 0 else ""))


def setting_picker(key: str) -> tuple:
    """Answer mode, plus a run-settings filter once answers with more than one combination of temperature,
    reasoning effort and seed are stored (they are never mixed)."""
    restore_widgets(st.session_state, [f"{key}_mode"])
    mode = mode_picker(f"{key}_mode")
    combos = [(float(t), r, int(s)) for t, r, s in store.run_settings(conn(), mode=mode)] or [DEFAULT_SETTING[1:]]
    default = DEFAULT_SETTING[1:] if DEFAULT_SETTING[1:] in combos else combos[0]
    combo = default
    if len(combos) > 1:
        keep_choice(st.session_state, f"{key}_combo", combos, default)
        combo = st.selectbox("Run settings", combos, key=f"{key}_combo",
                             format_func=_combo_label,
                             help="Answers with different temperature, reasoning effort or seed are scored separately.")
    return (mode, *combo)


def setting_picker_value(key: str) -> tuple:
    """The value setting_picker(key) will show, read before the widget is drawn (for layouts that draw it lower)."""
    restore_widgets(st.session_state, [f"{key}_mode", f"{key}_combo"])
    mode = st.session_state.get(f"{key}_mode") or "forced"
    combos = [(float(t), r, int(s)) for t, r, s in store.run_settings(conn(), mode=mode)] or [DEFAULT_SETTING[1:]]
    default = DEFAULT_SETTING[1:] if DEFAULT_SETTING[1:] in combos else combos[0]
    combo = st.session_state.get(f"{key}_combo", default)
    return (mode, *(combo if combo in combos else default))


def setting_label(setting: tuple) -> str:
    return f"{setting[0]} answers · {_combo_label(tuple(setting[1:]))}"


def empty_state(message: str, page: str | None = None, label: str | None = None) -> None:
    with st.container(border=True):
        st.markdown(message)
        if page:
            st.page_link(page, label=label or "Go there", icon=":material/arrow_forward:")


def sidebar_status() -> None:
    s = store.stats(conn())
    ts = testsets()
    decop_mock = manifest("decop").get("paraphraser") == "mock"
    rows = [
        ("Test sets", f"{len(ts)} / {len(METHODS)}", badge("DE-COP mock", "warn") if decop_mock else badge("ready", "good")),
        ("Real models run", str(s["api_models"]), ""),
        ("Stored answers", f"{s['answers']:,}", ""),
        ("OpenRouter spend", money(s["api_spend"]), ""),
        ("API key", "", badge("found", "good") if key_available() else badge("not set", "bad")),
    ]
    body = "".join(f'<div style="display:flex;justify-content:space-between;align-items:center;margin:0.28rem 0;'
                   f'font-size:0.84rem"><span>{_esc(k)}</span><span class="eb-num" style="font-weight:600">'
                   f'{_esc(v)} {b}</span></div>' for k, v, b in rows)
    with st.sidebar:
        st.html(f'<div class="eb-eyebrow" style="margin-top:0.5rem">Status</div>{body}')
        _run_indicator()


@st.fragment(run_every=5)
def _run_indicator() -> None:
    """On every page: the run going in the background, if any, linking to the Run page."""
    active = store.active_run(conn())
    if active:
        finished = active["done"] + active["failed"]
        st.page_link("pages/run.py", label=f"Run going: {finished:,} / {active['n_calls']:,}", icon=":material/sync:")
        st.caption(f"{money(active['spent'])} spent" + (" · stopping…" if active["stop_requested"] else "")
                   + " · open Run to stop it")


def run_progress_text(row: dict) -> str:
    """One line for a run in progress: finished calls (stored + failed) of all, money spent."""
    text = f"{row['done'] + row['failed']:,} / {row['n_calls']:,} calls · {money(row['spent'])} spent"
    if row["failed"]:
        text += f" · {row['failed']:,} failed"
    return text + (" · stopping…" if row.get("stop_requested") else "")


def run_outcome(row: dict) -> tuple[str, str]:
    """(message, callout kind) for a run that has ended, worded like the end of a run on the old Run page."""
    rid, status = row["run_id"], row["status"] or ""
    done = f"{row['done']:,} answers stored, {money(row['spent'])} spent."
    again = "run the same selection again to retry them (stored answers are never paid twice)."
    if status == "done":
        return f"<b>Run {rid} finished.</b> {done}", "good"
    if status.startswith("done with"):
        return f"<b>Run {rid} finished.</b> {done} {row['failed']:,} calls failed: {again}", "warn"
    if status == "stopped at cap":
        return f"<b>Run {rid} stopped at the spend cap.</b> {done} Raise the cap and run again to continue.", "warn"
    if status == "stopped by user":
        return f"<b>Run {rid}: Stopped.</b> {done} Run the same selection again to continue.", "warn"
    if status == "failed to start":
        return (f"<b>Run {rid} failed to start.</b> Nothing was paid. Details: results/jobs/{rid}.log", "bad")
    return (f"<b>Run {rid} was interrupted</b> (its worker stopped, e.g. the computer restarted). {done} To finish it, "
            f"{again} Details: results/jobs/{rid}.log", "bad")


def plain(text: str) -> str:
    """Escape markdown/LaTeX so article text (with $, *, _, # ...) renders literally in st.markdown."""
    out = str(text or "")
    for ch in "\\`*_{}[]<>()#+-.!|$~":
        out = out.replace(ch, "\\" + ch)
    return out


def method_title(method: str) -> str:
    return LABELS.get(method, method)


# ---------------------------------------------------------------- design layer

LOCAL_NAMES = {"mock/oracle": "Oracle (control)", "mock/random": "Random (control)",
               "baseline/blind": "Blind baseline"}


def display_name(model_id: str, meta: dict | None = None) -> str:
    """Human name: 'Oracle (control)', 'Llama 3.3 70B Instruct' … (vendor prefix dropped)."""
    kind = model_id.split("@")[0]
    if kind in LOCAL_NAMES:
        return LOCAL_NAMES[kind]
    name = ((meta if meta is not None else meta_by_id()).get(model_id) or {}).get("name") or model_id.split("/", 1)[-1]
    return name.split(": ", 1)[-1]


def vendor(model_id: str, meta: dict | None = None) -> str:
    kind = model_id.split("@")[0]
    if kind in LOCAL_NAMES:
        return "local · free"
    name = ((meta if meta is not None else meta_by_id()).get(model_id) or {}).get("name") or ""
    return name.split(": ", 1)[0] if ": " in name else model_id.split("/", 1)[0]


def _esc(text) -> str:
    import html as _html
    return _html.escape(str(text))


def inject_css() -> None:
    """Page-wide polish the theme cannot express: reading width, tabular numbers, cards, badges, callouts."""
    p = palette()
    dark = p is PALETTES["dark"]
    card = "#212120" if dark else "#ffffff"
    sub = "#c3c2b7" if dark else "#52514e"
    line = "#2c2c2a" if dark else "#e1e0d9"
    tints = {  # background, border, text for callouts and badges
        "info": ("#16263a", "#25466e", "#b7d3f6") if dark else ("#eef5fd", "#cde2fb", "#184f95"),
        "warn": ("#332a12", "#5c4a17", "#f5d58a") if dark else ("#fdf7e6", "#f3dfa2", "#7a5600"),
        "good": ("#12301c", "#1f5230", "#9fdcae") if dark else ("#ecf7ef", "#c3e6cc", "#0f6a2a"),
        "bad": ("#3a1818", "#6b2a2a", "#f4b3b3") if dark else ("#fdeeee", "#f5c9c8", "#9c2b2a"),
        "muted": ("#262625", "#383835", "#c3c2b7") if dark else ("#f4f3f0", "#e1e0d9", "#52514e"),
    }
    tint_css = "\n".join(
        f".eb-callout.{k}{{background:{bg};border-color:{bd};color:{fg}}}"
        f".eb-badge.{k}{{background:{bg};border-color:{bd};color:{fg}}}" for k, (bg, bd, fg) in tints.items())
    rc = READING_COLORS["dark" if dark else "light"]
    # a selected pill or segment whose label is in a role colour lights up in that colour instead of the theme violet
    # (border, 10% fill and text, as Streamlit does with the primary colour); the label is found by its colour span
    role_css = "\n".join(
        f'button:is([aria-pressed="true"], [aria-checked="true"]):has(.stMarkdownColoredText[style*="{rgb}"])'
        f"{{border-color:{hexc} !important; background-color:rgba({rgb}, 0.1) !important; color:{hexc} !important;}}"
        for hexc in (c[1] for c in ROLE_COLORS.values())
        for rgb in [", ".join(str(int(hexc[i:i + 2], 16)) for i in (1, 3, 5))])
    accent, accent_ink =("#9b85ff", "#b3a3ff") if dark else ("#6d4aff", "#5a3bdb")
    st.html(f"""<style>
    .block-container {{max-width: 1180px; padding-top: 4.2rem; padding-bottom: 4rem;}}
    [data-testid="stMetricValue"], .eb-num, td, [data-testid="stDataFrame"] {{font-variant-numeric: tabular-nums;}}
    h1 {{letter-spacing: -0.02em;}}
    h2, h3 {{letter-spacing: -0.01em;}}
    .eb-eyebrow {{font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
                  color: {p['muted']}; margin: 0 0 0.35rem 0;}}
    .eb-lead {{color: {sub}; font-size: 1rem; line-height: 1.55; max-width: 62rem; margin: 0.1rem 0 0.4rem 0;}}
    .eb-meta {{color: {p['muted']}; font-size: 0.85rem; margin-bottom: 0.6rem;}}
    .eb-title {{display:flex; align-items:center; gap:0.6rem; margin:0 0 0.3rem 0;}}
    .eb-title h1 {{margin:0; padding:0;}}
    .eb-badge {{display:inline-block; font-size:0.72rem; font-weight:600; padding:0.08rem 0.5rem; border-radius:999px;
                border:1px solid; line-height:1.4; white-space:nowrap; vertical-align:middle;}}
    .eb-callout {{border:1px solid; border-radius:8px; padding:0.65rem 0.9rem; font-size:0.9rem; line-height:1.5;
                  margin:0.4rem 0 0.9rem 0;}}
    .eb-cards {{display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:0.75rem; margin:0.3rem 0 0.8rem 0;}}
    .eb-card {{background:{card}; border:1px solid {line}; border-radius:10px; padding:0.85rem 1rem;}}
    .eb-card .v {{font-size:1.55rem; font-weight:650; font-variant-numeric:tabular-nums; margin:0.15rem 0;}}
    .eb-card .l {{font-size:0.78rem; color:{p['muted']}; font-weight:600; letter-spacing:0.04em; text-transform:uppercase;}}
    .eb-card .s {{font-size:0.84rem; color:{sub}; line-height:1.4;}}
    .eb-card .t {{font-weight:600; font-size:0.98rem;}}
    .eb-row {{display:flex; flex-wrap:wrap; gap:0.35rem; margin-top:0.45rem;}}
    .eb-section {{margin: 1.6rem 0 0.2rem 0;}}
    .eb-section h3 {{margin:0; padding:0; font-size:1.2rem;}}
    .eb-section p {{color:{sub}; margin:0.25rem 0 0 0; font-size:0.93rem; max-width: 60rem;}}
    .eb-steps {{display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:0.75rem;}}
    .eb-dot {{display:inline-block; width:0.66rem; height:0.66rem; border-radius:50%; margin-right:0.45rem;
               vertical-align:0.02rem; flex:none;}}
    .eb-card .v.bad {{color:{rc['detected']};}} .eb-card .v.warn {{color:{rc['warn']};}}
    .eb-card .v.muted {{color:{p['muted']}; font-weight:600;}}
    .eb-section h3 {{border-left:4px solid {accent}; padding-left:0.6rem;}}
    .eb-section p {{padding-left:0.85rem;}}
    .eb-eyebrow {{color:{accent_ink};}}
    .eb-model {{border-left:4px solid; padding:0.1rem 0 0.1rem 0.7rem; margin-bottom:0.35rem;}}
    .eb-key {{display:flex; flex-wrap:wrap; gap:0.35rem 1.1rem; align-items:center; font-size:0.84rem; color:{sub};
              margin:0.2rem 0 0.8rem 0;}}
    .eb-key .sw {{display:inline-block; width:0.9rem; height:0.9rem; border-radius:3px; vertical-align:-0.12rem;
                  margin-right:0.35rem;}}
    .kh .mk.above {{color:{rc['detected']}; font-size:0.8rem;}} .kh .mk.none {{color:{rc['none']};}}
    .kh .mk.unres {{color:{rc['warn']}; font-weight:700;}}
    .kh td.hit {{box-shadow: inset 0 0 0 2px {rc['detected']};}}
    .kh-wrap {{overflow-x:auto;}}
    .kh th .kh-marks {{display:block; white-space:normal; line-height:1.15; margin-top:1px;}}
    .kh th .kh-marks span {{display:inline-block;}}
    .kh.dense {{width:100%; table-layout:fixed; font-size:0.8rem; border-spacing:2px;}}
    .kh.dense thead th:first-child {{width:8.6rem;}}
    .kh.dense tbody th {{white-space:normal; line-height:1.2; padding:0.2rem 0.3rem;}}
    .kh.dense th {{padding:0.2rem 0.1rem;}}
    .kh.dense td {{min-width:0; padding:0.45rem 0.1rem;}}
    .kh.dense td .mk {{right:3px; font-size:0.62rem;}}
    .kh.dense .kh-cut, .kh.dense .kh-rel, .kh.dense .kh-habit {{font-size:0.62rem;}}
    .kh {{--kh-lo:{"#1a2433" if dark else "#eef4fc"}; --kh-hi:{"#5598e7" if dark else "#184f95"};
          border-collapse:separate; border-spacing:3px; font-size:0.9rem;}}
    .kh th {{font-weight:600; color:{sub}; padding:0.25rem 0.5rem; text-align:center; white-space:nowrap;}}
    .kh tbody th {{text-align:left; font-weight:500;}}
    .kh th.after {{color:{p['muted']}; font-style:italic; font-weight:500;}}
    .kh td {{position:relative; text-align:center; padding:0.5rem 0.35rem; min-width:4.4rem; border-radius:6px;
             font-variant-numeric:tabular-nums; cursor:default;}}
    .kh td .mk {{position:absolute; top:1px; right:5px; font-size:0.7rem;}}
    .kh td .v.dim {{font-weight:500; opacity:0.85;}}
    .kh td .n {{display:block; font-size:0.68rem; opacity:0.8;}}
    .kh td.hatched {{background:repeating-linear-gradient(45deg, {line} 0 4px, transparent 4px 8px); color:{p['muted']};}}
    .kh td.aftercut {{outline:1px dashed {p['muted']}; outline-offset:-2px;}}
    .kh td.kh-empty {{color:{p['muted']};}}
    .kh-cut {{font-size:0.68rem; font-weight:600; color:{p['diverging'][2]}; white-space:nowrap;}}
    .kh-habit {{font-size:0.68rem; font-weight:600; color:{p['muted']}; white-space:nowrap;}}
    .kh-rel {{font-size:0.68rem; font-weight:600; color:{p['ink']}; white-space:nowrap;}}
    {tint_css}
    {role_css}
    </style>""")


def badge(text: str, kind: str = "muted") -> str:
    return f'<span class="eb-badge {kind}">{_esc(text)}</span>'


def page_header(title: str, lead: str, meta: str = "", tag: tuple[str, str] | None = None) -> None:
    tag_html = badge(*tag) if tag else ""
    st.html(f'<div class="eb-title"><h1>{_esc(title)}</h1>{tag_html}</div>'
            f'<div class="eb-lead">{lead}</div>' + (f'<div class="eb-meta">{meta}</div>' if meta else ""))


def callout(html_text: str, kind: str = "info") -> None:
    st.html(f'<div class="eb-callout {kind}">{html_text}</div>')


def section(title: str, lead: str = "") -> None:
    st.html(f'<div class="eb-section"><h3>{_esc(title)}</h3>' + (f"<p>{lead}</p>" if lead else "") + "</div>")


def eyebrow(text: str, color: str | None = None) -> None:
    style = f' style="color:{color}"' if color else ""
    st.html(f'<div class="eb-eyebrow"{style}>{_esc(text)}</div>')


def cards_html(items: list[dict]) -> str:
    """items: {label, value, sub, title, badges(list of html), color (top edge), dot (colour before the title),
    tone (value colour: bad/warn/good/muted)} — any subset."""
    out = []
    for it in items:
        parts = []
        if it.get("label"):
            parts.append(f'<div class="l">{_esc(it["label"])}</div>')
        if it.get("title"):
            dot = f'<span class="eb-dot" style="background:{it["dot"]}"></span>' if it.get("dot") else ""
            parts.append(f'<div class="t">{dot}{_esc(it["title"])}</div>')
        if it.get("value") is not None:
            tone = f' {it["tone"]}' if it.get("tone") else ""
            parts.append(f'<div class="v{tone}">{_esc(it["value"])}</div>')
        if it.get("sub"):
            parts.append(f'<div class="s">{it["sub"]}</div>')
        if it.get("badges"):
            parts.append('<div class="eb-row">' + "".join(it["badges"]) + "</div>")
        style = f' style="border-top:3px solid {it["color"]}"' if it.get("color") else ""
        out.append(f'<div class="eb-card"{style}>{"".join(parts)}</div>')
    return f'<div class="eb-cards">{"".join(out)}</div>'


def cards(items: list[dict]) -> None:
    st.html(cards_html(items))


def style_chart(chart, height: int | None = None):
    """One consistent chart look (font, hairline grid, muted axes) in the viewer's light or dark mode."""
    p = palette()
    dark = p is PALETTES["dark"]
    grid, axis_line, label = ("#2c2c2a", "#383835", "#c3c2b7") if dark else ("#ecebe6", "#c3c2b7", "#52514e")
    if height:
        chart = chart.properties(height=height)
    return (chart.configure(background="transparent", font="Inter, system-ui, sans-serif")
            .configure_view(stroke=None)
            .configure_axis(gridColor=grid, domainColor=axis_line, tickColor=axis_line, labelColor=label,
                            titleColor=label, labelFontSize=12, titleFontSize=12, titleFontWeight=500,
                            labelLimit=260)
            .configure_legend(labelColor=label, titleColor=label, labelFontSize=12, titleFontSize=12,
                              orient="top", symbolType="circle")
            .configure_header(labelColor=label, titleColor=label, labelFontSize=12, labelFontWeight=600))


def show_chart(chart, height: int | None = None, stretch: bool = True) -> None:
    st.altair_chart(style_chart(chart, height), theme=None, width="stretch" if stretch else "content")


@st.cache_data(show_spinner=False, max_entries=64)
def _chart_spec(builder: str, theme: str, height: int | None, stamp: tuple, args: tuple) -> dict:
    """The finished Vega-Lite spec of ui.<builder>(*args): building a many-panel Altair chart takes about a second,
    so an unchanged chart is reused. Keyed by the theme (colours) and the database stamp (model colours)."""
    return style_chart(globals()[builder](*args), height).to_dict()


def show_cached_chart(builder: str, *args, height: int | None = None) -> None:
    """show_chart(ui.<builder>(*args)), rebuilt only when its inputs, the theme or the stored answers change."""
    theme = "dark" if palette() is PALETTES["dark"] else "light"
    st.vega_lite_chart(_chart_spec(builder, theme, height, store.db_stamp(conn()), args), theme=None, width="stretch")


def logo_svg() -> str:
    """A small wordmark as a data URI (st.logo needs an image)."""
    import base64
    ink = palette()["ink"]
    blue = palette()["categorical"][0]
    red = palette()["categorical"][7]
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="150" height="28" viewBox="0 0 150 28">'
           f'<rect x="1" y="5" width="5" height="18" rx="2" fill="{blue}"/>'
           f'<rect x="8" y="10" width="5" height="13" rx="2" fill="{blue}" opacity="0.6"/>'
           f'<rect x="15" y="2" width="5" height="21" rx="2" fill="{red}"/>'
           f'<text x="27" y="20" font-family="Inter, Segoe UI, sans-serif" font-size="17" font-weight="650" '
           f'fill="{ink}">evalBench</text></svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


# ---------------------------------------------------------------- charts

HEAT_READINGS = ["memorization detected", "refusal-driven", "better after the cutoff", "unresolved", "none detected",
                 "too few items", "not scorable"]
HEAT_FILLS = {"light": ["#f3b4b0", "#f1cf7a", "#f6e2ae", "#f6e2ae", "#f0efec", "#f8f8f7", "#f8f8f7"],
              "dark": ["#6b2a2a", "#7a5c12", "#5c4a17", "#5c4a17", "#383835", "#262625", "#262625"]}


def heat_reading(dkappa, p_holm) -> str:
    """A heatmap cell is coloured by its reading, so red only ever means a significant detection."""
    if pd.isna(dkappa) or pd.isna(p_holm):
        return "not scorable"
    if p_holm < 0.05:
        return "memorization detected" if dkappa > 0 else "better after the cutoff"
    return "none detected"


HEADER_LINE = 13      # px per line of a heatmap column header


def heat_label(d, q, reading: str) -> str:
    """A heatmap cell's number: none without a verdict (too few items), no stars where the guard holds it back."""
    if reading == "too few items":
        return "–"
    if reading == "unresolved":                   # too many unreadable answers: no verdict either
        return "?"
    if pd.isna(d):
        return "n/a"
    return f"{d:+.2f}" if reading == "refusal-driven" else f"{d:+.2f}{stars(q)}"


def heat_note(table: pd.DataFrame) -> str:
    """A line under the heatmap for the cells that invite questions, only when the table has them: stars on a grey
    cell (significant the other way), '–' (too few answers), '?' (too many unreadable answers)."""
    if not len(table):
        return ""
    reading = table["reading"] if "reading" in table else table.apply(cell_status, axis=1)
    grey = reading.isin(["none detected", "better after the cutoff"])
    other_way = (grey & (table["p_holm"] < 0.05) & (table["dkappa"] < 0)).any()
    parts = []
    if other_way:
        parts.append("Stars on a grey cell mean significant in the other direction, never memory: below the blind "
                     "guesser (M14, M16, M18) or better after the cutoff than before. On M14 it is mostly models "
                     "mixing up which index is asked (the all-share index for the OMXS30, or the scale of OMX "
                     "Helsinki).")
    if (reading == "too few items").any():
        parts.append("“–”: fewer than 20 answered questions on one side, so no number is shown.")
    if (reading == "unresolved").any():
        parts.append("“?”: more than 20% unreadable answers.")
    return " ".join(parts)


def heatmap_chart(table: pd.DataFrame, models: list[str], methods: list[str]):
    """Model x method grid of Δκ with the value and significance written in each cell."""
    p = palette()
    t = table.copy()
    t["name"] = t["model"].map(display_name)
    t["method_label"] = t["method"].map(method_short)
    t["reading"] = [cell_status(r) for _, r in t.iterrows()]
    t["label"] = [heat_label(d, q, r) for d, q, r in zip(t["dkappa"], t["p_holm"], t["reading"])]
    t["ci"] = [("too few items" if r == "too few items" else "refusals" if r == "refusal-driven"
                else f"[{lo:+.2f}, {hi:+.2f}]" if pd.notna(lo) else "")
               for lo, hi, r in zip(t["ci_lo"], t["ci_hi"], t["reading"])]
    names = [display_name(m) for m in models]
    t["method_axis"] = t["method"].map(lambda m: "|".join(method_lines(m)))   # one header line per "|"
    order = ["|".join(method_lines(m)) for m in methods]
    many = len(methods) > 7                       # narrow columns: slightly smaller cell text
    # Vega draws a multi-line label downwards from its first line: lift the headers by the height of the extra lines
    # of the tallest one, so no header reaches into the first row of cells
    lift = (max(len(method_lines(m)) for m in methods) - 1) * HEADER_LINE if methods else 0
    # each header in its method's role colour; main methods keep the theme's label colour (configure_axis)
    coloured = {"|".join(method_lines(m)): role_hex(m) for m in methods if role_hex(m)}
    header_color = {"expr": "".join(f"datum.value == '{k}' ? '{c}' : " for k, c in coloured.items())
                    + "'#c3c2b7'"} if coloured else alt.Undefined
    t["stat"] = [stat_name(r) for _, r in t.iterrows()]
    tip = [alt.Tooltip("name:N", title="model"), alt.Tooltip("method_label:N", title="method"),
           alt.Tooltip("stat:N", title="statistic"),
           alt.Tooltip("dkappa:Q", title="value", format="+.3f"), alt.Tooltip("ci:N", title="95% interval"),
           alt.Tooltip("p_holm:Q", title="p (Holm)", format=".4f"), alt.Tooltip("verdict:N"),
           alt.Tooltip("n_pre:Q", title="items before"), alt.Tooltip("n_post:Q", title="items after")]
    base = alt.Chart(t).encode(
        x=alt.X("method_axis:N", title=None, sort=order, axis=alt.Axis(
            orient="top", labelAngle=0, domain=False, ticks=False, labelFontWeight=600, labelLimit=0,
            labelExpr="split(datum.label, '|')", labelLineHeight=HEADER_LINE, labelPadding=6 + lift,
            labelFontSize=10 if many else 12, labelColor=header_color)),
        y=alt.Y("name:N", title=None, sort=names, axis=alt.Axis(domain=False, ticks=False, labelFontWeight=500,
                                                                labelPadding=18)))
    present = [r for r in HEAT_READINGS if r in set(t["reading"])]
    fills = dict(zip(HEAT_READINGS, HEAT_FILLS[theme_name()]))
    rect = base.mark_rect(cornerRadius=6, stroke=p["surface"], strokeWidth=3).encode(
        color=alt.Color("reading:N", title=None, scale=alt.Scale(domain=present, range=[fills[r] for r in present]),
                        legend=alt.Legend(orient="bottom", direction="horizontal", symbolType="square",
                                          symbolSize=180, labelLimit=640, labelExpr=(
                                              "datum.label == 'memorization detected' ? 'memorization detected (M14, "
                                              "M16, M18: beats the blind baseline before the cutoff)' : datum.label"
                                              if (t["stat"] == "model − blind baseline (before the cutoff)").any()
                                              else "datum.label"))),
        tooltip=tip)
    value = base.mark_text(fontSize=13 if many else 15, fontWeight=700, dy=-7, color=p["ink"]).encode(
        text="label:N", tooltip=tip)
    ci = base.mark_text(fontSize=10 if many else 11, dy=10, color=p["ink"], opacity=0.75).encode(text="ci:N", tooltip=tip)
    dots = alt.Chart(pd.DataFrame({"name": names})).mark_circle(size=90, opacity=1).encode(
        x=alt.value(-10), y=alt.Y("name:N", sort=names), color=alt.Color("name:N", scale=model_scale(models), legend=None))
    return (rect + value + ci + dots).resolve_scale(color="independent").properties(height=62 * len(models) + 60)


def forest_chart(df: pd.DataFrame, all_models: list[str], order: list[str] | None = None):
    """One method: models in a fixed order (by cutoff), never ranked; dot = estimate, line = 95% interval, value
    written above the dot."""
    p = palette()
    d = df.dropna(subset=["dkappa"]).copy()
    d["name"] = d["model"].map(display_name)
    names = [display_name(m) for m in fixed_order(d, order or list(d["model"]))]
    color = alt.Color("name:N", scale=model_scale(list(dict.fromkeys(d["model"]))), legend=None)
    y = alt.Y("name:N", title=None, sort=names, axis=alt.Axis(domain=False, ticks=False, labelFontWeight=500,
                                                              labelPadding=8))
    lo = min(-0.25, float(d["ci_lo"].min()) - 0.05) if len(d) else -1
    hi = max(0.25, float(d["ci_hi"].max()) + 0.05) if len(d) else 1
    x_title, x_scale = forest_axis_title(d), alt.Scale(domain=[lo, hi], nice=False)
    x = alt.X("dkappa:Q", title=x_title, scale=x_scale)
    tip = [alt.Tooltip("name:N", title="model"), alt.Tooltip("dkappa:Q", title=x_title.split("  (")[0], format="+.3f"),
           alt.Tooltip("ci_lo:Q", title="95% low", format="+.3f"), alt.Tooltip("ci_hi:Q", title="95% high", format="+.3f"),
           alt.Tooltip("p_holm:Q", title="p (Holm)", format=".4f"), alt.Tooltip("verdict:N")]
    base = alt.Chart(d).encode(y=y, color=color, tooltip=tip)
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4], color=p["muted"]).encode(x="x:Q")
    band = base.mark_rule(strokeWidth=2.5).encode(
        x=alt.X("ci_lo:Q", title=x_title, scale=x_scale), x2="ci_hi:Q")
    dot = base.mark_point(filled=True, size=110, opacity=1).encode(x=x)
    label = base.mark_text(dy=-12, fontSize=12, fontWeight=600).encode(x=x, text=alt.Text("dkappa:Q", format="+.2f"))
    return (zero + band + dot + label).properties(height=46 * len(d) + 40)


def quarter_chart(frames: pd.DataFrame, cut_rows: list[dict], models: list[str], all_models: list[str],
                  rel_rows: list[dict] | None = None):
    """Mean item score per quarter of the item date, one line per model, a dashed line at each model's cutoff and a
    solid one at its release."""
    p = palette()
    qd = frames.copy()
    qd["name"] = qd["model"].map(display_name)
    order = sorted(set(qd["quarter"]) | {r["quarter"] for r in cut_rows} | {r["quarter"] for r in rel_rows or []})
    present = [m for m in models if m in set(qd["model"])] or list(dict.fromkeys(qd["model"]))
    color = alt.Color("name:N", title=None, scale=model_scale(present),
                      legend=alt.Legend(values=[display_name(m) for m in present]))
    x = alt.X("quarter:O", title=None, sort=order,
              axis=alt.Axis(labelAngle=0, labelExpr="slice(datum.value, 4) == 'Q1' ? slice(datum.value, 0, 4) : ''",
                            ticks=False))
    lines = alt.Chart(qd).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
        x=x, y=alt.Y("mean:Q", title="mean item score", scale=alt.Scale(domain=[0, 1])), color=color,
        tooltip=[alt.Tooltip("name:N", title="model"), "quarter:O", alt.Tooltip("mean:Q", format=".3f"),
                 alt.Tooltip("n:Q", title="items")])
    if not cut_rows:
        return lines
    cuts = pd.DataFrame(cut_rows)
    cuts["name"] = cuts["model"].map(display_name)
    rules = alt.Chart(cuts).mark_rule(strokeDash=[5, 4], strokeWidth=2).encode(
        x=x, color=color, tooltip=[alt.Tooltip("name:N", title="model"), alt.Tooltip("quarter:O", title="cutoff")])
    if not rel_rows:
        return lines + rules
    rels = pd.DataFrame(rel_rows)
    rels["name"] = rels["model"].map(display_name)
    released = alt.Chart(rels).mark_rule(strokeWidth=2, opacity=0.8).encode(
        x=x, color=color, tooltip=[alt.Tooltip("name:N", title="model"), alt.Tooltip("quarter:O", title="released")])
    return lines + rules + released


# ---------------------------------------------------------------- profile view: one cell per model and year
PROFILE_LABELS = {  # mark, short text, hover text
    "recall_above_baseline": ("●", "above baseline", "Recall significantly above the model's own level on facts it "
                              "cannot know: facts from this period can be recovered."),
    "none_detected": ("○", "none detected", "This sample found no recall above the baseline. That is not proof of "
                      "absence and never means a period is clean."),
    "insufficient_coverage": ("–", "insufficient coverage", "Too few items for a meaningful assessment."),
    "unresolved": ("?", "unresolved", "Too many unreadable answers to interpret."),
    "after_cutoff": ("", "after cutoff", "After the model's training cutoff: its own 'cannot know' level, shown for "
                     "comparison."),
}
MARK_CLASS = {"recall_above_baseline": "above", "none_detected": "none", "insufficient_coverage": "thin",
              "unresolved": "unres", "after_cutoff": "after"}
PROFILE_BASELINES = {"cutoff": ("", "own after-cutoff level"), "release": ("_release", "own after-release level")}


def _pct(v) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:.0%}"


def fixed_order(df: pd.DataFrame, order: list[str]) -> list[str]:
    """Models in a fixed order (e.g. by cutoff), never by score: a diagnostic ranks nobody."""
    present = list(dict.fromkeys(df["model"]))
    return [m for m in order if m in present] + sorted(m for m in present if m not in order)


def _cell_label(c: dict, suffix: str) -> str:
    if c["label"] in ("after_cutoff", "insufficient_coverage", "unresolved") or not suffix:
        return c["label"]
    return c.get("label" + suffix) or c["label"]


def baseline_summary(cells: list[dict], baseline: str) -> str:
    """The baseline one model's grid is read against, e.g. 'baseline: 32% (63 items after the cutoff)'."""
    from .pilot import MIN_BASE
    suffix = PROFILE_BASELINES[baseline][0]
    c = next((c for c in cells if c.get("stratum", "all") == "all"), cells[0])
    if c.get("paired_blind"):
        return f"baseline: {PAIRED_LABEL.get(c.get('method', ''), 'a blind baseline on the same items')}, cell by cell"
    n = int(c.get("n_baseline" + suffix) or 0)
    refused = c.get("refused_baseline" + suffix)
    # the refusal share only where the baseline can be read: on a handful of questions it says nothing
    note = (f" · {refused:.0%} of the questions there refused, left out" if refused is not None
            and pd.notna(refused) and refused >= 0.05 and n >= MIN_BASE else "")
    where = "after the cutoff" if baseline == "cutoff" else "after release"
    fmt = value_fmt(c.get("method", ""))
    what = "correlation " if c.get("corr") else ""
    text = f"baseline: {what}{fmt(c.get('baseline' + suffix))} ({n} answer{'' if n == 1 else 's'} {where}{note})"
    return text + (" · too few for a reading" if n < MIN_BASE else "")


def baseline_changes(cells: list[dict]) -> tuple[int, int]:
    """(readings that differ between the after-cutoff and after-release baseline, readings tested)."""
    tested = [c for c in cells if c["label"] not in ("after_cutoff", "insufficient_coverage", "unresolved")]
    return sum(_cell_label(c, "") != _cell_label(c, "_release") for c in tested), len(tested)


def profile_grid_html(cells: list[dict], periods: list[str], strata: list[str], color: str = "recall",
                      baseline: str = "cutoff", cutoff=None, release=None, stratum_labels: dict | None = None,
                      habit: str | None = None) -> str:
    """One model's grid: rows = strata, columns = periods. The number is always the measured recall; the colour is
    the recall or its excess over the measured baseline; the corner mark is the reading; hovering shows counts,
    the interval, the baselines and the upper bound. Columns after the model's release are greyed. habit (M18): the
    month the model names most often; its year is marked, since fallback answers score there by luck."""
    import html as _html
    from datetime import date as _date

    suffix, base_name = PROFILE_BASELINES[baseline]
    by = {(c["stratum"], c["period"]): c for c in cells}
    method = cells[0].get("method", "") if cells else ""
    fmt = value_fmt(method)
    measure = measure_name(method)
    head = []
    for p in periods:
        after = release is not None and _date(int(p[:4]), 1, 1) > release
        mark = ' <span class="kh-cut">▾ cutoff</span>' if cutoff is not None and str(cutoff.year) == p[:4] else ""
        if release is not None and str(release.year) == p[:4]:
            mark += ' <span class="kh-rel">● release</span>'
        if habit and habit[:4] == p[:4]:
            mark += f' <span class="kh-habit" title="Its habitual answer, {_esc(habit)}">◇ habit</span>'
        cls = ' class="after"' if after else ""
        # markers on their own line under the year, so they never widen a column
        head.append(f"<th{cls}>{_esc(p)}" + (f'<span class="kh-marks">{mark.strip()}</span>' if mark else "") + "</th>")
    rows = []
    for s in strata:
        tds = []
        for p in periods:
            c = by.get((s, p))
            if c is None:
                tds.append('<td class="kh-empty">·</td>')
                continue
            label = _cell_label(c, suffix)
            mark, short, text = PROFILE_LABELS[label]
            mk = MARK_CLASS[label]
            base = c.get("baseline" + suffix)
            lines = [f"{c['facts']} facts / {c['responses']} responses",
                     f"{measure}: {fmt(c['recall'])} ({fmt(c['ci'][0])}–{fmt(c['ci'][1])})",
                     f"{PAIRED_LABEL.get(method, 'Blind baseline').capitalize()}: {fmt(base)}"
                     if c.get("paired_blind") else
                     f"Matched baseline ({base_name}): {fmt(base)} · n={c.get('n_baseline' + suffix, 0)}"]
            if c.get("refused_share") is not None and pd.notna(c["refused_share"]) and c["refused_share"] > 0:
                lines.append(f"Refused: {c['refused_share']:.0%} of this period's questions (left out of the numbers)")
            if habit and habit[:4] == p[:4]:
                lines.append(f"Its habitual answer, {habit}, falls in this year: when it does not know, its fallback "
                             "answer lands near these headlines and scores by luck.")
            if c.get("paired_blind"):
                pass                                     # the matched baseline line already is the blind baseline
            elif c.get("baseline_blind") is not None and not pd.isna(c["baseline_blind"]):
                lines.append(f"Blind baseline on the same items: {fmt(c['baseline_blind'])}")
            if label == "none_detected" and c.get("detection_upper_bound" + suffix) is not None:
                lines.append(f"Excess detection ≤ {fmt(c['detection_upper_bound' + suffix])} (95%)")
            lines.append(f"{mark} {short}: {text}".strip())
            if c.get("after_availability"):
                lines.append("Facts after this model became available: it cannot have seen them (a natural "
                             "negative control).")
            title = _html.escape("\n".join(lines), quote=True).replace("\n", "&#10;")
            if label in ("insufficient_coverage", "unresolved"):
                # no verdict, but the measured recall is still shown (greyed), as in the per-year cell format
                shown = (f'<span class="v dim">{fmt(c["recall"])}</span><span class="n">n={c["facts"]}</span>'
                         if c["facts"] and not pd.isna(c["recall"]) else f'<span class="v">n={c["facts"]}</span>')
                tds.append(f'<td class="hatched" title="{title}">{shown}<span class="mk {mk}">{mark}</span></td>')
                continue
            r = c["recall"]
            if color == "excess":
                b = 0.0 if base is None or pd.isna(base) else base
                t = (r - b) / max(1 - b, 1e-9)
            else:
                t = r
            t = 0.0 if pd.isna(t) else min(max(t, 0.0), 1.0)
            ink = "#fff" if t > 0.55 else "inherit"
            cls = {"after_cutoff": "aftercut", "recall_above_baseline": "hit"}.get(label, "")
            tds.append(f'<td class="{cls}" title="{title}" style="background:color-mix(in oklab, var(--kh-hi) '
                       f'{round(t * 100)}%, var(--kh-lo));color:{ink}"><span class="v">{fmt(r)}</span>'
                       f'<span class="mk {mk}">{mark}</span></td>')
        name = (stratum_labels or {}).get(s, s)
        rows.append(f"<tr><th>{_esc(name)}</th>{''.join(tds)}</tr>")
    # many periods (M18, the index methods: 2016-2026): a compact table that fills the card instead of scrolling
    table_cls = "kh dense" if len(periods) > DENSE_PERIODS else "kh"
    return (f'<div class="kh-wrap"><table class="{table_cls}"><thead><tr><th></th>{"".join(head)}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


# ---------------------------------------------------------------- profile and robustness data (cached)
BLIND_ID = "baseline/blind@2024-06-30"
SIZE_COLUMN = {"headline_date": "s_size", "price_recall": "s_size", "price_level": "s_size", "annual_return": "s_size", "price_choice": "s_size",
               "price_choice_prob": "s_size"}  # news: s_company_size
SIZE_LABELS = {"all": "all sizes", "large": "large caps", "mid": "mid caps", "small": "small caps",
               "none": "no listed company"}
# What each Profile row is about: the market and the population. The listings are Nordic (SE, FI, DK, NO), not only
# Swedish; the US methods are the private check; the index and headline methods have no company size at all.
US_METHODS = {"us_level", "us_index_level", "us_index_return"}
POPULATION = {"index_level": "indices", "index_return": "indices", "index_forecast": "indices", "path_date": "indices",
              "us_index_level": "indices",
              "us_index_return": "indices", "us_level": "large companies", "known_level": "15 largest companies",
              "known_return": "15 largest companies", "known_choice": "15 largest companies",
              "known_choice_prob": "15 largest companies", "headline_date": "news headlines"}
# methods whose questions have no Nordic company size (M18 headlines do: the company each one is about)
NO_SIZES = set(POPULATION) - {"headline_date"}


def row_label(method: str, stratum: str) -> str:
    """A Profile row's name, e.g. 'Nordic · large caps', 'US · indices', 'Nordic · 15 largest companies'."""
    region = "US" if method in US_METHODS else "Nordic"
    what = SIZE_LABELS.get(stratum, stratum) if stratum != "all" else POPULATION.get(method, SIZE_LABELS["all"])
    return f"{region} · {what}"


def splits_by_size(method: str) -> bool:
    """Whether the method's questions span company sizes: not the index, headline and US methods (no Nordic company
    size), nor the controls on the 15 largest companies (all large)."""
    return method not in NO_SIZES


def release_date(model_id: str, meta: dict):
    """When the model became available: a release override, else OpenRouter's listing date; None for local
    controls."""
    if is_local(model_id):
        return None
    return resolve_release(model_id, meta)


# Verdicts that count as a detection on the cards: the generic one, and the model - blind test (M14, M16, M18)
DETECTION_VERDICTS = {"memorization detected", "beats the blind baseline before the cutoff"}
# The Profile's baseline before anyone picks: the release line, the primary line since pilot E
PROFILE_DEFAULT_BASELINE = "release"
# Models that fail the positive controls (pilots D and E: gpt-oss-20b M7 57%, M9 6%): their blanks mean "too hard for
# this model", not "no memory"
CONTROL_FAILS = {"openai/gpt-oss-20b": "fails the controls: blanks = too hard"}
# Rolling charts pool every answered version only where the versions ask the same question the same way (the M5
# grids); elsewhere versions differ in design (M8 v1 has reverse splits, M3 versions use different rewriters)
ROLLING_POOLED = {"price_level"}


def is_detection(verdict_text: str) -> bool:
    return verdict_text in DETECTION_VERDICTS


def rolling_pools_versions(method: str) -> bool:
    return method in ROLLING_POOLED


def placebo_models(models: list[str], cutoffs: dict) -> list[str]:
    """The models that had seen every placebo item (all dated before 2023-10-01): BEFORE edge on or after it."""
    from .pilot import PLACEBO_UPPER
    return [m for m in models if cutoffs.get(m) and cutoffs[m] - timedelta(days=BUFFER_DAYS) >= PLACEBO_UPPER]


def load_cutoff_sources() -> dict:
    """config/model_cutoff_sources.yaml: listing (OpenRouter's cutoff, frozen) or estimate (ours), per manual cutoff."""
    return load_overrides(SOURCES)


def cutoff_source(model: str, overrides: dict, meta: dict | None = None, sources: dict | None = None) -> str:
    """'set by hand' for a cutoff we estimated; 'OpenRouter listing' for one OpenRouter states, also when frozen in
    config/model_cutoffs.yaml. The recorded source (config/model_cutoff_sources.yaml) wins, so the label does not
    change if OpenRouter later drops a listing; otherwise a frozen cutoff equal to the live listing counts as one."""
    over = overrides.get(model)
    if not over:
        return "OpenRouter listing"
    recorded = (load_cutoff_sources() if sources is None else sources).get(model)
    if recorded:
        return {"listing": "OpenRouter listing", "model card": "model card"}.get(recorded, "set by hand")
    listed = ((meta or {}).get(model) or {}).get("knowledge_cutoff") if meta is not None else None
    return "OpenRouter listing" if listed and str(listed)[:10] == str(over)[:10] else "set by hand"


def provisional_models(models: list[str], overrides: dict, meta: dict) -> list[str]:
    """Models whose cutoff is our own estimate (Qwen3.8 Flash): left out of tests that rest on the cutoff date."""
    return [m for m in models if cutoff_source(m, overrides, meta) == "set by hand"]


def assign_groups(df: pd.DataFrame, method: str, cut, rel, line: str) -> pd.DataFrame:
    """The registered before/after groups of one model x method frame: the grid plan's 12-month windows for M5, the
    index and dating plans' windows for M14-M18, and the micro-pilot blocks (Nordic index, M18) left out."""
    if not len(df):
        return df
    if method in analysis.GRID_REGISTERED and cut and (rel or line == "cutoff"):
        df = df.assign(group=analysis.grid_groups(df, *analysis.registered_window(cut, rel, line)).to_numpy())
    if method in analysis.REGISTERED and cut and (rel or line == "cutoff"):
        df = df.assign(group=analysis.registered_groups(df, *analysis.registered_window(cut, rel, line)).to_numpy())
    if method in analysis.REGISTERED:
        pilot = df["item_id"].map(lambda i: index_is_pilot(i) or headline_is_pilot(i))
        df = df.assign(group=df["group"].where(~pilot, -1))
    return df


def stat_name(row) -> str:
    """What a cell's value is: the registered statistic's name, or Δκ for the generic before/after test."""
    reg = row.get("registered") if hasattr(row, "get") else None
    if reg == "model − blind, before":
        return "model − blind baseline (before the cutoff)"
    if reg == "correlation drop":
        return "correlation drop (before − after)"
    return "Δκ"


def forest_axis_title(df: pd.DataFrame) -> str:
    regs = {r for r in df["registered"] if isinstance(r, str) and r} if "registered" in df else set()
    if regs == {"model − blind, before"}:
        return "model − blind baseline, before the cutoff  (0 = same as the baseline)"
    if regs == {"correlation drop"}:
        return "correlation before − after  (0 = no drop)"
    return "Δκ  (0 = no memorization signal)"


def placebo_note(models: list[str], cut: dict) -> str:
    """The placebo eyebrow: which shown models are left out because their data does not cover every placebo item."""
    out = [display_name(m) for m in models if m not in placebo_models(models, cut)]
    return ("Placebo cutoff · 2022-07-01, among items older than 2023-10-01 · "
            + (f"left out (their data ends before some placebo items): {', '.join(out)}" if out
               else "every model shown had seen all of them"))


def compare_defaults(answered: dict, order: list[str], methods: list[str]) -> tuple[str, str, str]:
    """Compare's first view: the first two models in cutoff order and the first method both answered."""
    a, b = order[0], order[1]
    for m in methods:
        if {a, b} <= answered.get(m, set()):
            return a, b, m
    return a, b, methods[0]


def side_label(group: int, line: str) -> str:
    """Which side of the test an item counts on, in the words of the chosen line and the registered windows."""
    if group == 1:
        return "before (in the test window)"
    if group == 0:
        return "after release" if line == "release" else "after cutoff"
    return "outside the test windows"


def registered_label(cells: pd.DataFrame) -> str:
    """The name of the registered test behind a set of cells; generic text when no cell carries one (no answers)."""
    vals = [v for v in cells["registered"] if isinstance(v, str) and v] if "registered" in cells else []
    return vals[0] if vals else "its registered test"


def generic_methods(methods) -> list[str]:
    """The methods read by the generic before/after score: the registered ones (M14-M18) have their own tests and
    stay out of the views that pool or split that score (staggered table, breakdown)."""
    return [m for m in methods if m not in analysis.REGISTERED]


def compare_sentence(name_a: str, name_b: str, mtitle: str, res: dict, answered: dict) -> tuple[str, str]:
    """A neutral reading of a paired comparison (whose before/after gap is larger, not who 'remembers more'), with
    the refusal guard: a difference that does not hold on the questions both answered is refusal-driven."""
    if pd.isna(res.get("ci_lo")):
        return f"Not enough shared items to compare {_esc(name_a)} and {_esc(name_b)} on {mtitle}.", "muted"
    if res["ci_lo"] <= 0 <= res["ci_hi"]:
        return (f"<b>No significant difference</b> between the before/after gaps of {_esc(name_a)} and "
                f"{_esc(name_b)} on {mtitle}.", "muted")
    who, other = (name_a, name_b) if res["diff"] > 0 else (name_b, name_a)
    text = (f"<b>The before/after gap is larger for {_esc(who)}</b> than for {_esc(other)} on {mtitle}: difference "
            f"{res['diff']:+.2f} [{res['ci_lo']:+.2f}, {res['ci_hi']:+.2f}].")
    lo, hi = answered.get("ci_lo"), answered.get("ci_hi")
    if pd.notna(lo) and (hi < 0 < res["diff"] or res["diff"] < 0 < lo):
        return text + " On the questions both answered it reverses: not confirmed.", "warn"
    if pd.isna(lo) or lo <= 0 <= hi:
        return text + " It does not hold on the questions both answered: refusal-driven, not confirmed.", "warn"
    return text, "bad"


def default_models(models: list[str], cutoffs: dict, k: int = 8) -> list[str]:
    """The models a page shows before anyone picks: the k earliest cutoffs (the same on Overview and Evidence)."""
    return sorted(models, key=lambda m: (str(cutoffs.get(m) or "9999"), m))[:k]


# Models run only for the cutoff-ladder extension (index questions, 2026-10-06): off by default on the Profile and the
# Market page, one click away
AUX_MODELS = {"deepseek/deepseek-v3.2-exp", "qwen/qwen3-235b-a22b-2507", "qwen/qwen3-next-80b-a3b-instruct"}


def default_visible(models: list[str]) -> list[str]:
    """The models shown before anyone picks: all but the auxiliary ones (unless nothing else is left)."""
    return [m for m in models if m not in AUX_MODELS] or list(models)


def profile_selection(models: list[str], hidden: set) -> list[str]:
    """The Profile's models: every model that answered the method, minus the ones the reader switched off."""
    return [m for m in models if m not in hidden]


def update_hidden(hidden: set, shown: list[str], selected: list[str]) -> set:
    """The models switched off after a change on one method: only the models shown there change; models switched off
    on another method stay off."""
    return (set(hidden) - set(shown)) | (set(shown) - set(selected))


def by_cutoff(models, meta: dict) -> list[str]:
    """A fixed order for every view: earliest cutoff first, controls last. Never by score."""
    cut = cutoffs(models, meta)
    return sorted(models, key=lambda m: (is_local(m), str(cut.get(m) or "9999"), display_name(m)))


DENSE_PERIODS = 8      # more year columns than this: the compact Profile grid
PROFILE_SCREEN = 2000   # draws per cell first; all 20,000 only where the reading can still change (pilot.SCREEN_P)
_WARMED: set = set()
_PAGE_BUSY = __import__("threading").Event()   # set while a page computes a profile: the warm-up waits


def warm_profiles(jobs: list[tuple[str, tuple]], setting: tuple, fp: tuple) -> bool:
    """Fill the per-model Profile cache for other methods in a background thread (once per data state), so that
    switching method on the Profile is instant once it has run. jobs: (method, models) in the order to compute.
    Returns whether a warm-up started."""
    import logging
    import threading
    key = (setting, fp)
    if key in _WARMED or not jobs:
        return False
    _WARMED.add(key)
    # cached functions called outside a page run log a harmless "missing ScriptRunContext" warning each time
    logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

    def run():
        import time as _time
        for method, models in jobs:
            for m in models:
                while _PAGE_BUSY.is_set():      # what the reader asked for goes first
                    _time.sleep(0.1)
                try:
                    profile_model(m, method, setting, fp, False)
                except Exception:       # a warm-up must never disturb the page; the page computes it itself
                    pass

    threading.Thread(target=run, daemon=True, name="profile-warmup").start()
    return True


def profile(models: tuple, method: str, setting: tuple, fp: tuple, by_size: bool) -> list[dict]:
    """The Profile's cells for the selected models, labelled with Holm across all of them. Each model's cells are
    computed and cached on their own, so switching a model on or off only computes that model."""
    from . import pilot
    cells: list[dict] = []
    _PAGE_BUSY.set()
    try:
        for m in models:
            cells += profile_model(m, method, setting, fp, by_size)
    finally:
        _PAGE_BUSY.clear()
    return pilot.label_profile(cells)


@st.cache_data(show_spinner="Computing a model's profile…", max_entries=400)
def profile_model(model: str, method: str, setting: tuple, fp: tuple, by_size: bool) -> list[dict]:
    """One model's unlabelled Profile cells (cached per model, so the selection does not matter)."""
    from . import pilot
    meta = meta_by_id()
    paired, corr = method in PAIRED_BLIND, method in CORR_METHODS
    df = scores(model, method, setting, fp)
    cut = resolve_cutoff(model, meta)
    if df.empty or cut is None:
        return []
    if paired and PAIRED_BLIND[method] is not None:
        _, items, _ = _cell_index(setting, store.db_stamp(conn()), testsets_fingerprint())
        blind = {i: PAIRED_BLIND[method](it) for i, it in items[method][1].items()} if method in items else None
    else:
        blind_df = scores(BLIND_ID, method, setting, fp)
        blind = dict(zip(blind_df["item_id"], blind_df["score"])) if len(blind_df) else None
    if method in analysis.REGISTERED:            # the M18 and index pilot blocks stay out, as in the registered tests
        df = df[~df["item_id"].map(lambda i: index_is_pilot(i) or headline_is_pilot(i))]
    col = SIZE_COLUMN.get(method, "s_company_size")
    parts = [("all", df)] if not by_size or col not in df or not splits_by_size(method) else \
        [(str(v), g) for v, g in df.groupby(col) if str(v) in ("large", "mid", "small", "none")]
    cells: list[dict] = []
    for stratum, part in parts:
        for c in pilot.profile_cells(part, method, model, cut, release_date(model, meta), blind, B=20000,
                                     paired=paired, drop_refused=True, corr=corr, screen=PROFILE_SCREEN):
            cells.append({**c, "stratum": stratum, "region": "US" if method in US_METHODS else "Nordic"})
    return cells


@st.cache_data(show_spinner="Running the placebo check…")
def placebo_table(models: tuple, methods: tuple, setting: tuple, fp: tuple) -> pd.DataFrame:
    from . import pilot
    seen = placebo_models(list(models), cutoffs(list(models), meta_by_id()))
    rows = [{"model": m, "method": me, **pilot.placebo(scores(m, me, setting, fp), me, B=2000)}
            for m in seen for me in methods if len(scores(m, me, setting, fp))]
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = stats.holm(out["p_value"].to_numpy(float))
    return out


@st.cache_data(show_spinner="Comparing models with different cutoffs…")
def staggered_table(models: tuple, methods: tuple, setting: tuple, fp: tuple, line: str = "release") -> pd.DataFrame:
    """Every pair of selected models with different cutoffs: the later one's advantage on items between the earlier
    model's release (cutoff line: its cutoff + 91 days) and the later model's cutoff - 91 days, minus its advantage on
    items before both (pooled over the selected methods). Also on the questions both answered (the refusal guard).
    Models whose cutoff we set by hand are left out."""
    from . import pilot
    meta = meta_by_id()
    cut = cutoffs(models, meta)
    shaky = set(provisional_models(list(models), load_overrides(), meta))
    methods = tuple(generic_methods(methods))
    ordered = [m for m in by_cutoff(models, meta) if cut.get(m) and not is_local(m) and m not in shaky]
    rows = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if cut[a] >= cut[b]:
                continue
            frames = {me: (scores(a, me, setting, fp), scores(b, me, setting, fp)) for me in methods
                      if len(scores(a, me, setting, fp)) and len(scores(b, me, setting, fp))}
            if not frames:
                continue
            rel = release_date(a, meta)
            start = pd.Period(rel, "M").end_time.date() if line == "release" and rel else None
            r = pilot.staggered(frames, cut[a], cut[b], B=2000, window_start=start)
            ans = pilot.staggered(frames, cut[a], cut[b], B=2000, window_start=start, answered_only=True)
            if r["pooled"] and r["pooled"]["n_window"]:
                rows.append({"earlier": a, "later": b, **r["pooled"],
                             "answered_did": (ans["pooled"] or {}).get("did", float("nan")),
                             "answered_p": (ans["pooled"] or {}).get("p_value", float("nan"))})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = stats.holm(out["p_value"].to_numpy(float))
        out["answered_p_holm"] = stats.holm(out["answered_p"].to_numpy(float))
        sig = (out["p_holm"] < 0.05) & (out["did"] > 0)
        held = (out["answered_p_holm"] < 0.05) & (out["answered_did"] > 0)
        out["reading"] = np.where(sig & held, "only the later model knows these months",
                                  np.where(sig, "refusal-driven, not confirmed", "not shown"))
    return out


# ---------------------------------------------------------------- model browser (Run page)
def model_table(meta: dict, overrides: dict | None = None) -> pd.DataFrame:
    """One row per model for the browser: name, maker, cutoff (a manual one marked), release date, prices, reasoning,
    context. Pure: no Streamlit calls."""
    rows = []
    for model_id, m in meta.items():
        manual = (overrides or {}).get(model_id) if cutoff_source(model_id, overrides or {}, meta) == "set by hand" \
            else None
        cutoff = f"{str(manual)[:10]} (manual)" if manual else (m.get("knowledge_cutoff") or "")[:10]
        pricing = m.get("pricing") or {}
        r = m.get("reasoning") or {}
        rows.append({"id": model_id, "name": display_name(model_id, meta), "maker": vendor(model_id, meta),
                     "cutoff": cutoff,
                     "released": str(resolve_release(model_id, meta) or ""),
                     "input $/M": float(pricing.get("prompt") or 0) * 1e6,
                     "output $/M": float(pricing.get("completion") or 0) * 1e6,
                     "reasoning": "must" if r.get("mandatory") else "optional" if r else "–",
                     "context": int(m.get("context_length") or 0),
                     "self-hosted": bool(m.get("self_hosted"))})
    cols = ["id", "name", "maker", "cutoff", "released", "input $/M", "output $/M", "reasoning", "context", "self-hosted"]
    return pd.DataFrame(rows, columns=cols)[["name", "maker", "cutoff", "released", "input $/M", "output $/M",
                                              "reasoning", "context", "self-hosted", "id"]]


def apply_browser_selection(current: list[str], shown: list[str], checked: list[str]) -> list[str]:
    """The new model selection: rows shown in the browser follow their tick boxes; models filtered out of view stay
    as they were. The existing order is kept and new models are appended."""
    shown_set, checked_set = set(shown), set(checked)
    kept = [m for m in current if m not in shown_set or m in checked_set]
    return kept + [m for m in checked if m not in kept]


def filter_model_table(table: pd.DataFrame, search: str = "", makers: list[str] | None = None,
                       cutoff_years: tuple[int, int] | None = None, known_cutoff_only: bool = False,
                       max_output: float | None = None) -> pd.DataFrame:
    """The browser's filters, all combined: text in the name or id, makers, a cutoff year range (a model without a
    known cutoff is outside every range), and a highest output price per M tokens. None / empty = no filter."""
    keep = pd.Series(True, index=table.index)
    if search.strip():
        s = search.strip().lower()
        keep &= table["name"].str.lower().str.contains(s, regex=False) | table["id"].str.lower().str.contains(s, regex=False)
    if makers:
        keep &= table["maker"].isin(makers)
    year = pd.to_numeric(table["cutoff"].str[:4], errors="coerce")
    if known_cutoff_only:
        keep &= year.notna()
    if cutoff_years:
        keep &= year.between(*cutoff_years)
    if max_output is not None:
        keep &= table["output $/M"] <= max_output
    return table[keep]


@st.cache_data(ttl=3600, show_spinner=False)
def browser_table(show_all: bool, overrides: tuple = ()) -> pd.DataFrame:
    """model_table for the models the picker offers. overrides = sorted (model, cutoff) pairs, part of the cache key."""
    meta = meta_by_id()
    return model_table({m: meta[m] for m in picker_models(show_all) if m in meta}, dict(overrides))


def cutoff_overrides() -> tuple:
    return tuple(sorted((k, str(v)) for k, v in load_overrides().items()))


# ---------------------------------------------------------------- choices that survive visiting another page
def remember_widgets(state, keys) -> None:
    """Streamlit deletes a widget's value when a page without that widget is shown. Keep a copy under a key no
    widget uses (call after the widgets are drawn)."""
    for k in keys:
        if k in state:
            state[f"_kept_{k}"] = state[k]


def restore_widgets(state, keys) -> None:
    """Put back what remember_widgets kept, for values Streamlit dropped (call before the widgets are drawn)."""
    for k in keys:
        if k not in state and f"_kept_{k}" in state:
            state[k] = state[f"_kept_{k}"]


# every page's own choices (widget keys by page prefix; the setting pickers use <page>_mode / <page>_combo)
KEEP_PREFIXES = ("run_", "prof_", "ev_", "cmp_", "profile_", "results_", "compare_", "mkt_", "market_")
KEEP_NEVER = ("run_approve_",)          # a spend approval must be given again, never carried back


def remember_page_choices(state) -> None:
    """Call on a page right after its widgets are drawn, and inside a fragment after the fragment's widgets:
    Streamlit drops a page's widget values as soon as another page is opened, so the copy must already exist."""
    for k in list(state.keys()):
        if isinstance(k, str) and k.startswith(KEEP_PREFIXES) and not k.startswith(KEEP_NEVER):
            state[f"_kept_{k}"] = state[k]


def keep_choice(state, key: str, options, default, multi: bool = False) -> None:
    """Call right before drawing widget `key`, and draw it without default= / index=: it shows the choice made
    last time if that is still offered, else `default`. A deliberately emptied multi-choice stays empty."""
    restore_widgets(state, [key])
    if key in state:
        value = state[key]
        if multi:
            state[key] = [v for v in (value or []) if v in options]
            return
        if value in options:
            return
    state[key] = default


# ---------------------------------------------------------------- method pickers
SHORT_NAMES = {"known_return": "Known returns", "known_level": "Known levels", "us_level": "US levels",
               "us_index_level": "US index level", "us_index_return": "US index change",
               "price_choice": "Which price fits", "known_choice": "Known price choice",
               "price_choice_prob": "Which price fits", "known_choice_prob": "Known price choice",
               "headline_date": "Headline date", "index_forecast": "Index forecast",
               "path_date": "Date a path"}


# One colour per method role wherever a method is named (pickers, card titles, chart headers); main methods keep the
# normal ink. Light tints, kept apart from the reading colours (red, amber) and the vivid model colours.
# .streamlit/config.toml sets the same hexes as Streamlit's green/blue/orange/gray text, so markdown labels match.
# (markdown name, hex, emoji for pickers that cannot colour text: selectbox and multiselect)
ROLE_COLORS = {"control": ("green", "#4cc38a", "🟢"), "probe": ("blue", "#6aa8ff", "🔵"),
               "private": ("orange", "#f0a35e", "🟠"), "secondary": ("gray", "#a3a19a", "⚪")}


def method_role(method: str) -> str:
    """'control', 'private', 'secondary', 'probe', or '' for a main method."""
    return ("control" if method in POSITIVE_CONTROLS else "private" if method in PRIVATE_CHECKS
            else "secondary" if method in SECONDARY else "probe" if method in PROBES else "")


def role_hex(method: str) -> str | None:
    role = method_role(method)
    return ROLE_COLORS[role][1] if role else None


def method_short(method: str) -> str:
    """A picker label short enough for a row of buttons: number, short name and its role, if it has one."""
    number, name = LABELS[method].split(" (")[0].split(" ", 1)
    role = method_role(method)
    return f"{number} {SHORT_NAMES.get(method, name)}" + (f" · {role}" if role else "")


def method_md(method: str) -> str:
    """method_short in its role colour, for pickers that render markdown (st.pills, st.segmented_control)."""
    role = method_role(method)
    return f":{ROLE_COLORS[role][0]}[{method_short(method)}]" if role else method_short(method)


def method_option(method: str) -> str:
    """method_short behind a coloured dot, for pickers that show plain text only (st.selectbox, st.multiselect)."""
    role = method_role(method)
    return f"{ROLE_COLORS[role][2]} {method_short(method)}" if role else method_short(method)


def method_legend(methods) -> None:
    """A one-line key to the role colours, listing only the roles among `methods`."""
    roles = [r for r in ROLE_COLORS if any(method_role(m) == r for m in methods)]
    if roles:
        st.caption("Method colours: " + " · ".join(f":{ROLE_COLORS[r][0]}[{r}]" for r in roles)
                   + " · main methods in white.")


def method_eyebrow(method: str, suffix: str = "") -> None:
    """A card's small capitals title: the method's full label in its role colour."""
    eyebrow(LABELS[method] + suffix, color=role_hex(method))


def methods_in_order(methods) -> list[str]:
    """Main methods first (in their M-number order), then positive controls, secondary and private ones."""
    rank = {m: i for i, m in enumerate(METHODS)}
    role = lambda m: (1 if m in POSITIVE_CONTROLS else 2 if m in SECONDARY else 3 if m in PRIVATE_CHECKS
                      else 4 if m in PROBES else 0)
    return sorted(methods, key=lambda m: (role(m), rank.get(m, 99)))


def method_lines(method: str, width: int = 11) -> list[str]:
    """A method name broken into short lines for narrow column headers: the number, the name wrapped at `width`
    characters, and its role (control / private / secondary) on its own line."""
    label = method_short(method)
    role = f" · {method_role(method)}" if method_role(method) else ""
    number, name = label[: len(label) - len(role)].split(" ", 1)
    lines, line = [], ""
    for word in name.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return [number, *lines, line] + ([role.strip()] if role else [])


# ---------------------------------------------------------------- model timeline (cutoff, grey zone, release)
TIMELINE_PHASES = ("may have seen (before cutoff)", "grey zone (cutoff → release)", "cannot know (after release)")
TIMELINE_FILLS = {"light": ["#a3a098", "#d6d3c9", "#efede7"], "dark": ["#7a7973", "#55544f", "#383835"]}
DATA_START, DATA_END = date(2016, 1, 1), date(2026, 8, 10)


def timeline_rows(cutoffs: dict, releases: dict, start=DATA_START, end=DATA_END) -> pd.DataFrame:
    """Three phases per model with a known cutoff and release, clipped to our data window (start..end)."""
    rows = []
    for m, cut in cutoffs.items():
        rel = releases.get(m)
        if cut is None or rel is None:
            continue
        bounds = [(TIMELINE_PHASES[0], start, cut), (TIMELINE_PHASES[1], cut, rel), (TIMELINE_PHASES[2], rel, end)]
        for phase, a, b in bounds:
            a, b = max(a, start), min(b, end)
            if a < b:
                rows.append({"model": m, "phase": phase, "start": a.isoformat(), "end": b.isoformat()})
    return pd.DataFrame(rows, columns=["model", "phase", "start", "end"])


def timeline_chart(models: list[str], meta: dict, cutoff_map: dict | None = None, end=DATA_END):
    """One row per model on a shared date axis: what it may have seen (before its cutoff), the grey zone (cutoff to
    release) and what it cannot know (after release), with our data window marked. Controls have no release date and
    are left out. Rows in cutoff order, never by score."""
    p = palette()
    cut = cutoff_map if cutoff_map is not None else cutoffs(models, meta)
    real = [m for m in models if release_date(m, meta) is not None and cut.get(m) is not None]
    real = sorted(real, key=lambda m: (cut[m], display_name(m, meta)))
    releases = {m: release_date(m, meta) for m in real}
    rows = timeline_rows({m: cut[m] for m in real}, releases, end=end)
    rows["name"] = rows["model"].map(lambda m: display_name(m, meta))
    names = [display_name(m, meta) for m in real]
    last = max([end, *releases.values()])
    x_scale = alt.Scale(domain=[DATA_START.isoformat(), date(last.year + 1, 1, 1).isoformat()])
    y = alt.Y("name:N", title=None, sort=names, axis=alt.Axis(domain=False, ticks=False, labelFontWeight=500))
    fills = TIMELINE_FILLS["dark" if p is PALETTES["dark"] else "light"]
    bars = alt.Chart(rows).mark_bar(height=18, cornerRadius=3, stroke=p["muted"], strokeWidth=0.8).encode(
        x=alt.X("start:T", title=None, scale=x_scale, axis=alt.Axis(format="%Y", tickCount="year")), x2="end:T", y=y,
        color=alt.Color("phase:N", title=None, scale=alt.Scale(domain=list(TIMELINE_PHASES), range=fills),
                        legend=alt.Legend(orient="bottom", direction="horizontal", symbolType="square", labelLimit=0,
                                          symbolStrokeColor=p["muted"], symbolStrokeWidth=0.5)),
        tooltip=[alt.Tooltip("name:N", title="model"), alt.Tooltip("phase:N", title="period"),
                 alt.Tooltip("start:T", title="from", format="%Y-%m-%d"),
                 alt.Tooltip("end:T", title="to", format="%Y-%m-%d")])
    marks = pd.DataFrame([{"name": display_name(m, meta), "date": d.isoformat(), "what": what,
                           "shape": shape, "model": m}
                          for m in real for what, d, shape in (("cutoff", cut[m], "triangle-down"),
                                                               ("release", releases[m], "circle"))])
    color = alt.Color("name:N", scale=model_scale(real), legend=None)
    points = alt.Chart(marks).mark_point(size=110, filled=True, opacity=1, stroke=p["surface"], strokeWidth=1).encode(
        x=alt.X("date:T", scale=x_scale), y=y, color=color,
        shape=alt.Shape("shape:N", scale=None),
        tooltip=[alt.Tooltip("name:N", title="model"), alt.Tooltip("what:N", title="marker"),
                 alt.Tooltip("date:T", title="date", format="%Y-%m-%d")])
    edge = pd.DataFrame({"date": [end.isoformat()], "label": [f"our data ends {end:%Y-%m}"]})
    rule = alt.Chart(edge).mark_rule(strokeDash=[4, 3], color=p["muted"]).encode(x=alt.X("date:T", scale=x_scale))
    label = alt.Chart(edge).mark_text(align="left", dx=4, dy=-6, fontSize=11, color=p["muted"]).encode(
        x=alt.X("date:T", scale=x_scale), y=alt.value(0), text="label:N")
    return (bars + points + rule + label).resolve_scale(color="independent").properties(height=34 * len(real) + 50)


# ---------------------------------------------------------------- DE-COP blind-baseline audit
@st.cache_data(show_spinner="Auditing the DE-COP sets…")
def decop_audit_table(fingerprint: tuple) -> pd.DataFrame:
    """Rules x DE-COP versions: how often each no-model rule picks the original, and whether the set passes."""
    from . import decop_audit
    from .items import TESTSETS
    from .methods import decop
    folder = TESTSETS / decop.NAME
    versions = sorted((p.name[: -len(".jsonl")] for p in folder.glob("*.jsonl") if "mock" not in p.name),
                      key=lambda v: int(v[1:]) if v[1:].isdigit() else 0) if folder.exists() else []
    cols = {}
    for v in versions:
        rows = decop_audit.audit(load_testset(decop.NAME, v))
        verdict = "passed" if decop_audit.passed(rows) else "FAILED"
        cols[f"{v} · {verdict}"] = [("–" if r["rate"] != r["rate"] else f"{r['rate']:.0%}") + (" ✓" if r["passed"]
                                    else " ✗") + f" ({r['decided']})" for r in rows]
    return pd.DataFrame({"no-model rule": list(decop_audit.RULES), **cols})


# ---------------------------------------------------------------- rolling window over time
def with_segments(curve: pd.DataFrame) -> pd.DataFrame:
    """Number the runs of consecutive months, so a line or band is never drawn across months without a window."""
    if curve.empty:
        return curve.assign(segment=[])
    months = pd.PeriodIndex(curve["month"], freq="M")
    step = pd.Series(months.asi8).diff().fillna(1).to_numpy()
    return curve.assign(segment=(step > 1).cumsum())


def gap_spans(curve: pd.DataFrame) -> pd.DataFrame:
    """The runs of months missing inside a curve (windows left out for too few answers), as date spans
    [start, end) for shading. Months before the first or after the last point are not gaps."""
    if curve.empty:
        return pd.DataFrame(columns=["start", "end"])
    have = sorted(pd.PeriodIndex(curve["month"], freq="M").unique())
    rows = [{"start": (a + 1).to_timestamp().strftime("%Y-%m-%d"), "end": b.to_timestamp().strftime("%Y-%m-%d")}
            for a, b in zip(have, have[1:]) if (b - a).n > 1]
    return pd.DataFrame(rows, columns=["start", "end"])


def rolling_chart(frames: pd.DataFrame, models: list[str], cut_rows: list[dict], rel_rows: list[dict],
                  value_label: str = "mean score"):
    """One small panel per model (in the given order): its rolling-window mean with a 95% band, the blind guesser as a
    grey dashed reference (how hard the questions of each period are without memory), a dashed cutoff and a solid
    release rule. All panels share the time axis and the score scale, so they can be compared at a glance."""
    p = palette()
    d = frames.copy()
    d["date"] = pd.PeriodIndex(d["month"], freq="M").to_timestamp().strftime("%Y-%m-%d")
    present = [m for m in models if m in set(d["model"])]
    top = float(min(1.0, max(0.05, d["hi"].max() * 1.05))) if len(d) else 1.0
    lo_y = float(min(0.0, d["lo"].min())) if len(d) else 0.0
    dates = pd.to_datetime(d["date"]) if len(d) else pd.Series(dtype="datetime64[ns]")
    x_scale = alt.Scale(domain=[dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")]) if len(d) else \
        alt.Scale()
    x = alt.X("date:T", title=None, scale=x_scale, axis=alt.Axis(format="%Y", tickCount="year"))
    y = alt.Y("mean:Q", title=None, scale=alt.Scale(domain=[lo_y, top]))
    blind = with_segments(d[d["model"] == BLIND_ID].sort_values("month"))
    colors = {m: model_color(m) for m in present}
    panels = []
    for m in present:
        c = colors.get(m, p["ink"])
        own = with_segments(d[d["model"] == m].sort_values("month"))
        tip = [alt.Tooltip("month:N", title="window around"), alt.Tooltip("mean:Q", title=value_label, format=".3f"),
               alt.Tooltip("lo:Q", title="95% low", format=".3f"), alt.Tooltip("hi:Q", title="95% high", format=".3f"),
               alt.Tooltip("n:Q", title="questions in window")]
        gaps = gap_spans(own).assign(note="no line here: too few answers in these windows")
        layers = [alt.Chart(gaps).mark_rect(color=p["muted"], opacity=0.12).encode(
                      x=alt.X("start:T", scale=x_scale), x2="end:T", tooltip=[alt.Tooltip("note:N", title="gap")]),
                  alt.Chart(own).mark_area(opacity=0.18, color=c).encode(
                      x=x, y=alt.Y("lo:Q", title=None, scale=alt.Scale(domain=[lo_y, top])), y2="hi:Q",
                      detail="segment:N"),
                  alt.Chart(own).mark_line(strokeWidth=2, color=c).encode(x=x, y=y, detail="segment:N", tooltip=tip)]
        if len(blind):
            layers.append(alt.Chart(blind).mark_line(strokeWidth=1.3, strokeDash=[3, 3], color=p["muted"]).encode(
                x=x, y=y, detail="segment:N",
                tooltip=[alt.Tooltip("month:N", title="window around"),
                         alt.Tooltip("mean:Q", title="blind guesser", format=".3f")]))
        for rows, dash, what in ((cut_rows, [5, 4], "cutoff"), (rel_rows, None, "released")):
            mine = [r for r in rows if r["model"] == m]
            if mine:
                r = pd.DataFrame(mine)
                r["date"] = pd.PeriodIndex(r["month"], freq="M").to_timestamp().strftime("%Y-%m-%d")
                mark = alt.Chart(r).mark_rule(strokeWidth=1.8, color=c, **({"strokeDash": dash} if dash else {}))
                layers.append(mark.encode(x=x, tooltip=[alt.Tooltip("month:N", title=what)]))
        panels.append(alt.layer(*layers).properties(title=alt.TitleParams(display_name(m), anchor="start",
                                                                            fontSize=12, color=c),
                                                  width=400, height=150))
    return alt.concat(*panels, columns=2).resolve_scale(y="shared")


# Methods whose Profile cells compare with a blind baseline on the same items, as their registered tests do: M18 with the
# text-only guess (a model's own after-cutoff level is near 0 by construction for news dating), M14/M16 with the blind
# guesser's answers (None: read from its stored answers)
PAIRED_BLIND = {"headline_date": headline_blind_score, "index_level": None, "us_index_level": None}
PAIRED_LABEL = {"headline_date": "the text-only guess on the same headlines",
                "index_level": "the blind guesser (last year's level) on the same months",
                "us_index_level": "the blind guesser (last year's level) on the same months"}


def dating_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Answered M18 items: the true publication month and the answered month (first of the month, for a time axis),
    and whether the answer is within 3 months."""
    if df.empty or "answer" not in df:
        return pd.DataFrame(columns=["item_id", "true", "answered", "close", "err_months"])
    a = df[~df["refused"].astype(bool)].dropna(subset=["answer"]).copy()
    a["true"] = pd.PeriodIndex(pd.to_datetime(a["date"]), freq="M").to_timestamp().strftime("%Y-%m-%d")
    a["answered"] = pd.PeriodIndex(a["answer"], freq="M").to_timestamp().strftime("%Y-%m-%d")
    a["close"] = a["err_months"].abs().le(3).map({True: "within 3 months", False: "further off"})
    return a[["item_id", "true", "answered", "close", "err_months"]].reset_index(drop=True)


def dating_years(df: pd.DataFrame) -> pd.DataFrame:
    """Per publication year (main items only, the pilot block left out): how many headlines were asked and the share
    the model dated within 3 months of the truth (a refusal or an unreadable answer counts as a miss)."""
    if df.empty:
        return pd.DataFrame(columns=["year", "n", "share"])
    d = df[~df["item_id"].map(headline_is_pilot)]
    within = d["within_3"] if "within_3" in d else pd.Series(False, index=d.index)   # absent when all refused
    hit = within.fillna(False).astype(bool) & ~d["refused"].astype(bool)
    year = pd.to_datetime(d["date"]).dt.year
    g = pd.DataFrame({"year": year, "hit": hit}).groupby("year")["hit"].agg(["size", "mean"])
    return pd.DataFrame({"year": g.index.astype(int), "n": g["size"].to_numpy(), "share": g["mean"].to_numpy()})


def default_month(df: pd.DataFrame) -> str | None:
    """The month a model names most often: its fallback answer when it does not know."""
    a = df[~df["refused"].astype(bool)]["answer"].dropna() if len(df) and "answer" in df else pd.Series(dtype=str)
    return str(a.mode().iloc[0]) if len(a) else None


def dating_year_chart(frame: pd.DataFrame, models: list[str], defaults: dict, cut_rows: list[dict],
                      rel_rows: list[dict]):
    """One panel per model: the share of its headlines dated within 3 months, per publication year (its colour), next
    to the blind baseline's (grey dashed). The shaded band is the year around the model's default answer month, where
    a fallback answer lands within 3 months by luck; above the baseline outside that band is real dating. Dashed
    rule: cutoff; solid rule: release."""
    p = palette()
    d = frame.copy()
    d["date"] = pd.to_datetime(d["year"].astype(str) + "-07-01").dt.strftime("%Y-%m-%d")
    span = ["2016-01-01", "2026-12-31"]
    x = alt.X("date:T", title="year published", scale=alt.Scale(domain=span), axis=alt.Axis(format="%Y", tickCount="year"))
    top = float(min(1.0, max(0.2, d["share"].max() * 1.1))) if len(d) else 1.0
    y = alt.Y("share:Q", title="share dated within 3 months", scale=alt.Scale(domain=[0, top]),
              axis=alt.Axis(format="%"))
    blind = d[d["model"] == BLIND_ID]
    panels = []
    for m in [m for m in models if m in set(d["model"])]:
        c = model_color(m)
        own = d[d["model"] == m]
        layers = []
        if defaults.get(m):
            mid = pd.Period(defaults[m], "M")
            band = pd.DataFrame({"start": [(mid - 6).to_timestamp().strftime("%Y-%m-%d")],
                                 "end": [(mid + 6).to_timestamp().strftime("%Y-%m-%d")],
                                 "note": [f"default answer {mid}: the month this model names when it does not "
                                          "recognise a headline; correct answers here may be luck"]})
            layers.append(alt.Chart(band).mark_rect(color=p["muted"], opacity=0.16).encode(
                x=alt.X("start:T", scale=alt.Scale(domain=span)), x2="end:T",
                tooltip=[alt.Tooltip("note:N", title="shaded")]))
        if len(blind):
            layers.append(alt.Chart(blind).mark_line(strokeWidth=1.3, strokeDash=[3, 3], color=p["muted"]).encode(
                x=x, y=y, tooltip=[alt.Tooltip("year:Q", title="year"),
                                   alt.Tooltip("share:Q", title="text-only baseline", format=".0%")]))
        tip = [alt.Tooltip("year:Q", title="year"), alt.Tooltip("share:Q", title="within 3 months", format=".0%"),
               alt.Tooltip("n:Q", title="headlines")]
        layers += [alt.Chart(own).mark_line(strokeWidth=2, color=c).encode(x=x, y=y),
                   alt.Chart(own).mark_point(filled=True, size=40, color=c).encode(x=x, y=y, tooltip=tip)]
        for rows, dash, what in ((cut_rows, [5, 4], "cutoff"), (rel_rows, None, "released")):
            mine = [r for r in rows if r["model"] == m]
            if mine:
                r = pd.DataFrame(mine)
                r["date"] = pd.PeriodIndex(r["month"], freq="M").to_timestamp().strftime("%Y-%m-%d")
                mark = alt.Chart(r).mark_rule(strokeWidth=1.6, color=c, **({"strokeDash": dash} if dash else {}))
                layers.append(mark.encode(x=alt.X("date:T", scale=alt.Scale(domain=span)),
                                          tooltip=[alt.Tooltip("month:N", title=what)]))
        panels.append(alt.layer(*layers).properties(
            title=alt.TitleParams(display_name(m), anchor="start", fontSize=12, color=c), width=260, height=150))
    return alt.concat(*panels, columns=3).resolve_scale(y="shared")


def dating_chart(frame: pd.DataFrame, models: list[str], cut_rows: list[dict], rel_rows: list[dict]):
    """One panel per model: each answered headline as a dot at (true month, answered month). The diagonal is a
    perfect answer; a horizontal band is the model's default month; dots near the diagonal far from that band are
    real dating. Dashed rule: cutoff; solid rule: release. Dots within 3 months of the truth are drawn stronger."""
    p = palette()
    span = ["2016-01-01", "2026-08-01"]
    x = alt.X("true:T", title="true publication month", scale=alt.Scale(domain=span),
              axis=alt.Axis(format="%Y", tickCount="year"))
    y = alt.Y("answered:T", title="answered month",   # answers outside the range sit on its edge (clamped)
              scale=alt.Scale(domain=["2014-01-01", "2027-01-01"], clamp=True),
              axis=alt.Axis(format="%Y", tickCount="year"))
    diag = pd.DataFrame({"true": span, "answered": span})
    present = [m for m in models if m in set(frame["model"])]
    panels = []
    for m in present:
        c = model_color(m)
        own = frame[frame["model"] == m]
        layers = [alt.Chart(diag).mark_line(strokeWidth=1, color=p["muted"], strokeDash=[2, 2]).encode(x=x, y=y),
                  alt.Chart(own).mark_circle(size=14, color=c).encode(
                      x=x, y=y,
                      opacity=alt.Opacity("close:N", scale=alt.Scale(domain=["within 3 months", "further off"],
                                                                      range=[0.85, 0.22]), legend=None),
                      tooltip=[alt.Tooltip("true:T", title="published", format="%Y-%m"),
                               alt.Tooltip("answered:T", title="answered", format="%Y-%m"),
                               alt.Tooltip("err_months:Q", title="months off")])]
        for rows, dash, what in ((cut_rows, [5, 4], "cutoff"), (rel_rows, None, "released")):
            mine = [r for r in rows if r["model"] == m]
            if mine:
                r = pd.DataFrame(mine)
                r["true"] = pd.PeriodIndex(r["month"], freq="M").to_timestamp().strftime("%Y-%m-%d")
                mark = alt.Chart(r).mark_rule(strokeWidth=1.6, color=c, **({"strokeDash": dash} if dash else {}))
                layers.append(mark.encode(x=alt.X("true:T", scale=alt.Scale(domain=span)),
                                          tooltip=[alt.Tooltip("month:N", title=what)]))
        panels.append(alt.layer(*layers).properties(
            title=alt.TitleParams(display_name(m), anchor="start", fontSize=12, color=c), width=260, height=200))
    return alt.concat(*panels, columns=3)


@st.cache_data(show_spinner=False)
def scores_all_versions(model: str, method: str, setting: tuple, fp: tuple) -> pd.DataFrame:
    """Item scores for every version of a method this model answered (e.g. M5 v1 and v2 together: more items per
    month for the rolling view), with each item's chance level in p0."""
    calls = load_calls(setting)
    if calls.empty:
        return pd.DataFrame()
    sub = calls[(calls["model"] == model) & (calls["method"] == method)]
    frames = []
    for name in sorted(set(sub["testset"])):
        m, v = name.split("/")
        items = {it.item_id: it for it in load_testset(m, v)}
        df = app_scores(analysis.item_scores(sub[sub["testset"] == name], items, method,
                                             resolve_cutoff(model, meta_by_id()),
                                             extra_strata={"company_size": company_sizes()}), method,
                        drop_hedges=False)       # every answer, for the Items page; M5 (pooled) has no hedges
        if len(df):
            frames.append(df.assign(testset=name))
    return pd.concat(frames, ignore_index=True).assign(p0=METHODS[method].P0) if frames else pd.DataFrame()


def over_time_score(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """The score the over-time views plot: closeness for price levels (as the reports since pilot E: 1 = the exact
    price, 0 = off by a factor of 2 or more), otherwise the method's own score. Refused questions are left out
    (scoring spec §8): the views describe what a model answered."""
    if len(df) and "refused" in df:
        df = df[~df["refused"].astype(bool)]
    return with_score(df, method, "primary")


def rolling_from_scores(df: pd.DataFrame, method: str, window: int = 6, size: str = "all") -> pd.DataFrame:
    if size != "all" and len(df):
        df = df[size_of(df, method) == size]
    return analysis.rolling_curve(over_time_score(df, method), window=window, min_n=15, B=500)


def rolling_by_size(df: pd.DataFrame, method: str, window: int = 6) -> pd.DataFrame:
    """One rolling curve per company size (large, mid, small), stacked with a size column."""
    parts = [rolling_from_scores(df, method, window, size).assign(size=size) for size in SIZES]
    parts = [x for x in parts if len(x)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["month", "mean", "lo", "hi", "n",
                                                                                     "size"])


@st.cache_data(show_spinner="Computing rolling windows…")
def rolling(model: str, method: str, setting: tuple, fp: tuple, window: int = 6, size: str = "all") -> pd.DataFrame:
    """The rolling-window curve of one model over one method (every answered version for M5, the analysed version
    otherwise: see ROLLING_POOLED); size: 'all', one size, or
    'compare' (one curve per size, with a size column)."""
    df = scores_all_versions(model, method, setting, fp) if rolling_pools_versions(method) else \
        scores(model, method, setting, fp)
    if not len(df):
        return pd.DataFrame(columns=["month", "mean", "lo", "hi", "n"])
    return rolling_by_size(df, method, window) if size == "compare" else rolling_from_scores(df, method, window, size)


# company sizes are ordered: two hues with a neutral middle (validated: CVD, normal vision and contrast pass in both
# modes; the grey middle reads grey on purpose), plus a line style per size as a second cue
SIZE_SHADES = {"light": {"large": "#2a78d6", "mid": "#8a8880", "small": "#eb6834"},
               "dark": {"large": "#3987e5", "mid": "#8a8880", "small": "#d95926"}}
SIZE_DASH = {"large": [1, 0], "mid": [6, 3], "small": [2, 2]}


def size_shades() -> dict:
    return SIZE_SHADES["dark" if palette() is PALETTES["dark"] else "light"]


def rolling_size_chart(frames: pd.DataFrame, models: list[str], cut_rows: list[dict], rel_rows: list[dict]):
    """As rolling_chart, but each model's panel has one line per company size (darker = larger), no bands, so the
    three can be compared; the cutoff and release rules as before."""
    p = palette()
    d = frames.copy()
    d["date"] = pd.PeriodIndex(d["month"], freq="M").to_timestamp().strftime("%Y-%m-%d")
    present = [m for m in models if m in set(d["model"])]
    top = float(min(1.0, max(0.05, d["mean"].max() * 1.1))) if len(d) else 1.0
    dates = pd.to_datetime(d["date"]) if len(d) else pd.Series(dtype="datetime64[ns]")
    x_scale = alt.Scale(domain=[dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")]) if len(d) else \
        alt.Scale()
    x = alt.X("date:T", title=None, scale=x_scale, axis=alt.Axis(format="%Y", tickCount="year"))
    y = alt.Y("mean:Q", title=None, scale=alt.Scale(domain=[0, top]))
    color = alt.Color("size:N", title="Company size", sort=list(SIZES),
                      scale=alt.Scale(domain=list(SIZES), range=[size_shades()[s] for s in SIZES]),
                      legend=alt.Legend(orient="top", labelExpr="datum.label + ' caps'", symbolType="stroke",
                                        symbolStrokeWidth=2))
    size_dash = alt.StrokeDash("size:N", sort=list(SIZES), legend=None,
                               scale=alt.Scale(domain=list(SIZES), range=[SIZE_DASH[s] for s in SIZES]))
    panels = []
    for m in present:
        c = model_color(m)
        own = pd.concat([with_segments(g.sort_values("month")) for _, g in d[d["model"] == m].groupby("size")],
                        ignore_index=True)
        tip = [alt.Tooltip("size:N", title="size"), alt.Tooltip("month:N", title="window around"),
               alt.Tooltip("mean:Q", title="mean score", format=".3f"), alt.Tooltip("lo:Q", title="95% low", format=".3f"),
               alt.Tooltip("hi:Q", title="95% high", format=".3f"), alt.Tooltip("n:Q", title="questions in window")]
        layers = [alt.Chart(own).mark_line(strokeWidth=2).encode(x=x, y=y, color=color, strokeDash=size_dash,
                                                                 detail="segment:N", tooltip=tip)]
        for rows, dash, what in ((cut_rows, [5, 4], "cutoff"), (rel_rows, None, "released")):
            mine = [r for r in rows if r["model"] == m]
            if mine:
                r = pd.DataFrame(mine)
                r["date"] = pd.PeriodIndex(r["month"], freq="M").to_timestamp().strftime("%Y-%m-%d")
                mark = alt.Chart(r).mark_rule(strokeWidth=1.8, color=p["muted"], **({"strokeDash": dash} if dash else {}))
                layers.append(mark.encode(x=x, tooltip=[alt.Tooltip("month:N", title=what)]))
        panels.append(alt.layer(*layers).properties(title=alt.TitleParams(display_name(m), anchor="start",
                                                                            fontSize=12, color=c),
                                                  width=400, height=150))
    return alt.concat(*panels, columns=2).resolve_scale(y="shared")


# ---------------------------------------------------------------- the Market page (M14/M15)
@st.cache_data(show_spinner="Computing rolling correlations…")
def market_rolling(model: str, setting: tuple, fp: tuple, window: int = 12,
                   method: str = "index_return") -> pd.DataFrame:
    """The rolling correlation between a model's monthly-change answers and the true changes, all indices of the
    method's market pooled (index_return: Nordic, us_index_return: US)."""
    calls, ts = load_calls(setting), analysis_testsets(setting)
    if calls.empty or method not in ts:
        return pd.DataFrame(columns=["month", "r", "lo", "hi", "n"])
    name, items = ts[method]
    df = market.answers_frame(calls[calls["testset"] == name], items, model, method)
    return market.rolling_corr(df, window=window, min_n=20, B=500)


def market_comparison_row(us_before: pd.DataFrame, nordic_before: pd.DataFrame, us_move: pd.Series) -> dict:
    """One model's Nordic-vs-US numbers, by the report's own functions and seeds (index_report: the paired US - Nordic
    test, US plan §4.2, and E4 "Nordic beyond US", §6.1), so the page shows exactly the report's values."""
    from . import index_report as ir
    pc = ir.paired_corr_diff(us_before, nordic_before)
    pt = ir.partial_test(nordic_before, us_move)
    return {"r_us": pc["r_us"], "r_nordic": pc["r_nordic"], "diff": pc["diff"], "lo": pc["lo"], "hi": pc["hi"],
            "p": pc["p"], "n_months": pc["n_months"], "beyond": pt["r"], "beyond_lo": pt["lo"],
            "beyond_hi": pt["hi"], "beyond_p": pt["p"]}


@st.cache_data(show_spinner="Comparing the Nordic and US answers…")
def market_comparison(models: tuple, setting: tuple, fp: tuple) -> pd.DataFrame:
    """Per model, on its BEFORE months: r Nordic, r US, the paired difference and the Nordic correlation left after
    removing the month's US move; Holm across the models shown, as in the report (which uses all six)."""
    calls, ts = load_calls(setting), analysis_testsets(setting)
    if calls.empty or "index_return" not in ts or "us_index_return" not in ts:
        return pd.DataFrame()
    meta = meta_by_id()
    (n_name, n_items), (u_name, u_items) = ts["index_return"], ts["us_index_return"]
    n_items = [it for it in n_items if not index_is_pilot(it.item_id)]
    move = pd.DataFrame([(pd.Period(it.payload["month"], "M"), it.truth["return_pct"]) for it in u_items],
                        columns=["month", "r"]).groupby("month")["r"].mean()
    rows = []
    for m in models:
        cut, rel = resolve_cutoff(m, meta), release_date(m, meta)
        if not cut or not rel:
            continue
        last, _ = market.sides(cut, rel)
        nb = market.answers_frame(calls[calls["testset"] == n_name], n_items, m, "index_return")
        ub = market.answers_frame(calls[calls["testset"] == u_name], u_items, m, "us_index_return")
        nb, ub = nb[nb["month"] <= last], ub[ub["month"] <= last]
        if len(nb) < 20 or len(ub) < 20:
            continue
        rows.append({"model": m, **market_comparison_row(ub, nb, move)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = stats.holm(out["p"].to_numpy(float))
        out["beyond_p_holm"] = stats.holm(out["beyond_p"].to_numpy(float))
    return out


@st.cache_data(show_spinner=False)
def market_repeats(model: str, setting: tuple, fp: tuple, method: str = "index_return") -> pd.DataFrame:
    """The stock-answer share over time (market.repeat_share) for one model's monthly-change answers."""
    calls, ts = load_calls(setting), analysis_testsets(setting)
    if calls.empty or method not in ts:
        return pd.DataFrame(columns=["month", "share", "value", "distinct", "n"])
    name, items = ts[method]
    return market.repeat_share(market.answers_frame(calls[calls["testset"] == name], items, model, method))


def stock_answer_key() -> str:
    p = palette()
    return chart_key([("line", p["muted"], "share of the answers that are one repeated value (in the model's colour)"),
                      ("gap", p["muted"], "no line: too few answers in the window"),
                      ("rule-dashed", p["muted"], "stated cutoff"), ("rule-solid", p["muted"], "release")])


def stock_answer_chart(frames: pd.DataFrame, models: list[str], cut_rows: list[dict], rel_rows: list[dict]):
    """One panel per model: in a 12-month window around each month, the share of its answers that are its most
    repeated value (0-100%), with the value and the number of different answers on hover; gaps shaded."""
    p = palette()
    d = frames.copy()
    d["date"] = pd.PeriodIndex(d["month"], freq="M").to_timestamp().strftime("%Y-%m-%d")
    present = [m for m in models if m in set(d["model"])]
    dates = pd.to_datetime(d["date"]) if len(d) else pd.Series(dtype="datetime64[ns]")
    x_scale = alt.Scale(domain=[dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")]) if len(d) else \
        alt.Scale()
    x = alt.X("date:T", title=None, scale=x_scale, axis=alt.Axis(format="%Y", tickCount="year"))
    y = alt.Y("share:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=4))
    panels = []
    for m in present:
        c = model_color(m)
        own = with_segments(d[d["model"] == m].sort_values("month"))
        gaps = gap_spans(own).assign(note="no line here: too few answers in these windows")
        tip = [alt.Tooltip("month:N", title="window around"), alt.Tooltip("share:Q", title="share", format=".0%"),
               alt.Tooltip("value:Q", title="most repeated answer", format="+.2f"),
               alt.Tooltip("distinct:Q", title="different answers"), alt.Tooltip("n:Q", title="answers in window")]
        layers = [alt.Chart(gaps).mark_rect(color=p["muted"], opacity=0.12).encode(
                      x=alt.X("start:T", scale=x_scale), x2="end:T", tooltip=[alt.Tooltip("note:N", title="gap")]),
                  alt.Chart(own).mark_line(strokeWidth=2, color=c).encode(x=x, y=y, detail="segment:N", tooltip=tip)]
        for rows, dash, what in ((cut_rows, [5, 4], "cutoff"), (rel_rows, None, "released")):
            mine = [r for r in rows if r["model"] == m]
            if mine:
                r = pd.DataFrame(mine)
                r["date"] = pd.PeriodIndex(r["month"], freq="M").to_timestamp().strftime("%Y-%m-%d")
                mark = alt.Chart(r).mark_rule(strokeWidth=1.8, color=c, **({"strokeDash": dash} if dash else {}))
                layers.append(mark.encode(x=x, tooltip=[alt.Tooltip("month:N", title=what)]))
        panels.append(alt.layer(*layers).properties(title=alt.TitleParams(display_name(m), anchor="start",
                                                                            fontSize=12, color=c),
                                                  width=400, height=130))
    return alt.concat(*panels, columns=2)


MARKET_SERIES = ("Nordic", "US")
# two categorical slots of the validated reference palette (blue, orange; CVD and contrast pass), plus a line style
MARKET_COLORS = {"light": {"Nordic": "#2a78d6", "US": "#eb6834"}, "dark": {"Nordic": "#3987e5", "US": "#d95926"}}
MARKET_DASH = {"Nordic": [1, 0], "US": [6, 3]}


def rolling_market_chart(frames: pd.DataFrame, models: list[str], cut_rows: list[dict], rel_rows: list[dict]):
    """One panel per model: the rolling correlation for the Nordic and the US indices (column 'series'), with the
    model's cutoff (dashed rule) and release (solid rule)."""
    p = palette()
    shades = MARKET_COLORS["dark" if p is PALETTES["dark"] else "light"]
    d = frames.copy()
    d["date"] = pd.PeriodIndex(d["month"], freq="M").to_timestamp().strftime("%Y-%m-%d")
    present = [m for m in models if m in set(d["model"])]
    lo_y = float(min(-0.2, d["mean"].min())) if len(d) else -0.2
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount="year"))
    y = alt.Y("mean:Q", title=None, scale=alt.Scale(domain=[lo_y, 1.0]))
    color = alt.Color("series:N", title="Market", sort=list(MARKET_SERIES),
                      scale=alt.Scale(domain=list(MARKET_SERIES), range=[shades[k] for k in MARKET_SERIES]),
                      legend=None)   # market_compare_key() above the chart
    dash = alt.StrokeDash("series:N", sort=list(MARKET_SERIES), legend=None,
                          scale=alt.Scale(domain=list(MARKET_SERIES), range=[MARKET_DASH[k] for k in MARKET_SERIES]))
    panels = []
    for m in present:
        c = model_color(m)
        own = pd.concat([with_segments(g.sort_values("month")) for _, g in d[d["model"] == m].groupby("series")],
                        ignore_index=True)
        tip = [alt.Tooltip("series:N", title="market"), alt.Tooltip("month:N", title="window around"),
               alt.Tooltip("mean:Q", title="correlation", format=".2f"), alt.Tooltip("lo:Q", title="95% low", format=".2f"),
               alt.Tooltip("hi:Q", title="95% high", format=".2f"), alt.Tooltip("n:Q", title="answers in window")]
        layers = [alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color=p["muted"], strokeWidth=1).encode(y="zero:Q"),
                  alt.Chart(own).mark_line(strokeWidth=2).encode(x=x, y=y, color=color, strokeDash=dash,
                                                                 detail="segment:N", tooltip=tip)]
        for rows, rule_dash, what in ((cut_rows, [5, 4], "cutoff"), (rel_rows, None, "released")):
            mine = [r for r in rows if r["model"] == m]
            if mine:
                r = pd.DataFrame(mine)
                r["date"] = pd.PeriodIndex(r["month"], freq="M").to_timestamp().strftime("%Y-%m-%d")
                mark = alt.Chart(r).mark_rule(strokeWidth=1.8, color=c,
                                              **({"strokeDash": rule_dash} if rule_dash else {}))
                layers.append(mark.encode(x=x, tooltip=[alt.Tooltip("month:N", title=what)]))
        panels.append(alt.layer(*layers).properties(title=alt.TitleParams(display_name(m), anchor="start",
                                                                            fontSize=12, color=c),
                                                  width=400, height=150))
    return alt.concat(*panels, columns=2).resolve_scale(y="shared")


def chart_key(entries: list[tuple[str, str, str]]) -> str:
    """A chart key as one row of small CSS swatches (st.html strips inline SVG); labels stay in the text colour.
    entries: (kind, colour, label); kind: line, dashed, band, dots, triangles, rule-dashed, rule-solid, gap."""
    ring = palette()["surface"]

    def swatch(kind: str, c: str) -> str:
        box = "display:inline-block;vertical-align:middle"
        return {
            "line": f'<span style="{box};width:24px;height:2px;background:{c};box-shadow:0 0 0 2px {ring}"></span>',
            "dashed": f'<span style="{box};width:24px;height:0;border-top:2px dashed {c}"></span>',
            "band": f'<span style="{box};width:24px;height:12px;background:{c};opacity:0.3;border-radius:2px"></span>',
            "dots": "".join(f'<span style="{box};width:7px;height:7px;border-radius:50%;background:{c};'
                            f'margin-right:3px"></span>' for _ in range(2)),
            "triangles": f'<span style="color:{c};font-size:0.8rem;letter-spacing:1px">▲▼</span>',
            "rule-dashed": f'<span style="{box};width:0;height:14px;border-left:2px dashed {c}"></span>',
            "rule-solid": f'<span style="{box};width:0;height:14px;border-left:2px solid {c}"></span>',
            "gap": f'<span style="{box};width:18px;height:14px;background:{c};opacity:0.25;border-radius:2px"></span>',
        }[kind]
    items = "".join(f'<span style="display:inline-flex;align-items:center;gap:7px;margin-right:20px;'
                    f'white-space:nowrap">{swatch(k, c)}{label}</span>' for k, c, label in entries)
    return f'<div style="font-size:0.85rem;line-height:2;margin:0 0 2px 0">{items}</div>'


def path_chart_key() -> str:
    """The key for market_path_chart."""
    p = palette()
    return chart_key([("line", p["ink"], "real index"),
                      ("dots", p["muted"], "the model's answer for that month (in its colour)"),
                      ("triangles", p["muted"], "answer off the scale (hover for its value)"),
                      ("rule-dashed", p["muted"], "stated cutoff"), ("rule-solid", p["muted"], "release")])


def rolling_chart_key(value: str = "score", blind: bool = True) -> str:
    """The key for rolling_chart: value names what the coloured line is (e.g. 'correlation')."""
    p = palette()
    return chart_key([("line", p["muted"], f"the model's {value} (in its colour)"),
                      ("band", p["muted"], "95% range"),
                      *([("dashed", p["muted"], "blind guesser (no memory)")] if blind else []),
                      ("gap", p["muted"], "no line: too few answers in the window"),
                      ("rule-dashed", p["muted"], "stated cutoff"), ("rule-solid", p["muted"], "release")])


def market_compare_key() -> str:
    """The key for rolling_market_chart."""
    p, shades = palette(), MARKET_COLORS["dark"]
    return chart_key([("line", shades["Nordic"], "Nordic indices"), ("dashed", shades["US"], "US indices"),
                      ("rule-dashed", p["muted"], "stated cutoff"), ("rule-solid", p["muted"], "release"),
                      ("gap", p["muted"], "a break in a line: too few answers in the window")])


def market_path_chart(truth: pd.DataFrame, answers: pd.DataFrame, models: list[str], cut_rows: list[dict],
                      rel_rows: list[dict], y_title: str, kind: str = "change"):
    """One small panel per model: the model's answers (its colour, dots only: a line through answers that jump between
    a few stock values reads as noise) over the real index path (an ink line with a ring in the page colour); each dot has a
    ring too, so a dot right on the path stays visible, with the model's cutoff (dashed) and release (solid). path_chart_key() explains it.
    truth: month, value; answers: model, month, answer (and truth). kind 'level' or 'change' sets how far beyond the
    real path the scale may reach (market.path_scale); answers beyond it sit at the edge as triangles."""
    p = palette()
    t = truth.assign(date=pd.PeriodIndex(truth["month"], freq="M").to_timestamp().strftime("%Y-%m-%d"))
    a = answers.assign(date=pd.PeriodIndex(answers["month"], freq="M").to_timestamp().strftime("%Y-%m-%d"))
    a, lo, hi = market.path_scale(t["value"], a, kind)
    a["off"] = a["off"].map({"": "on scale", "above": "above the scale", "below": "below the scale"})
    present = [m for m in models if m in set(a["model"])]
    pad = (hi - lo) * 0.04 or 1.0
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount="year"))
    y_scale = alt.Scale(domain=[lo - pad, hi + pad], zero=False, nice=False)
    fmt = ",.0f" if kind == "level" else "+.1f"
    shape = alt.Shape("off:N", legend=None, scale=alt.Scale(domain=["on scale", "above the scale", "below the scale"],
                                                           range=["circle", "triangle-up", "triangle-down"]))
    size = alt.Size("off:N", legend=None, scale=alt.Scale(domain=["on scale", "above the scale", "below the scale"],
                                                         range=[26, 70, 70]))
    panels = []
    for m in present:
        c = model_color(m)
        own = with_segments(a[a["model"] == m].sort_values("month"))   # no line across months without an answer
        tip = [alt.Tooltip("month:N", title="month"), alt.Tooltip("answer:Q", title="answer", format=fmt),
               alt.Tooltip("truth:Q", title="real", format=fmt), alt.Tooltip("off:N", title="drawn")]
        layers = []
        if kind == "change":
            layers.append(alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color=p["muted"], strokeWidth=1,
                                                                            opacity=0.6).encode(y=alt.Y("zero:Q", scale=y_scale)))
        path = alt.Chart(t).encode(x=x, y=alt.Y("value:Q", scale=y_scale),
                                   tooltip=[alt.Tooltip("month:N", title="month"),
                                            alt.Tooltip("value:Q", title="real", format=fmt)])
        layers += [path.mark_line(strokeWidth=5, color=p["surface"]),     # a ring in the page colour around the path
                   path.mark_line(strokeWidth=1.6, color=p["ink"]),
                   # answers on top, each with a ring in the page colour: a dot right on the path stays visible
                   alt.Chart(own).mark_point(filled=True, color=c, opacity=1, stroke=p["surface"], strokeWidth=1.2).encode(
                       x=x, y=alt.Y("shown:Q", scale=y_scale, title=y_title), shape=shape, size=size, tooltip=tip)]
        for rows, dash, what in ((cut_rows, [5, 4], "cutoff"), (rel_rows, None, "released")):
            mine = [r for r in rows if r["model"] == m]
            if mine:
                r = pd.DataFrame(mine)
                r["date"] = pd.PeriodIndex(r["month"], freq="M").to_timestamp().strftime("%Y-%m-%d")
                mark = alt.Chart(r).mark_rule(strokeWidth=1.8, color=c, **({"strokeDash": dash} if dash else {}))
                layers.append(mark.encode(x=x, tooltip=[alt.Tooltip("month:N", title=what)]))
        panels.append(alt.layer(*layers).properties(title=alt.TitleParams(display_name(m), anchor="start", fontSize=12,
                                                                            color=c), width=400, height=190))
    return alt.concat(*panels, columns=2)
