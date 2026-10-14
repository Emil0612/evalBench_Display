"""DE-COP v3: the blind-baseline audit, the two-way form filter, the hub-breaking selection and the v3 builder
(pilot D/E: the original was found without memory by being the most central option or the only one with quotes)."""
import numpy as np
import pytest

from evalbench import build_testsets as bt
from evalbench import decop_audit as da
from evalbench.items import Item, load_manifest
from evalbench.methods import SECONDARY, decop

ORIG = ('Exempelbanken upprepar köp på Fjällnet. "Vi tror att Fjällnet kan växa snabbare än marknaden", skriver '
        "banken, som ser en uppsida på 20 procent (riktkurs 85 kronor).")


# ---------------------------------------------------------------- the two-way form filter
def test_a_rewrite_that_drops_the_quote_marks_is_a_tell():
    dropped = ("Exempelbanken behåller köp på Fjällnet. Banken tror att Fjällnet kan växa snabbare än marknaden och ser "
               "en uppsida på 20 procent (riktkurs 85 kronor).")
    assert any("quote" in r for r in decop.style_tells(ORIG, [dropped]))


def test_a_rewrite_must_use_the_same_special_characters():
    no_brackets = ('Exempelbanken behåller köp på Fjällnet. "Fjällnet kan växa snabbare än marknaden", skriver banken, '
                   "som ser en uppsida på 20 procent med riktkurs 85 kronor.")
    assert any("punctuation" in r for r in decop.style_tells(ORIG, [no_brackets]))
    same = ('Exempelbanken behåller sitt köpråd för Fjällnet. "Vi räknar med att Fjällnet växer snabbare än marknaden", '
            "skriver banken, som ser en uppsida på 20 procent (riktkurs 85 kronor).")
    assert decop.style_tells(ORIG, [same]) == []


# ---------------------------------------------------------------- blind rules and the audit
def _hub_item(i, rng):
    """An original and three rewrites that each change different words of it: the original is the hub."""
    words = [f"w{k}" for k in range(40)]
    rewrites = []
    for _ in range(3):
        w = list(words)
        for k in rng.choice(40, 12, replace=False):
            w[k] = f"x{rng.integers(10**6)}"
        rewrites.append(" ".join(w))
    return Item(item_id=f"dc-{i:04d}", method="decop", date="2023-01-01", cluster=f"c{i}", payload={},
                truth={"passage": " ".join(words), "paraphrases": rewrites, "newsid": str(i)})


def test_the_central_rule_finds_a_hub_original_and_the_audit_fails_it():
    rng = np.random.default_rng(1)
    items = [_hub_item(i, rng) for i in range(40)]
    rows = {r["rule"]: r for r in da.audit(items)}
    assert rows[da.CENTRAL]["rate"] > 0.9 and not rows[da.CENTRAL]["passed"]
    assert not da.passed(list(rows.values()))


def test_a_form_rule_fails_the_audit_as_soon_as_it_decides_anything():
    it = Item(item_id="dc-0", method="decop", date="2023-01-01", cluster="c0", payload={},
              truth={"passage": 'Han sa "ja" till affären.', "paraphrases": ["Han sa ja till affären nu.",
                                                                            "Han godkände affären i dag.",
                                                                            "Han accepterade affären då."]})
    rows = {r["rule"]: r for r in da.audit([it])}
    assert rows[da.ONLY_QUOTES]["decided"] == 1 and not rows[da.ONLY_QUOTES]["passed"]


# ---------------------------------------------------------------- the hub-breaking selection
def _candidates(rng, words):
    """Six rewrites, in pairs that share their changes: a pair member can be more central than the original."""
    out = []
    for pair in range(3):
        shared = rng.choice(40, 10, replace=False)
        for member in range(2):
            w = list(words)
            for k in shared:
                w[k] = f"p{pair}_{k}"
            for k in rng.choice(40, 3, replace=False):
                w[k] = f"y{rng.integers(10**6)}"
            out.append((" ".join(w), f"model{member + pair % 2}"))
    return out


def test_selection_brings_the_central_rule_down_to_chance():
    rng = np.random.default_rng(7)
    words = [f"w{k}" for k in range(40)]
    hits = 0
    n = 200
    for i in range(n):
        cands = _candidates(rng, words)
        pick = da.select_rewrites(" ".join(words), cands, rng)
        opts = [" ".join(words), *[cands[j][0] for j in pick]]
        assert len(set(pick)) == 3
        hits += da.blind_picks(opts)[da.CENTRAL] == 0
    assert 0.12 <= hits / n <= 0.38


# ---------------------------------------------------------------- the v3 builder (fake rewriters, no calls)
def _fake(tag):
    def fn(batch):
        return [[f"{p} {tag} variant one", f"{p} {tag} variant two"] for p, _lang in batch]
    return fn


def test_v3_builder_keeps_three_rewrites_from_at_least_two_models_and_records_the_audit(data, tmp_path):
    fns = {"a/one": _fake("alfa"), "b/two": _fake("beta"), "c/three": _fake("gamma")}
    path, items, audit_rows = bt.generate_decop_v3(data, 6, seed=3, fns=fns, root=tmp_path)
    assert items and all(len(it.truth["paraphrases"]) == 3 for it in items)
    assert all(len(set(it.truth["paraphrasers"])) >= 2 for it in items)
    manifest = load_manifest("decop", path.stem, tmp_path)
    assert manifest["paraphraser"] == sorted(fns) and manifest["selection"] == "hub-balanced"
    assert "blind_audit" in manifest and manifest["paraphrase_prompt"] == "v3"


def test_de_cop_is_secondary_until_a_set_passes_the_audit():
    assert "decop" in SECONDARY


def test_the_command_line_entry_point_comes_after_every_function():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(bt))
    last = tree.body[-1]
    assert isinstance(last, ast.If) and "__main__" in ast.unparse(last.test)   # run_decop_v3 exists when main() runs


# ---------------------------------------------------------------- balancing an existing set (free, no new rewrites)
def test_balancing_removes_only_sets_the_rule_gets_right_until_it_is_at_chance():
    rng = np.random.default_rng(3)
    hub = [_hub_item(i, rng) for i in range(60)]              # the central rule finds these
    plain = []
    for i in range(60, 160):                                   # the hub is a rewrite here: the rule misses the original
        it = _hub_item(i, rng)
        it.truth["passage"], it.truth["paraphrases"][0] = it.truth["paraphrases"][0], it.truth["passage"]
        plain.append(it)
    items = hub + plain
    before = {r["rule"]: r for r in da.audit(items)}[da.CENTRAL_WORDS]["rate"]
    kept = da.balance(items, da.CENTRAL_WORDS, seed=1)
    after = {r["rule"]: r for r in da.audit(kept)}[da.CENTRAL_WORDS]
    assert before > 0.3 and abs(after["rate"] - 0.25) <= 0.02
    removed = {it.item_id for it in items} - {it.item_id for it in kept}
    assert removed and all(da.blind_picks([it.truth["passage"], *it.truth["paraphrases"]])[da.CENTRAL_WORDS] == 0
                           for it in items if it.item_id in removed)


def test_selection_balances_both_kinds_of_centrality():
    rng = np.random.default_rng(11)
    words = [f"w{k}" for k in range(40)]
    hits_c = hits_w = 0
    for _ in range(200):
        cands = _candidates(rng, words)
        pick = da.select_rewrites(" ".join(words), cands, rng)
        picks = da.blind_picks([" ".join(words), *[cands[j][0] for j in pick]])
        hits_c += picks[da.CENTRAL] == 0
        hits_w += picks[da.CENTRAL_WORDS] == 0
    assert hits_c / 200 <= 0.38 and hits_w / 200 <= 0.38


def test_balancing_a_version_saves_the_next_version_and_records_what_was_removed(tmp_path):
    from evalbench.items import save_testset
    rng = np.random.default_rng(5)
    items = [_hub_item(i, rng) for i in range(40)] + [_hub_item(i, rng) for i in range(40, 90)]   # rule at 44%: fails
    for it in items[40:]:
        it.truth["passage"], it.truth["paraphrases"][0] = it.truth["paraphrases"][0], it.truth["passage"]
    save_testset(items, "decop", "v3", {"paraphraser": ["a/one"]}, root=tmp_path)
    path, kept, rows = da.balance_version("v3", root=tmp_path, seed=2)
    assert path.stem == "v4" and da.passed(rows) and len(kept) < len(items)
    manifest = load_manifest("decop", "v4", tmp_path)
    assert manifest["balanced_from"] == "v3" and len(manifest["removed"]) == len(items) - len(kept)
    assert manifest["paraphraser"] == ["a/one"] and manifest["blind_audit"]["passed"]


def test_v3_builds_keep_every_usable_candidate_for_free_reselection(data, tmp_path):
    fns = {"a/one": _fake("alfa"), "b/two": _fake("beta")}
    _, items, _ = bt.generate_decop_v3(data, 4, seed=3, fns=fns, root=tmp_path)
    assert all(len(it.truth["candidates"]) >= 3 for it in items)


# ---------------------------------------------------------------- re-selecting from stored candidates (free)
def _item_with_candidates(i, rng, words):
    cands = _candidates(rng, words)
    return Item(item_id=f"dc-{i:04d}", method="decop", date="2023-01-01", cluster=f"c{i}", payload={},
                truth={"passage": " ".join(words), "paraphrases": [cands[k][0] for k in (0, 2, 4)],   # one per pair:
                       "paraphrasers": [cands[k][1] for k in (0, 2, 4)],                          # the original is the hub
                       "candidates": [list(c) for c in cands],
                       "newsid": str(i)})


def test_reselecting_brings_both_central_rules_near_chance_without_dropping_sets(tmp_path):
    from evalbench.items import save_testset
    rng = np.random.default_rng(4)
    words = [f"w{k}" for k in range(40)]
    items = [_item_with_candidates(i, rng, words) for i in range(120)]
    save_testset(items, "decop", "v5", {"paraphraser": ["model0", "model1"]}, root=tmp_path)
    assert {r["rule"]: r for r in da.audit(items)}[da.CENTRAL]["rate"] > 0.5          # first three: the hub wins
    path, kept, rows = da.reselect_version("v5", root=tmp_path, seed=1)
    by = {r["rule"]: r for r in rows}
    assert path.stem == "v6" and len(kept) == len(items)
    assert 0.15 <= by[da.CENTRAL]["rate"] <= 0.35 and by[da.CENTRAL_WORDS]["rate"] <= 0.35
    manifest = load_manifest("decop", "v6", tmp_path)
    assert manifest["reselected_from"] == "v5" and "blind_audit" in manifest
    assert all(set(it.truth["paraphrases"]) <= {c[0] for c in it.truth["candidates"]} for it in kept)
