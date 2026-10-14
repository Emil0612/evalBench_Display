"""The DE-COP blind-baseline audit, and the selection that keeps the original from standing out.

Pilots D and E showed that the original passage can be found without any model: it is the "hub" that every rewrite
was made from (the most central option picked it 47-55% of the time, chance 25%), and in 1 set in 6 it was the only
option with quotation marks. Blind baselines like these beat published membership tests (Das, Zhang & Tramèr 2024),
so every DE-COP set is audited with them before any model is asked, and v3 chooses its rewrites to defeat them.

uv run python -m evalbench.decop_audit            # audit every DE-COP version
uv run python -m evalbench.decop_audit v3         # one version
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from itertools import combinations

import numpy as np

from .items import Item, TESTSETS, load_manifest, load_testset, next_version, save_testset
from .methods import decop

CENTRAL = "most central option (character overlap)"
CENTRAL_WORDS = "most central option (shared words)"
LONGEST, SHORTEST = "longest option", "shortest option"
PUNCT = "most kinds of punctuation"
ONLY_QUOTES = "only option with quote marks"
ONLY_SPECIAL = "only option with some punctuation mark"
RULES = (CENTRAL, CENTRAL_WORDS, LONGEST, SHORTEST, PUNCT, ONLY_QUOTES, ONLY_SPECIAL)
FORM_RULES = (ONLY_QUOTES, ONLY_SPECIAL)      # a form tell must not exist at all
PASS_MAX = 0.35                               # chance is 0.25; a rule above this finds the original without memory
MIN_DECIDED = 10                              # a non-form rule that decides fewer sets than this cannot fail


def _grams(text: str, n: int = 4) -> Counter:
    t = re.sub(r"\s+", " ", text.lower())
    return Counter(t[i:i + n] for i in range(len(t) - n + 1))


def _cos(a: Counter, b: Counter) -> float:
    num = sum(a[k] * b[k] for k in a.keys() & b.keys())
    return num / (np.sqrt(sum(v * v for v in a.values())) * np.sqrt(sum(v * v for v in b.values())) + 1e-12)


def centrality(opts: list[str]) -> list[float]:
    """Mean character-4-gram similarity of each option to the others."""
    g = [_grams(o) for o in opts]
    return [float(np.mean([_cos(g[i], g[j]) for j in range(len(opts)) if j != i])) for i in range(len(opts))]


def _word_centrality(opts: list[str]) -> list[float]:
    ws = [set(re.findall(r"\w+", o.lower())) for o in opts]
    return [float(np.mean([len(ws[i] & ws[j]) / max(1, len(ws[i] | ws[j])) for j in range(len(opts)) if j != i]))
            for i in range(len(opts))]


def _unique_max(values: list[float]) -> int | None:
    best = max(values)
    return values.index(best) if values.count(best) == 1 else None


def _only_one(flags: list[bool]) -> int | None:
    return flags.index(True) if sum(flags) == 1 else None


def blind_picks(opts: list[str]) -> dict[str, int | None]:
    """The option each no-model rule would pick (None when it cannot decide)."""
    specials = [set(o) & set(decop.SPECIALS) for o in opts]
    lone = [bool(s - set().union(*(specials[:i] + specials[i + 1:]))) for i, s in enumerate(specials)]
    return {
        CENTRAL: _unique_max(centrality(opts)),
        CENTRAL_WORDS: _unique_max(_word_centrality(opts)),
        LONGEST: _unique_max([float(len(o)) for o in opts]),
        SHORTEST: _unique_max([-float(len(o)) for o in opts]),
        PUNCT: _unique_max([float(len(s)) for s in specials]),
        ONLY_QUOTES: _only_one([any(c in o for c in decop.QUOTES) for o in opts]),
        ONLY_SPECIAL: _only_one(lone),
    }


def audit(items: list[Item]) -> list[dict]:
    """Per rule: how many sets it decides and how often it picks the original. Passed when it is near chance
    (at most PASS_MAX); a form rule passes only if it never decides."""
    decided, hits = Counter(), Counter()
    for it in items:
        for rule, pick in blind_picks([it.truth["passage"], *it.truth["paraphrases"]]).items():
            if pick is not None:
                decided[rule] += 1
                hits[rule] += pick == 0
    rows = []
    for rule in RULES:
        n = decided[rule]
        rate = hits[rule] / n if n else float("nan")
        if rule in FORM_RULES:
            ok = n == 0
        else:
            ok = n < MIN_DECIDED or rate <= PASS_MAX
        rows.append({"rule": rule, "decided": n, "items": len(items), "rate": rate, "passed": bool(ok)})
    return rows


def passed(rows: list[dict]) -> bool:
    return all(r["passed"] for r in rows)


def select_rewrites(original: str, candidates: list[tuple[str, str]], rng: np.random.Generator) -> list[int]:
    """Choose 3 of the candidate rewrites (text, model) so that the original sits at a random rank of centrality (by
    characters and by words: v3 balanced only the first, and the second then found the original 39% of the time) and
    of length among the four options, so neither 'most central' nor 'longest/shortest' finds it more often than
    chance. Among equally good choices, prefer rewrites from different models."""
    target_central, target_length = int(rng.integers(4)), int(rng.integers(4))
    best, best_cost = None, None
    for combo in combinations(range(len(candidates)), 3):
        opts = [original, *[candidates[i][0] for i in combo]]
        cen, cen_w = centrality(opts), _word_centrality(opts)
        rank_c = sorted(range(4), key=lambda i: -cen[i]).index(0)          # 0 = most central (characters)
        rank_w = sorted(range(4), key=lambda i: -cen_w[i]).index(0)        # 0 = most central (words)
        rank_l = sorted(range(4), key=lambda i: -len(opts[i])).index(0)    # 0 = longest
        models = len({candidates[i][1] for i in combo})
        cost = (3 * abs(rank_c - target_central) + 3 * abs(rank_w - target_central) + abs(rank_l - target_length)
                + (3 - models))
        if best_cost is None or cost < best_cost:
            best, best_cost = list(combo), cost
    return best


def balance(items: list[Item], rule: str, target: float = 0.25, seed: int = 0) -> list[Item]:
    """A subset of a built set on which `rule` finds the original only at chance: remove, at random, just enough of
    the sets it gets right. Removing only its hits leaves the rule uninformative, not reversed. Order is kept."""
    hits = [it.item_id for it in items
            if blind_picks([it.truth["passage"], *it.truth["paraphrases"]])[rule] == 0]
    decided = sum(blind_picks([it.truth["passage"], *it.truth["paraphrases"]])[rule] is not None for it in items)
    k = int(np.ceil((len(hits) - target * decided) / (1 - target)))
    if k <= 0:
        return list(items)
    drop = set(np.random.default_rng(seed).choice(hits, size=min(k, len(hits)), replace=False))
    return [it for it in items if it.item_id not in drop]


def balance_version(version: str, root=TESTSETS, seed: int = 0):
    """Free: balance a failed DE-COP version on each failing non-form rule, in turn, and save the subset as the next
    version (a frozen set is never changed). Returns (path, items, audit rows of the new version)."""
    items = load_testset(decop.NAME, version, root)
    kept = list(items)
    for _ in range(3):                          # balancing one rule can move another: re-check a few times
        failing = [r["rule"] for r in audit(kept) if not r["passed"] and r["rule"] not in FORM_RULES]
        if not failing:
            break
        for rule in failing:
            kept = balance(kept, rule, seed=seed)
    rows = audit(kept)
    removed = sorted({it.item_id for it in items} - {it.item_id for it in kept})
    manifest = {k: v for k, v in load_manifest(decop.NAME, version, root).items()
                if k not in ("method", "version", "n_items", "n_fabricated", "built_at", "items_md5", "items_per_year",
                             "strata_counts", "blind_audit")}
    new = next_version(decop.NAME, root)
    path = save_testset(kept, decop.NAME, new, {
        **manifest, "balanced_from": version, "removed": removed, "balance_seed": seed,
        "blind_audit": {"passed": passed(rows),
                        "rules": {r["rule"]: {"decided": r["decided"], "rate": None if r["rate"] != r["rate"]
                                              else round(r["rate"], 3), "passed": r["passed"]} for r in rows}},
    }, root=root)
    return path, kept, rows


def _combos(original: str, cands: list) -> list[dict]:
    """Every 3-rewrite choice with what the blind rules would see: is the original the most central (by characters,
    by words), and how many different models wrote the three."""
    out = []
    for combo in combinations(range(len(cands)), 3):
        opts = [original, *[cands[i][0] for i in combo]]
        out.append({"combo": list(combo), "hit_c": _unique_max(centrality(opts)) == 0,
                    "hit_w": _unique_max(_word_centrality(opts)) == 0, "models": len({cands[i][1] for i in combo})})
    return out


def reselect(items: list[Item], seed: int = 0, low: float = 0.20, target: float = 0.25) -> list[Item]:
    """Choose each set's three rewrites again from its stored candidates (free): wherever possible a choice in which
    the original is the most central option on neither measure; the passages where it must be the hub then set the
    overall rate. If that falls below `low` (a reversed tell), random sets are given back a central original until
    the rate reaches `target`. Prefers rewrites from more models. Sets without stored candidates are kept as they are."""
    rng = np.random.default_rng(seed)
    options, choice = {}, {}
    for it in items:
        cands = it.truth.get("candidates")
        if not cands or len(cands) < 3:
            continue
        combos = _combos(it.truth["passage"], cands)
        options[it.item_id] = combos
        clean = [c for c in combos if not c["hit_c"] and not c["hit_w"]] or                 [c for c in combos if not c["hit_c"]] or combos
        choice[it.item_id] = max(clean, key=lambda c: (c["models"], -c["combo"][0]))
    n = len(items)
    hits = sum(c["hit_c"] for c in choice.values())
    if n and hits / n < low:
        flexible = [i for i, c in choice.items() if not c["hit_c"] and any(o["hit_c"] for o in options[i])]
        for iid in rng.permutation(flexible):
            if hits / n >= target:
                break
            choice[iid] = max((o for o in options[iid] if o["hit_c"]), key=lambda c: c["models"])
            hits += 1
    out = []
    for it in items:
        c = choice.get(it.item_id)
        if c is None:
            out.append(it)
            continue
        cands = it.truth["candidates"]
        truth = {**it.truth, "paraphrases": [cands[i][0] for i in c["combo"]],
                 "paraphrasers": [cands[i][1] for i in c["combo"]]}
        out.append(Item(item_id=it.item_id, method=it.method, date=it.date, cluster=it.cluster, payload=it.payload,
                        truth=truth, strata=it.strata, fabricated=it.fabricated))
    return out


def _save_derived(items: list[Item], version: str, root, extra: dict):
    rows = audit(items)
    manifest = {k: v for k, v in load_manifest(decop.NAME, version, root).items()
                if k not in ("method", "version", "n_items", "n_fabricated", "built_at", "items_md5", "items_per_year",
                             "strata_counts", "blind_audit", "balanced_from", "removed", "balance_seed",
                             "reselected_from", "reselect_seed")}
    path = save_testset(items, decop.NAME, next_version(decop.NAME, root), {
        **manifest, **extra,
        "blind_audit": {"passed": passed(rows),
                        "rules": {r["rule"]: {"decided": r["decided"], "rate": None if r["rate"] != r["rate"]
                                              else round(r["rate"], 3), "passed": r["passed"]} for r in rows}},
    }, root=root)
    return path, items, rows


def reselect_version(version: str, root=TESTSETS, seed: int = 0):
    """Free: re-select every set of VERSION from its stored candidates and save the result as the next version."""
    items = reselect(load_testset(decop.NAME, version, root), seed=seed)
    return _save_derived(items, version, root, {"reselected_from": version, "reselect_seed": seed})


def _print(version: str, rows: list[dict]) -> None:
    print(f"\ndecop/{version} ({rows[0]['items']} sets): {'PASSED' if passed(rows) else 'FAILED'} the blind-baseline audit")
    for r in rows:
        rate = "–" if np.isnan(r["rate"]) else f"{r['rate']:.0%}"
        print(f"  {'ok  ' if r['passed'] else 'FAIL'} {r['rule']:40s} picks the original {rate:>4} "
              f"(decides {r['decided']} of {r['items']})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("versions", nargs="*", help="e.g. v2 v3 (default: every version)")
    ap.add_argument("--reselect", metavar="VERSION",
                    help="free: choose every set's rewrites again from its stored candidates, as the next version")
    ap.add_argument("--balance", metavar="VERSION",
                    help="free: save a subset of VERSION on which every failing rule is at chance, as the next version")
    args = ap.parse_args(argv)
    if args.reselect:
        path, kept, rows = reselect_version(args.reselect)
        print(f"decop/{args.reselect} -> {path.stem}: re-selected {len(kept)} sets from their stored candidates")
        _print(path.stem, rows)
        return
    if args.balance:
        path, kept, rows = balance_version(args.balance)
        print(f"decop/{args.balance} -> {path.stem}: kept {len(kept)} sets")
        _print(path.stem, rows)
        return
    folder = TESTSETS / decop.NAME
    versions = args.versions or sorted(p.name[: -len(".jsonl")] for p in folder.glob("*.jsonl") if "mock" not in p.name)
    for v in versions:
        _print(v, audit(load_testset(decop.NAME, v)))


if __name__ == "__main__":
    main()
