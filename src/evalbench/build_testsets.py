"""Build the frozen, versioned test sets from data/.

uv run python -m evalbench.build_testsets --mock-paraphrases          # DE-COP with fake distractors ('mock')
uv run python -m evalbench.build_testsets --methods decop --paraphraser <openrouter model id>   # PAID

The paid DE-COP generation prints an upper-bound cost and asks for confirmation before any call is made.
Generated DE-COP sets never overwrite an existing version (v1, v2, ...); review them on the Test sets page.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from pathlib import Path

import numpy as np

from .data import DATA_DIR, DATE_MAX, DATE_MIN, Data, load_data, str_or_none
from .items import TESTSETS, Item, file_md5, load_testset, next_version, save_testset
from .llm.openrouter import OpenRouterClient, build_body, fetch_models, models_by_id
from .methods import METHODS, decop, headline_date, index_level, us_index_level
from .textutil import extract_json

SOURCE_FILES = ("universe.parquet", "prices_daily.parquet", "news_sample.parquet")
CHARS_PER_TOKEN = 3.5
# v2 (after pilot C): the original must not stand out by its form, only by its exact wording
PARAPHRASE_PROMPT_VERSION = "v2"
PARAPHRASE_PROMPT = (
    "Rewrite the passage below in three different ways, in {language}. Each version must read as if a human editor "
    "at the same company had written it for the same release: the same tone, register and level of detail, "
    "roughly the same length (within 10%), and the same punctuation conventions as the original. Do not add "
    "dashes, quotation marks, lists or any formatting the original does not use. Copy every number exactly as "
    "written, character for character, including decimal commas or points, spaces in thousands and units, and keep "
    "every fact and name. Change the wording and sentence structure so that no version copies the original "
    "phrasing, and avoid phrasing typical of AI-written text.\n\nPassage:\n{passage}\n\n"
    'Respond only with JSON: {{"paraphrases": ["<version 1>", "<version 2>", "<version 3>"]}}'
)


# v3 (after pilot E): two versions per model and three models per passage; quotations stay quotations and every
# punctuation mark is kept, because the v2 rewriter turned direct quotes into reported speech (a free tell)
PARAPHRASE_PROMPT_V3_VERSION = "v3"
PARAPHRASE_PROMPT_V3 = (
    "Rewrite the passage below in two different ways, in {language}. Each version must read as if the same human "
    "writer at the same company had written it for the same release: the same tone, register, level of detail and "
    "sentence rhythm, including any slightly awkward or repetitive phrasing typical of the original. Keep roughly the "
    "same length (within 10%). Use exactly the same punctuation marks as the original and no others: if the original "
    "quotes someone in quotation marks, keep it as a direct quotation with the same quotation marks; keep brackets, "
    "colons, percent signs and slashes where the original has them. Copy every number exactly as written, character "
    "for character, and keep every fact and name. Change the wording and word order so that neither version copies "
    "the original phrasing, but do not polish, simplify or improve the text, and do not vary vocabulary more than the "
    "original does.\n\nPassage:\n{passage}\n\n"
    'Respond only with JSON: {{"paraphrases": ["<version 1>", "<version 2>"]}}'
)
# v5 (after the v4 gate test): Mistral's rewrites were the easiest to spot as machine-written (picked 14% of the time
# when Gemini was wrong, against 20-21% for GLM and Kimi), so it is replaced by Claude Haiku 4.5
V3_PARAPHRASERS = ["z-ai/glm-4.5-air", "moonshotai/kimi-k2-0905", "anthropic/claude-haiku-4.5"]


def source_md5(data_dir: Path = DATA_DIR) -> dict[str, str]:
    return {f: file_md5(data_dir / f) for f in SOURCE_FILES if (data_dir / f).exists()}


def paraphrase_messages(passage: str, lang: str, prompt: str = PARAPHRASE_PROMPT) -> list[dict]:
    return [{"role": "user", "content": prompt.format(language=decop.LANG_NAMES[lang], passage=passage)}]


def paraphrase_max_tokens(passage: str, n_versions: int = 3) -> int:
    """Room for n rewrites of the passage plus JSON: caps output, so the cost estimate is an upper bound."""
    return int(len(passage) / CHARS_PER_TOKEN * (n_versions + 1)) + 200


def used_articles(root: Path = TESTSETS) -> set:
    """newsids of every article in any earlier DE-COP version (real or mock)."""
    folder = Path(root) / decop.NAME
    return {it.truth["newsid"] for p in folder.glob("*.jsonl")
            for it in load_testset(decop.NAME, p.name[: -len(".jsonl")], root)} if folder.exists() else set()


def estimate_paraphrase_cost(data: Data, n: int, seed: int, model: str, meta_by_id: dict,
                             exclude: set | None = None, n_versions: int = 3, prompt: str = PARAPHRASE_PROMPT) -> float:
    """Upper bound for paraphrasing the passages generate_decop would pick."""
    meta = meta_by_id.get(model) or {}
    pricing = meta.get("pricing") or {}
    total = 0.0
    for r, passage in decop.pick_articles(data, n, seed, exclude):
        body = build_body(meta, model, paraphrase_messages(passage, r["lang_group"], prompt),
                          paraphrase_max_tokens(passage, n_versions))
        chars = sum(len(m["content"]) for m in body["messages"])
        total += chars / CHARS_PER_TOKEN * float(pricing.get("prompt") or 0)
        total += body["max_tokens"] * float(pricing.get("completion") or 0)
    return total


def make_llm_paraphraser(model: str, meta_by_id: dict | None = None, on_progress=None, n_versions: int = 3,
                         prompt: str = PARAPHRASE_PROMPT) -> decop.ParaphraseFn:
    """PAID: calls the OpenRouter model once per passage."""
    meta = (meta_by_id or models_by_id(fetch_models())).get(model)
    if meta is None:
        raise ValueError(f"unknown OpenRouter model: {model}")

    def paraphrase(batch: list[tuple[str, str]]) -> list[list[str] | None]:
        spent, done = 0.0, 0

        async def main():
            client = OpenRouterClient()
            try:
                async def one(passage: str, lang: str):
                    nonlocal spent, done
                    body = build_body(meta, model, paraphrase_messages(passage, lang, prompt),
                                      paraphrase_max_tokens(passage, n_versions))
                    try:
                        res = await client.chat(body)
                    except Exception as e:
                        print(f"  paraphrase failed: {e}")
                        return None
                    finally:
                        done += 1
                    spent += res.cost
                    if on_progress:
                        on_progress(done, len(batch), spent)
                    obj = extract_json(res.text)
                    ps = obj.get("paraphrases") if obj else None
                    if isinstance(ps, list) and len(ps) == n_versions and all(isinstance(p, str) and p.strip()
                                                                              for p in ps):
                        return [" ".join(p.split()) for p in ps]
                    return None

                return await asyncio.gather(*(one(p, lang) for p, lang in batch))
            finally:
                await client.aclose()

        out = asyncio.run(main())
        print(f"  paraphrasing cost ${spent:.4f} ({model})")
        return out

    return paraphrase


def generate_decop(data: Data, n: int, seed: int, paraphrase_fn: decop.ParaphraseFn, paraphraser: str,
                   root: Path = TESTSETS, fresh: bool = False) -> tuple[Path, list[Item]]:
    """Build a DE-COP test set as the next free version; returns its path and items. fresh: no article that any
    earlier DE-COP version used."""
    items = decop.build_items(data, n, seed, paraphrase_fn=paraphrase_fn,
                              exclude=used_articles(root) if fresh else None)
    path = save_testset(items, decop.NAME, next_version(decop.NAME, root), {
        "seed": seed, "source_md5": source_md5(), "prompt_version": decop.PROMPT_VERSION, "filters": decop.FILTERS,
        "date_window": [DATE_MIN, DATE_MAX], "paraphraser": paraphraser, "n_requested": n,
        "paraphrase_prompt": PARAPHRASE_PROMPT_VERSION, "fresh": fresh,
    }, root=root)
    return path, items


# M14-M17 are built from the daily index closes, not from data/ as a whole
INDEX_SOURCES = {"index_level": index_level.load_index, "index_return": index_level.load_index,
                 "us_index_level": us_index_level.load_index, "us_index_return": us_index_level.load_index}
# M18 is built from the headline index (data/news_index.parquet)
NEWS_SOURCES = {"headline_date": headline_date.load_headlines}
# M12/M13 and their probability probes take the company-months of another test set
DERIVED_SETS = {"price_choice", "known_choice", "price_choice_prob", "known_choice_prob", "index_forecast", "path_date"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--paraphraser", help="PAID: OpenRouter model id that writes the DE-COP paraphrases")
    ap.add_argument("--mock-paraphrases", action="store_true", help="fake DE-COP distractors (pipeline test only)")
    ap.add_argument("--fresh", action="store_true", help="DE-COP: skip every article used by an earlier version")
    ap.add_argument("--decop-v3", action="store_true",
                    help="PAID: DE-COP v3 with three rewriting models (two versions each), fresh passages, audited")
    ap.add_argument("--paraphrasers", nargs="+", default=V3_PARAPHRASERS, help="--decop-v3: the rewriting models")
    ap.add_argument("--unfreeze", action="store_true", help="build a new DE-COP version although M3 is frozen at v7")
    args = ap.parse_args(argv)
    if (args.decop_v3 or ("decop" in args.methods and args.paraphraser)) and not args.unfreeze:
        print("M3 DE-COP is frozen at v7 (decided 2026-10-06: every version failed the after-release gate, because "
              "models spot machine rewrites by style). Nothing was built or spent; pass --unfreeze to override.")
        args.methods = [m for m in args.methods if m != "decop"]
        if args.decop_v3 or not args.methods:
            return
    if args.decop_v3:
        run_decop_v3(args.n, args.seed, args.paraphrasers)
        return

    print("loading data/ ...")
    data = load_data()
    md5 = source_md5()
    for method in args.methods:
        mod = METHODS[method]
        if method == "decop":
            if args.paraphraser:
                meta = models_by_id(fetch_models())
                if args.paraphraser not in meta:
                    print(f"unknown OpenRouter model: {args.paraphraser} - nothing was spent")
                    continue
                exclude = used_articles() if args.fresh else None
                bound = estimate_paraphrase_cost(data, args.n, args.seed, args.paraphraser, meta, exclude)
                print(f"DE-COP paraphrases: {args.n} passages with {args.paraphraser}, next version "
                      f"decop/{next_version(decop.NAME)}"
                      + (f", fresh (skipping {len(exclude)} articles used before)." if args.fresh else "."))
                print(f"This is PAID (OpenRouter). Upper bound: ${bound:.4f} (output is capped per passage).")
                answer = input(f"Continue and spend up to ${bound:.4f}? [y/N] ").strip().lower()
                if answer not in ("y", "yes"):
                    print("Cancelled - nothing was spent.")
                    continue
                path, items = generate_decop(data, args.n, args.seed, make_llm_paraphraser(args.paraphraser, meta),
                                             args.paraphraser, fresh=args.fresh)
                print(f"{method}: {len(items)} items -> {path}")
                continue
            if not args.mock_paraphrases:
                print("skipping decop: pass --paraphraser MODEL or --mock-paraphrases")
                continue
            items = mod.build_items(data, args.n, args.seed, paraphrase_fn=decop.mock_paraphrases)
            version, extra = "mock", {"paraphraser": "mock"}
        elif method in DERIVED_SETS:
            print(f"{method}: skipped - built from another test set (its manifest's filters say which), not from data/")
            continue
        elif method in NEWS_SOURCES:
            try:
                items = mod.build_items(NEWS_SOURCES[method](), args.seed)
            except FileNotFoundError as e:
                print(f"{method}: skipped - {e}")
                continue
            version, extra = args.version, mod.manifest_extra()
        else:
            try:   # the index methods ask one question per index-month, so --n does not apply
                items = (mod.build_items(INDEX_SOURCES[method](), args.seed) if method in INDEX_SOURCES
                         else mod.build_items(data, args.n, args.seed))
            except FileNotFoundError as e:     # the private US data (M10, M16, M17) is fetched separately
                hint = (" - fetch it: uv run --with yfinance python -m evalbench.fetch_us_indices"
                        if method in ("us_index_level", "us_index_return") else "")
                print(f"{method}: skipped - {e}{hint}")
                continue
            version, extra = args.version, {}
        try:
            path = save_testset(items, method, version, {
                "seed": args.seed, "source_md5": md5, "prompt_version": mod.PROMPT_VERSION,
                "filters": mod.FILTERS, "date_window": [DATE_MIN, DATE_MAX], **extra,
            })
        except FileExistsError:
            print(f"{method}: {method}/{version} already exists - left unchanged (pass --version to build a new one)")
            continue
        print(f"{method}: {len(items)} items -> {path}")



def generate_decop_v3(data: Data, n: int, seed: int, fns: dict[str, decop.ParaphraseFn], root: Path = TESTSETS,
                      fresh: bool = True) -> tuple[Path, list[Item], list[dict]]:
    """DE-COP v3: every rewriting model writes two versions of each passage; each rewrite is checked on its own for
    form tells; three are chosen so the original is neither the most central nor the longest/shortest option more
    often than chance, preferring different models. The set is audited with the blind rules and saved as the next
    version with the audit in its manifest. Returns (path, items, audit rows)."""
    from . import decop_audit
    rng = np.random.default_rng(seed)
    picked = decop.pick_articles(data, n, seed, used_articles(root) if fresh else None)
    batch = [(p, r["lang_group"]) for r, p in picked]
    per_model = {model: fn(batch) for model, fn in fns.items()}
    items: list[Item] = []
    dropped = Counter()
    for k, (r, passage) in enumerate(picked):
        company = str_or_none(r["company"])
        cands, seen = [], {passage}
        for model, outs in per_model.items():
            for text in (outs[k] or []):
                if text in seen:
                    continue
                seen.add(text)
                tells = decop.style_tells(passage, [text], company)
                if tells:
                    dropped[tells[0].split(": ", 1)[1].split(" (")[0]] += 1
                    continue
                cands.append((text, model))
        if len(cands) < 3:
            dropped["fewer than 3 usable rewrites"] += 1
            continue
        chosen = decop_audit.select_rewrites(passage, cands, rng)
        items.append(Item(
            item_id=f"dc-{len(items):04d}", method=decop.NAME, date=r["date"].date().isoformat(), cluster=r["cluster"],
            payload={"distributor": r["distributor"], "company": company, "language": r["lang_group"],
                     "source_group": r["source_group"]},
            truth={"passage": passage, "paraphrases": [cands[i][0] for i in chosen],
                   "paraphrasers": [cands[i][1] for i in chosen], "newsid": r["newsid"],
                   "candidates": [list(c) for c in cands]},        # every usable rewrite: re-selection is free
            strata={"language": r["lang_group"], "source_group": r["source_group"]},
        ))
    audit_rows = decop_audit.audit(items)
    path = save_testset(items, decop.NAME, next_version(decop.NAME, root), {
        "seed": seed, "source_md5": source_md5(), "prompt_version": decop.PROMPT_VERSION, "filters": decop.FILTERS,
        "date_window": [DATE_MIN, DATE_MAX], "paraphraser": sorted(fns), "n_requested": n,
        "paraphrase_prompt": PARAPHRASE_PROMPT_V3_VERSION, "fresh": fresh, "selection": "hub-balanced",
        "rewrites_dropped": dict(dropped),
        "blind_audit": {"passed": decop_audit.passed(audit_rows),
                        "rules": {r["rule"]: {"decided": r["decided"], "rate": None if r["rate"] != r["rate"]
                                              else round(r["rate"], 3), "passed": r["passed"]} for r in audit_rows}},
    }, root=root)
    return path, items, audit_rows


def run_decop_v3(n: int, seed: int, paraphrasers: list[str]) -> None:
    """PAID: show the upper bound per rewriting model, ask y/N, build DE-COP v3 and print its blind-baseline audit."""
    from . import decop_audit
    print("loading data/ ...")
    data = load_data()
    meta = models_by_id(fetch_models())
    unknown = [m for m in paraphrasers if m not in meta]
    if unknown:
        print(f"unknown OpenRouter model(s): {', '.join(unknown)} - nothing was spent")
        return
    exclude = used_articles()
    bounds = {m: estimate_paraphrase_cost(data, n, seed, m, meta, exclude, n_versions=2, prompt=PARAPHRASE_PROMPT_V3)
              for m in paraphrasers}
    total = sum(bounds.values())
    print(f"DE-COP v3: {n} fresh passages (skipping {len(exclude)} used before), two rewrites each from "
          f"{len(paraphrasers)} models -> decop/{next_version(decop.NAME)}")
    for m, b in bounds.items():
        print(f"  {m}: up to ${b:.4f}")
    print(f"This is PAID (OpenRouter). Upper bound: ${total:.4f} (output is capped per passage).")
    if input(f"Continue and spend up to ${total:.4f}? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Cancelled - nothing was spent.")
        return
    fns = {m: make_llm_paraphraser(m, meta, n_versions=2, prompt=PARAPHRASE_PROMPT_V3) for m in paraphrasers}
    path, items, rows = generate_decop_v3(data, n, seed, fns)
    print(f"decop: {len(items)} items -> {path}")
    decop_audit._print(path.stem, rows)
    if not decop_audit.passed(rows):
        print("The set FAILED the blind-baseline audit: do not run models on it (see the rules marked FAIL).")


if __name__ == "__main__":
    main()
