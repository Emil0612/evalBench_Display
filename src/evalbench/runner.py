"""Plan calls, skip cached ones, estimate cost, and execute with a hard spend cap."""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field

import pandas as pd

from . import store
from .cutoffs import group_for
from .items import Item
from .llm.mock import is_local, local_answer
from .llm import selfhosted
from .llm.openrouter import ChatResult, OpenRouterClient, build_body
from .methods import METHODS

CHARS_PER_TOKEN = 3.5


def is_paid(model_id: str) -> bool:
    """Only OpenRouter models cost money: the local controls and self-hosted models are free."""
    return not is_local(model_id) and not selfhosted.is_self_hosted(model_id)


@dataclass(frozen=True)
class CallSpec:
    model: str
    method: str
    item: Item
    variant: str
    mode: str
    testset: str
    temperature: float = 0.0
    reasoning: str = "auto"
    seed: int = -1          # -1 = no seed
    sample: int = 0         # repeat index when several samples per item are requested
    route: str = ""         # pinned provider endpoint ('' = default routing); recorded, not part of the key

    @property
    def key(self) -> tuple:  # same order as store.KEY_COLS
        return (self.model, self.method, self.item.item_id, self.variant, self.mode, self.temperature, self.reasoning,
                self.seed, self.sample, self.testset, METHODS[self.method].PROMPT_VERSION)

    @property
    def sent_seed(self) -> int | None:
        """Each sample gets its own seed (seed + sample), otherwise repeats would be identical."""
        return None if self.seed < 0 else self.seed + self.sample


@dataclass
class RunState:
    spent: float = 0.0
    done: int = 0
    errors: list[str] = field(default_factory=list)
    stopped: bool = False
    reserved: float = 0.0   # upper-bound cost of calls in flight
    skipped: int = 0        # already stored or being paid for by another run
    user_stopped: bool = False   # Stop was pressed (from any tab): no new calls start


def plan(models, methods, n: int, mode: str, testsets: dict[str, tuple[str, list[Item]]],
         temperature: float = 0.0, reasoning: str = "auto", seed: int | None = None,
         samples: int = 1, pins: dict | None = None) -> list[CallSpec]:
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if seed is not None and not 0 <= int(seed) + samples - 1 < 2**31:
        raise ValueError("seed + samples must stay below 2**31")
    temperature = round(float(temperature), 2)
    seed_value = -1 if seed is None else int(seed)
    calls = []
    for model in models:
        route = "" if is_local(model) else (pins or {}).get(model, "")
        for method in methods:
            name, items = testsets[method]
            for item in items[:n]:
                for sample in range(samples):
                    for variant in METHODS[method].VARIANTS:
                        calls.append(CallSpec(model, method, item, variant, mode, name, temperature, reasoning,
                                              seed_value, sample, route))
    return calls


def _is_done(c: CallSpec, cached: dict[tuple, str]) -> bool:
    """Stored already; a pinned call only counts an answer that came from the same provider endpoint."""
    return c.key in cached and (not c.route or cached[c.key] == c.route)


def pending(calls: list[CallSpec], conn) -> list[CallSpec]:
    cached = store.cached_routes(conn)
    return [c for c in calls if not _is_done(c, cached)]


def plan_summary(calls: list[CallSpec], conn, meta_by_id: dict, cutoffs: dict) -> pd.DataFrame:
    """One row per model: what a run would do and cost, and how its items split around the model's cutoff."""
    cached = store.cached_routes(conn)
    rows = []
    for model in dict.fromkeys(c.model for c in calls):
        mc = [c for c in calls if c.model == model]
        todo = [c for c in mc if not _is_done(c, cached)]
        items = {(c.method, c.item.item_id): c.item for c in mc}.values()
        groups = [group_for(it.date, cutoffs.get(model)) for it in items if not it.fabricated]
        rows.append({"model": model, "calls": len(mc), "cached": len(mc) - len(todo), "to_run": len(todo),
                     "est_cost": estimate_cost(todo, meta_by_id), "cutoff": cutoffs.get(model),
                     "items_pre": groups.count(1), "items_post": groups.count(0), "items_excluded": groups.count(-1)})
    return pd.DataFrame(rows)


def estimate_cost(calls: list[CallSpec], meta_by_id: dict) -> float:
    """Upper bound: prompt chars / 3.5 x input price + max_tokens (incl. reasoning headroom) x output price
    + any per-request fee. Pass prices from with_worst_case_prices for a bound that holds on every provider."""
    total = 0.0
    for c in calls:
        if not is_paid(c.model):
            continue
        meta = meta_by_id.get(c.model) or {}
        pricing = meta.get("pricing") or {}
        mod = METHODS[c.method]
        body = build_body(meta, c.model, mod.render(c.item, c.mode, c.variant), mod.MAX_TOKENS, c.temperature,
                          c.reasoning, c.sent_seed)
        chars = sum(len(m["content"]) for m in body["messages"])
        total += chars / CHARS_PER_TOKEN * float(pricing.get("prompt") or 0)
        total += body["max_tokens"] * float(pricing.get("completion") or 0)
        total += float(pricing.get("request") or 0)
    return total


def ceiling_still_approved(calls: list[CallSpec], meta_now: dict, approved: float) -> tuple[bool, float]:
    """Re-check the ceiling with prices fetched at the moment of paying: False if it rose above what was approved."""
    now = estimate_cost(calls, meta_now)
    return now <= approved * 1.0001 + 1e-9, now


async def _execute(calls, conn, run_id, meta_by_id, client, spend_cap, on_progress, concurrency,
                   state: RunState, privacy: str = "standard", self_clients: dict | None = None,
                   poll_s: float = 2.0) -> RunState:
    sem = asyncio.Semaphore(concurrency)
    last_poll = [float("-inf")]

    def fail(message: str) -> None:
        state.errors.append(message)
        store.add_run_error(conn, run_id, message)

    def stop_pressed() -> bool:
        # the Stop button writes to the database; read it at most every poll_s seconds
        if not state.user_stopped and time.monotonic() - last_poll[0] >= poll_s:
            last_poll[0] = time.monotonic()
            state.user_stopped = store.stop_requested(conn, run_id)
        return state.user_stopped

    def notify() -> None:
        if on_progress:
            try:
                on_progress(state, len(calls))
            except Exception:  # a broken progress display must never cost paid results
                pass

    async def one(c: CallSpec) -> None:
        async with sem:
            if stop_pressed():
                return
            bound = estimate_cost([c], meta_by_id)
            # hard cap: start a call only if even its worst case keeps total spend within the cap
            # (a model without a known price has bound 0: then fall back to stopping once real spend reaches the cap)
            over = state.spent + state.reserved + bound > spend_cap if bound > 0 else state.spent >= spend_cap
            if over and is_paid(c.model):
                state.stopped = True
                return
            if not store.claim_call(conn, c.key, run_id):   # another live run is paying for it
                state.skipped += 1
                return
            try:
                if store.has_call(conn, c.key, c.route):
                    state.skipped += 1
                    return
                state.reserved += bound
                try:
                    await call(c)
                finally:
                    state.reserved -= bound
            finally:
                store.release_call(conn, c.key)

    async def call(c: CallSpec) -> None:
        mod = METHODS[c.method]
        messages = mod.render(c.item, c.mode, c.variant)
        t0 = time.perf_counter()
        try:
            if is_local(c.model):
                res = ChatResult(text=local_answer(c.model, c.method, c.item, c.variant, c.mode, c.sample), cost=0.0,
                                 prompt_tokens=None, completion_tokens=None, provider="local", model=c.model,
                                 finish_reason="stop")
            elif selfhosted.is_self_hosted(c.model):
                meta = meta_by_id[c.model]
                res = await self_clients[c.model].chat(selfhosted.build_body(
                    meta, messages, mod.MAX_TOKENS, c.temperature, c.sent_seed))
                res.text = selfhosted.strip_thinking(res.text)
                res.cost, res.provider = 0.0, f"self-hosted · {meta['self_hosted']['base_url']}"
            else:
                res = await client.chat(build_body(meta_by_id.get(c.model), c.model, messages, mod.MAX_TOKENS,
                                                   c.temperature, c.reasoning, c.sent_seed, privacy,
                                                   c.route or None, logprobs=getattr(mod, "LOGPROBS", 0)))
        except Exception as e:  # recorded, not cached: a re-run retries it
            fail(f"{c.model} | {c.method} | {c.item.item_id} | {c.variant}: {e}")
            notify()
            return
        if not res.cost and is_paid(c.model):
            res.cost = _token_cost(res, meta_by_id.get(c.model))
        state.spent += res.cost
        if res.finish_reason == "length" and not (res.text or "").strip():
            # the output budget ran out (usually on reasoning) before any answer: not a model outcome, so it is
            # not cached as one; it was billed, so it still counts against the cap
            fail(f"{c.model} | {c.method} | {c.item.item_id} | {c.variant}: reply cut off at "
                 f"max_tokens before any answer (reasoning used the budget)")
            notify()
            return
        try:
            parsed = mod.parse(res.text)
        except Exception as e:  # keep the paid response; score it as a malformed answer
            parsed = {"refused": True, "outcome": "malformed", "parse_error": str(e)}
        if hasattr(mod, "with_logprobs"):   # probability probes: the letters' probabilities at the answer
            parsed = mod.with_logprobs(parsed, getattr(res, "logprobs", None))
        store.insert_call(conn, {
            "model": c.model, "method": c.method, "item_id": c.item.item_id, "variant": c.variant,
            "mode": c.mode, "temperature": c.temperature, "reasoning": c.reasoning, "seed": c.seed,
            "sample": c.sample, "testset": c.testset,
            "privacy": ("local" if is_local(c.model) else "self-hosted" if selfhosted.is_self_hosted(c.model)
                        else privacy),
            "route": c.route,
            "prompt_version": mod.PROMPT_VERSION, "run_id": run_id,
            "messages": json.dumps(messages, ensure_ascii=False), "response": res.text,
            "parsed": json.dumps(parsed, ensure_ascii=False), "refused": int(parsed["refused"]),
            "cost": res.cost, "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
            "provider": res.provider, "model_returned": res.model,
            "latency_s": round(time.perf_counter() - t0, 3), "finish_reason": res.finish_reason,
            "logprobs": json.dumps(res.logprobs, ensure_ascii=False) if getattr(res, "logprobs", None) else None,
        })
        state.done += 1
        notify()

    await asyncio.gather(*(one(c) for c in calls))
    return state


def _token_cost(res: ChatResult, meta: dict | None) -> float:
    """Fallback when the provider reports no usage.cost: tokens x list price."""
    pricing = (meta or {}).get("pricing") or {}
    return ((res.prompt_tokens or 0) * float(pricing.get("prompt") or 0)
            + (res.completion_tokens or 0) * float(pricing.get("completion") or 0))


def self_hosted_clients(model_ids, meta_by_id: dict, transport=None) -> dict:
    """One client per self-hosted model, talking to its own server (no OpenRouter key involved). A local server is
    often slow and handles few requests at once: concurrency (default 2) and timeout_s (default 600) per entry."""
    out = {}
    for m in sorted(set(model_ids)):
        e = meta_by_id[m]["self_hosted"]
        out[m] = OpenRouterClient(api_key=selfhosted.api_key(e), transport=transport, key_required=False,
                                  base_url=e["base_url"], concurrency=int(e.get("concurrency") or 2),
                                  timeout=float(e.get("timeout_s") or 600))
    return out


def run_info(calls: list[CallSpec], meta_by_id: dict, run_id: str) -> dict:
    """The run's row in the run log."""
    return {"run_id": run_id, "models": ",".join(sorted({c.model for c in calls})),
            "methods": ",".join(sorted({c.method for c in calls})), "n_calls": len(calls),
            "est_cost": estimate_cost(calls, meta_by_id), "cost": 0.0}


def run(calls: list[CallSpec], conn, meta_by_id: dict, spend_cap: float = 2.0, on_progress=None, client=None,
        concurrency: int = 8, privacy: str = "standard", transport=None, run_id: str | None = None,
        poll_s: float = 2.0, heartbeat_s: float = 5.0) -> tuple[str, RunState]:
    """Progress, a heartbeat and the failed calls go to the database (any tab can follow the run and press Stop);
    run_id: a run registered with store.start_run_exclusive (the background worker). transport: an httpx transport
    for the self-hosted servers (tests)."""
    run_id = run_id or uuid.uuid4().hex[:10]
    info = {**run_info(calls, meta_by_id, run_id), "status": "running"}
    store.upsert_run(conn, info)
    active = {r[0] for r in conn.execute(f"SELECT run_id FROM runs WHERE status IN {store.ACTIVE}")}
    store.clear_claims(conn, keep=active | {run_id})   # claims of runs that died never block a question

    state = RunState()

    def beat() -> None:
        store.heartbeat(conn, run_id, done=state.done, failed=len(state.errors), spent=state.spent, pid=os.getpid())

    async def heartbeats() -> None:   # while every call waits on the network
        while True:
            beat()
            await asyncio.sleep(heartbeat_s)

    last_beat = [time.monotonic()]

    def progress(st: RunState, n_calls: int) -> None:
        # after each finished call too: instant answers (the free controls) never let the task above run
        if time.monotonic() - last_beat[0] >= heartbeat_s:
            last_beat[0] = time.monotonic()
            beat()
        if on_progress:
            on_progress(st, n_calls)

    async def main() -> RunState:
        own = client is None and any(is_paid(c.model) for c in calls)   # the OpenRouter key: only when needed
        cl = OpenRouterClient() if own else client
        selfs = self_hosted_clients([c.model for c in calls if selfhosted.is_self_hosted(c.model)], meta_by_id,
                                    transport)
        hb = asyncio.create_task(heartbeats())
        try:
            return await _execute(calls, conn, run_id, meta_by_id, cl, spend_cap, progress, concurrency, state,
                                  privacy, selfs, poll_s)
        finally:
            hb.cancel()
            if own:
                await cl.aclose()
            for s in selfs.values():
                await s.aclose()

    status = "interrupted"
    try:
        asyncio.run(main())
        status = "stopped by user" if state.user_stopped else "stopped at cap" if state.stopped else "done"
    finally:
        if state.errors and status == "done":
            status = f"done with {len(state.errors)} errors"
        beat()
        store.upsert_run(conn, {**info, "cost": state.spent, "status": status})
    return run_id, state
