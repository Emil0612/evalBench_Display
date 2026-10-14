from evalbench import runner, store
from evalbench.llm.openrouter import ChatResult, OpenRouterError
from evalbench.methods import decop, price_recall


class FakeClient:
    def __init__(self, cost=1.0, fail=False):
        self.cost, self.fail, self.bodies = cost, fail, []

    async def chat(self, body):
        self.bodies.append(body)
        if self.fail:
            raise OpenRouterError(402, "insufficient credits")
        return ChatResult(text='{"price": 1}', cost=self.cost, prompt_tokens=10, completion_tokens=2,
                          provider="fake", model=body["model"], finish_reason="stop")


def _testsets(data):
    return {"price_recall": ("price_recall/t", price_recall.build_items(data, 10, 1)),
            "decop": ("decop/t", decop.build_items(data, 5, 1, decop.mock_paraphrases))}


def test_plan_counts_variants(data):
    ts = _testsets(data)
    assert len(runner.plan(["a", "b"], ["decop"], 5, "forced", ts)) == 2 * 5 * 4
    assert len(runner.plan(["a"], ["price_recall"], 3, "forced", ts)) == 3


def test_pending_skips_cached(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    calls = runner.plan(["mock/random@2024-06-30"], ["price_recall"], 4, "forced", _testsets(data))
    runner.run(calls, conn, {})
    assert runner.pending(calls, conn) == []
    more = runner.plan(["mock/random@2024-06-30"], ["price_recall"], 6, "forced", _testsets(data))
    assert len(runner.pending(more, conn)) == 2


def test_estimate_cost(data):
    ts = _testsets(data)
    meta = {"x/m": {"pricing": {"prompt": "0.000001", "completion": "0.000002"}, "supported_parameters": ["temperature"]}}
    assert runner.estimate_cost(runner.plan(["mock/oracle@2024-06-30"], ["price_recall"], 5, "forced", ts), meta) == 0
    assert runner.estimate_cost(runner.plan(["x/m"], ["price_recall"], 5, "forced", ts), meta) > 0


def test_spend_cap_stops_run(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    calls = runner.plan(["x/m"], ["price_recall"], 10, "forced", _testsets(data))
    _, state = runner.run(calls, conn, {}, spend_cap=2.5, client=FakeClient(cost=1.0), concurrency=1)
    assert state.done == 3 and state.stopped and state.spent == 3.0


def test_errors_are_not_cached(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    calls = runner.plan(["x/m"], ["price_recall"], 4, "forced", _testsets(data))
    _, state = runner.run(calls, conn, {}, client=FakeClient(fail=True))
    assert state.done == 0 and len(state.errors) == 4 and "402" in state.errors[0]
    assert len(runner.pending(calls, conn)) == 4
    assert store.load_runs(conn).status.iloc[0] == "done with 4 errors"


def test_progress_callback_failure_does_not_lose_paid_calls(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    calls = runner.plan(["x/m"], ["price_recall"], 4, "forced", _testsets(data))

    def bad_progress(state, total):
        raise RuntimeError("ui went away")

    _, state = runner.run(calls, conn, {}, client=FakeClient(cost=0.1), on_progress=bad_progress)
    assert state.done == 4 and runner.pending(calls, conn) == []


def test_finish_reason_is_stored(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    calls = runner.plan(["x/m"], ["price_recall"], 1, "forced", _testsets(data))
    runner.run(calls, conn, {}, client=FakeClient(cost=0.1))
    assert store.load_calls(conn).finish_reason.iloc[0] == "stop"


def test_run_status_is_finalised_when_interrupted(data, tmp_path):
    import pytest
    conn = store.connect(tmp_path / "t.sqlite")
    calls = runner.plan(["x/m"], ["price_recall"], 2, "forced", _testsets(data))

    class Interrupt(BaseException):
        pass

    def interrupt(state, total):
        raise Interrupt()

    with pytest.raises(Interrupt):
        runner.run(calls, conn, {}, client=FakeClient(cost=0.1), on_progress=interrupt, concurrency=1)
    assert store.load_runs(conn).status.iloc[0] == "interrupted"


import asyncio as _asyncio
import threading

PRICED = {"x/m": {"pricing": {"prompt": "0", "completion": "0.01"}, "supported_parameters": []}}


class SlowClient(FakeClient):
    async def chat(self, body):
        await _asyncio.sleep(0.02)
        return await super().chat(body)


def test_spend_cap_is_never_exceeded_with_concurrency(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    calls = runner.plan(["x/m"], ["price_recall"], 10, "forced", _testsets(data))
    # each call's upper bound is 60 tokens x $0.01 = $0.60; actual cost equals the bound
    _, state = runner.run(calls, conn, PRICED, spend_cap=2.0, client=SlowClient(cost=0.6), concurrency=8)
    assert state.spent <= 2.0 and state.done == 3 and state.stopped


def test_missing_cost_falls_back_to_token_pricing(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    calls = runner.plan(["x/m"], ["price_recall"], 1, "forced", _testsets(data))
    _, state = runner.run(calls, conn, PRICED, client=FakeClient(cost=0.0))
    assert state.spent == 2 * 0.01  # 2 completion tokens x $0.01


def test_zero_cap_still_runs_free_local_models(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    calls = runner.plan(["mock/random@2024-06-30"], ["price_recall"], 3, "forced", _testsets(data))
    _, state = runner.run(calls, conn, {}, spend_cap=0.0)
    assert state.done == 3


def test_two_concurrent_runs_never_pay_twice(data, tmp_path):
    db = tmp_path / "t.sqlite"
    calls = runner.plan(["x/m"], ["price_recall"], 6, "forced", _testsets(data))
    clients = [SlowClient(cost=0.01), SlowClient(cost=0.01)]
    threads = [threading.Thread(target=runner.run, args=(calls, store.connect(db), {}),
                                kwargs={"client": cl, "spend_cap": 10.0}) for cl in clients]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(len(cl.bodies) for cl in clients) == len(calls)


def test_plan_summary_per_model(data, tmp_path):
    from datetime import date
    conn = store.connect(tmp_path / "t.sqlite")
    ts = _testsets(data)
    calls = runner.plan(["mock/oracle@2024-06-30", "x/m"], ["price_recall"], 10, "forced", ts)
    runner.run([c for c in calls if c.model.startswith("mock")], conn, {})
    s = runner.plan_summary(calls, conn, PRICED, {"mock/oracle@2024-06-30": date(2024, 6, 30), "x/m": None})
    row = s.set_index("model")
    assert row.loc["mock/oracle@2024-06-30", "to_run"] == 0 and row.loc["x/m", "to_run"] == 10
    assert row.loc["x/m", "est_cost"] > 0 and row.loc["mock/oracle@2024-06-30", "est_cost"] == 0
    assert row.loc["mock/oracle@2024-06-30", "items_pre"] > 0 and row.loc["x/m", "items_pre"] == 0
