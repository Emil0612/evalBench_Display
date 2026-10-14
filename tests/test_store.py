from evalbench import store


def _rec(item_id="i1", **kw):
    rec = {"model": "m", "method": "price_recall", "item_id": item_id, "variant": "main", "mode": "forced",
           "testset": "price_recall/v1", "prompt_version": "p1", "run_id": "r1", "messages": "[]",
           "response": '{"price": 1}', "parsed": '{"value": 1.0, "refused": false}', "refused": 0, "cost": 0.001,
           "prompt_tokens": 10, "completion_tokens": 2, "provider": "P", "model_returned": "m", "latency_s": 0.5}
    rec.update(kw)
    return rec


def test_insert_load_and_cache(tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    store.insert_call(conn, _rec("i1"))
    store.insert_call(conn, _rec("i2", mode="abstain"))
    assert ("m", "price_recall", "i1", "main", "forced", 0.0, "auto", -1, 0, "price_recall/v1", "p1") \
        in store.cached_keys(conn)
    assert len(store.load_calls(conn)) == 2
    assert list(store.load_calls(conn, mode="forced").item_id) == ["i1"]


def test_runs(tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    store.upsert_run(conn, {"run_id": "r1", "models": "m", "methods": "x", "n_calls": 3, "est_cost": 0.1,
                            "cost": 0.0, "status": "running"})
    store.upsert_run(conn, {"run_id": "r1", "models": "m", "methods": "x", "n_calls": 3, "est_cost": 0.1,
                            "cost": 0.05, "status": "done"})
    runs = store.load_runs(conn)
    assert len(runs) == 1 and runs.status.iloc[0] == "done"


def test_stats_counts_in_the_database(tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    store.insert_call(conn, _rec("i1", model="mock/oracle@2024-06-30", provider="local", cost=0.0))
    store.insert_call(conn, _rec("i2", model="x/m", provider="P", cost=0.25))
    store.insert_call(conn, _rec("i3", model="x/m", provider="P", cost=0.5))
    s = store.stats(conn)
    assert s == {"answers": 3, "models": 2, "api_models": 1, "api_spend": 0.75}
    assert store.stats(store.connect(tmp_path / "empty.sqlite")) == {"answers": 0, "models": 0, "api_models": 0,
                                                                     "api_spend": 0.0}


def test_db_stamp_changes_only_when_answers_change(tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    empty = store.db_stamp(conn)
    base = {"model": "m", "method": "price_recall", "item_id": "i", "variant": "main", "mode": "forced",
            "testset": "t", "prompt_version": "p1", "parsed": "{}", "cost": 0.01}
    store.insert_call(conn, base)
    one = store.db_stamp(conn)
    assert one != empty and store.db_stamp(conn) == one
    store.insert_call(conn, {**base, "item_id": "j"})
    assert store.db_stamp(conn) != one
