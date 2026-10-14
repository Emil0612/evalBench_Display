"""Cost ceilings use current prices from the most expensive provider, and DE-COP leaves room for a short preamble."""
import httpx
import pytest

from evalbench import runner
from evalbench.llm import openrouter as orr
from evalbench.methods import decop, price_recall

ENDPOINTS = [
    {"provider_name": "Cheap", "pricing": {"prompt": "0.0000001", "completion": "0.00000032"}},
    {"provider_name": "Pricey", "pricing": {"prompt": "0.0000003", "completion": "0.0000022", "request": "0.0001"}},
    {"provider_name": "Odd", "pricing": {"prompt": "n/a", "completion": None}},
]
LISTED = {"x/m": {"pricing": {"prompt": "0.0000001", "completion": "0.00000032"}, "supported_parameters": []}}


def test_fetch_endpoints_reads_the_public_endpoint_list():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={"data": {"id": "x/m", "endpoints": ENDPOINTS}})

    eps = orr.fetch_endpoints("x/m", transport=httpx.MockTransport(handler))
    assert seen == ["/api/v1/models/x/m/endpoints"] and len(eps) == 3


def test_worst_case_pricing_takes_the_most_expensive_provider_per_price():
    worst = orr.worst_case_pricing(ENDPOINTS)
    assert worst == {"prompt": pytest.approx(3e-7), "completion": pytest.approx(2.2e-6), "request": pytest.approx(1e-4)}


def test_prices_are_raised_to_the_worst_provider_without_touching_the_shared_list():
    priced, info = orr.with_worst_case_prices(LISTED, ["x/m"], fetch=lambda m: ENDPOINTS)
    assert priced["x/m"]["pricing"]["completion"] == pytest.approx(2.2e-6)
    assert LISTED["x/m"]["pricing"]["completion"] == "0.00000032"      # the cached model list is not mutated
    assert info["x/m"]["ok"] and info["x/m"]["providers"] == 3


def test_a_failed_price_check_keeps_the_listed_price_and_says_so():
    def boom(model):
        raise httpx.ConnectError("offline")

    priced, info = orr.with_worst_case_prices(LISTED, ["x/m"], fetch=boom)
    assert priced["x/m"]["pricing"] == LISTED["x/m"]["pricing"] and not info["x/m"]["ok"]


def test_ceiling_grows_with_the_worst_provider_and_counts_per_request_fees(data):
    calls = runner.plan(["x/m"], ["price_recall"], 3, "forced",
                        {"price_recall": ("t", price_recall.build_items(data, 3, 1))})
    listed = runner.estimate_cost(calls, LISTED)
    priced, _ = orr.with_worst_case_prices(LISTED, ["x/m"], fetch=lambda m: ENDPOINTS)
    worst = runner.estimate_cost(calls, priced)
    assert worst > listed * 3
    no_fee, _ = orr.with_worst_case_prices(LISTED, ["x/m"], fetch=lambda m: ENDPOINTS[:1])
    assert runner.estimate_cost(calls, priced) - runner.estimate_cost(
        calls, {"x/m": {**priced["x/m"], "pricing": {**priced["x/m"]["pricing"], "request": 0}}}) == \
        pytest.approx(len(calls) * 1e-4)
    assert runner.estimate_cost(calls, no_fee) == pytest.approx(listed)


def test_decop_reply_budget_leaves_room_for_a_short_preamble():
    # the smoke test's Llama answer ran out after 20 tokens of "To determine the original text, ..."
    assert decop.MAX_TOKENS >= 60


def test_a_higher_ceiling_at_the_moment_of_paying_needs_new_approval(data):
    calls = runner.plan(["x/m"], ["price_recall"], 3, "forced",
                        {"price_recall": ("t", price_recall.build_items(data, 3, 1))})
    approved = runner.estimate_cost(calls, LISTED)
    ok, now = runner.ceiling_still_approved(calls, LISTED, approved)
    assert ok and now == pytest.approx(approved)
    pricier = {"x/m": {**LISTED["x/m"], "pricing": {"prompt": "0.000001", "completion": "0.00001"}}}
    ok, now = runner.ceiling_still_approved(calls, pricier, approved)
    assert not ok and now > approved
