"""The M12/M13 report (docs/preregistration/2026-10-01-price-choice.md)."""
from evalbench import choice_report as cr


def test_sensitive_means_the_lower_bound_beats_both_chance_and_the_blind_guesser():
    assert cr.sensitive(0.40, blind=0.23)
    assert not cr.sensitive(0.24, blind=0.10)          # not above chance (0.25)
    assert not cr.sensitive(0.30, blind=0.32)          # not above the blind guesser


def test_a_clean_after_question_has_every_option_from_after_the_release():
    from datetime import date
    from evalbench.items import Item
    it = Item("pc-1", "price_choice", "2025-11-28", "c1", {"month": "2025-11"},
              {"decoy_months": ["2025-09", "2026-03", "2026-07"]})
    assert cr.clean_after(it, date(2025, 8, 21))            # all after August 2025
    assert not cr.clean_after(it, date(2025, 9, 2))         # September 2025 is the release month itself
