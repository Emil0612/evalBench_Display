import pytest

from evalbench.textutil import (
    extract_json, extract_numbers, has_unit, is_templated, is_year, numbers_equal,
    parse_answer_number, parse_number, significant_digits,
)


def test_extract_numbers_mixed_formats():
    text = "Net sales rose to SEK 1 234,5 million (1 100) in 2023, up 12,5%."
    assert [m.group() for m in extract_numbers(text)] == ["1 234,5", "1 100", "2023", "12,5"]


def test_extract_numbers_skips_dates():
    assert extract_numbers("Dated 15.06.2021 and 2021-06-15") == []


@pytest.mark.parametrize("s,lang,expected", [
    ("1 234,5", "sv", 1234.5),
    ("1,234.5", "en", 1234.5),
    ("1.234,5", "sv", 1234.5),
    ("12,5", "sv", 12.5),
    ("1.234", "sv", 1234.0),
    ("SEK 450 million", "en", 450.0),
])
def test_parse_number(s, lang, expected):
    assert parse_number(s, lang) == pytest.approx(expected)


def test_numbers_equal_uses_article_language_for_truth():
    assert numbers_equal("1234.5", "1 234,5", "sv")
    assert numbers_equal("1 234,5 MSEK", "1 234,5", "sv")
    assert not numbers_equal("1234", "1 234,5", "sv")
    assert not numbers_equal("no idea", "1 234,5", "sv")


@pytest.mark.parametrize("value,expected", [
    ("210,65", 210.65), ("1,234", 1234.0), ("SEK 1 234.50", 1234.5), (99, 99.0), (12.5, 12.5), ("n/a", None),
])
def test_parse_answer_number(value, expected):
    assert parse_answer_number(value) == (pytest.approx(expected) if expected is not None else None)


def test_has_unit():
    text = "sales of SEK 450 million and page 451 of the report"
    m1, m2 = [m for m in extract_numbers(text)]
    assert has_unit(text, m1.start(), m1.end())
    assert not has_unit(text, m2.start(), m2.end())


def test_is_templated():
    assert is_templated("Sampo plc's share buybacks 02/01/2023")
    assert is_templated("Kallelse till årsstämma i Volvo")
    assert not is_templated("Ny stororder på hybridbussar till Scania")


def test_digits_and_years():
    assert significant_digits("0,045") == 2
    assert is_year("2023") and not is_year("1 234")


def test_extract_json():
    assert extract_json('Sure! ```json\n{"price": 12.5}\n```') == {"price": 12.5}
    assert extract_json('{"answer": null}') == {"answer": None}
    assert extract_json("no json here") is None


def test_is_boilerplate():
    from evalbench.textutil import is_boilerplate
    assert is_boilerplate("This information is information that Impala Bondco plc is obliged to make public pursuant to the EU Market Abuse Regulation.")
    assert is_boilerplate("Denna information är sådan som Bolaget är skyldigt att offentliggöra enligt EU:s marknadsmissbruksförordning.")
    assert is_boilerplate("För ytterligare information, vänligen kontakta: Anna Svensson, VD")
    assert not is_boilerplate("The order intake increased 51 per cent year-on-year and totalled EUR 14.3 million.")


def test_is_boilerplate_about_sections():
    from evalbench.textutil import is_boilerplate
    assert is_boilerplate("Om Analyst Group: Ett av Sveriges främsta analyshus.")
    assert is_boilerplate("About Volvo Group: The Volvo Group is one of the world's leading manufacturers.")
    assert not is_boilerplate("About 400 employees were affected by the change.")


def test_is_boilerplate_disclaimers():
    from evalbench.textutil import is_boilerplate
    assert is_boilerplate("Gartner Disclaimer: GARTNER is a registered trademark of Gartner, Inc.")
    assert is_boilerplate("This release contains forward-looking statements that involve risks.")
    assert is_boilerplate("Detta pressmeddelande innehåller framåtriktade uttalanden.")
