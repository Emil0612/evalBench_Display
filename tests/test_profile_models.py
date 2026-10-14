"""The Profile's model picker: every model that answered the method is shown by default; models the reader switches
off stay off across methods and pages until switched back on."""
from evalbench import ui


def test_every_model_run_is_shown_by_default():
    assert ui.profile_selection(["a", "b", "c"], set()) == ["a", "b", "c"]
    assert ui.profile_selection(["a", "b", "c"], {"b", "x"}) == ["a", "c"]


def test_switching_a_model_off_or_on_changes_only_the_models_on_this_method():
    hidden = ui.update_hidden({"x"}, shown=["a", "b"], selected=["a"])
    assert hidden == {"x", "b"}                               # x was hidden on another method and stays hidden
    assert ui.update_hidden(hidden, shown=["a", "b"], selected=["a", "b"]) == {"x"}


def test_all_and_none_act_on_the_models_of_this_method():
    assert ui.update_hidden({"x", "b"}, shown=["a", "b"], selected=["a", "b"]) == {"x"}
    assert ui.update_hidden({"x"}, shown=["a", "b"], selected=[]) == {"x", "a", "b"}
