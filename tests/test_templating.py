import pytest

from modelbench.templating import render, required_vars


def test_render_substitutes():
    assert render("用{{tone}}写{{topic}}", {"tone": "builder", "topic": "agent"}) == "用builder写agent"


def test_render_repeated_var():
    assert render("{{x}}-{{x}}", {"x": "a"}) == "a-a"


def test_render_missing_var_raises():
    with pytest.raises(KeyError):
        render("hi {{name}}", {})


def test_required_vars():
    assert required_vars("{{a}} and {{ b }}") == {"a", "b"}
    assert required_vars("no vars here") == set()
