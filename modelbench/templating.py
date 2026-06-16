"""Tiny {{var}} template rendering. Deliberately not a full template engine —
we want substitution to be obvious and failures (missing vars) to be loud."""

from __future__ import annotations

import re

_PATTERN = re.compile(r"{{\s*(\w+)\s*}}")


def required_vars(template: str) -> set:
    """Variable names referenced by the template."""
    return set(_PATTERN.findall(template))


def render(template: str, variables: dict) -> str:
    """Substitute {{name}} with variables[name]. Raises KeyError if any are missing."""

    def repl(match: re.Match) -> str:
        name = match.group(1)
        if name not in variables:
            raise KeyError(f"missing template variable: {name!r}")
        return str(variables[name])

    return _PATTERN.sub(repl, template)
