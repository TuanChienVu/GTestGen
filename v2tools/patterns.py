"""Behat Turnip patterns, Cucumber Expressions and raw regexes -> compiled Python regexes."""
from __future__ import annotations

import re

_TURNIP_ARG = r'(?:"[^"]*"|\'[^\']*\'|[^\s"\']+)'
_CUKE_ARGS = {
    "string": r'(?:"[^"]*"|\'[^\']*\')', "int": r"-?\d+", "long": r"-?\d+", "byte": r"-?\d+", "short": r"-?\d+",
    "float": r"-?\d*[.,]?\d+", "double": r"-?\d*[.,]?\d+", "bigdecimal": r"-?\d*[.,]?\d+", "biginteger": r"-?\d+",
    "word": r"\S+", "": r".*",
}
_TOKEN = re.compile(r":\w+|\{(\w*)\}|\((\w[^()]*)\)|[^\s/(){}:]+(?:/[^\s/(){}:]+)+")


def expression_to_regex(pattern: str, flavor: str) -> str:
    out: list[str] = []
    pos = 0
    for m in _TOKEN.finditer(pattern):
        out.append(re.escape(pattern[pos:m.start()]))
        token = m.group(0)
        if token.startswith(":") and flavor == "turnip":
            out.append(_TURNIP_ARG)
        elif token.startswith("{") and flavor == "cucumber":
            out.append(_CUKE_ARGS.get(m.group(1), r".*?"))
        elif m.group(2) is not None:
            out.append(f"(?:{re.escape(m.group(2))})?")
        elif "/" in token and not token.startswith((":", "{")):
            out.append("(?:" + "|".join(re.escape(part) for part in token.split("/")) + ")")
        else:
            out.append(re.escape(token))
        pos = m.end()
    out.append(re.escape(pattern[pos:]))
    return "^" + "".join(out) + "$"


# Port of Behat's TurnipPatternPolicy: placeholder token, optional "(word)" with its own optional surrounding
# whitespace, "a/b" alternation of two words, matched case-insensitively. The optional group's content excludes
# every character preg_quote would escape, as in Behat (which applies its rules to the preg_quote'd pattern).
_BEHAT_TOKEN = r"""["']?(?:(?<=")[^"]*(?=")|(?<=')[^']*(?=')|-?[\w.,]+)['"]?"""
_BEHAT_RULES = re.compile(r":\w+|(\s)?\(([^.\\+*?\[^\]$(){}=!<>|:\-#/]+)\)(\s)?|(\w+)/(\w+)")


def turnip_to_regex(pattern: str) -> str:
    out: list[str] = []
    pos = 0
    for m in _BEHAT_RULES.finditer(pattern):
        out.append(re.escape(pattern[pos:m.start()]))
        if m.group(0).startswith(":"):
            out.append(_BEHAT_TOKEN)
        elif m.group(2) is not None:
            out.append("".join(f"(?:{re.escape(part)})?" for part in m.group(1, 2, 3) if part))
        else:
            out.append(f"(?:{re.escape(m.group(4))}|{re.escape(m.group(5))})")
        pos = m.end()
    out.append(re.escape(pattern[pos:]))
    return "^" + "".join(out) + "$"


def _ruby_to_python(pattern: str) -> str:
    pattern = re.sub(r"\(\?<(?![=!])", "(?P<", pattern)
    return pattern.replace(r"\A", "^").replace(r"\z", "$").replace(r"\Z", "$").replace(r"\h", "[0-9a-fA-F]")


def compile_pattern(kind: str, pattern: str) -> tuple[re.Pattern, str] | None:
    try:
        if kind == "turnip":
            return re.compile(turnip_to_regex(pattern), re.IGNORECASE), "fullmatch"
        if kind == "cucumber":
            return re.compile(expression_to_regex(pattern, kind)), "fullmatch"
        if kind == "regex":
            return re.compile(_ruby_to_python(pattern)), "search"
        if kind == "regex_full":
            return re.compile(_ruby_to_python(pattern)), "fullmatch"
    except re.error:
        return None
    raise ValueError(f"unknown pattern kind {kind!r}")
