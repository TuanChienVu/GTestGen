"""When each class member NAME was first declared in a file (rule 3 of v2tools/state.py). A name counts only in a
declaration context: a member declaration found by the outline parser (property, constant, promoted constructor
parameter, method), or an assignment `$this->x =` / `this.x =`. Constructor parameter names (promoted or not) are kept
apart, under "params", so that a parameter name never makes a later member look old. Every version of the file is
parsed, following renames; the first time is the minimum author time over the versions that declare the name. A version
whose blob is missing is an error (gitio.cat_batch). The version list itself is kept (rule 5: ④ shows each file as it
was before T), with, for each catalog function of the file, whether its HEAD code (without annotation and comment
lines: body._text, the normal form of body_key) is present in each version."""
from __future__ import annotations

import re
from pathlib import Path

from . import gitio
from .body import _COMMENT, _text as core_text
from .outline import parse

ASSIGN = {"php": re.compile(r"\$this->(\w+)\s*=(?![=>])"), "java": re.compile(r"\bthis\.(\w+)\s*=(?!=)"),
          "csharp": re.compile(r"\bthis\.(\w+)\s*=(?!=)")}


def declarations(language: str, source: str, cls: str | None = None) -> tuple[set[str], set[str], set[str]]:
    """(member names, constructor parameter names, private member names) that one version of a file declares in its
    class named `cls` (the HEAD class; else its first class)."""
    o = parse(language, source, cls)
    if o is None:
        return set(), set(), set()
    names = {n for m in o.members for n in m.names} | set(ASSIGN[language].findall(source))
    private = {n for m in o.members if m.visibility == "private" for n in m.names}
    return names, {n for ps in o.params.values() for _, _, n in ps if n}, private


def flat(text: str) -> str:
    """A whole file version in the normal form of body._text(): comment and docblock lines dropped, whitespace
    collapsed, so that `core_text(fn) in flat(version)` means the function's code is unchanged in that version."""
    return " ".join(" ".join(line for line in text.splitlines() if not _COMMENT.match(line)).split())


def first_times(repo: Path, commit: str, path: str, language: str, fns: list[dict] = (), cls: str | None = None) -> dict:
    """For a file at `commit`: {"file": first author time, "names": {member name: first author time}, "params":
    {constructor parameter name: first author time}, "versions": [[commit, author time, path there], ...] newest first
    (git log --follow order), and, aligned with "versions": "vnames" (member names each version declares), "vprivate"
    (its private member names), "present" {fn_id: "0"/"1" per version} for the catalog functions `fns` of that file}."""
    vs = gitio.versions(repo, commit, path)
    contents = gitio.cat_batch(repo, [f"{sha}:{p}" for sha, _, p in vs])
    names: dict[str, int] = {}
    params: dict[str, int] = {}
    cores = {f["fn_id"]: core_text(f) for f in fns}
    bits: dict[str, list[str]] = {fid: [] for fid in cores}
    vnames, vprivate = [], []
    for (_, at, _), text in zip(vs, contents):
        found, ctor, private = declarations(language, text, cls)
        vnames.append(sorted(found))
        vprivate.append(sorted(private))
        for table, got in ((names, found), (params, ctor)):
            for n in got:
                if n not in table or at < table[n]:
                    table[n] = at
        if cores:
            ft = flat(text)
            for fid, c in cores.items():
                bits[fid].append("1" if c in ft else "0")
    return {"file": min((at for _, at, _ in vs), default=None), "names": names, "params": params,
            "versions": [[sha, at, p] for sha, at, p in vs], "vnames": vnames, "vprivate": vprivate,
            "present": {fid: "".join(b) for fid, b in sorted(bits.items())}}
