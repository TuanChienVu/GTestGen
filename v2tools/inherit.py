"""Resolve the parents (extends / implements / traits) of a step class to files of the same repo (rule 4 of
v2tools/state.py). Every declared parent is classified as "resolved" (in-repo file found), "external" (outside the repo's own
namespaces: vendor, framework, standard library) or "unresolved" (inside a repo namespace but no file). The gate
requires "unresolved" == 0, so a wrong notion of "in repo" cannot hide missing parents."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .outline import Outline


@dataclass(frozen=True)
class Parent:
    relation: str  # "extends" | "implements" | "trait"
    written: str  # the name as written in the source
    fqn: str
    status: str  # "resolved" | "external" | "unresolved"
    path: str | None


class Resolver:
    def __init__(self, language: str, tree: list[str], composer_json: str | None = None):
        self.language = language
        self.tree = set(tree)
        self.prefixes: list[tuple[str, str]] = []
        if language == "php":
            if composer_json is None:  # without it every parent would look external and the gate would pass
                raise ValueError("a PHP repo needs its composer.json to resolve parents")
            data = json.loads(composer_json)
            for section in ("autoload", "autoload-dev"):
                for prefix, dirs in (data.get(section, {}).get("psr-4") or {}).items():
                    for d in ([dirs] if isinstance(dirs, str) else dirs):
                        self.prefixes.append((prefix, d.rstrip("/") + "/" if d else ""))
            self.prefixes.sort(key=lambda p: -len(p[0]))
        elif language == "java":
            self.by_suffix: dict[str, str] = {}
            self.packages: set[str] = set()
            for p in self.tree:
                if p.endswith(".java"):
                    parts = p[:-5].split("/")
                    for i in range(len(parts)):
                        self.by_suffix.setdefault("/".join(parts[i:]), p)
                        self.packages.add("/".join(parts[i:-1]))
        elif language == "csharp":
            self.by_name: dict[str, list[str]] = {}
            for p in self.tree:
                if p.endswith(".cs"):
                    self.by_name.setdefault(p.rsplit("/", 1)[-1][:-3], []).append(p)

    def resolve(self, o: Outline, relation: str, written: str) -> Parent:
        name = written.split("<", 1)[0].strip()  # drop Java/C# generic arguments
        if self.language == "php":
            if name.startswith("\\"):
                fqn = name[1:]
            else:
                head, _, rest = name.partition("\\")
                fqn = (o.imports[head] + ("\\" + rest if rest else "")) if head in o.imports else \
                    (o.namespace + "\\" + name if o.namespace else name)
            for prefix, d in self.prefixes:
                if fqn.startswith(prefix):
                    path = d + fqn[len(prefix):].replace("\\", "/") + ".php"
                    return Parent(relation, written, fqn, "resolved" if path in self.tree else "unresolved",
                                  path if path in self.tree else None)
            return Parent(relation, written, fqn, "external", None)
        if self.language == "java":
            if "." in name:
                cands = [name]
            elif name in o.imports:
                cands = [o.imports[name]]
            else:
                cands = [f"{o.namespace}.{name}" if o.namespace else name]
            for fqn in cands:
                path = self.by_suffix.get(fqn.replace(".", "/"))
                if path:
                    return Parent(relation, written, fqn, "resolved", path)
            pkg = cands[0].rsplit(".", 1)[0].replace(".", "/") if "." in cands[0] else ""
            imported = name in o.imports or "." in name
            # an explicitly imported (or fully qualified) class from a repo package must exist. An unqualified name
            # that is not imported and has no file in the class's own package comes from java.lang or a wildcard
            # import: external
            status = "unresolved" if imported and pkg in self.packages else "external"
            return Parent(relation, written, cands[0], status, None)
        paths = self.by_name.get(name.rsplit(".", 1)[-1], [])
        if len(paths) == 1:
            return Parent(relation, written, name, "resolved", paths[0])
        return Parent(relation, written, name, "unresolved" if len(paths) > 1 else "external", None)
