"""Overlay manifest contract shared with scripts/write_overlays.py, which writes it.

<root>/<run>/manifest.jsonl, one line per (record, condition):
  {record_id, condition, dir, base, base_files: {relpath: sha256}, files: {relpath: sha256}, suite, target,
   scenario_name, t_vis, stats, extract_status, insert_status, skip}
- base: <run>/<slug>/base, the ER tree (v2tools/ertree.py: HEAD minus the record's answer and future code), shared
  by all conditions of the record;
- dir: <run>/<slug>/<condition>, FULL contents of the target files after insertion (empty for neg);
- skip != null: no directory, the runner does not run it.
<root>/<run>/MANIFEST_SHA256 holds "sha256  manifest.jsonl". Paths in the manifest are relative to <root>.
Every tree file is a .php file under src/Sylius/Behat/; condition files are a subset of the record's CREATE target paths.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class ManifestError(Exception):
    pass


CONDITION = re.compile(r"^(neg|gold|stub|[a-z0-9-]+\.(s\d+|base)\.[a-z0-9_]+(\.[a-z0-9_]+)?)$")
TREE_PATH = re.compile(r"^src/Sylius/Behat/[A-Za-z0-9_/]+\.php$")


@dataclass(frozen=True)
class Entry:
    record_id: str
    condition: str
    dir: str | None
    base: str | None
    base_files: dict[str, str]
    files: dict[str, str]
    skip: str | None
    line: str  # raw manifest line, for overlay_sha256

    @property
    def overlay_sha256(self) -> str:
        return hashlib.sha256(self.line.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(run_dir: Path) -> list[Entry]:
    manifest = run_dir / "manifest.jsonl"
    want = (run_dir / "MANIFEST_SHA256").read_text(encoding="utf-8").split()[0]
    if sha256_file(manifest) != want:
        raise ManifestError(f"{manifest}: sha256 does not match MANIFEST_SHA256")
    entries, seen = [], set()
    # split("\n"), not splitlines(): U+2028/U+2029/U+0085 survive json.dumps(ensure_ascii=False) inside strings
    for line in manifest.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        o = json.loads(line)
        key = (o["record_id"], o["condition"])
        if key in seen:
            raise ManifestError(f"duplicate manifest line for {key}")
        seen.add(key)
        entries.append(Entry(o["record_id"], o["condition"], o.get("dir"), o.get("base"), o.get("base_files") or {},
                             o.get("files") or {}, o.get("skip"), line))
    return entries


def _safe_rel(rel: str) -> str:
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ManifestError(f"unsafe path in manifest: {rel!r}")
    return rel


def _check_tree(root: Path, rel_dir: str | None, want: dict[str, str], what: str) -> None:
    for rel in want:
        _safe_rel(rel)
    if rel_dir is None:
        if want:
            raise ManifestError(f"{what}: files listed but no directory")
        return
    d = root / _safe_rel(rel_dir)
    have = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()} if d.is_dir() else set()
    if have != set(want):
        raise ManifestError(f"{what}: files on disk differ from manifest "
                            f"(extra {sorted(have - set(want))[:3]}, missing {sorted(set(want) - have)[:3]})")
    for rel, sha in want.items():
        if sha256_file(d / rel) != sha:
            raise ManifestError(f"{what}: sha256 mismatch for {rel}")


def verify_entry(root: Path, e: Entry, record: dict | None = None) -> None:
    """Raise ManifestError unless base and condition trees are exactly the files and hashes the manifest lists, every
    file is a .php file under src/Sylius/Behat/, and (given the record) condition files are CREATE target paths."""
    if not CONDITION.match(e.condition):
        raise ManifestError(f"bad condition name {e.condition!r}")
    if e.skip is not None:
        return
    if e.base is None or not e.base_files:
        raise ManifestError(f"{e.record_id} {e.condition}: no base tree")
    for rel in [*e.base_files, *e.files]:
        if not TREE_PATH.match(rel):
            raise ManifestError(f"{e.condition}: {rel!r} is not a .php file under src/Sylius/Behat/")
    if record is not None:
        targets = {c["path"] for c in record["create"]}
        if not set(e.files) <= targets:
            raise ManifestError(f"{e.condition}: files outside the CREATE targets: {sorted(set(e.files) - targets)[:3]}")
    _check_tree(root, e.base, e.base_files, f"{e.condition} base")
    _check_tree(root, e.dir, e.files, e.condition)


# ---------- tamper scan (flags only; flagged outputs are still run and scored) ----------

TAMPER_PATTERNS = {  # flag-only (threat model: model code is not adversarial); flagged PASS outputs are checked by hand
    "harness": re.compile(r"ErTrace|Recorder::|BEHAT_PARAMS|\bER_[A-Z_]+|['\"]ER_"),
    "out_dir": re.compile(r"['\"]/out(/|['\"])"),
    "process_exit": re.compile(r"\b(exit|die)\s*[(;]"),
    "global_hook": re.compile(r"\b(register_shutdown_function|set_error_handler|set_exception_handler)\s*\(|"
                              r"__destruct|\$GLOBALS"),
    "behat_hook": re.compile(r"\b(Before|After)(Scenario|Step|Feature|Suite)\b"),
    "env": re.compile(r"\b(getenv|putenv)\s*\(|\$_(ENV|SERVER)\b|DATABASE_URL"),
    "shell": re.compile(r"\b(shell_exec|exec|system|proc_open|popen|passthru|pcntl_\w+|posix_kill)\s*\(|`"),
    "file_write": re.compile(r"\b(file_put_contents|fopen|fwrite|unlink|rename|rmdir|copy|symlink|link|mkdir|touch|"
                             r"chmod|tempnam)\s*\("),
    "reflection": re.compile(r"\bReflection\w*|Closure::bind|->bindTo\s*\(|\bdebug_backtrace\b"),
    "raw_db": re.compile(r"new\s+\\?PDO\b|\bmysqli\b|->(exec|executeStatement|executeQuery|query)\s*\(|"
                         r"\b(CREATE|DROP|ALTER)\s+(TRIGGER|TABLE|VIEW|PROCEDURE|FUNCTION|EVENT|USER|DATABASE)\b|"
                         r"\bTRUNCATE\s", re.I),
    "timing": re.compile(r"\b(sleep|usleep|time_nanosleep|set_time_limit|ini_set)\s*\("),
    # a function name in a variable, called; built by concatenation; called through ($f)(...); or built by one of these
    # functions and called at once (a built string that is not called, e.g. an exception message, is not flagged)
    "dynamic_call": re.compile(r"\$\w+\s*\(|\bcall_user_func(_array)?\s*\(|['\"]\s*\.\s*['\"]|"
                               r"\b(eval|create_function|assert)\s*\(|\b(include|require)(_once)?\b|"
                               r"\(\s*\$\w+\s*\)\s*\(|"
                               r"\b(implode|sprintf|str_replace|strrev|base64_decode|chr)\s*\(.*\)\s*\("),
}


def added_lines(before: str, after: str) -> list[str]:
    """Lines of `after` not in `before`, as a multiset difference (order of `after` kept)."""
    left = Counter(before.splitlines())
    out = []
    for line in after.splitlines():
        if left[line]:
            left[line] -= 1
        else:
            out.append(line)
    return out


def tamper_flags(root: Path, e: Entry, head_root: Path | None = None) -> list[str]:
    """Pattern names found in code the condition adds on top of the base tree (or of HEAD for files not in base)."""
    if e.skip is not None or e.dir is None:
        return []
    flags = set()
    for rel in e.files:
        after = (root / e.dir / rel).read_text(encoding="utf-8", errors="replace")
        if rel in e.base_files:
            before = (root / e.base / rel).read_text(encoding="utf-8", errors="replace")
        elif head_root is not None and (head_root / rel).is_file():
            before = (head_root / rel).read_text(encoding="utf-8", errors="replace")
        else:
            before = ""
        for line in added_lines(before, after):
            flags.update(name for name, rx in TAMPER_PATTERNS.items() if rx.search(line))
    return sorted(flags)


# ---------- writing a run (tests, layout conversion) ----------

def write_run(root: Path, run: str, items: list[dict]) -> Path:
    """items: {record, condition, slug, base: {rel: text}, files: {rel: text}, skip?}. Writes trees, manifest, sha."""
    run_dir = root / run
    lines = []
    for it in items:
        rec, slug = it["record"], it["slug"]
        base_rel, dir_rel = f"{run}/{slug}/base", f"{run}/{slug}/{it['condition']}"
        base_files, files = {}, {}
        if it.get("skip") is None:
            for tree, rel_dir, hashes in ((it["base"], base_rel, base_files), (it["files"], dir_rel, files)):
                for rel, text in tree.items():
                    p = root / rel_dir / _safe_rel(rel)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(text, encoding="utf-8")
                    hashes[rel] = sha256_file(p)
        lines.append(json.dumps({
            "record_id": rec["id"], "condition": it["condition"],
            "dir": dir_rel if it.get("skip") is None and files else None,
            "base": base_rel if it.get("skip") is None else None,
            "base_files": base_files, "files": files, "suite": rec["suite"],
            "target": f"{rec['feature_path']}:{rec['scenario_line']}", "scenario_name": rec["scenario_name"],
            "t_vis": rec.get("t_vis"), "stats": it.get("stats", {}), "extract_status": it.get("extract_status", "ok"),
            "insert_status": it.get("insert_status", "ok"), "skip": it.get("skip")}, ensure_ascii=False, sort_keys=True))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    (run_dir / "MANIFEST_SHA256").write_text(f"{sha256_file(run_dir / 'manifest.jsonl')}  manifest.jsonl\n",
                                             encoding="utf-8")
    return run_dir
