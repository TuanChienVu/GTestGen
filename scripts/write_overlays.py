"""Write the ER overlays of one run for the Sylius ER runner (the manifest contract of er/manifest.py; PHP
repositories only, as ER runs only Sylius: another --repo is refused):
<out>/<run>/<slug>/base/<relpath> holds the record's ER tree (every file the removal changes) and
<out>/<run>/<slug>/<condition>/<relpath> only the target files with the code inserted; both hold repository files only,
full content, paths relative to the repo root; the runner copies base, then the condition. <out>/<run>/manifest.jsonl
has one row per (record, condition), with sha256 of every file, and <out>/<run>/MANIFEST_SHA256 its digest.
Conditions: neg (the ER tree), gold (the gold answer inserted), stub (declared before any output: one NEW method per
CREATE annotation, stub1, stub2, ..., as many untyped parameters as the pattern has placeholders, empty body;
preexisting shells stay as in the ER tree) and, with --outputs, one per model run: evaluate.run_name(), i.e.
<model>.<seed|base>.<condition>[.<variant>]; a model run's manifest has no gold (the runner uses the frozen reference
of the controls run, --ref-run). A run whose output yields no usable block, or collides (a clash with a method of the
file, or the same method key twice in the output), gets a manifest row with `skip` and no directory; the runner writes
it an er_results row with er = 0 and er_status SKIP. --skip-gold lists records whose gold stops the controls run again
and again while the other records run (the runner's contract): their gold row gets skip "flaky_gold" and no
directory; the runner scores that gold row SKIP (er = 0) and marks every row of the record with ref_error "gold is SKIP
(flaky_gold)" and null trace fields, so the analysis leaves the record out of ER (its gold does not PASS).
Refused with exit 2 before any file is written: neither --controls nor
--outputs; an <out>/<run> that exists (a run directory is never written over); a missing input; a --repo absent from
--config or not PHP; a --full-root without the repo's clone at the records' commit; --ids unknown, repeated, without a
t_vis, or at more than one commit; with a dev id, a record of the repo without a t_vis (the composer reads them); an
output row lacking a field; a run name the harness's CONDITION refuses, or one run twice for a record."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2tools import symcache  # noqa: E402
from v2tools.argcheck import step_args  # noqa: E402
from v2tools.compose import Composer, gold_answer  # noqa: E402
from v2tools.controls import shown4  # noqa: E402
from v2tools.evaluate import Repo, future_keys, prepare, run_name  # noqa: E402
from v2tools.extract import extract  # noqa: E402
from v2tools.ertree import er_tree, phantoms, removal  # noqa: E402
from v2tools.insert import duplicate_keys, insert, shell_keys  # noqa: E402
from v2tools.state import RepoContext  # noqa: E402

# the condition names of the ER harness, a copy of er/manifest.py's CONDITION; one name it refuses makes the harness
# refuse the whole manifest at load
CONDITION = re.compile(r"^(neg|gold|stub|[a-z0-9-]+\.(s\d+|base)\.[a-z0-9_]+(\.[a-z0-9_]+)?)$")


def slug(record_id: str) -> str:
    _, feature, line, _ = record_id.split("|")
    return f"{Path(feature).stem}_{line}_{hashlib.sha1(record_id.encode()).hexdigest()[:6]}"


def stub_params(pattern: str, is_regex: bool) -> list[str]:
    """Parameter names of an ER stub: Turnip placeholders, or every capture group of a regex (named groups by name)."""
    if not is_regex:
        return re.findall(r":(\w+)", pattern)
    names, i, in_class = [], 0, False
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            i += 2
            continue
        if in_class:
            in_class = ch != "]"
        elif ch == "[":
            in_class = True
        elif ch == "(":
            m = re.match(r"\(\?P?<([A-Za-z_]\w*)>", pattern[i:])
            if m:
                names.append(m.group(1))
            elif not pattern.startswith("(?", i):
                names.append(f"arg{len(names) + 1}")
        i += 1
    return names


def er_stub(record: dict, fns: dict[str, dict], labels_row: dict, tree: dict[str, str], text_of) -> dict[str, str]:
    """The ER stub, in the format validated in Docker with the harness: for each CREATE annotation, in record
    order, a NEW method stub<n> (prefixed "er" while the name exists in the class) with one untyped parameter per
    placeholder or capture group, plus `$multiline = null` when a step bound to it has a table or doc string, and an
    empty body; the annotation line is copied (a docblock line stays a docblock). Preexisting methods are untouched."""
    args = step_args(record)
    multi = {d for ids, a in zip(labels_row["step_def_ids"], args) if a for d in ids}
    cdefs, out, n = set(record["create_def_ids"]), {}, 0
    for c in record["create"]:
        fn = fns[c["fn_id"]]
        text = tree.get(fn["path"]) or text_of(fn["path"])
        existing = {m.lower() for m in re.findall(r"function\s+(\w+)\s*\(", text)}
        existing |= {x.lower() for code in out.get(fn["path"], []) for x in re.findall(r"function\s+(\w+)\s*\(", code)}
        lines = text_of(fn["path"]).split("\n")
        for a in fn["annotations"]:
            if a["def_id"] not in cdefs:
                continue
            n += 1
            name = f"stub{n}"
            while name.lower() in existing:
                name = "er" + name
            line = lines[a["line"] - 1].strip()
            is_regex = bool(re.search(r"""(\(\s*['"]|@(Given|When|Then)\s+)/""", line))
            params = [f"${p}" for p in stub_params(a["pattern"], is_regex)] + (["$multiline = null"] if a["def_id"] in multi else [])
            head = line if line.startswith("#[") else f"/**\n * {line.lstrip('*').strip()}\n */"
            out.setdefault(fn["path"], []).append(f"{head}\npublic function {name}({', '.join(params)})\n{{\n}}")
    return {p: "\n\n".join(v) for p, v in out.items()}


def targets_after(record: dict, fns: dict, ctx, tree: dict[str, str], blocks: dict[str, str],
                  future: dict[str, frozenset] | None = None) -> tuple[dict[str, str], str]:
    """(full text of each target file with its block inserted into the ER tree, insert status). `future` = per path the
    keys of the record's FUTURE methods of that class (evaluate.future_keys): an output method with such a key replaces
    it, as in the scorers."""
    out, status = {}, "ok"
    for path, code in blocks.items():
        shells = shell_keys("php", [fns[c["fn_id"]]["code"] for c in record["create"] if c["preexisting"] and c["path"] == path])
        ins = insert("php", tree.get(path, ctx.text(path)), code, shells, (future or {}).get(path, frozenset()))
        status = "collision" if ins.collisions or duplicate_keys("php", code) else status
        out[path] = ins.text
    return out, status


def put(root: Path, rel: str, files: dict[str, str]) -> dict[str, str]:
    digests = {}
    for path, text in sorted(files.items()):
        target = root / rel / path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = text.encode("utf-8")
        target.write_bytes(data)
        digests[path] = hashlib.sha256(data).hexdigest()
    return digests


def jsonl(path: Path, ap: argparse.ArgumentParser) -> list[dict]:
    """Rows of a JSONL input, read by the shared reader (symcache.jsonl_rows, as score_run.load): U+2028, U+0085 or a
    lone
    "\r" of an output text stay in it; a blank line or a line that is not JSON is refused (exit 2)."""
    if not path.is_file():
        ap.error(f"missing input {path}")
    try:
        return symcache.jsonl_rows(path)
    except ValueError as e:
        ap.error(f"{path}: not JSONL ({e})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase2", type=Path, default=Path("out/phase2"))
    ap.add_argument("--full-root", type=Path, default=Path("repos-full"))
    ap.add_argument("--config", type=Path, default=Path("config/sources.json"))
    ap.add_argument("--repo", default="Sylius/Sylius")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run", required=True, help="e.g. controls, dev_pilot_v0, test")
    ap.add_argument("--ids", type=Path, help="record ids, one per line (default: test exec_core + dev Sylius api*)")
    ap.add_argument("--controls", action="store_true", help="write neg, gold and stub")
    ap.add_argument("--outputs", type=Path, nargs="*", default=[], help="generation JSONL files")
    ap.add_argument("--skip-gold", type=Path, help="record ids, one per line, whose gold is flaky (skip: flaky_gold)")
    ap.add_argument("--symtimes", type=Path, default=Path("out/phase3/symtimes.jsonl"), help="the sealed symbol times")
    args = ap.parse_args()
    if not args.controls and not args.outputs:  # else an empty manifest, exit 0
        ap.error("nothing to write: give --controls and/or --outputs")
    if (args.out / args.run).exists():  # a frozen run (hours of Docker ER) is never written over
        ap.error(f"{args.out / args.run} exists: choose a new --out or --run (a run directory is never written over)")
    inputs = [args.symtimes, args.config, args.phase2 / "forbidden.json"] + ([args.skip_gold] if args.skip_gold else [])
    for path in inputs:  # a clean refusal, not a traceback
        if not path.is_file():
            ap.error(f"missing input {path}")
    every = {}  # the repo's records of every partition: the composer's masked infrastructure needs the held-out ones
    for part in ("train", "dev", "test", "aux_test"):
        for r in jsonl(args.phase2 / "records" / f"{part}.jsonl", ap):
            if r["repo"] == args.repo:
                every[r["id"]] = r
    records = {i: r for i, r in every.items() if r["partition"] in ("test", "dev")}
    if args.ids:
        if not args.ids.is_file():
            ap.error(f"missing input {args.ids}")
        ids = [l.strip() for l in args.ids.read_text().splitlines() if l.strip()]
    else:
        ids = sorted(i for i, r in records.items() if r["exec_core"] or (r["partition"] == "dev" and r["suite"].startswith("api")))
    fns = {}
    for f in jsonl(args.phase2 / "catalog.jsonl", ap):
        if f["repo"] == args.repo:
            fns[f["fn_id"]] = f
    labels = {}
    for lab in jsonl(args.phase2 / "labels.jsonl", ap):
        if lab["repo"] == args.repo:
            labels[f"{args.repo}|{lab['feature_path']}|{lab['scenario_line']}|{lab['suite']}"] = lab
    src = next((s for s in json.loads(args.config.read_text()) if s["repo"] == args.repo), None)
    if src is None:  # a clean refusal, not a StopIteration traceback
        ap.error(f"--repo {args.repo} is not in {args.config}")
    if src["language"] != "php":  # the ER runner is Sylius only, so the tree and insertion below are PHP
        ap.error(f"--repo {args.repo} is {src['language']}: ER overlays are written for PHP repositories only")
    unknown = [i for i in ids if i not in records]  # refuse, never a traceback
    if not ids or unknown:
        ap.error("--ids: no record id" if not ids else
                 f"--ids: {len(unknown)} ids are not test/dev records of {args.repo}, e.g. {unknown[0]}")
    if len(set(ids)) != len(ids):  # the harness refuses a (record, condition) twice
        ap.error(f"--ids: {len(ids) - len(set(ids))} repeated ids, e.g. {next(i for i in ids if ids.count(i) > 1)}")
    untimed = [i for i in ids if not isinstance(records[i].get("t_vis"), int)]  # as score_run's timed()
    if untimed:
        ap.error(f"--ids: {len(untimed)} records have no t_vis, e.g. {untimed[0]}")
    commits = sorted({records[i]["commit"] for i in ids})  # RepoContext reads the repo at one commit
    if len(commits) > 1:
        ap.error(f"--ids: the records are at {len(commits)} commits ({commits[0]}, {commits[1]}, …); a run reads one")
    dev = any(records[i]["partition"] == "dev" for i in ids)
    untimed = sorted(i for i, x in every.items() if not isinstance(x.get("t_vis"), int)) if dev else []
    if untimed:  # the composer places every record of the repo at its t_vis (else a KeyError)
        ap.error(f"{args.repo}: {len(untimed)} records have no t_vis, e.g. {untimed[0]} (3A's composer, for a dev id)")
    missing_labels = [i for i in ids if i not in labels]
    if missing_labels:
        ap.error(f"labels.jsonl has no row for {len(missing_labels)} ids, e.g. {missing_labels[0]}")
    sym, git_dir = symcache.load(args.symtimes).get(args.repo, {}), args.full_root / f"{src['dir']}.git"
    try:
        ctx = RepoContext(args.repo, "php", git_dir, commits[0], fns, symtimes=sym)
    except subprocess.CalledProcessError as e:  # no clone there, or not that commit: a clean refusal
        ap.error(f"--full-root: {git_dir} cannot be read at {commits[0]} ({e.stderr.strip()})")
    try:
        ghost = phantoms("php", fns, ctx.text)
    except ValueError as e:  # the catalog disagrees with the HEAD files, or a row has no annotation; the repo
        ap.error(f"{args.repo}: {str(e).replace(f' (repo {args.repo})', '')}")  # the repo named once
    forbidden = json.loads((args.phase2 / "forbidden.json").read_text(encoding="utf-8"))
    repo = Repo(args.repo, "php", src["styles"], fns, ctx, ghost, {}, forbidden)  # for future_keys only
    comp = Composer({args.repo: ctx}, {args.repo: fns}, forbidden, every) if dev else None
    outputs = {}
    for f in args.outputs:
        for o in jsonl(f, ap):
            lack = {"record_id", "model", "seed", "condition", "text"} - set(o)  # a clean refusal, not a KeyError
            if lack:
                ap.error(f"{f}: an output row lacks {sorted(lack)}")
            if o["record_id"] in records:
                outputs.setdefault(o["record_id"], []).append(o)
    for rid in ids:  # the names the harness would refuse at load, refused before writing
        names = Counter(run_name(o) for o in outputs.get(rid, []))
        bad = sorted(n for n in names if not CONDITION.match(n))
        if bad:
            ap.error(f"{rid}: run name {bad[0]!r} does not match the harness's CONDITION {CONDITION.pattern}")
        twice = sorted(n for n, k in names.items() if k > 1)
        if twice:
            ap.error(f"{rid}: run {twice[0]} twice in --outputs")
    flaky = {l.strip() for l in args.skip_gold.read_text().splitlines() if l.strip()} if args.skip_gold else set()
    if flaky - set(ids):
        ap.error(f"--skip-gold: {len(flaky - set(ids))} ids are not in this run, e.g. {min(flaky - set(ids))}")
    rows = []
    for rid in ids:
        r = records[rid]
        rem = removal(r, fns, ghost)
        tree = er_tree(rem, ctx.text)
        base = f"{args.run}/{slug(rid)}/base"
        base_files = put(args.out, base, tree)
        stats = {"answer_fns_removed": rem.answer_fns, "answer_annotations_removed": rem.answer_annotations,
                 "future_fns_removed": rem.future_fns, "phantom_fns": rem.phantom_fns}
        targets = list(dict.fromkeys(c["path"] for c in r["create"]))
        masked = shown4(comp, r).masked if comp is not None and r["partition"] == "dev" else frozenset()
        p = prepare(r, repo, labels[rid], masked)
        future = {path: future_keys(p, path) for path in targets}
        runs = []
        if args.controls:
            runs += [("neg", None, "ok"), ("gold", extract(gold_answer(r, fns), targets).by_path(), "ok"),
                     ("stub", er_stub(r, fns, labels[rid], tree, ctx.text), "ok")]
        for o in outputs.get(rid, []):
            e = extract(o["text"], targets)
            runs.append((run_name(o), e.by_path(), e.status))
        for condition, blocks, ext in runs:
            row = {"record_id": rid, "condition": condition, "suite": r["suite"], "scenario_name": r["scenario_name"],
                   "target": f"{r['feature_path']}:{r['scenario_line']}", "t_vis": r["t_vis"], "stats": stats,
                   "base": base, "base_files": base_files, "dir": None, "files": {}, "extract_status": ext,
                   "insert_status": "ok", "skip": None}
            if condition == "gold" and rid in flaky:
                row["skip"] = "flaky_gold"
            elif blocks is not None and not blocks:
                row.update(insert_status="empty", skip="no_usable_block")
            elif blocks is not None:
                files, status = targets_after(r, fns, ctx, tree, blocks, future)
                if status == "collision":
                    row.update(insert_status=status, skip="collision")
                else:
                    row["dir"] = f"{args.run}/{slug(rid)}/{condition}"
                    row["files"] = put(args.out, row["dir"], files)
            rows.append(row)
    manifest = args.out / args.run / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (args.out / args.run / "MANIFEST_SHA256").write_text(f"{digest}  manifest.jsonl\n")
    print(json.dumps({"rows": len(rows), "records": len(ids), "skipped": sum(x["skip"] is not None for x in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
