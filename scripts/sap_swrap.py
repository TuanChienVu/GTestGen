"""S-wrap (declared 05/10/2026, before the SAP lock): a DESCRIPTIVE sensitivity analysis of RQ1 to outputs that wrap
their step definitions in a namespace/class shell. The main measure keeps the pre-registered scoring rule of 3C (such an
output is inserted as written and fails condition 1, so BV = CA = 0, in both scorers and in ER); S-wrap reports, beside
RQ1, what the anchor model's base runs (Zero-shot, RAG-only) score once the shell is removed. It is outside the Holm
families, sap_run never runs it, it has no p-value, and it changes neither the "supported" rule nor any conclusion on an
H. "Probe rows are never in the SAP" still holds for the main measure: S-wrap is the one declared, descriptive exception
(its rows are scored by 3C's probe mode in score_run, since an unwrapped row no longer hashes to its manifest line).

unwrap: reads generation outputs that pass the generation contract (each file's sha256 is the outputs_sha256 of its
manifest lines, each row an `ok` line whose line_sha256 is the sha256 of the row's raw bytes, the coverage hashes that
manifest and says the generation is complete), and applies ONE fixed rule to every FILE segment of a fence (§1.4 of the
design specification: a FILE line opens a segment; a `<?php` line before the first FILE line counts as blank; a
fence-open line right before a FILE line stays where it is) whose path is a CREATE target of the record (PHP and Java,
the languages of RQ1's test repositories): code that parses without error into an optional `<?php`, an optional
namespace (statement or braced) or package, import lines (`use`, `import`), comments and EXACTLY ONE type declaration
(class, trait, interface, enum, record) becomes its import lines, a blank line and the text inside the type's braces;
anything else is left as it is. Writes <out>/swrap_outputs.jsonl (the anchor model's base-condition rows of the
partition's records, model "probe", the original in swrap_model) and <out>/SWRAP_META.json (the sha256 of this script,
of every input and of the output; per model|condition|repo the rows, the rows with a wrapped target segment and the rows
unwrapped: the "tỷ lệ bọc vỏ" column; the keys of the unwrapped rows).
Then: score_run.py --partition <p> --outputs <out>/swrap_outputs.jsonl, with the locked tool commit and controls.
report: the official score rows, the S-wrap score rows (3C's probe mode in score_run, from the swrap_outputs.jsonl of
SWRAP_META) and the inputs sap_run reads for the CA denominator; for each RQ1 comparison of Full-aware with a base
condition and for CA and BV: the estimate, the 95% cluster-bootstrap interval and n, official and with the base rows
unwrapped, computed as sap_run computes RQ1 and the BV beside it: the rows of both sides go through check_complete with
the design records, the --accepted-missing pairs and the locked flags (P4: an accepted pair scores CA = BV = 0, an
unaccepted gap stops the report), then the SAP's denominators (usable, usable_records, compose_dropped), seed pairing
and bootstrap with sap_run's seeds (stats.compare without a p-value), so the official column IS sap_run's, interval
included; and each base condition's rate (record level) both ways. Each SCORES_META must hash its scores; the S-wrap one
must have the official ones' code_sha256, a3_sha256, partition and controls (the ca_ref, exclusions and future_methods
files it read), --controls-dir must hold those exclusions, and --accepted-missing must be the file the official scoring
accepted. A row S-wrap left unchanged must score as the official row (else the scoring used other code or controls).
Writes SWRAP.json.
Exit 3 on any refused input; nothing is written then."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import tree_sitter as ts

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2tools.sap import (SapInputError, check_complete, compare, compose_dropped, design_records,  # noqa: E402
                         jsonl_bytes, jsonl_rows, locked_flags, nan_none, per_seed_pairs, pooled, seed_values, usable,
                         usable_records)

OPEN = re.compile(r"^\s*(`{3,})[^`]*$")  # design specification §1.4 (as 3C's extractor; 3D imports no 3C module)
CLOSE = re.compile(r"^\s*(`{3,})\s*$")
FILE_LINE = re.compile(r"^\s*(?://|#)\s*FILE:\s*(\S+)\s*$")
PHP_TAG_LINE = re.compile(r"^\s*<\?php\s*$", re.I)
PHP_OPEN = re.compile(r"\A\s*<\?php", re.I)
TYPES = {"class_declaration", "trait_declaration", "interface_declaration", "enum_declaration", "record_declaration"}
SHELLS = {"namespace_definition", "package_declaration"}
IMPORTS = {"namespace_use_declaration", "import_declaration"}
ALLOWED = {"php": {"php_tag", "comment", "namespace_definition", "text_interpolation", "text"},
           "java": {"package_declaration", "line_comment", "block_comment"}}
KEY = ("model", "condition", "seed", "variant", "record_id", "fmt")  # the generation run's manifest key
METRICS = ("ca", "bv")


def parser(language: str) -> ts.Parser:
    if language == "php":
        import tree_sitter_php as m
        return ts.Parser(ts.Language(m.language_php()))
    import tree_sitter_java as m
    return ts.Parser(ts.Language(m.language()))


def _parse(language: str, code: str):
    src = (code if language != "php" or PHP_OPEN.match(code) else "<?php\n" + code).encode("utf-8")
    return src, parser(language).parse(src).root_node


def _top(language: str, root) -> list:
    """The top-level nodes of a file, a braced PHP namespace's own included."""
    out = []
    for ch in root.named_children:
        out.append(ch)
        body = ch.child_by_field_name("body") if language == "php" and ch.type == "namespace_definition" else None
        if body is not None:
            out += body.named_children
    return out


def wrapped(language: str, code: str) -> bool:
    """A namespace, package or type declaration at the top of the code (parse errors or not)."""
    return any(n.type in TYPES | SHELLS for n in _top(language, _parse(language, code)[1]))


def unwrap_code(language: str, code: str) -> str | None:
    """The rule: the import lines, a blank line and the text inside the one type's braces; None when it does not apply."""
    src, root = _parse(language, code)
    if root.has_error:
        return None
    imports, types = [], []
    for n in _top(language, root):
        if n.type in TYPES:
            types.append(n)
        elif n.type in IMPORTS:
            imports.append(src[n.start_byte:n.end_byte].decode("utf-8"))
        elif n.type not in ALLOWED[language]:
            return None
    body = types[0].child_by_field_name("body") if len(types) == 1 else None
    if body is None:
        return None
    inner = src[body.start_byte + 1:body.end_byte - 1].decode("utf-8").split("\n")
    while inner and not inner[0].strip():  # the blank (or indent-only) lines next to the braces
        inner.pop(0)
    while inner and not inner[-1].strip():
        inner.pop()
    return "\n".join(imports) + ("\n\n" if imports else "") + "\n".join(inner)


def _closes(line: str, ticks: str) -> bool:
    m = CLOSE.match(line)
    return m is not None and len(m.group(1)) >= len(ticks)


def unwrap_text(text: str, language: str, targets: list[str]) -> tuple[str, int, int]:
    """(text with the rule applied to every target FILE segment, segments wrapped, segments unwrapped)."""
    goal = {t.removeprefix("./") for t in targets}
    lines, out, i, n_wrapped, n_unwrapped = text.split("\n"), [], 0, 0, 0
    while i < len(lines):
        m = OPEN.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not _closes(lines[j], m.group(1)):
            j += 1
        body = lines[i + 1:j]
        starts = [k for k, x in enumerate(body) if FILE_LINE.match(x)]
        head = [x for x in body[:starts[0]] if x.strip() and not PHP_TAG_LINE.match(x)] if starts else ["-"]
        if language in ALLOWED and not head:
            new = body[:starts[0]]
            for a, b in zip(starts, starts[1:] + [len(body)]):
                seg, tail = body[a:b], []
                if b < len(body) and len(seg) > 1 and OPEN.match(seg[-1]):  # a reopen line before the next FILE
                    tail.insert(0, seg.pop())
                while len(seg) > 1 and not seg[-1].strip():
                    tail.insert(0, seg.pop())
                if FILE_LINE.match(seg[0]).group(1).removeprefix("./") in goal and wrapped(language, "\n".join(seg[1:])):
                    n_wrapped += 1
                    code = unwrap_code(language, "\n".join(seg[1:]))
                    if code is not None:
                        n_unwrapped += 1
                        seg = seg[:1] + code.split("\n")
                new += seg + tail
            body = new
        out.append(lines[i])
        out.extend(body)
        if j < len(lines):
            out.append(lines[j])
        i = j + 1
    return "\n".join(out), n_wrapped, n_unwrapped


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_rows(outputs: list[Path], manifest: Path, coverage: Path) -> tuple[list[dict], dict]:
    """The rows of the output files, each checked against the generation run's manifest, and {path: sha256}."""
    mbytes = manifest.read_bytes()
    cov = json.loads(coverage.read_text(encoding="utf-8"))
    if not isinstance(cov, dict) or cov.get("manifest_sha256") != sha256(mbytes):
        raise SapInputError(f"{coverage}: manifest_sha256 is not the sha256 of {manifest}")
    if cov.get("status") != "PASS" and cov.get("status_except_missing") != "PASS":
        raise SapInputError(f"{coverage}: status {cov.get('status')}: the generation is not complete")
    plan = {tuple(x.get(k) for k in KEY): x for x in jsonl_bytes(mbytes, manifest)}
    rows, sources = [], {}
    for f in outputs:
        data = f.read_bytes()
        sources[str(f)] = sha256(data)
        raw = data.split(b"\n")
        if raw and raw[-1] == b"":
            raw.pop()
        for row, line in zip(jsonl_bytes(data, f), raw):
            k = tuple(row.get(x) for x in KEY)
            m = plan.get(k)
            if m is None or m.get("status") != "ok":
                raise SapInputError(f"{f}: row {k} is not an ok line of {manifest}")
            if m.get("outputs_sha256") != sources[str(f)]:
                raise SapInputError(f"{f}: its sha256 is not the outputs_sha256 of its manifest lines")
            if m.get("line_sha256") != sha256(line):
                raise SapInputError(f"{f}: row {k}: the sha256 of its raw bytes is not its line_sha256")
            rows.append(row)
    return rows, sources


def unwrap(args) -> int:
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    rows, sources = manifest_rows(args.outputs, args.manifest, args.coverage)
    records = {r["id"]: r for r in jsonl_rows(args.records)}
    wrap: dict = {}
    out, unwrapped, other = [], [], 0
    for r in rows:
        rec = records.get(r["record_id"])
        if rec is None:  # a row of the partition's test/aux_test partner (one generation workdir holds both)
            other += 1
            continue
        new, n_wrapped, n_unwrapped = unwrap_text(r["text"], rec["language"], [c["path"] for c in rec["create"]])
        c = wrap.setdefault(f"{r['model']}|{r['condition']}|{rec['repo']}", Counter())
        c.update({"rows": 1, "wrapped": int(n_wrapped > 0), "unwrapped": int(n_unwrapped > 0)})
        if r["model"] == cfg["anchor_model"] and r["condition"] in cfg["base_conditions"]:
            out.append({**r, "model": "probe", "swrap_model": r["model"], "text": new})
            if n_unwrapped:
                unwrapped.append([r["record_id"], r["condition"], r["seed"], r.get("variant") or "main"])
    args.out.mkdir(parents=True)
    data = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out).encode("utf-8")
    (args.out / "swrap_outputs.jsonl").write_bytes(data)
    meta = {"name": "S-wrap", "descriptive_only": True, "script_sha256": sha256(Path(__file__).read_bytes()),
            "config_sha256": sha256(args.config.read_bytes()), "sources": sources,
            "manifest_sha256": sha256(args.manifest.read_bytes()), "coverage_sha256": sha256(args.coverage.read_bytes()),
            "records_sha256": sha256(args.records.read_bytes()), "out_sha256": sha256(data), "rows_written": len(out),
            "skipped_other_partition": other, "wrap": {k: {x: v[x] for x in ("rows", "wrapped", "unwrapped")}
                                                       for k, v in sorted(wrap.items())},
            "unwrapped": unwrapped}
    (args.out / "SWRAP_META.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("rows_written", "skipped_other_partition", "out_sha256")}))
    return 0


CONTROL_INPUTS = ("ca_ref.jsonl", "exclusions.jsonl", "future_methods.jsonl")  # what score_run --outputs reads of controls


def scores_meta(scores: Path) -> dict:
    """The SCORES_META.json next to a scores.jsonl, which must hash that file (scores_sha256)."""
    meta = json.loads((scores.parent / "SCORES_META.json").read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or meta.get("scores_sha256") != sha256(scores.read_bytes()):
        raise SapInputError(f"{scores}: the scores_sha256 of its SCORES_META.json is not its sha256")
    return meta


def controls_read(meta: dict) -> dict:
    """{file name: sha256} of the controls files a scoring read (its SCORES_META inputs)."""
    return {Path(p).name: v for p, v in (meta.get("inputs") or {}).items() if Path(p).name in CONTROL_INPUTS}


def report(args) -> int:
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    anchor, bases = cfg["anchor_model"], cfg["base_conditions"]
    meta = json.loads(args.swrap_meta.read_text(encoding="utf-8"))
    # the S-wrap scores come from the swrap_outputs.jsonl of SWRAP_META (S-wrap as declared in the analysis plan at
    # 56b5b45, 05/10/2026), and were scored with the official scoring's code, 3A files and controls; each SCORES_META
    # hashes its scores
    metas = [scores_meta(f) for f in args.scores]
    scored = scores_meta(args.swrap_scores)
    if meta.get("out_sha256") not in (scored.get("inputs") or {}).values():
        raise SapInputError(f"{args.swrap_scores} was not scored from the swrap_outputs.jsonl of {args.swrap_meta} "
                            "(SWRAP_META out_sha256 is not among the inputs of its SCORES_META.json)")
    for k in ("code_sha256", "a3_sha256", "partition"):
        if any(m.get(k) != scored.get(k) for m in metas):
            raise SapInputError(f"the S-wrap scores were scored with another {k} than the official scores")
    ctl = controls_read(scored)
    if set(ctl) != set(CONTROL_INPUTS) or any(controls_read(m) != ctl for m in metas):
        raise SapInputError("the S-wrap scores were scored with other controls than the official scores "
                            f"({', '.join(CONTROL_INPUTS)} of their SCORES_META inputs)")
    if ctl["exclusions.jsonl"] != sha256((args.controls_dir / "exclusions.jsonl").read_bytes()):
        raise SapInputError(f"--controls-dir {args.controls_dir} is not the controls the scores were scored with")
    # the accepted-missing pairs, filled as sap_run fills them (check_complete, P4); the file must be the
    # one the official scoring accepted (its SCORES_META accepted_missing_sha256)
    took = {m.get("accepted_missing_sha256") for m in metas}
    given = None if args.accepted_missing is None else sha256(args.accepted_missing.read_bytes())
    if took != {given}:
        raise SapInputError(f"--accepted-missing ({given}) is not the file the official scores accepted "
                            f"(SCORES_META accepted_missing_sha256 {sorted(map(str, took))})")
    accepted = {(e["record_id"], e["model"], e["condition"], e["seed"])
                for e in (jsonl_rows(args.accepted_missing) if args.accepted_missing else [])
                if (e.get("variant") or "main") == "main"}
    official = [r for f in args.scores for r in jsonl_rows(f)]
    swrap = jsonl_rows(args.swrap_scores)
    if any(r.get("model") != "probe" for r in swrap):
        raise SapInputError(f"{args.swrap_scores}: S-wrap score rows are 3C probe rows (model \"probe\")")
    key = lambda r: (r["record_id"], r["condition"], r["seed"], r.get("variant") or "main")
    base = {key(r): r for r in official if r["model"] == anchor and r["condition"] in bases}
    sw = {key(r): r for r in swrap}
    if len(sw) != len(swrap) or set(sw) != set(base):
        raise SapInputError(f"the S-wrap score rows are not the official base rows of {anchor}: "
                            f"{len(set(base) - set(sw))} missing, {len(set(sw) - set(base))} extra, "
                            f"{len(swrap) - len(sw)} twice")
    done = {tuple(k) for k in meta.get("unwrapped", [])}
    same = ("ca", "bv", "ca_defined", "excl_bv", "excl_ca")
    for k, r in sorted(sw.items()):
        if k not in done and any(r.get(x) != base[k].get(x) for x in same):
            raise SapInputError(f"row {k} was not unwrapped but scores otherwise than the official row: the S-wrap "
                                "scoring used other code or other controls")
    controls = jsonl_rows(args.controls_dir / "controls.jsonl")
    exclusions = jsonl_rows(args.controls_dir / "exclusions.jsonl")
    compose = jsonl_rows(args.compose_excluded)
    ids = {m: design_records(controls, compose, m) for m in cfg["seeds"]}
    flags = locked_flags(controls, exclusions)
    after = [r for r in official if key(r) not in base or r["model"] != anchor] + [{**r, "model": anchor} for r in swrap]
    official, info = check_complete(official, ids, cfg, accepted, flags)  # as sap_run (P4): unaccepted gaps stop it
    after, _ = check_complete(after, ids, cfg, accepted, flags)
    keep = usable_records(controls, exclusions, compose, anchor)
    dropped = compose_dropped(compose, anchor)
    excluded = {r["record_id"] for r in official} - keep  # as sap_run, after the filled rows
    f = list(cfg["families"]).index("RQ1")
    comparisons, rates = {}, {}
    for i, (a, b) in enumerate(cfg["families"]["RQ1"]):
        if b not in bases:
            continue
        comparisons[f"RQ1:{a}_vs_{b}"] = {}
        for m, off in (("ca", 0), ("bv", 50)):  # sap_run's seeds (confirmatory: CA, and BV beside it)
            keep_m = usable(m, excluded, dropped=dropped)
            res = {}
            for side, rs in (("official", official), ("swrap", after)):
                c = compare(pooled(per_seed_pairs(seed_values(rs, m, keep_m), (anchor, a, "main"), (anchor, b, "main"))),
                            cfg["reps_bootstrap"], 0, cfg["seed"] + 10000 * f + 100 * i + off, with_p=False)
                res[side] = {"n_scenarios": c.n_scenarios, "n_clusters": c.n_clusters, "estimate": nan_none(c.estimate),
                             "ci95": [nan_none(x) for x in c.ci]}
                ours = [r for r in rs if r["model"] == anchor and r["condition"] == b and keep_m(r)]
                rates.setdefault(b, {}).setdefault(m, {})[side] = [sum(int(bool(r[m])) for r in ours), len(ours)]
            comparisons[f"RQ1:{a}_vs_{b}"][m] = res
    out = {"name": "S-wrap", "descriptive_only": True,
           "note": "descriptive sensitivity (05/10): outside the Holm families, not run by sap_run, no p-value, "
                   "no change to the supported rule or to any H; the confirmatory RQ1 is sap_run's",
           "wrap": meta.get("wrap"), "unwrapped_rows": len(done), "filled_missing": info["missing_filled"],
           "inputs": {str(p): sha256(p.read_bytes()) for p in (*args.scores, args.swrap_scores, args.swrap_meta,
                                                               args.controls_dir / "controls.jsonl",
                                                               args.controls_dir / "exclusions.jsonl",
                                                               args.compose_excluded, args.config,
                                                               *([args.accepted_missing] if args.accepted_missing else []))},
           "comparisons": comparisons, "rates": rates}
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(comparisons))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("unwrap")
    u.add_argument("--config", type=Path, default=Path("config/sap.json"))
    u.add_argument("--outputs", type=Path, nargs="+", required=True, help="the generation run's outputs.jsonl files")
    u.add_argument("--manifest", type=Path, required=True)
    u.add_argument("--coverage", type=Path, required=True)
    u.add_argument("--records", type=Path, required=True, help="out/phase2/records/<partition>.jsonl")
    u.add_argument("--out", type=Path, required=True, help="a new directory")
    r = sub.add_parser("report")
    r.add_argument("--config", type=Path, default=Path("config/sap.json"))
    r.add_argument("--scores", type=Path, nargs="+", required=True, help="the official scores.jsonl of the partition")
    r.add_argument("--swrap-scores", type=Path, required=True, help="score_run's scores.jsonl of swrap_outputs.jsonl")
    r.add_argument("--swrap-meta", type=Path, required=True)
    r.add_argument("--controls-dir", type=Path, required=True)
    r.add_argument("--compose-excluded", type=Path, required=True)
    r.add_argument("--accepted-missing", type=Path, help="the accepted_missing.jsonl the official scores accepted")
    r.add_argument("--out", type=Path, required=True, help="SWRAP.json (new)")
    args = ap.parse_args()
    if args.out.exists():
        raise SapInputError(f"{args.out} exists (S-wrap never writes over a run)")
    return unwrap(args) if args.cmd == "unwrap" else report(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SapInputError, OSError, ValueError, KeyError) as e:  # a refused or malformed input: nothing written
        print(f"refusing: {e}", file=sys.stderr)
        raise SystemExit(3)
