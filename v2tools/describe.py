"""Descriptive measures of Phase 3 (§2.4, outside the Holm families) and the per-(record, run) score row that the SAP
reads: similarity to the gold answer, hallucinated members (used but not in V(r) and not defined by the output),
adherence to the target file's conventions, the dose of gold symbols shown in ④, and the record's subsets. Every input
field the row needs must be there: a missing one stops scoring (InputError), never a silent None or default."""
from __future__ import annotations

import hashlib
import re
from collections import Counter

from .compose import gold_answer
from .evaluate import CONDITIONS, Prepared, Score, body_open, ca, item_defs
from .extract import extract
from .insert import split_code
from .outline import parse
from .records import hidden_for
from .state import t_view
from .stepdefs import extract as extract_defs
from .stepdefs import file_imports
from .symbols import php_promoted, refs, scope

DAY = 86400
KNOWLEDGE = "Project knowledge:"  # heading of block ④ in compose.py


def tokens(code: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", code)


def rouge_l(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    rec, prec = lcs / len(a), lcs / len(b)
    return 2 * rec * prec / (rec + prec)


def similarity(p: Prepared, blocks: dict[str, str]) -> dict:
    gold = extract(gold_answer(p.record, p.repo.fns), p.targets).by_path()
    g = "\n".join(gold.get(t, "") for t in p.targets)
    o = "\n".join(blocks.get(t, "") for t in p.targets)
    return {"rouge_l": round(rouge_l(tokens(g), tokens(o)), 4), "exact": " ".join(g.split()) == " ".join(o.split())}


def naming(name: str) -> str:
    return "snake" if "_" in name.strip("_") else ("camel" if name[:1].islower() else "pascal")


def _majority(c: Counter):
    if not c:
        return None
    (a, na), *rest = c.most_common(2) + [(None, -1)]
    return a if na > rest[0][1] else None


def conventions(p: Prepared, s: Score, blocks: dict[str, str] | None = None) -> dict:
    """Share of generated step definitions that follow the majority annotation style, pattern syntax and method naming
    of their target file in the ER tree (None when the file has no majority). The items are the output's own (its
    `blocks`, own_items: before insertion, without the HEAD annotation lines a rewritten shell keeps; 26/09/2026);
    without `blocks`, the inserted items of `s` (as before)."""
    repo = p.repo
    got = {"conv_style": [], "conv_pattern": [], "conv_naming": []}
    items = [(q, k, x) for q, k, x, _ in own_items(repo.language, blocks)] if blocks is not None else s.items.values()
    for path, kind, text in items:
        base = p.er_files.get(path, repo.ctx.text(path)) or ""
        styles, kinds = Counter(), Counter()
        for st in repo.styles:
            for d in extract_defs(path, base, st, file_imports(repo.language, base)):
                styles[st] += 1
                kinds["regex" if d.kind.startswith("regex") else "placeholder"] += 1
        o = parse(repo.language, base) if repo.language != "ruby" else None
        names = Counter(naming(m.names[0]) for m in (o.members if o else ()) if m.kind == "method" and not m.names[0].startswith("__"))
        want = (_majority(styles), _majority(kinds), _majority(names))
        for st in repo.styles:
            try:
                found = extract_defs(path, text, st, own_imports(p, path, blocks or {}))
            except ValueError:  # code cut off (an output that hit the token cap): skipped, as in evaluate.score_blocks
                continue
            for d in found:
                if want[0]:
                    got["conv_style"].append(st == want[0])
                if want[1]:
                    got["conv_pattern"].append(("regex" if d.kind.startswith("regex") else "placeholder") == want[1])
        if kind == "method" and want[2]:
            m = next((it for it in split_code(repo.language, text)[1] if it.kind == "method"), None)
            if m is not None:
                got["conv_naming"].append(naming(m.name) == want[2])
    return {k: (round(sum(v) / len(v), 4) if v else None) for k, v in got.items()}


def member_names(language: str, text: str) -> set[str]:
    """Member names of the file's class; in PHP also the parameters a constructor written `__CONSTRUCT` in another case
    promotes (symbols.php_promoted; 3A's outline only knows `__construct`)."""
    o = parse(language, text)
    return ({n for m in o.members for n in m.names} | (set(php_promoted(text)) if language == "php" else set())
            if o else set())


def method_names(text: str) -> set[str]:
    """Casefolded names of the PHP methods and constructor of the file's class (PHP method names ignore case)."""
    o = parse("php", text)
    return {n.casefold() for m in o.members if m.kind in ("method", "constructor") for n in m.names} if o else set()


def own_members(language: str, s: Score, path: str) -> tuple[set[str], set[str]]:
    """(member names, casefolded method and constructor names) that the output's own items for `path` declare: every
    method and member item, one that replaces a future or shell method of the ER tree included (the output defines it,
    though the ER tree had it)."""
    prefix = "<?php\nclass __X {\n" if language == "php" else "class __X {\n"
    names, methods = set(), set()
    for q, kind, text in s.items.values():
        if q == path and kind in ("method", "member"):
            o = parse(language, prefix + text + "\n}\n")
            names |= member_names(language, prefix + text + "\n}\n")
            methods |= {n.casefold() for m in (o.members if o else ()) if m.kind in ("method", "constructor")
                        for n in m.names}
    return names, methods


def hallucinated(p: Prepared, s: Score, masked: frozenset = frozenset()) -> tuple[int, int]:
    """(member references of the generated code that are neither visible in V(r) nor defined by the output nor imported
    statically, member references); a class with an ancestor outside the repo is not judged. A name condition 6 resolves
    outside V(r) is not invented either, as symbols.unresolved: an explicit static import that is external or added by
    the output, a member of the root object, Lombok's `log`. V(r) is ④(a) as 3A shows it (state.t_view, rule 5 of
    v2tools/state.py): members in their pre-T form, nothing for a file that is new at t_vis, without the infrastructure
    3A masks for the record (`masked`, Composer.masked; train/dev only). PHP, as condition 6: a METHOD call is also not
    hallucinated when its casefold is that of a method or constructor V(r) shows or the output adds; a property name
    compares as written. The output defines every member its own items declare (own_members: a method replacing a future
    one included)."""
    repo, r = p.repo, p.record
    if repo.language == "ruby":
        return 0, 0
    gone = {c["fn_id"] for c in r["create"] if not c["preexisting"]} | hidden_for(r["partition"], repo.forbidden.get(r["repo"], {}))
    bad = used = 0
    for path, kind, text in s.items.values():
        if kind == "member":
            continue
        base = p.er_files.get(path, repo.ctx.text(path)) or ""
        sc = scope(repo.ctx, path, base, s.after.get(path, base), p.er_files)
        if sc is None or sc.open_all:
            continue
        view = t_view(repo.ctx, path, r["t_vis"], gone, masked)  # ④(a) as shown (rule 5); a new file shows nothing
        visible = {n for x in view or () for n in x.member.names}
        rf = refs(repo.language, text)
        mine, mine_methods = own_members(repo.language, s, path)
        new = (member_names(repo.language, s.after.get(path, base)) - member_names(repo.language, base)) | mine
        missing = (set(rf.members) - visible - new - set(sc.statics) - sc.open_statics - sc.root
                   - ({"log"} if sc.lombok_log else set()) - (set(rf.calls) if sc.open_calls else set()))
        if repo.language == "php":
            methods = {n.casefold() for x in view or () if x.member.kind in ("method", "constructor")
                       for n in x.member.names}
            methods |= (method_names(s.after.get(path, base)) - method_names(base)) | mine_methods
            missing -= {n for n in rf.calls if n.casefold() in methods}
        used += len(rf.members)
        bad += len(missing)
    return bad, used


def dose(p: Prepared, prompt_text: str) -> int:
    """Distinct project members used by the gold answer that appear as identifiers in block ④ of an F2 prompt."""
    g = p.gold()
    used = set()
    for ident, (path, kind, text) in g.items.items():
        used |= set(refs(p.repo.language, text).members)
    j = prompt_text.find(KNOWLEDGE)
    shown = set(re.findall(r"[A-Za-z_]\w*", prompt_text[j:])) if j >= 0 else set()
    return len(used & shown)


SUBSETS = ("post_cutoff_all", "time_coherent_30d", "residual_clean")
ER_FIELDS = ("er", "er_status", "trace_action", "trace_then", "er_trace", "n_then_evaluable")  # the ER harness's er_results
PROVENANCE = ("prompts_sha256", "prompt_sha256", "adapter_tensor_sha256", "leakcheck_receipt_sha256", "finish_reason",
              "decision_sha256", "gen_code_sha256", "line_sha256")  # the last three: copied from the generation manifest (v2gpu)
# (26/09/2026); line_sha256 = the manifest ok line's own line_sha256; None for a probe row (no manifest, so `run`
# lacks the field)


class InputError(ValueError):
    """An input lacks a field or a row it must have: scoring stops."""


def need(d: dict, fields, what: str) -> None:
    missing = [f for f in fields if f not in d]
    if missing:
        raise InputError(f"{what}: missing {missing}")


def subsets(record: dict, sealed: dict) -> dict:
    """The three sensitivity subsets of §5, from out/phase2/sensitivity_subsets.json (subset -> record ids, possibly
    split by partition); a subset missing from the file is an error."""
    need(sealed, SUBSETS, "sensitivity_subsets.json")
    ids = lambda v: set(v) if isinstance(v, list) else {i for x in v.values() for i in x}
    return {f"sub_{k}": record["id"] in ids(sealed[k]) for k in SUBSETS}


def call_overlap(p: Prepared, s: Score) -> dict:
    """Descriptive: n_calls = |O|, call_precision = |O ∩ G| / |O|, with O the project calls of the output functions that
    bind CREATE steps and G those of the gold functions, both after helper expansion (Score.calls: the CA call sets).
    PHP, as calls.agreement: calls are (receiver, casefold(method)), so two calls that differ only in the method's case
    are one; the receiver stays as written."""
    g = p.gold()
    key = (lambda cs: {(r, f.casefold()) for r, f in cs}) if p.repo.language == "php" else set
    o = key(set().union(*(s.calls.get(f, set()) for f in set(s.binders.values()))))
    gold = key(set().union(*(g.calls.get(f, set()) for f in set(g.binders.values()))))
    return {"n_calls": len(o), "call_precision": round(len(o & gold) / len(o), 4) if o else None}


CA_REFS = ("ca_ref_v3m", "ca_ref_v3all", "ca_ref_v3names", "ca_ref_top4")  # lazy references (D11: v3names, top4)
CA_PREC_TAUS = (0.1, 0.25, 0.5)  # D11 (c): τ = 0.25 is pre-registered; 0.1 and 0.5 for the controls
CA_PREC_CHECKED = ("gold", "gold-helpers")  # D11: controls whose CA 1 must keep CA_prec at τ = 0.25


def ca_prec(ca_value, call_precision: float | None, tau: float) -> int:
    """CA_prec (D11 (c), fixed 25/09/2026, before any output) = CA × [call_precision ≥ τ], with call_overlap's
    call_precision (after helper expansion); an output with no call (call_precision None) gives 0."""
    return int(bool(ca_value) and call_precision is not None and call_precision >= tau)


def ca_prec_excluded(rows: list[dict], tau: float = 0.25) -> list[dict]:
    """D11: the controls rows of gold and gold-helpers whose CA is 1 but CA_prec at `tau` is 0, each listed for the
    check before the lock (expected: none)."""
    return [{k: x[k] for k in ("record_id", "repo", "control", "ca", "call_precision", "n_calls")} for x in rows
            if x["control"] in CA_PREC_CHECKED and x["ca"] and not ca_prec(x["ca"], x["call_precision"], tau)]


def own_items(language: str, blocks: dict[str, str]) -> list[tuple[str, str, str, str]]:
    """(path, kind, text, name) of every item the OUTPUT wrote (its own blocks, split as insertion splits them), before
    insertion: a rewritten shell there carries only the annotations the output wrote, not the HEAD lines insertion keeps
    (insert's kept-annotation rule); the descriptive annotation-form columns read these (26/09/2026)."""
    return [(path, it.kind, it.text, it.name) for path, code in blocks.items() for it in split_code(language, code)[1]]


def own_imports(p: Prepared, path: str, blocks: dict[str, str]):
    """The imports an output item of `path` resolves its annotations through (§2.2): the output block's own and those of
    the target file in the ER tree, as insertion merges them."""
    return file_imports(p.repo.language, blocks.get(path, "") + "\n" + (p.er_files.get(path) or p.repo.ctx.text(path) or ""))


def own_step_styles(p: Prepared, blocks: dict[str, str]) -> Counter:
    """Style -> the step definitions the output's own method, lambda and block items register (evaluate.item_defs: the
    annotations before the `{` that opens the body, as BV binds them), in the repo's styles."""
    out: Counter = Counter()
    for path, kind, text, _ in own_items(p.repo.language, blocks):
        if kind in ("method", "lambda", "block"):
            out.update(d.style for d in item_defs(p.repo.language, kind, path, text, p.repo.styles,
                                                  own_imports(p, path, blocks)))
    return out


def php_steps(p: Prepared, run: dict, blocks: dict[str, str]) -> dict:
    """Descriptive PHP columns of a scores row with a text (None otherwise; 26/09/2026): php_tag_stripped = a block of
    the text had a `<?php` / `?>` tag that the extractor removes, per block, before insertion
    (Extraction.php_tag_stripped); php_docblock_steps, php_attribute_steps = the step definitions the output wrote, by
    style (own_step_styles: its own blocks, not the file after insertion; §2.3: the ER image, Behat 4.0.0-alpha1,
    registers attribute steps only)."""
    if p.repo.language != "php" or "text" not in run:
        return {"php_tag_stripped": None, "php_docblock_steps": None, "php_attribute_steps": None}
    styles = own_step_styles(p, blocks)
    return {"php_tag_stripped": extract(run["text"], p.targets).php_tag_stripped,  # per block (§1.4)
            "php_docblock_steps": styles.get("php_docblock", 0), "php_attribute_steps": styles.get("php_attribute", 0)}


def unread_steps(p: Prepared, run: dict, blocks: dict[str, str]) -> dict:
    """Pre-registered (§2.2, 27/09/2026, measured on the dev pilot before the SAP lock): unread_step_annotations = over
    the output's own items (own_items), the step-looking candidates that the reader itself does not take, in each item's
    head (a method or a java8 lambda: before the `{` that opens its body; a Ruby block: its first line; any other item,
    which never binds: all of it, the definitions read there included): what looks like a step declaration but is not
    read, per output; None without a text. Since 28/09/2026 it is counted (§2.2) with the very reader BV binds with,
    stepdefs.extract's `rejected`, through the file's imports (own_imports), not with a pattern of its own, so an alias
    of a foreign class (`use Foo\\Then as T; #[T('x')]`) and a foreign `#[\\Foo\\Then('x')]` count, and a "then" in a
    docblock's prose does not. The column measures the reader's blind spots, not outputs cut off, so no annotation of a
    method cut mid-body is counted, in every language and every order of the annotations (fixed 28/09/2026, before any
    output): stepdefs.extract records no candidate of a cut method, and the definitions it read before it stopped there
    (`read`) still count in an item that never binds, as do the candidates it declined before. Reported per run and
    condition by the SAP (descriptive)."""
    if "text" not in run:
        return {"unread_step_annotations": None}
    lang, n = p.repo.language, 0
    for path, kind, text, _ in own_items(lang, blocks):
        imports = own_imports(p, path, blocks)
        if kind == "block":
            head = len(text.split("\n", 1)[0])
        elif kind in ("method", "lambda"):
            opener = body_open(lang, kind, text)
            head = len(text) if opener is None else opener
        else:
            head = len(text)
        for style in p.repo.styles:
            rejected, defs = [], []
            try:
                extract_defs(path, text, style, imports, rejected=rejected, read=defs)
            except ValueError:  # code cut off: what the reader read or declined before the cut still counts
                pass
            n += sum(1 for i, _ in rejected if i < head)
            if kind not in ("method", "lambda", "block"):
                n += len(defs)  # read, but an item that is no method, lambda or block never binds
    return {"unread_step_annotations": n}


CUKE_TYPES = {"string", "int", "long", "byte", "short", "float", "double", "bigdecimal", "biginteger", "word", ""}


def cuke_patterns(p: Prepared, run: dict, blocks: dict[str, str]) -> dict:
    """Pre-registered (27/09/2026), per condition: among the Cucumber-expression patterns the reader reads in the
    output's own items (evaluate.item_defs, through the file's imports; Java, Ruby), how many hold an escaped `\\(`,
    `\\{` or `\\/` (cuke_escaped_patterns) and how many a parameter type the Phase 1 binder does not know
    (cuke_custom_types; it falls back to `.*?`): the binder's documented limits (a) to (d), which the gold never meets
    but an output may. None without a text."""
    if "text" not in run:
        return {"cuke_escaped_patterns": None, "cuke_custom_types": None}
    esc = custom = 0
    for path, kind, text, _ in own_items(p.repo.language, blocks):
        if kind not in ("method", "lambda", "block"):
            continue
        for d in item_defs(p.repo.language, kind, path, text, p.repo.styles, own_imports(p, path, blocks)):
            if d.kind == "cucumber":
                esc += bool(re.search(r"\\[({/]", d.pattern))
                custom += any(x not in CUKE_TYPES for x in re.findall(r"\{(\w*)\}", d.pattern))
    return {"cuke_escaped_patterns": esc, "cuke_custom_types": custom}


def open_resolution_rates(rows: list[dict], by: tuple[str, ...] = ("repo", "condition")) -> dict[str, dict]:
    """Share of scored runs whose condition 6 relied on an "open" heuristic (symbols.open_paths), per group, with its
    denominator (runs that reached condition 6, i.e. parsed)."""
    out: dict[str, list[int]] = {}
    for x in rows:
        if not x.get("bv_syntax"):
            continue
        g = out.setdefault("|".join(str(x.get(k)) for k in by), [0, 0])
        g[0] += bool(x.get("open_resolution"))
        g[1] += 1
    return {k: {"open": a, "n": n, "rate": round(a / n, 4)} for k, (a, n) in sorted(out.items())}


def scenario_key(record: dict) -> str:
    return f"{record['repo']}|{record['feature_path']}|{record['scenario_line']}"


def row(p: Prepared, run: dict, s: Score, blocks: dict[str, str], refs_: dict, excl: dict, sealed_subsets: dict,
        self_contained: bool | None = None, er: dict | None = None, prompt_f2: str | None = None,
        masked: frozenset = frozenset()) -> dict:
    """One row of scores.jsonl. `refs_` = {ca_ref_v3m, ca_ref_v3all, ca_ref_v3names, ca_ref_top4} of the record
    (controls run; D11: ca_ref_v3names is None for a Ruby record, which has no stub-v3names); `excl` = {excl_bv,
    excl_ca} (§2.2: BV leaves out only the records whose gold fails BV or whose HEAD does not parse, CA also the records
    whose CA is undefined); `er` = the record's er_results row for this run, when the run is in an ER manifest; `masked`
    = what 3A masks in the record's prompt (Composer.masked), for `hallucinated`."""
    r = p.record
    need(refs_, CA_REFS, f"ca_ref of {r['id']}")
    need(excl, ("excl_bv", "excl_ca"), f"exclusions of {r['id']}")
    if er is not None:
        need(er, ER_FIELDS, f"er_results row of {r['id']}")
        if er["er_status"] == "SKIP" and er["er"] != 0:  # a skipped run (no block, collision) scores ER = 0 (§2.3b)
            raise InputError(f"er_results row of {r['id']}: SKIP with er = {er['er']}")
    ca_defined, ca_value = ca(p, s)
    bad, used = hallucinated(p, s, masked)
    out = {"record_id": r["id"], "partition": r["partition"], "repo": r["repo"], "language": r["language"],
           "suite": r["suite"], "feature_path": r["feature_path"], "scenario_key": scenario_key(r),
           "exec_core": r["exec_core"], "self_contained": self_contained,
           **{k: run.get(k) for k in ("model", "seed", "condition", "variant")},
           "extract_status": s.extract_status, "insert_status": s.insert_status,
           # a fence of the run's text detectably closed early (extract.Extraction.truncated)
           "fence_truncated": extract(run["text"], p.targets).truncated if "text" in run else None,
           "ca": ca_value, "ca_defined": ca_defined, **{k: refs_[k] for k in CA_REFS},
           "excl_bv": excl["excl_bv"], "excl_ca": excl["excl_ca"], **call_overlap(p, s),
           "bv": s.bv, **{f"bv_{c}": int(v) for c, v in s.conds.items()},
           **{k: None if er is None else er[k] for k in ER_FIELDS},
           "duplicate_reuse": bool(s.detail.get("duplicate_reuse")), "unresolved": len(s.detail.get("unresolved", [])),
           "open_resolution": bool(s.detail.get("open_resolution")),
           "hallucinated": bad, "members_used": used, **conventions(p, s, blocks), **similarity(p, blocks),
           "dose": None if prompt_f2 is None else dose(p, prompt_f2),
           "cap_hit": run.get("cap_hit"), "prompt_tokens": run.get("prompt_tokens"),
           "len_bucket": None if run.get("prompt_tokens") is None else ("le8192" if run["prompt_tokens"] <= 8192 else "gt8192"),
           "preexisting": any(c["preexisting"] for c in r["create"]), **subsets(r, sealed_subsets),
           **php_steps(p, run, blocks), **unread_steps(p, run, blocks), **cuke_patterns(p, run, blocks),
           **{k: run.get(k) for k in PROVENANCE},
           "text_sha256": hashlib.sha256(run["text"].encode("utf-8")).hexdigest() if "text" in run else None}
    assert set(CONDITIONS) <= {k[3:] for k in out if k.startswith("bv_")}
    return out
