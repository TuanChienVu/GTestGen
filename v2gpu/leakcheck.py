"""Independent re-check of compose's anti-leak rules, run by train_job's preflight on
every training and dev prompt and, on the laptop, by `python -m v2gpu.leakcheck` on test and aux prompts (v2gpu.audit).
Written on purpose without importing v2tools, so a bug in compose's helpers cannot hide itself; function facts (code,
annotation lines and times, suites, step patterns) come from the sealed Phase 2 catalog.

Two layers, as the spec asks: through the provenance (examples ③, knowledge ④(b), skeleton ④(a) by id) and through the
text of the final prompt.

New CREATE items (preexisting = False):
- answer code: no aligned 200-char slice of the whitespace-normalised code appears in the normalised prompt (so no
  400-char window can appear either); when no such slice matched, neither the tail slice of the code nor an aligned
  slice of body_key may appear either (compose blocks the same definition, decision 24/09);
- twins: no ③ CREATE function and no ④ function may be a twin of the item: same body_key OR same body_key_nosig;
- method name: never a whole identifier token anywhere in the prompt, except tokens of the record's own Background,
  Scenario and step text (② is the only place such a token may legitimately come from); no ③/④ function may carry it;
  lambda steps (cucumber-java8: the annotation line is a call such as `When("...", (..) -> {`) have no method name, so the name rules skip them;
  any other non-Ruby function without a found name is a problem, never a silent skip;
- signature line: the FIRST line of the method's declaration (the kept line holding `name(`), whitespace-normalised, never
  appears as a whole prompt line. On both sides a trailing ` /* … */ }` (U+2026; compose's ④(a) form, decision 24/09),
  then a trailing `{` or `;`, is ignored.
Every CREATE item, preexisting included:
- CREATE patterns: the patterns of the annotations in the record's `create_def_ids`, normalised (whitespace collapsed,
  casefolded) and compared WHOLE, never as substrings. No ③ example may share one (by its own create_def_ids); no ④(b)
  function may carry one (the record's own preexisting function only on the annotations it may keep); and no annotation
  line of the repo's catalog that carries one appears as a whole prompt line (catches ④(a) and anything provenance
  cannot list).
Provenance rules:
- ③: in the retrieval pool, same repo, not the same .feature, not a direct neighbour (computed here and read from the
  `neighbours` field), no shared CREATE fn_id, has create_def_ids, no CREATE function hidden for the query's partition,
  and visible before the query: example t_vis < query t_vis, both known (decision of 24/09, every partition);
- ④(b): in the catalog, not hidden, in the record's snapshot (independent re-implementation of Phase 2 snapshot_fns) and
  not a future step function (rule 2: every annotation at or after t_vis); its pattern rule looks only at the
  annotations V(r) keeps (def_id not in create_def_ids, dated before t_vis), as compose renders them;
- ④(a) and ④(b) never list a new CREATE function of the query itself;
- ④(a) skeleton: in the catalog, not hidden, not a future step function, name and twin rules; members of other suites
  may be shown (a class frame is not a snapshot) and annotations are not shown, so the pattern rule does not apply.
The record's own preexisting functions may be shown with their old body and their own method name.
Report only, train and dev queries: (path, name) pairs of test-answer infrastructure (ctx.infra, from the prompt
directory's infra.json) whose name is an identifier token outside the record's own Gherkin and whose file is shown (a
target file of the record, or the file of a ④ function), in `counts["test_infra_name"]`; compose hides them and drops
the records that use them, and a same-named member of another class would make a refusal unsafe.
render_answer (decision of 24/09): the gold of a preexisting item is the catalog function minus the
annotation lines whose def_id is not in create_def_ids and whose time is None or >= t_vis (t_vis None: only undated
lines go); a new item's gold is create[].code.
body_key (one definition with the Phase 2 build): catalog code minus annotation lines (annotations[].line from
first_line) and minus
lines matching ^\\s*(/\\*\\*|\\*|\\*/|//|#(?!\\[)); signature KEPT; joined, whitespace collapsed; None under 60 chars.
body_key_nosig: the same string before the threshold; for PHP, Java and C#, when the part before the first `{` holds a
`(` and no `;` (a declaration), that part and the `{` are dropped; Ruby unchanged; None under 60 chars.
Functions without `code` (the training pod's catalog drops the code of test and aux answers) are checked by id only.
Hidden answer code in the text (fixed 25/09; laptop audit only, v2gpu.audit): the catalog code of
a function hidden for the query's partition (hidden_for: train hides dev, test and aux answers, dev hides test and aux
answers, test and aux hide nothing) must not appear in the prompt text: its declaration (the normalised signature
line, the kept line holding `name(`, as a whole prompt line; for Ruby or a Java lambda, an annotation line) together
with an aligned 200-char slice (at least 60 chars) of its whitespace-normalised code. Evidence the partition may see
anyway does not count: a slice also held by visible code (the catalog code of a function of the repo that is not
hidden, or the create[].code of a retrieval-pool record whose function is not hidden), a declaration also carried by a
visible function. A slice alone is not enough: ③ shows an example's CREATE functions in their pre-T form (compose's
example_forms, from the repo history this audit does not have), and on real v0 such a pre-T body shares 200-char
boilerplate slices with the current code of two dev answers in 112 train prompts, without their declarations. The
training pod's catalog has no code for test and aux answers, so train_job cannot run this rule; the laptop audit runs
it on every prompt file that goes to a pod, train included.
After the positive control's 11 misses: catalog entries compose never
renders (never_rendered, the criteria of snapshot_reason: catch-all, dead code, a span containing another
function) are no visible code (a gpconnect method whose code only a commented span also holds is checked
again); the anchored slice, the hidden function's code from its own signature line on (first 200 chars, at least 60,
held by no visible code), is evidence on its own (a signature a visible Api/Ui twin shares no longer makes the
function uncheckable). One rule stays: a declaration another function that is not hidden carries is no declaration
mark, never-rendered ones included (④(a) may show a catch-all's signature). On real data (v0 dev + train, 9,075 lines):
0 lines flagged; the positive control (seed 0) catches 237 of 240, the 3 left are near or exact copies of a visible
function; planting every hidden function catches 1,411 of 1,426 (train) and 974 of 983 (dev, repos with dev prompts).
The other misses are exact or near copies of a visible function too, but for two fineract dead spans that hold 18
other hidden functions whole: their plant is flagged through those 18.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

SLICE = 200
BODY_MIN = 60
DECLARING = ("php", "java", "csharp")
IDENT = re.compile(r"[A-Za-z_]\w*")
COMMENT = re.compile(r"^\s*(/\*\*|\*|\*/|//|#(?!\[))")
NOT_NAMES = {"if", "for", "foreach", "while", "switch", "catch", "return", "new", "throw", "else", "do", "try", "using",
             "lock", "synchronized", "typeof", "sizeof", "nameof", "base", "this", "super", "function"}


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def code_slices(code: str, width: int = SLICE) -> list[str]:
    n = normalize_ws(code)
    if len(n) <= width:
        return [n]
    return [n[i:i + width] for i in range(0, len(n) - width + 1, width)]


def _kept_lines(fn: dict) -> list[str]:
    annotation_lines = {a["line"] for a in fn.get("annotations", [])}
    kept = []
    for number, line in enumerate(fn.get("code", "").splitlines(), start=fn.get("first_line", 1)):
        if number in annotation_lines or COMMENT.match(line):
            continue
        kept.append(line.strip())
    return kept


def _body_text(fn: dict) -> str:
    return normalize_ws(" ".join(_kept_lines(fn)))


def body_key(fn: dict) -> str | None:
    text = _body_text(fn)
    return text if len(text) >= BODY_MIN else None


def _declaration(text: str, language: str) -> str | None:
    """The part before the first `{` when it holds a `(` and no `;` (PHP, Java, C#), else None."""
    if language not in DECLARING or "{" not in text:
        return None
    head = text[:text.index("{")]
    return head if "(" in head and ";" not in head else None


def body_key_nosig(fn: dict, language: str) -> str | None:
    text = _body_text(fn)
    head = _declaration(text, language)
    if head is not None:
        text = text[len(head) + 1:].strip()
    return text if len(text) >= BODY_MIN else None


LAMBDA_STEP = re.compile(r"^\s*(Given|When|Then|And|But)\s*\(")


def is_lambda(fn: dict, language: str) -> bool:
    """A cucumber-java8 step: the annotation line is a call, When("...", (String a) -> {, not an @When annotation (the
    lambda's `->` may sit on a later line)."""
    lines = fn.get("code", "").splitlines()
    return language == "java" and any(LAMBDA_STEP.match(lines[a["line"] - fn.get("first_line", 1)])
                                      for a in fn.get("annotations", [])
                                      if 0 <= a["line"] - fn.get("first_line", 1) < len(lines))


def method_name(fn: dict, language: str) -> str | None:
    head = _declaration(_body_text(fn), language)
    if head is None or (language != "php" and ("->" in head or "=>" in head)):  # a Java/C# lambda declares nothing
        return None
    if language == "php":
        m = re.search(r"\bfunction\s+&?\s*([A-Za-z_]\w*)\s*\(", head)
        return m.group(1) if m else None
    for m in re.finditer(r"(?<![@\[\w.])([A-Za-z_]\w*)\s*\(", head):
        if m.group(1) not in NOT_NAMES:
            return m.group(1)
    return None


SIG_STUB = re.compile(r"\s*\{?\s*/\*\s*\u2026\s*\*/\s*\}$")


def _norm_sig(line: str) -> str:
    return re.sub(r"[\s{;]+$", "", SIG_STUB.sub("", normalize_ws(line)))


def signature_line(fn: dict, language: str) -> str | None:
    """The normalised kept line that declares the method (the one holding `name(`), or None (lambda, Ruby)."""
    name = method_name(fn, language)
    if name is None:
        return None
    for line in _kept_lines(fn):
        if re.search(rf"\b{re.escape(name)}\s*\(", line):
            return _norm_sig(line)
    return None


def render_answer(fn: dict, t_vis: int | None, create_def_ids) -> str:
    drop = {a["line"] for a in fn.get("annotations", []) if a.get("def_id") not in create_def_ids
            and (a.get("time") is None or (t_vis is not None and a["time"] >= t_vis))}
    return "\n".join(l for i, l in enumerate(fn["code"].splitlines(), start=fn.get("first_line", 1)) if i not in drop)


def norm_pattern(pattern: str) -> str:
    return normalize_ws(pattern).casefold()


def patterns(fn: dict, skip_def_ids=()) -> set[str]:
    return {norm_pattern(a["pattern"]) for a in fn.get("annotations", [])
            if a.get("pattern") and a.get("def_id") not in skip_def_ids}


def kept_patterns(fn: dict, create_def_ids, t_vis: int | None) -> set[str]:
    """Patterns of the annotations V(r) shows in ④(b): not a CREATE annotation of the query, dated before t_vis."""
    return {norm_pattern(a["pattern"]) for a in fn.get("annotations", []) if a.get("pattern")
            and a.get("def_id") not in create_def_ids and a.get("time") is not None and t_vis is not None
            and a["time"] < t_vis}


def pattern_lines(catalog_repo: dict[str, dict]) -> dict[str, list[tuple[str, str]]]:
    """Normalised pattern -> [(fn_id, whitespace-normalised source line of the annotation)] over one repo's catalog;
    functions without code are skipped."""
    out: dict[str, list[tuple[str, str]]] = {}
    for fid, fn in catalog_repo.items():
        lines = fn.get("code", "").splitlines()
        for a in fn.get("annotations", []):
            i = a["line"] - fn.get("first_line", 1)
            if a.get("pattern") and 0 <= i < len(lines):
                out.setdefault(norm_pattern(a["pattern"]), []).append((fid, normalize_ws(lines[i])))
    return out


def neighbours(records: list[dict]) -> dict[str, set[str]]:
    """Direct links only: the same .feature file or at least one shared CREATE function."""
    by_key: dict[tuple, set[str]] = {}
    for r in records:
        for k in [("feature", r["repo"], r["feature_path"])] + [("fn", r["repo"], c["fn_id"]) for c in r["create"]]:
            by_key.setdefault(k, set()).add(r["id"])
    out = {r["id"]: set() for r in records}
    for ids in by_key.values():
        for i in ids:
            out[i] |= ids - {i}
    return out


def neighbour_problems(train_records: list[dict], other_records: list[dict]) -> list[str]:
    """Train: the `neighbours` field equals the direct links among train records. Dev, test, aux: no train record is a
    direct neighbour (decision of 24/09: report, never drop)."""
    nb = neighbours(train_records + other_records)
    train_ids = {r["id"] for r in train_records}
    problems = []
    for r in train_records:
        if "neighbours" not in r:
            problems.append(f"train record {r['id']} has no neighbours field")
            continue
        field_ids, direct = set(r["neighbours"] or []) - {r["id"]}, nb[r["id"]] & train_ids
        if field_ids != direct:
            problems.append(f"train record {r['id']}: neighbours field differs from the direct links "
                            f"(missing {sorted(direct - field_ids)}, extra {sorted(field_ids - direct)})")
    for r in other_records:
        linked = nb[r["id"]] & train_ids
        if linked:
            problems.append(f"{r['partition']} record {r['id']} is a direct neighbour of train record(s) {sorted(linked)[:5]}")
    return problems


def hidden_for(partition: str, forbidden_repo: dict) -> set[str]:
    parts = {"train": ("test", "dev", "aux"), "dev": ("test", "aux")}.get(partition, ())
    return {f for p in parts for f in forbidden_repo.get(p, [])}


def never_rendered(fn: dict) -> str | None:
    """Why compose never shows fn's catalog code in ④(b) for any record, or None (3A records.snapshot_fns): a
    catch-all, dead code, or a span that contains another function (`catch_all`/`dead`/`contains_fn`, absent = False).
    ③ shows a pool record's own create[].code, which hidden_slices counts as visible code on its own."""
    if fn.get("catch_all"):
        return "catch-all"
    if fn.get("dead"):
        return "dead code"
    if fn.get("contains_fn"):
        return "contains another function"
    return None


def snapshot_reason(record: dict, fn: dict, hidden: set[str]) -> str | None:
    """Why fn is not in the record's snapshot, or None (Phase 2 snapshot_fns, re-implemented; tags compared casefolded;
    via never_rendered: also skips catch-alls, dead code and spans that contain another function)."""
    if fn["fn_id"] in hidden:
        return "hidden"
    reason = never_rendered(fn)
    if reason:
        return reason
    if record.get("scoping") == "specflow_scope":
        tags = {t.casefold() for t in record.get("tags", [])}
        if fn.get("scope_tags") and not any({t.casefold() for t in group} & tags for group in fn["scope_tags"]):
            return "outside the scope tags"
    elif record.get("suite") not in fn.get("suites", []):
        return "other suite"
    if not any(a.get("time") is not None and a["time"] < record["scenario_time"] for a in fn.get("annotations", [])):
        return "no pattern before the scenario"
    return None


def is_future(fn: dict, t_vis: int | None) -> bool:
    """V(r) rule 2: a step function whose every annotation is at or after T (= t_vis)."""
    times = [a.get("time") for a in fn.get("annotations", [])]
    return t_vis is not None and bool(times) and all(t is not None and t >= t_vis for t in times)


@dataclass
class Context:
    pool: dict[str, dict]            # retrieval pool (③), by record id
    catalog: dict[str, dict]         # repo -> fn_id -> catalog entry
    hidden: dict[str, set]           # repo -> fn_ids hidden for the query's partition
    language: dict[str, str]         # repo -> language
    neighbours: dict[str, set]       # direct neighbours of every record the check can meet
    infra: dict[str, set] = field(default_factory=dict)   # repo -> {(path, name)} of test-answer infrastructure
    _lines: dict = field(default_factory=dict)
    _hidden_slices: dict = field(default_factory=dict)

    def annotation_lines(self, repo: str) -> dict[str, list[tuple[str, str]]]:
        if repo not in self._lines:
            self._lines[repo] = pattern_lines(self.catalog[repo])
        return self._lines[repo]

    def hidden_slices(self, repo: str, hidden) -> list[tuple[str, list[str], set[str], set[str], set[str]]]:
        """[(fn_id, slices, signature lines, annotation lines, anchored slices)] of the hidden functions of the repo
        that have code:
        - the aligned slices of at least BODY_MIN chars that no visible code holds. Visible code is a function of the
          repo that is not hidden and that compose can render, i.e. not never_rendered, or the
          create[].code of a retrieval-pool record of the repo whose function is not hidden;
        - the declaration marks no function that is not hidden carries: its normalised signature line, or
          for a function without one (Ruby, a Java lambda) its annotation lines;
        - the anchored slice: the first SLICE chars of its normalised code from its own
          signature line on, at least BODY_MIN, held by no visible code."""
        key = (repo, frozenset(hidden))
        if key not in self._hidden_slices:
            catalog, language = self.catalog.get(repo, {}), self.language.get(repo, "")
            seen = [f for fid, f in catalog.items() if fid not in key[1] and "code" in f]
            shown = [f["code"] for f in seen if not never_rendered(f)] + [
                c["code"] for r in self.pool.values() if r.get("repo") == repo for c in r.get("create", [])
                if c.get("code") and c.get("fn_id") not in key[1]]
            visible = "\n".join(normalize_ws(code) for code in shown)
            seen_sigs = {signature_line(f, language) for f in seen} - {None}
            seen_lines = {line for f in seen for line in _annotation_texts(f)}
            out = []
            for fid in sorted(key[1]):
                f = catalog.get(fid, {})
                if "code" not in f:
                    continue
                slices = [s for s in code_slices(f["code"]) if len(s) >= BODY_MIN and s not in visible]
                sig = signature_line(f, language)
                sigs = {sig} - seen_sigs - {None}
                lines = set() if sig else set(_annotation_texts(f)) - seen_lines
                anchor = _anchored(f, sig)
                anchors = {anchor} if anchor and len(anchor) >= BODY_MIN and anchor not in visible else set()
                if (slices and (sigs or lines)) or anchors:
                    out.append((fid, slices, sigs, lines, anchors))
            self._hidden_slices[key] = out
        return self._hidden_slices[key]


def _anchored(fn: dict, sig: str | None) -> str | None:
    """The first SLICE chars of fn's whitespace-normalised code from its own signature line on:
    the declaration joined to its own body, so a signature a visible twin shares cannot hide it."""
    if not sig:
        return None
    lines = fn["code"].splitlines()
    for i, line in enumerate(lines):
        if _norm_sig(line) == sig:
            return normalize_ws("\n".join(lines[i:]))[:SLICE]
    return None


def _annotation_texts(fn: dict) -> list[str]:
    lines = fn.get("code", "").splitlines()
    return [normalize_ws(lines[a["line"] - fn.get("first_line", 1)]) for a in fn.get("annotations", [])
            if 0 <= a["line"] - fn.get("first_line", 1) < len(lines)]


def _twin_keys(fn: dict, language: str) -> set[tuple[str, str]]:
    return {(kind, k) for kind, k in (("sig", body_key(fn)), ("nosig", body_key_nosig(fn, language))) if k}


def _clashes(label: str, fns: list[dict], language: str, names: set, pats: set, twins: set) -> list[str]:
    out = []
    for f in fns:
        name = method_name(f, language)
        if name and name in names:
            out.append(f"{label}: method name {name} of a new CREATE function")
        shared = patterns(f) & pats
        if shared:
            out.append(f"{label}: CREATE step pattern {sorted(shared)}")
        if _twin_keys(f, language) & twins:
            out.append(f"{label}: twin of a new CREATE function ({f['fn_id']})")
    return out


def check(record: dict, messages: list[dict], provenance: dict, ctx: Context, counts: Counter | None = None) -> list[str]:
    counts = Counter() if counts is None else counts
    repo, rid = record["repo"], record["id"]
    catalog, language = ctx.catalog[repo], ctx.language[repo]
    hidden = ctx.hidden.get(repo, set())
    problems = []
    new_items = [c for c in record["create"] if not c["preexisting"]]
    preexisting = {c["fn_id"] for c in record["create"] if c["preexisting"]}
    own = {c["fn_id"] for c in record["create"]}
    new_fns = []
    for c in new_items:
        if c["fn_id"] not in catalog:
            problems.append(f"{rid}: CREATE function {c['fn_id']} is not in the catalog")
        elif "code" not in catalog[c["fn_id"]]:
            problems.append(f"{rid}: CREATE function {c['fn_id']} has no code in the catalog")
        else:
            new_fns.append(catalog[c["fn_id"]])
            if language != "ruby" and not is_lambda(catalog[c["fn_id"]], language) and \
                    method_name(catalog[c["fn_id"]], language) is None:
                problems.append(f"{rid}: no method name found for {c['fn_id']}, which is neither a lambda nor Ruby")
    problems += [f"{rid}: record has no {k}" for k in ("create_def_ids", "t_vis") if k not in record]
    create_ids = set(record.get("create_def_ids") or [])
    create_anns = {a["def_id"]: a for c in record["create"] for a in catalog.get(c["fn_id"], {}).get("annotations", [])
                   if a.get("def_id") in create_ids}
    problems += [f"{rid}: CREATE annotation {d} is not on a CREATE function of the record"
                 for d in sorted(create_ids - set(create_anns))]
    create_pats = {norm_pattern(a["pattern"]) for a in create_anns.values() if a.get("pattern")}
    names = {n for f in new_fns if (n := method_name(f, language))}
    twins = set().union(*(_twin_keys(f, language) for f in new_fns)) if new_fns else set()

    # layer 1: the text of the prompt
    text = "\n".join(m["content"] for m in messages)
    prompt = normalize_ws(text)
    lines = [line for m in messages for line in m["content"].splitlines()]
    line_set, sig_set = {normalize_ws(x) for x in lines}, {_norm_sig(x) for x in lines}
    for c in new_items:
        full_hit = any(s in prompt for s in code_slices(c["code"]))
        if full_hit:
            problems.append(f"{rid}: answer code of {c['fn_id']} appears in the prompt")
        f = catalog.get(c["fn_id"], {})
        short = [normalize_ws(c["code"])[-SLICE:]] + (code_slices(body_key(f)) if body_key(f) else [])
        if not full_hit and any(s and s in prompt for s in short):
            problems.append(f"{rid}: short answer slice of {c['fn_id']} appears in the prompt")
        sig = signature_line(f, language) if "code" in f else None
        if sig and sig in sig_set:
            problems.append(f"{rid}: signature line of {c['fn_id']} appears in the prompt")
    for p in sorted(create_pats):
        for fid, line in ctx.annotation_lines(repo).get(p, []):
            if line in line_set:
                problems.append(f"{rid}: annotation of CREATE pattern {p!r} ({fid}) appears in the prompt")
    allowed = set(IDENT.findall(" ".join([record.get("background") or "", record.get("scenario") or "",
                                          *record.get("steps", [])])))
    tokens = set(IDENT.findall(text)) - allowed
    for name in sorted(names & tokens):
        problems.append(f"{rid}: method name {name} of a new CREATE function appears in the prompt")
    if record.get("partition") in ("train", "dev"):
        shown = {c.get("path") for c in record["create"]} | {f.rsplit(":", 1)[0] for f in
                                                             provenance["knowledge"] + provenance["skeleton"]}
        counts["test_infra_name"] += sum(1 for path, name in ctx.infra.get(repo, set()) if name in tokens and path in shown)

    # layer 2: the provenance
    direct = ctx.neighbours.get(rid, set()) | set(record.get("neighbours") or [])
    for eid in provenance["examples"]:
        ex = ctx.pool.get(eid)
        label = f"{rid} example {eid}"
        if ex is None:
            problems.append(f"{label}: not in the retrieval pool")
            continue
        if ex["repo"] != repo:
            problems.append(f"{label}: other repo")
        if ex["feature_path"] == record["feature_path"]:
            problems.append(f"{label}: same .feature")
        if eid in direct:
            problems.append(f"{label}: direct neighbour of the query")
        shared = own & {c["fn_id"] for c in ex["create"]}
        if shared:
            problems.append(f"{label}: shares CREATE function {sorted(shared)}")
        if "create_def_ids" not in ex:
            problems.append(f"{label}: example record has no create_def_ids")
        if ex.get("t_vis") is None or record.get("t_vis") is None or not ex["t_vis"] < record["t_vis"]:
            problems.append(f"{label}: t_vis {ex.get('t_vis')} is not before the query's {record.get('t_vis')}")
        ex_ids = set(ex.get("create_def_ids") or [])
        ex_pats = {norm_pattern(a["pattern"]) for c in ex["create"] for a in catalog.get(c["fn_id"], {}).get("annotations", [])
                   if a.get("def_id") in ex_ids and a.get("pattern")}
        if ex_pats & create_pats:
            problems.append(f"{label}: CREATE step pattern {sorted(ex_pats & create_pats)}")
        problems += [f"{label}: CREATE function {c['fn_id']} is hidden for partition {record['partition']}"
                     for c in ex["create"] if c["fn_id"] in hidden]
        ex_fns = [catalog[c["fn_id"]] for c in ex["create"] if c["fn_id"] in catalog and c["fn_id"] not in preexisting]
        problems += _clashes(label, ex_fns, language, names, set(), twins)
    for kind, key in (("knowledge", "knowledge"), ("skeleton", "skeleton")):
        for fid in provenance[key]:
            label = f"{rid} {kind} {fid}"
            if fid not in catalog:
                problems.append(f"{label}: not in the catalog")
                continue
            f = catalog[fid]
            if fid in hidden:
                problems.append(f"{label}: hidden for partition {record['partition']}")
                continue
            if fid in own and fid not in preexisting:
                problems.append(f"{label}: a new CREATE function of the query")
                continue
            if is_future(f, record.get("t_vis")) and fid not in own:
                problems.append(f"{label}: future step function (every pattern at or after t_vis)")
            if kind == "knowledge":
                reason = snapshot_reason(record, f, hidden)
                if reason:
                    problems.append(f"{label}: not in the snapshot ({reason})")
            shown_pats = kept_patterns(f, create_ids, record.get("t_vis")) & create_pats if kind == "knowledge" else set()
            if shown_pats:
                problems.append(f"{label}: CREATE step pattern {sorted(shown_pats)}")
            if fid in preexisting:  # its old body and name may be shown
                continue
            problems += _clashes(label, [f], language, names, set(), twins)
    return problems


def hidden_code(record: dict, messages: list[dict], ctx: Context, hidden) -> list[str]:
    """The hidden functions (fn ids, sorted) whose catalog code appears in the prompt text; see the docstring."""
    text = "\n".join(m["content"] for m in messages)
    prompt, lines = normalize_ws(text), text.splitlines()
    line_set, sig_set = {normalize_ws(x) for x in lines}, {_norm_sig(x) for x in lines}
    return [fid for fid, slices, sigs, marks, anchors in ctx.hidden_slices(record["repo"], hidden)
            if ((sigs & sig_set or marks & line_set) and any(s in prompt for s in slices))
            or any(a in prompt for a in anchors)]


if __name__ == "__main__":  # the laptop audit of test and aux prompts lives in v2gpu.audit
    from .audit import main

    raise SystemExit(main())
