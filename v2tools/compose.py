"""Prompt composer of Phase 3: one user turn made of ① instructions, ② the scenario and its undefined steps, ③ retrieved
examples (F1, F2) and ④ project knowledge (F2), with fixed token budgets, the leak rules and a provenance record that
the GPU pipeline's preflight (v2gpu) re-checks independently. Budgets count every token of a block, headers and
fences included. gold_answer() is separate and never called when building an inference prompt."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable

from .bm25 import BM25, tokens
from .body import body_key, body_key_nosig
from .outline import parse_snippet
from .records import hidden_for, snapshot_fns
from .state import (SkeletonItem, annotation_pattern, future_ref, head_members, member_refs, pre_t_code, render_answer,
                    render_skeleton, render_visible, skeleton_parts)

FENCE = {"php": "php", "java": "java", "ruby": "ruby", "csharp": "csharp"}
COMMENT = {"php": "//", "java": "//", "csharp": "//", "ruby": "#"}
LANG_NAME = {"php": "PHP", "java": "Java", "ruby": "Ruby", "csharp": "C#"}
PLACEBO_REPO = {"Sylius/Sylius": "PrestaShop/PrestaShop", "PrestaShop/PrestaShop": "Sylius/Sylius",
                "apache/fineract": "openMF/ph-ee-integration-test", "openMF/ph-ee-integration-test": "apache/fineract"}
FORMATS = ("F0", "F1", "F2")
MODES = ("full", "4a", "4b", "placebo")
HELD_OUT_PARTS = ("test", "aux_test")  # their answers define the answer infrastructure
TICKS = "`" * 3  # a Markdown code fence, never written literally, so this file can sit inside one
SLICE = 200
SCAN_EXAMPLES = 60  # ranked ③ candidates considered per query (after the eligibility filters)
SCAN_KNOWLEDGE = 200  # ranked ④(b) candidates considered per query
EXAMPLES_HEAD = "Examples from this repository:\n\n"
KNOWLEDGE_HEAD = "Project knowledge:\n\n"


@dataclass(frozen=True)
class Params:
    k: int = 8  # maximum number of ④(b) functions
    n_examples: int = 3  # maximum number of ③ examples
    ratio: float = 0.5  # share of the ④ budget reserved for ④(a); the rest, plus what ④(a) leaves, goes to ④(b)
    budget_examples: int = 1536
    budget_knowledge: int = 1536
    mode: str = "full"  # "full" | "4a" (only ④a) | "4b" (only ④b) | "placebo" (④ = random functions of another repo)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown compose mode {self.mode!r}")


def norm_ws(s: str) -> str:
    return " ".join(s.split())


def norm_pattern(p: str) -> str:
    return " ".join(p.split()).casefold()


def aligned_slices(code: str, width: int = SLICE) -> list[str]:
    n = norm_ws(code)
    return [n[i:i + width] for i in range(0, len(n) - width + 1, width)]


def short_slices(code: str, key: str | None) -> list[str]:
    """Slices that catch a short answer shown without its annotation (blocking since 24/09/2026): the last 200
    characters of the whitespace-normalised code and the aligned 200-character slices of its body_key, exactly as
    v2gpu's leakcheck counts them."""
    return [norm_ws(code)[-SLICE:]] + (aligned_slices(key) if key else [])


def method_names(language: str, code: str) -> list[str]:
    """Declared method names of a CREATE code block; [] for Ruby and for Java lambda steps (no method)."""
    o = parse_snippet(language, code)
    return [] if o is None else [m.names[0] for m in o.members if m.kind == "method"]


def signature_lines(language: str, code: str) -> list[str]:
    o = parse_snippet(language, code)
    return [] if o is None else [norm_ws(m.text.splitlines()[0]) for m in o.members if m.kind == "method"]


def gold_answer(record: dict, catalog: dict | None = None, forms: dict[str, str] | None = None) -> str:
    """The training target and the reference answer: one fenced block per target file, in order of first appearance,
    `FILE:` line first, then the CREATE functions of that file separated by blank lines. A new function keeps its
    Phase 2 code (create[].code); a preexisting item is rendered from the repo's catalog ({fn_id: row}) as V(r) shows
    it at t_vis plus its CREATE annotations (state.render_answer). `forms` ({fn_id: code}) replaces the code of those
    functions: the pre-T form an example shows (Composer.example_forms). A record with a preexisting item and no
    catalog raises: Phase 2 code would be the wrong gold."""
    if catalog is None and any(c["preexisting"] for c in record["create"]):
        raise ValueError(f"{record.get('id')}: a preexisting CREATE item needs the catalog for its gold answer")
    lang = record["language"]
    cdefs = frozenset(record.get("create_def_ids", ()))
    order, by_path = [], {}
    for c in record["create"]:
        if c["path"] not in by_path:
            order.append(c["path"])
            by_path[c["path"]] = []
        code = render_answer(catalog[c["fn_id"]], record["t_vis"], cdefs) if c["preexisting"] else c["code"]
        by_path[c["path"]].append(forms[c["fn_id"]] if forms and c["fn_id"] in forms else code)
    return "\n\n".join(f"{TICKS}{FENCE[lang]}\n{COMMENT[lang]} FILE: {p}\n" + "\n\n".join(by_path[p]) + f"\n{TICKS}"
                       for p in order)


def text_keys(language: str, code: str) -> set[tuple[str, str]]:
    """Twin keys (body_key, body_key_nosig) of a text as it is SHOWN, e.g. a pre-T form whose annotation lines are not
    known by catalog line numbers: those lines (state.annotation_pattern) go first (so a renamed copy of a new answer
    shown in its pre-T form is still a twin)."""
    fn = {"code": "\n".join(l for l in code.splitlines() if annotation_pattern(language, l) is None),
          "annotations": [], "first_line": 1}
    return {k for k in (("sig", body_key(fn)), ("nosig", body_key_nosig(fn, language))) if k[1] is not None}


def annotation_lines(language: str, code: str) -> list[tuple[str, str | None]]:
    """(line, normalised pattern of its annotation or None) for each line of a pre-T text (state.annotation_pattern)."""
    return [(l, None if (p := annotation_pattern(language, l)) is None else norm_pattern(p)) for l in code.splitlines()]


def undefined_steps(record: dict) -> list[str]:
    seen, out = set(), []
    for s, k in zip(record["steps"], record["step_kinds"]):
        if k == "create" and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def gherkin(record: dict) -> str:
    return "\n".join(x for x in (record["background"], record["scenario"]) if x)


@dataclass
class Answer:
    """Everything the leak rules need about a query record's answer. names, sig_lines, slices and keys come from the
    CREATE items with preexisting=False only; patterns from every CREATE item, normalised. The rules apply to every item
    shown in ③ and ④, the record's own preexisting functions included."""
    new_fns: set[str]
    pre_fns: set[str]
    names: set[str]
    sig_lines: set[str]
    slices: list[str]
    keys: set[tuple[str, str]]
    patterns: set[str]
    short: list[str] = field(default_factory=list)


@dataclass
class Composer:
    contexts: dict  # repo -> RepoContext
    catalog: dict  # repo -> {fn_id: catalog row}
    forbidden: dict  # repo -> {"test": [...], "dev": [...], "aux": [...]}
    records: dict  # record id -> record (every partition)
    params: Params = field(default_factory=Params)
    _pool_index: dict = field(default_factory=dict)
    _keys: dict = field(default_factory=dict)
    _infra: dict | None = None

    # ----- facts about functions and records -------------------------------------------------------------------
    def fn_keys(self, repo: str, fid: str) -> set[tuple[str, str]]:
        if (repo, fid) not in self._keys:
            fn, lang = self.catalog[repo][fid], self.contexts[repo].language
            ks = {("sig", body_key(fn)), ("nosig", body_key_nosig(fn, lang))}
            self._keys[(repo, fid)] = {k for k in ks if k[1] is not None}
        return self._keys[(repo, fid)]

    def patterns_of(self, record: dict) -> set[str]:
        ids = set(record["create_def_ids"])
        return {norm_pattern(a["pattern"]) for c in record["create"] for a in self.catalog[record["repo"]][c["fn_id"]]["annotations"]
                if a["def_id"] in ids}

    def shown_patterns(self, record: dict) -> set[str]:
        """Every pattern whose annotation appears in the record's gold answer (gold_answer with the catalog): the CREATE
        patterns, plus the older annotations kept by a preexisting item (before t_vis, state.render_answer) or by a new
        function (before scenario_time, records.render)."""
        ids = set(record["create_def_ids"])
        out = set()
        for c in record["create"]:
            limit = record["t_vis"] if c["preexisting"] else record["scenario_time"]
            out |= {norm_pattern(a["pattern"]) for a in self.catalog[record["repo"]][c["fn_id"]]["annotations"]
                    if a["def_id"] in ids or (a["time"] is not None and a["time"] < limit)}
        return out

    def answer(self, r: dict) -> Answer:
        lang, new = r["language"], [c for c in r["create"] if not c["preexisting"]]
        return Answer(new_fns={c["fn_id"] for c in new}, pre_fns={c["fn_id"] for c in r["create"] if c["preexisting"]},
                      names={n for c in new for n in method_names(lang, c["code"])},
                      sig_lines={s for c in new for s in signature_lines(lang, c["code"])},
                      slices=[s for c in new for s in aligned_slices(c["code"])],
                      keys={k for c in new for k in self.fn_keys(r["repo"], c["fn_id"])},
                      patterns=self.patterns_of(r),
                      short=[x for c in new for x in short_slices(c["code"], body_key(self.catalog[r["repo"]][c["fn_id"]]))])

    def hidden(self, r: dict) -> set[str]:
        return hidden_for(r["partition"], self.forbidden.get(r["repo"], {}))

    def gold(self, r: dict, forms: dict[str, str] | None = None) -> str:
        return gold_answer(r, self.catalog[r["repo"]], forms)

    # ----- test-answer infrastructure ------------------------------------------------------------------------------
    def uses(self, r: dict, preexisting: bool = False, new_since_t: bool = False) -> set[tuple[str, str]]:
        """(declaring path at HEAD, name) of the non-step class members that the record's new CREATE code uses; with
        preexisting=True, the HEAD code of its preexisting items too (shown as gold when the record is an example). With
        new_since_t=True, only the members that no file of the class chain declared before the record's t_vis
        (RepoContext.declared_before, as rule 3): members written with the answer. An old member that the class reached
        through a parent added at t_vis is NOT counted: infrastructure is masked in every train/dev prompt of the repo,
        and such members (e.g. PrestaShop referenceToId) are old helpers of many classes; the record's own prompt does
        not show them anyway (chain_before), and its self_contained flag is False."""
        ctx, t = self.contexts[r["repo"]], r["t_vis"]
        if ctx.language == "ruby":
            return set()
        defined = {n for c in r["create"] for n in method_names(ctx.language, c["code"])}
        out: set[tuple[str, str]] = set()
        for c in r["create"]:
            if (c["preexisting"] and not preexisting) or ctx.text(c["path"]) is None:
                continue
            for n in (member_refs(ctx.language, c["code"]) & head_members(ctx, c["path"])) - defined:
                q = ctx.declarer(c["path"], n)
                if q is None:
                    continue
                m = next(m for m in ctx.outline(q).members if n in m.names)
                if m.kind == "method" and ctx.step_at.get((q, m.body_line)):
                    continue  # a step function: forbidden.json already handles it
                if not new_since_t or n not in ctx.declared_before(c["path"], t):
                    out.add((q, n))
        return out

    def infra(self) -> dict:
        """{"members": {repo: {(path, name)}}, "users": {record ids}}: the ANSWER INFRASTRUCTURE of the held-out sets
        (test and aux_test), i.e. non-step members that a held-out answer uses although no file of its class chain
        declared them before that record's t_vis (uses(new_since_t=True): written with the answer, so its own
        prompt masks them), and the train/dev records whose CREATE code (new, or the HEAD code of a preexisting item,
        which an example shows) uses one of them. Train and dev prompts mask these members; their users leave training
        and every pool."""
        if self._infra is None:
            members: dict[str, set] = {}
            for r in self.records.values():
                if r["partition"] in HELD_OUT_PARTS:
                    members.setdefault(r["repo"], set()).update(self.uses(r, new_since_t=True))
            members = {repo: v for repo, v in members.items() if v}
            users = {r["id"] for r in self.records.values() if r["partition"] in ("train", "dev")
                     and self.uses(r, preexisting=True) & members.get(r["repo"], set())}
            self._infra = {"members": members, "users": users}
        return self._infra

    def masked(self, r: dict) -> frozenset:
        return frozenset(self.infra()["members"].get(r["repo"], ())) if r["partition"] in ("train", "dev") else frozenset()

    # ----- leak rules ----------------------------------------------------------------------------------------------
    @staticmethod
    def text_rule(ans: Answer, text: str) -> str | None:
        """Leak rules that look only at text: aligned 200-character slices, the short-answer slices, signature lines,
        method names as identifiers (a signature line holds its name, so "signature" is tested first)."""
        n = norm_ws(text)
        if any(s in n for s in ans.slices):
            return "slice"
        if any(s and s in n for s in ans.short):
            return "short_slice"
        if ans.sig_lines and any(norm_ws(line) in ans.sig_lines for line in text.splitlines()):
            return "signature"
        for name in ans.names:
            if re.search(rf"(?<![\w$]){re.escape(name)}(?!\w)", text):
                return "name"
        return None

    def fn_rule(self, ans: Answer, repo: str, fid: str, kept_patterns: set[str], shown: str | None = None) -> str | None:
        """The rule "pattern" or "twin" for a ④(b) function; `shown` is its pre-T form when that is what the prompt shows, and
        is compared too (a twin of a new answer before T need not be one at HEAD)."""
        if kept_patterns & ans.patterns:
            return "pattern"
        if self.fn_keys(repo, fid) & ans.keys:  # also for the record's own preexisting function
            return "twin"
        if shown is not None and text_keys(self.contexts[repo].language, shown) & ans.keys:
            return "twin"
        return None

    # ----- ③ examples --------------------------------------------------------------------------------------------
    def pool_index(self, pool: list[str]) -> BM25:
        key = hashlib.sha256("\n".join(sorted(pool)).encode()).hexdigest()
        if key not in self._pool_index:
            docs = [(rid, tokens(gherkin(self.records[rid]) + "\n" + "\n".join(undefined_steps(self.records[rid]))))
                    for rid in sorted(pool)]
            self._pool_index[key] = BM25(docs)
        return self._pool_index[key]

    def eligible_example(self, r: dict, e: dict) -> bool:
        """The pool filters that are not leak rules (not recorded): another record of the same repo, not the same
        .feature file, strictly older t_vis (later examples are future code), not a user of answer infrastructure, not a
        direct neighbour, no shared CREATE function."""
        if e["id"] == r["id"] or e["repo"] != r["repo"] or e["feature_path"] == r["feature_path"]:
            return False
        if e["t_vis"] >= r["t_vis"] or e["id"] in self.infra()["users"] or e["id"] in (r.get("neighbours") or ()):
            return False
        return not {c["fn_id"] for c in e["create"]} & {c["fn_id"] for c in r["create"]}

    def example_forms(self, r: dict, e: dict) -> dict[str, str] | None:
        """Rule 5 on ③: {fn_id: shown code} for the CREATE functions of the example e changed since the query's T, each
        in its pre-T form (state.pre_t_code) keeping only the annotation lines of the patterns e's gold shows
        (shown_patterns); None when one of them cannot be located, or when its pre-T form lacks one of e's own CREATE
        patterns (the step text changed after T: the answer would show no annotation for its step)."""
        ctx, t, keep = self.contexts[r["repo"]], r["t_vis"], self.shown_patterns(e)
        cdefs, fns = set(e["create_def_ids"]), self.catalog[r["repo"]]
        out = {}
        for f in sorted({c["fn_id"] for c in e["create"]}):
            code = pre_t_code(ctx, f, t)
            if code == "":
                return None
            if code is not None:
                lines = annotation_lines(ctx.language, code)
                if not {norm_pattern(a["pattern"]) for a in fns[f]["annotations"] if a["def_id"] in cdefs} <= \
                        {p for _, p in lines}:
                    return None
                out[f] = "\n".join(l for l, p in lines if p is None or p in keep)
        return out

    def example_rule(self, r: dict, ans: Answer, e: dict, hidden: set[str]) -> str | None:
        """The leak rules on a whole example, recorded in provenance.dropped: a CREATE function hidden for the
        query's partition, a pattern shared with any CREATE item of the query, a twin of a new answer; and V(r) of the
        QUERY applied to the example's answer: a CREATE function changed since the query's T whose pre-T form cannot be
        located (rule 5, example_forms), or one whose text shown uses a class member first declared at or after T
        (rule 3)."""
        e_fns = sorted({c["fn_id"] for c in e["create"]})
        if set(e_fns) & hidden:
            return "hidden"
        if self.shown_patterns(e) & ans.patterns:
            return "pattern"
        if any(self.fn_keys(e["repo"], f) & ans.keys for f in e_fns):
            return "twin"
        ctx, t, fns = self.contexts[r["repo"]], r["t_vis"], self.catalog[r["repo"]]
        forms = self.example_forms(r, e)
        if forms is not None and any(text_keys(ctx.language, code) & ans.keys for code in forms.values()):
            return "twin"
        if forms is None:
            return "changed_after_t"
        if any(future_ref(ctx, fns[f], t, self.masked(r), forms.get(f)) for f in e_fns):
            return "future_ref"
        return None

    def example_text(self, e: dict, n: int, forms: dict[str, str] | None = None) -> str:
        steps = "\n".join(f"- {s}" for s in undefined_steps(e))
        return (f"Example {n} ({e['feature_path']}):\n{TICKS}gherkin\n{gherkin(e)}\n{TICKS}\nUndefined steps:\n{steps}\n"
                f"Answer:\n{self.gold(e, forms)}")

    def examples(self, r: dict, ans: Answer, pool: list[str], tok: Callable[[str], int],
                 dropped: list) -> tuple[list[str], str]:
        """(chosen record ids, the rendered ③ block or ""). Greedy by BM25 rank; the whole block, header included,
        stays within budget_examples."""
        hidden, budget, sep = self.hidden(r), self.params.budget_examples, tok("\n\n")
        ranked = self.pool_index(pool).rank(tokens("\n".join(undefined_steps(r))))
        chosen, texts, used, scanned = [], [], tok(EXAMPLES_HEAD), 0
        for rid, _ in ranked:
            if len(chosen) >= self.params.n_examples or scanned >= SCAN_EXAMPLES:
                break
            e = self.records[rid]
            if not self.eligible_example(r, e):
                continue
            scanned += 1
            rule, text = self.example_rule(r, ans, e, hidden), None
            if rule is None:
                text = self.example_text(e, len(chosen) + 1, self.example_forms(r, e))
                rule = self.text_rule(ans, text)
            if rule:
                dropped.append({"kind": "example", "id": rid, "rule": rule})
                continue
            cost = tok(text) + (sep if texts else 0)
            if used + cost <= budget:
                chosen.append(rid)
                texts.append(text)
                used += cost
        while texts and tok(EXAMPLES_HEAD + "\n\n".join(texts)) > budget:  # joins may tokenize differently
            chosen.pop()
            texts.pop()
        return chosen, (EXAMPLES_HEAD + "\n\n".join(texts)) if texts else ""

    # ----- ④ knowledge -------------------------------------------------------------------------------------------
    def pre_t_knowledge(self, ans: Answer, fid: str, code: str, language: str) -> tuple[str, set[str]]:
        """Rule 1 by pattern on the pre-T text of a ④(b) function: the record's own preexisting item loses the annotation
        lines of the query's CREATE patterns. (the text shown, the normalised patterns of the annotation lines it
        keeps), so that fn_rule drops any other function that carries one of them."""
        own = fid in ans.pre_fns
        lines = [(l, p) for l, p in annotation_lines(language, code) if not (own and p in ans.patterns)]
        return "\n".join(l for l, _ in lines), {p for _, p in lines if p is not None}

    def knowledge_candidates(self, r: dict, ans: Answer, dropped: list) -> list[tuple[str, str]]:
        """(fn_id, rendered code) of ④(b) candidates, BM25-ranked on that code: snapshot ∩ V(r) (rule 5: the HEAD code
        when it is unchanged since T, else the pre-T form, and nothing when that form cannot be located; rule 3: the
        text shown uses no future member), leak rules applied."""
        ctx, t = self.contexts[r["repo"]], r["t_vis"]
        fns = self.catalog[r["repo"]]
        cdefs = frozenset(r["create_def_ids"])
        docs = []
        masked = self.masked(r)
        for fid in snapshot_fns(r, fns, self.hidden(r)):
            if fid in ans.new_fns:
                continue
            pre = pre_t_code(ctx, fid, t)
            if pre == "" or future_ref(ctx, fns[fid], t, masked, pre):
                continue
            if pre is None:
                code = render_visible(fns[fid], t, cdefs)
                kept = {norm_pattern(a["pattern"]) for a in fns[fid]["annotations"]
                        if a["def_id"] not in cdefs and a["time"] is not None and a["time"] < t}
            else:
                code, kept = self.pre_t_knowledge(ans, fid, pre, ctx.language)
            # the record's own preexisting function may show its old body, but never code, names or signatures of
            # the NEW answers (a new answer may be a copy of it)
            rule = self.fn_rule(ans, r["repo"], fid, kept, None if pre is None else code) or self.text_rule(ans, code)
            if rule:
                dropped.append({"kind": "knowledge", "id": fid, "rule": rule})
                continue
            docs.append((fid, code))
        if not docs:
            return []
        bm = BM25([(fid, tokens(code)) for fid, code in docs])
        code_of = dict(docs)
        return [(fid, code_of[fid]) for fid, _ in bm.rank(tokens("\n".join(undefined_steps(r))))[:SCAN_KNOWLEDGE]]

    def skeletons(self, r: dict, ans: Answer, budget: int, tok: Callable[[str], int], dropped: list,
                  used: int) -> tuple[list[tuple[str, list[str]]], int]:
        """([(rendered outline, step fn_ids it shows)], tokens used so far) for the target files that exist at T,
        starting from `used` tokens (the ④ header). Items are added in order (own members, then inherited) while they
        fit; methods whose signature text matches a leak rule are dropped as frame lines."""
        ctx, t = self.contexts[r["repo"]], r["t_vis"]
        gone, sep = ans.new_fns | self.hidden(r), tok("\n\n")
        out = []
        for path in dict.fromkeys(c["path"] for c in r["create"]):
            parts = skeleton_parts(ctx, path, t, gone, self.masked(r))
            if parts is None:
                continue
            head, items = parts
            gap = sep if out else 0
            wrap = f"Class outline of {path}:\n{TICKS}{FENCE[r['language']]}\n" + "{}\n" + TICKS
            if used + gap + tok(wrap.format(render_skeleton(path, head, []))) > budget:
                continue
            kept: list[SkeletonItem] = []
            for it in items:
                rule = self.text_rule(ans, it.text) or ("twin" if it.fn_id and self.fn_keys(r["repo"], it.fn_id) & ans.keys else None)
                if rule:
                    dropped.append({"kind": "frame_line", "id": f"{it.origin}:{it.member.line}", "rule": rule})
                    continue
                if used + gap + tok(wrap.format(render_skeleton(path, head, kept + [it]))) <= budget:
                    kept.append(it)
            text = wrap.format(render_skeleton(path, head, kept))
            out.append((text, [it.fn_id for it in kept if it.fn_id]))
            used += gap + tok(text)
        return out, used

    def placebo(self, r: dict, target: int, tok: Callable[[str], int]) -> list[str]:
        other = PLACEBO_REPO.get(r["repo"])
        if other is None or target <= 0:
            return []
        fns = self.catalog[other]
        order = sorted(fns, key=lambda f: hashlib.sha256(f"{r['id']}|{f}".encode()).hexdigest())
        out, used = [], 0
        for fid in order[:SCAN_KNOWLEDGE]:
            text = f"{COMMENT[r['language']]} {fns[fid]['path']}\n{fns[fid]['code']}"
            cost = tok(text)
            if used + cost <= target:
                out.append(text)
                used += cost
        return out

    def knowledge(self, r: dict, ans: Answer, tok: Callable[[str], int], dropped: list) -> tuple[str, list[str], list[str]]:
        """(the rendered ④ block or "", ④(b) fn_ids, step fn_ids shown in ④(a)). ④(a) gets ratio × budget first; ④(b)
        gets the rest; the whole block, headers and fences included, stays within budget_knowledge."""
        p, fence = self.params, FENCE[r["language"]]
        budget, sep = p.budget_knowledge, tok("\n\n")
        used = tok(KNOWLEDGE_HEAD)
        skels: list[tuple[str, list[str]]] = []
        if p.mode in ("full", "4a"):
            skels, used = self.skeletons(r, ans, int(budget * (1.0 if p.mode == "4a" else p.ratio)), tok, dropped, used)

        def section(texts: list[str]) -> str:
            return "Existing step definitions:\n" + TICKS + fence + "\n" + "\n\n".join(texts) + "\n" + TICKS

        def block() -> str:
            parts = [x for x, _ in skels] + ([section(fn_texts)] if fn_texts else [])
            return (KNOWLEDGE_HEAD + "\n\n".join(parts)) if parts else ""
        chosen, fn_texts = [], []
        if p.mode in ("full", "4b"):
            for fid, code in self.knowledge_candidates(r, ans, dropped):
                if len(chosen) >= p.k:
                    break
                text = f"{COMMENT[r['language']]} {self.catalog[r['repo']][fid]['path']}\n{code}"
                cost = tok(text) + (sep if fn_texts else tok(section([""])) + (sep if skels else 0))
                if used + cost <= budget:
                    chosen.append(fid)
                    fn_texts.append(text)
                    used += cost
        if p.mode == "placebo":
            full = Composer(self.contexts, self.catalog, self.forbidden, self.records,
                            Params(p.k, p.n_examples, p.ratio, p.budget_examples, p.budget_knowledge, "full"),
                            self._pool_index, self._keys, self._infra)
            budget = min(budget, tok(full.knowledge(r, ans, tok, [])[0]))  # no more tokens than the real knowledge
            fn_texts = self.placebo(r, budget - used - tok(section([""])), tok)
        while fn_texts and tok(block()) > budget:  # joins may tokenize differently: trim the lowest-ranked item
            fn_texts.pop()
            if chosen:
                chosen.pop()
        while skels and tok(block()) > budget:
            skels.pop()
        return block(), chosen, [f for _, ids in skels for f in ids]

    # ----- the prompt ----------------------------------------------------------------------------------------------
    def instructions(self, r: dict) -> str:
        lang, ctx = r["language"], self.contexts[r["repo"]]
        lines = []
        for path in dict.fromkeys(c["path"] for c in r["create"]):
            o = ctx.outline(path)
            desc = f"class {o.name}" if o is not None else ""
            if ctx.is_new_file(path, r["t_vis"]):
                desc = (desc + ", " if desc else "") + "new file" + (f", namespace {o.namespace}" if o and o.namespace else "")
            lines.append(f"- {path}" + (f" ({desc})" if desc else ""))
        c = COMMENT[lang]
        return (f"You are an expert {r['framework']} developer working in the {r['repo']} repository "
                f"({LANG_NAME[lang]}). Write the step definitions that are missing for the scenario below.\n\n"
                "Output format:\n"
                f"- One fenced code block per target file. The first line inside each block is `{c} FILE: <path>`.\n"
                "- A block may start with import lines, followed by the new step definitions with their step "
                "annotations.\n"
                "- Output only the code to add; do not repeat existing code.\n\n"
                "Target files:\n" + "\n".join(lines))

    def scenario_block(self, r: dict) -> str:
        steps = "\n".join(f"- {s}" for s in undefined_steps(r))
        return f"Scenario ({r['feature_path']}):\n{TICKS}gherkin\n{gherkin(r)}\n{TICKS}\nUndefined steps:\n{steps}"

    def compose_prompt(self, record: dict, fmt: str, tok: Callable[[str], int], pool: list[str]) -> tuple[list[dict], dict]:
        """The prompt of one record in one format. `pool` may hold train ids of every repo: only the query's own repo
        is kept, before BM25 is built, so that every caller (build, held_out_exclusions) composes the same prompt."""
        if fmt not in FORMATS:
            raise ValueError(fmt)
        pool = [p for p in pool if self.records[p]["repo"] == record["repo"]]
        ans = self.answer(record)
        dropped: list[dict] = []
        blocks = [self.instructions(record), self.scenario_block(record)]
        examples, knowledge, skel = [], [], []
        if fmt in ("F1", "F2"):
            examples, text = self.examples(record, ans, pool, tok, dropped)
            if text:
                blocks.append(text)
        if fmt == "F2":
            text, knowledge, skel = self.knowledge(record, ans, tok, dropped)
            if text:
                blocks.append(text)
        blocks.append("Now write the code for the undefined steps.")
        return ([{"role": "user", "content": "\n\n".join(blocks)}],
                {"examples": examples, "knowledge": knowledge, "skeleton": skel, "dropped": dropped})
