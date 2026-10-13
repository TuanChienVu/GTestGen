"""Re-check the Phase 2 dataset from its files alone (no v2tools import); exit 1 if any check is non-zero. It is an
independent re-check of the RULES given the catalog: annotation times, suites, scopes and catch-alls (catalog.jsonl) and
step bindings (labels.jsonl) are trusted, every rule built on them is recomputed here; the dead-code flags are
recomputed from the step files themselves (--root)."""
import argparse
import datetime as dt
import difflib
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--phase2", type=Path, default=Path("out/phase2"))
ap.add_argument("--pairs", type=Path, default=Path("out/hist/pairs.jsonl"))
ap.add_argument("--config", type=Path, default=Path("config/phase2.json"))
ap.add_argument("--sources", type=Path, default=Path("config/sources.json"))
ap.add_argument("--root", type=Path, default=Path("repos-hist"))
args = ap.parse_args()
cfg = json.loads(args.config.read_text())
cut = int(dt.datetime.fromisoformat(cfg["cutoff_utc"].replace("Z", "+00:00")).timestamp())
load = lambda p: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
key = lambda r: (r["repo"], r["feature_path"], r["scenario_line"], r["suite"])
pairs = {key(r): r for r in load(args.pairs)}
labels = {key(l): l for l in load(args.phase2 / "labels.jsonl")}
cat = defaultdict(dict)
for f in load(args.phase2 / "catalog.jsonl"):
    cat[f["repo"]][f["fn_id"]] = f
ann = {(r, a["def_id"]): a["time"] for r, fs in cat.items() for f in fs.values() for a in f["annotations"]}
recs = {p: load(args.phase2 / "records" / f"{p}.jsonl") for p in ("train", "dev", "test")}
recs["aux_test"] = load(args.phase2 / "records" / "aux_test.jsonl") if (args.phase2 / "records" / "aux_test.jsonl").exists() else []
forbidden_file = {r: {k: set(v) for k, v in parts.items()} for r, parts in json.loads((args.phase2 / "forbidden.json").read_text()).items()}
split = json.loads((args.phase2 / "split.json").read_text())
subsets_file = json.loads((args.phase2 / "sensitivity_subsets.json").read_text())
sources = {s["repo"]: s for s in json.loads(args.sources.read_text())}
lang = {r: s["language"] for r, s in sources.items()}
excluded = {r: {t.lower() for t in v} for r, v in cfg["excluded_tags"].items()}
required = {r: {t.lower() for t in v} for r, v in cfg.get("required_tags_any", {}).items()}
c = defaultdict(int)
COMMENT = re.compile(r"^\s*(/\*\*|\*|\*/|//|#(?!\[))")
LINE_COMMENT = re.compile(r"^\s*(//|#(?!\[))")


def block_lines(text: str) -> set[int]:
    """Lines inside /* ... */ blocks that are not /** docblocks, skipping string literals and // comments (5d)."""
    out, i, n, line = set(), 0, len(text), 1
    while i < n:
        ch = text[i]
        if ch == "\n":
            line, i = line + 1, i + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif ch in "\"'":
            j = i + 1
            while j < n and text[j] not in (ch, "\n"):
                j += 2 if text[j] == "\\" else 1
            i = j + 1
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j
            end = line + text.count("\n", i, j)
            if not (text.startswith("/**", i) and not text.startswith("/**/", i)):
                out.update(range(line, end + 1))
            line, i = end, j + 2
        else:
            i += 1
    return out


dead_re, contains_re = set(), set()  # dead and contains_fn flags, recomputed from the step files at HEAD
for repo, fs in cat.items():
    by_path = defaultdict(list)
    for f in fs.values():
        by_path[f["path"]].append(f)
    for path, group in by_path.items():
        text = (args.root / sources[repo]["dir"] / path).read_text(encoding="utf-8", errors="replace")
        lines, blocks = text.splitlines(), (set() if lang[repo] == "ruby" else block_lines(text))
        def_lines = {a["line"] for g in group for a in g["annotations"]}  # every definition's annotation line in the file
        for f in group:
            commented = all(LINE_COMMENT.match(lines[a["line"] - 1]) or a["line"] in blocks for a in f["annotations"])
            bodiless = all(i in def_lines or not lines[i - 1].strip() or COMMENT.match(lines[i - 1])
                           for i in range(f["first_line"], f["body_end"] + 1))
            if commented or bodiless:
                dead_re.add((repo, f["fn_id"]))
            if any(g is not f and f["first_line"] <= g["first_line"] and g["body_end"] <= f["body_end"] for g in group):
                contains_re.add((repo, f["fn_id"]))
c["dead_flag_wrong"] = sum(f["dead"] != ((r, fid) in dead_re) for r, fs in cat.items() for fid, f in fs.items())
c["contains_fn_flag_wrong"] = sum(f["contains_fn"] != ((r, fid) in contains_re) for r, fs in cat.items() for fid, f in fs.items())


def texts_of(repo: str, fn: dict) -> tuple[str, str]:
    """The body_key and body_key_nosig strings before their threshold, written again from the twin definitions
    (deliberately not imported)."""
    ann_lines = {a["line"] for a in fn["annotations"]}
    kept = [l for i, l in enumerate(fn["code"].splitlines(), start=fn["first_line"]) if i not in ann_lines and not COMMENT.match(l)]
    s = t = " ".join(" ".join(kept).split())
    if lang.get(repo) != "ruby":
        j = t.find("{")
        if j >= 0 and "(" in t[:j] and ";" not in t[:j]:
            t = t[j + 1:].strip()
    return s, t


def keys_of(repo: str, fn: dict) -> set[tuple]:
    s, t = texts_of(repo, fn)
    return ({("sig", s)} if len(s) >= 60 else set()) | ({("nosig", t)} if len(t) >= 60 else set())


def best_near(answers: dict[str, str], candidates: dict[str, str]) -> dict[str, tuple[str, float]]:
    """The near-duplicate rule with the stdlib only: candidate -> (best answer, ratio) for ratio(candidate, answer) >= 0.90, both texts
    at least 60 characters, autojunk off; answers visited in id order, a tie keeps the first."""
    best: dict[str, tuple[str, float]] = {}
    sm = difflib.SequenceMatcher(None, autojunk=False)
    for f, b in sorted(answers.items()):
        if len(b) < 60:
            continue
        sm.set_seq2(b)
        for g, a in candidates.items():
            if len(a) < 60:
                continue
            sm.set_seq1(a)
            if sm.real_quick_ratio() >= 0.9 and sm.quick_ratio() >= 0.9:
                ratio = sm.ratio()
                if ratio >= 0.9 and ratio > best.get(g, ("", 0.0))[1]:
                    best[g] = (f, ratio)
    return best


def lcs(a: str, b: str) -> int:  # bit-parallel longest common subsequence: an upper bound of SequenceMatcher's matches
    masks: dict[str, int] = {}
    for i, ch in enumerate(b):
        masks[ch] = masks.get(ch, 0) | (1 << i)
    full = (1 << len(b)) - 1
    v = full
    for ch in a:
        u = v & masks.get(ch, 0)
        v = ((v + u) | (v - u)) & full
    return len(b) - bin(v).count("1")


def best_container(answers: dict[str, str], candidates: dict[str, str]) -> dict[str, tuple[str, float]]:
    """The containment rule with the stdlib only: candidate -> (answer, share) where share = matched characters of
    SequenceMatcher(None, answer, candidate, autojunk=False) / len(answer) >= 0.90, answer >= 200 characters."""
    best: dict[str, tuple[str, float]] = {}
    sm = difflib.SequenceMatcher(None, autojunk=False)
    long = [(f, a, Counter(a)) for f, a in sorted(answers.items()) if len(a) >= 200]
    for g, b in sorted(candidates.items()):
        counts_b = Counter(b)
        sm.set_seq2(b)
        for f, a, counts_a in long:
            need = 0.9 * len(a)
            if len(b) < need or sum((counts_a & counts_b).values()) < need or lcs(a, b) < need:
                continue
            sm.set_seq1(a)
            share = sum(m.size for m in sm.get_matching_blocks()) / len(a)
            if share >= 0.9 and share > best.get(g, ("", 0.0))[1]:
                best[g] = (f, share)
    return best


def after_cut(repo: str, f: str) -> bool:  # earliest DATED annotation after the cutoff; undated functions are not
    times = [a["time"] for a in cat[repo][f]["annotations"] if a["time"] is not None]
    return bool(times) and min(times) > cut


fkeys = {(r, f): keys_of(r, fn) for r, fs in cat.items() for f, fn in fs.items()}
forbidden, dev_answer, aux_answer = defaultdict(set), defaultdict(set), defaultdict(set)
for r in recs["test"]:
    forbidden[r["repo"]] |= set(pairs[key(r)]["new_target_fns"])
    forbidden[r["repo"]] |= {x["fn_id"] for x in r["create"]}
answers_of = {r["id"]: set(pairs[key(r)]["new_target_fns"]) | {x["fn_id"] for x in r["create"]} for r in recs["test"]}
test_keys = {(r, k) for r, fs in forbidden.items() for f in fs for k in fkeys.get((r, f), ())}
twins = {(r, f) for (r, f), ks in fkeys.items()  # post-cutoff twins of a test answer join it
         if f not in forbidden.get(r, set()) and any((r, k) in test_keys for k in ks) and after_cut(r, f)}
near_rows, contain_rows = [], []  # post-cutoff near-duplicates (5c) and containers (5d) of a test answer join it
nosig = {r: {f: texts_of(r, fn)[1] for f, fn in fs.items()} for r, fs in cat.items()}
for r, fs in sorted(forbidden.items()):
    answer_texts = {f: nosig[r][f] for f in fs if f in nosig[r]}
    candidates = {g: t for g, t in nosig[r].items() if g not in fs and after_cut(r, g)}
    near_rows += [[r, g, f, round(x, 4)] for g, (f, x) in best_near(answer_texts, candidates).items()]
    contain_rows += [[r, g, f, round(x, 4)] for g, (f, x) in best_container(answer_texts, candidates).items()]
near = {(r, g) for r, g, _, _ in near_rows}
contain = {(r, g) for r, g, _, _ in contain_rows}
listed_rows = lambda name: {tuple(x) for x in split.get(name, [])}
c["near_list_mismatch"] = len(listed_rows("test_answer_near_duplicates") ^ {tuple(x) for x in near_rows})
c["container_list_mismatch"] = len(listed_rows("test_answer_containers") ^ {tuple(x) for x in contain_rows})
for r, f in twins | near | contain:
    forbidden[r].add(f)
for r in recs["dev"]:
    dev_answer[r["repo"]] |= {x["fn_id"] for x in r["create"]}
for r in recs["aux_test"]:
    aux_answer[r["repo"]] |= {x["fn_id"] for x in r["create"]}
empty = {"test": set(), "dev": set(), "aux": set()}
c["forbidden_file_mismatch"] = sum(forbidden[r] != forbidden_file.get(r, empty)["test"] or dev_answer[r] != forbidden_file.get(r, empty)["dev"]
                                   or aux_answer[r] != set(forbidden_file.get(r, empty).get("aux", set()))
                                   for r in set(forbidden) | set(dev_answer) | set(aux_answer) | set(forbidden_file))
aux_files = {(r["repo"], r["feature_path"]) for r in recs["aux_test"]}

scen_part, file_part, file_parts, skey_scen = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
lc = [f"lc{n}" for n in sorted(cfg["learning_curve"])]
other_files, other_creates = set(), set()
for part in ("dev", "test", "aux_test"):
    for r in recs[part]:
        other_files.add((r["repo"], r["feature_path"]))
        other_creates |= {(r["repo"], x["fn_id"]) for x in r["create"]}
test_code = {hashlib.sha1(" ".join(x["code"].split()).encode()).hexdigest() for r in recs["test"] for x in r["create"]}
reached = defaultdict(set)  # functions train/dev reach: targets, bound steps, and the producer's snapshots
HIDE = {"train": ("test", "dev", "aux"), "dev": ("test", "aux"), "test": (), "aux_test": ()}
for part, rs in recs.items():
    for r in rs:
        k = key(r)
        lab, pr = labels[k], pairs[k]
        scen_part[k[:3]].add(part)
        file_part[k[:2]].add("test" if part == "test" else "train_dev")
        file_parts[k[:2]].add(part)
        skey_scen[pr["structure_key"]].add(k[:3])
        c["file_shared_with_aux"] += part != "aux_test" and k[:2] in aux_files
        c["uses_aux_answer"] += part in ("train", "dev") and any(f in aux_answer[r["repo"]] for f in r["step_fns"])
        c["aux_ambiguous"] += part == "aux_test" and pr["ambiguous_steps"] > 0
        kinds, fns = r["step_kinds"], r["step_fns"]
        created = {f for f, kd in zip(fns, kinds) if kd == "create"}
        c["no_create"] += not r["create"]
        c["create_mismatch"] += created != {x["fn_id"] for x in r["create"]}
        c["preexisting_flag_wrong"] += sum(x["preexisting"] != any(a["time"] is not None and a["time"] < r["scenario_time"]
                                                                   for a in cat[r["repo"]][x["fn_id"]]["annotations"]) for x in r["create"])
        c["post_cutoff_code_wrong"] += sum(x["post_cutoff_code"] != (x["fn_id"] in set(pr["new_target_fns"])) for x in r["create"])
        c["create_def_ids_wrong"] += r["create_def_ids"] != sorted(lab["create_def_ids"])
        c["t_vis_wrong"] += r["t_vis"] != max((t for t in (ann[(r["repo"], d)] for d in lab["create_def_ids"]) if t is not None), default=None)
        c["reuse_mismatch"] += set(r["reuse_fns"]) != {f for f, kd in zip(fns, kinds) if kd == "reuse"} - created
        for ids, kd in zip(lab["step_def_ids"], kinds):
            times = [ann[(r["repo"], i)] for i in ids]
            c["create_step_existed_before"] += kd == "create" and any(t is not None and t < r["scenario_time"] for t in times)
            c["reuse_step_did_not_exist"] += kd == "reuse" and not any(t is not None and t < r["scenario_time"] for t in times)
        c["excluded_tag_present"] += bool({t.lower() for t in r["tags"]} & excluded.get(r["repo"], set()))
        c["subset_outside_train"] += bool(r["subsets"]) and part != "train"
        c["lc_not_nested"] += any(a in r["subsets"] and b not in r["subsets"] for a, b in zip(lc, lc[1:]))
        prefix = cfg["exec_suite_prefix"].get(r["repo"])
        c["bad_exec_core"] += r["exec_core"] != (part == "test" and prefix is not None and r["suite"].startswith(prefix))
        if not re.search(r"Scenario (Outline|Template)|Examples:", r["scenario"]):
            c["step_text_missing"] += any(s not in r["scenario"] and s not in r["background"] for s in r["steps"])
        tags = {t.lower() for t in r["tags"]}
        c["required_tag_missing"] += r["repo"] in required and not tags & required[r["repo"]]
        reachable = []  # active in the record's suite/scope, not a catch-all, with a pattern older than the scenario
        for fid, f in cat[r["repo"]].items():
            if f["catch_all"] or not any(a["time"] is not None and a["time"] < r["scenario_time"] for a in f["annotations"]):
                continue
            if r["scoping"] == "specflow_scope":
                if f["scope_tags"] and not any(set(t) & tags for t in f["scope_tags"]):
                    continue
            elif r["suite"] not in f["suites"]:
                continue
            reachable.append(fid)
        visible = [fid for fid in reachable  # hidden by the rules recomputed here
                   if (r["repo"], fid) not in dead_re and (r["repo"], fid) not in contains_re
                   and not (part in ("train", "dev") and fid in forbidden[r["repo"]] or part == "train" and fid in dev_answer[r["repo"]]
                            or part in ("train", "dev") and fid in aux_answer[r["repo"]])]
        c["snapshot_size_mismatch"] += len(visible) != r["snapshot_size"]
        fh = forbidden_file.get(r["repo"], {})  # the snapshot as the PRODUCER builds it: catalog flags and forbidden.json
        hide = set().union(*(fh.get(x, set()) for x in HIDE[part]))
        producer_view = [fid for fid in reachable if not cat[r["repo"]][fid]["dead"] and not cat[r["repo"]][fid]["contains_fn"]
                         and fid not in hide]
        c["dead_fn_in_snapshot"] += any((r["repo"], fid) in dead_re for fid in producer_view)
        c["span_contains_fn_in_snapshot"] += any((r["repo"], fid) in contains_re for fid in producer_view)
        if part == "test":
            c["test_not_post_cutoff"] += not r["scenario_time"] > cut
            c["test_create_without_new_code"] += not set(lab["create_def_ids"]) & set(pr["new_target_ids"])
        else:
            c["train_dev_uses_forbidden"] += any(f in forbidden[r["repo"]] for f in fns)
            c["train_creates_dev_answer"] += part == "train" and any(x["fn_id"] in dev_answer[r["repo"]] for x in r["create"])
            c["train_uses_dev_answer"] += part == "train" and any(f in dev_answer[r["repo"]] for f in fns)
            c["train_dev_answer_code_equals_test"] += any(
                hashlib.sha1(" ".join(x["code"].split()).encode()).hexdigest() in test_code for x in r["create"])
        if part in ("train", "dev"):  # a post-cutoff copy (computed here) as target, bound step, or producer snapshot entry
            exposed = {x["fn_id"] for x in r["create"]} | set(fns) | set(producer_view)
            reached[r["repo"]] |= exposed
            c["test_answer_twin_in_training"] += any((r["repo"], f) in twins for f in exposed)
            c["near_duplicate_in_training"] += any((r["repo"], f) in near for f in exposed)
            c["test_answer_container_in_training"] += any((r["repo"], f) in contain for f in exposed)
        c["train_neighbour_of_other_partition"] += part == "train" and (
            k[:2] in other_files or any((r["repo"], x["fn_id"]) in other_creates for x in r["create"]))
c["scenario_in_2_parts"] = sum(len(v) > 1 for v in scen_part.values())
c["file_shared_train_dev"] = sum({"train", "dev"} <= v for v in file_parts.values())
c["structure_key_on_2_scenarios"] = sum(len(v) > 1 for v in skey_scen.values())  # check_split's rule, now with aux_test
by_file, by_fn = defaultdict(set), defaultdict(set)  # direct neighbours among train records, not transitive
for r in recs["train"]:
    by_file[(r["repo"], r["feature_path"])].add(r["id"])
    for x in r["create"]:
        by_fn[(r["repo"], x["fn_id"])].add(r["id"])
neighbours = {r["id"]: (by_file[(r["repo"], r["feature_path"])].union(*(by_fn[(r["repo"], x["fn_id"])] for x in r["create"])))
              - {r["id"]} for r in recs["train"]}
c["neighbours_wrong"] = (sum(r["neighbours"] != sorted(neighbours[r["id"]]) for r in recs["train"])
                         + sum(r["neighbours"] is not None for p in ("dev", "test", "aux_test") for r in recs[p]))
listed = {r["id"]: set(r["neighbours"] or ()) for r in recs["train"]}
c["neighbours_asymmetric"] = sum(i not in listed.get(n, set()) for i, ns in listed.items() for n in ns)

touched = set()  # sensitivity subsets: residual relations of test answers with what train/dev reach
sm = difflib.SequenceMatcher(None, autojunk=False)
for repo in sorted({r["repo"] for r in recs["test"]}):
    for f in sorted({f for r in recs["test"] if r["repo"] == repo for f in answers_of[r["id"]] if f in nosig[repo]}):
        b = nosig[repo][f]
        if len(b) < 60:
            continue
        sm.set_seq2(b)
        for g in sorted(reached[repo]):
            a = nosig[repo].get(g, "")
            if g == f or len(a) < 60:
                continue
            post = after_cut(repo, g)
            low = 0.85 if post else 0.9
            sm.set_seq1(a)
            if sm.real_quick_ratio() >= low and sm.quick_ratio() >= low:
                ratio = sm.ratio()
                if (low <= ratio < 0.9) if post else ratio >= 0.9:
                    touched.add((repo, f))
                    break
expected_subsets = {
    "post_cutoff_all": sorted(r["id"] for r in recs["test"] if all(x["post_cutoff_code"] for x in r["create"])),
    "residual_clean": sorted(r["id"] for r in recs["test"] if not any((r["repo"], f) in touched for f in answers_of[r["id"]])),
    "time_coherent_30d": sorted(r["id"] for r in recs["test"] if r["t_vis"] - r["scenario_time"] <= 30 * 86400)}
c["sensitivity_subsets_mismatch"] = sum(subsets_file.get(n) != v for n, v in expected_subsets.items()) + len(set(subsets_file) - set(expected_subsets))
aux_by_repo = defaultdict(set)
for r in recs["aux_test"]:
    aux_by_repo[r["repo"]].add(key(r)[:3])
c["aux_quota_mismatch"] = sum(len(aux_by_repo[repo]) != k for repo, k in cfg.get("aux_holdout", {}).items())
c["file_shared_with_test"] = sum(len(v) > 1 for v in file_part.values())
uniq = {p: len({key(r)[:3] for r in rs}) for p, rs in recs.items()}
c["below_minimum"] = sum(uniq[p] < m for p, m in cfg["minimum_unique_scenarios"].items())
by_repo = defaultdict(set)
for r in recs["test"]:
    by_repo[r["repo"]].add(key(r)[:3])
c["test_over_quota"] = sum(len(v) > cfg["test_quota"].get(repo, 0) for repo, v in by_repo.items())
print(json.dumps({"unique_scenarios": uniq, "checks": dict(sorted(c.items()))}, indent=2))
raise SystemExit(1 if any(c.values()) else 0)
