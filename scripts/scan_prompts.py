"""Independent acceptance scan of a built prompt directory: imports no v2tools module, reads only the prompt files, the
Phase 2 records and the catalog. For every row it checks that ③ and ④ contain none of the record's CREATE annotation
lines, none of its normalised CREATE patterns (quoted with ', " or /, or as the text of a docblock annotation such as
`@Given <pattern>`), none of the method names of its preexisting=False items as identifiers and not the last 200
normalised characters of their code; that every row is one user turn and test/aux rows carry no answer; and that
PROMPTS_SHA256SUMS covers every other file. Exit 1 on any finding. --report writes the provenance.dropped counts by
partition, kind and rule (drops are reported per partition)."""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HEADS = ("Examples from this repository:", "Project knowledge:")
DOC = re.compile(r"^\s*(?:/\*\*|\*)?\s*@(?:given|when|then|and|but)\s+(.*?)\s*(?:\*/)?\s*$", re.I)
SLICE = 200
DECL = {"php": re.compile(r"function\s+(\w+)\s*\("), "java": re.compile(r"^\s*(?:public|protected|private)[\w<>\[\], ]*\s(\w+)\s*\(", re.M),
        "csharp": re.compile(r"^\s*(?:public|protected|private|internal)[\w<>\[\], ]*\s(\w+)\s*\(", re.M)}


def norm(s: str) -> str:
    return " ".join(s.split())


def jsonl_rows(path: Path) -> list[dict]:
    """JSONL rows split on "\\n" only, from the decoded bytes, as phase3io.jsonl_rows does (this script imports no
    v2tools module). A blank line inside the file still raises."""
    lines = path.read_bytes().decode("utf-8").split("\n")
    if lines[-1] == "":
        lines.pop()
    return [json.loads(line) for line in lines]


def tail(text: str) -> str:
    """The part of the prompt from ③ (or ④ when there is no ③) on; "" for F0."""
    idx = [text.find(h) for h in HEADS if h in text]
    return text[min(idx):] if idx else ""


def scan(prompt_dir: Path, records: dict, catalog: dict, drops: Counter | None = None) -> Counter:
    found: Counter = Counter()
    sums = dict(reversed(l.split("  ", 1)) for l in (prompt_dir / "PROMPTS_SHA256SUMS").read_text().splitlines())
    files = {p.name for p in prompt_dir.iterdir()} - {"PROMPTS_SHA256SUMS"}
    if set(sums) != files:
        found["sums_file_set"] += 1
    for name, h in sums.items():
        if (prompt_dir / name).exists() and hashlib.sha256((prompt_dir / name).read_bytes()).hexdigest() != h:
            found["sums_mismatch"] += 1
    for f in sorted(prompt_dir.glob("*.jsonl")):
        if f.name == "excluded.jsonl":
            continue
        for row in jsonl_rows(f):
            r = records[row["record_id"]]
            if r["partition"] in ("test", "aux_test") and row["answer"] is not None:
                found["held_out_answer"] += 1
            if [m["role"] for m in row["messages"]] != ["user"]:
                found["not_one_user_turn"] += 1
            if drops is not None:
                for d in row["provenance"]["dropped"]:
                    drops[(r["partition"], d["kind"], d["rule"])] += 1
            t = tail(row["messages"][0]["content"])
            lines = {norm(l) for l in t.splitlines()}
            flat = norm(t).casefold()
            doc = {norm(m.group(1)).casefold() for l in t.splitlines() if (m := DOC.match(l))}
            fns = catalog[r["repo"]]
            cdefs = set(r["create_def_ids"])
            for c in r["create"]:
                fn = fns[c["fn_id"]]
                src = fn["code"].splitlines()
                for a in fn["annotations"]:
                    if a["def_id"] in cdefs:
                        if norm(src[a["line"] - fn["first_line"]]) in lines:
                            found["create_annotation_line"] += 1
                        p = norm(a["pattern"]).casefold()
                        if any(re.search(re.escape(q + p + q), flat) for q in ("'", '"', "/")):
                            found["create_pattern_quoted"] += 1
                        if p in doc:
                            found["create_pattern_docblock"] += 1
                if not c["preexisting"]:
                    if norm(c["code"])[-SLICE:] in norm(t):
                        found["short_answer_slice"] += 1
                    for m in DECL.get(r["language"], re.compile(r"(?!)")).findall(c["code"]):
                        if re.search(rf"(?<![\w$]){re.escape(m)}(?!\w)", t):
                            found["answer_method_name"] += 1
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--phase2", type=Path, default=Path("out/phase2"))
    ap.add_argument("--report", type=Path, help="write provenance.dropped counts by partition, kind and rule (JSON)")
    args = ap.parse_args()
    records = {}
    for part in ("train", "dev", "test", "aux_test"):
        for r in jsonl_rows(args.phase2 / "records" / f"{part}.jsonl"):
            records[r["id"]] = r
    catalog: dict = defaultdict(dict)
    for row in jsonl_rows(args.phase2 / "catalog.jsonl"):
        catalog[row["repo"]][row["fn_id"]] = row
    drops: Counter = Counter()
    found = scan(args.prompts, records, catalog, drops)
    if args.report:
        table: dict = {}
        for (part, kind, rule), n in sorted(drops.items()):
            table.setdefault(part, {}).setdefault(kind, {})[rule] = n
        args.report.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(sorted(found.items())) or {"findings": 0}))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
