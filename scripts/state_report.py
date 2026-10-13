"""Freeze the self_contained flag of every record and write out/phase3/STATE_REPORT.md: the flag by
repo and partition, new-file targets, the members and ④(a) lines that rules 1–3 remove and the ④(b) snapshot functions
that rule 3 drops (by partition and repo), the parent gate, catalog step functions that match no outline method, the
answer infrastructure, and a fixed sample of 30 records (sha256 order, 10 Sylius exec_core, 10 PrestaShop, 5 fineract,
5 ph-ee test records) printed with their answer and ④(a) outlines for the manual check that must happen BEFORE the
flag is used. Refuses to run when code or config is uncommitted."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2tools.compose import TICKS  # noqa: E402
from v2tools.phase3io import dirty_code, load, sha256_file  # noqa: E402
from v2tools.records import snapshot_fns  # noqa: E402
from v2tools.state import future_ref, member_use, pre_t_code, self_contained, skeleton, t_view  # noqa: E402

SAMPLE = (("Sylius/Sylius", True, 10), ("PrestaShop/PrestaShop", None, 10), ("apache/fineract", None, 5),
          ("openMF/ph-ee-integration-test", None, 5))


def gone_of(comp, r) -> set:
    return {c["fn_id"] for c in r["create"] if not c["preexisting"]} | comp.hidden(r)


def flags(comp) -> dict[str, bool]:
    out = {}
    for rid, r in sorted(comp.records.items()):
        out[rid] = self_contained(comp.contexts[r["repo"]], r, gone_of(comp, r), comp.masked(r))
    return out


def sample(records: dict) -> list[str]:
    out = []
    for repo, exec_core, n in SAMPLE:
        pool = sorted((i for i, r in records.items() if r["partition"] == "test" and r["repo"] == repo
                       and (exec_core is None or r["exec_core"] == exec_core)), key=lambda i: hashlib.sha256(i.encode()).hexdigest())
        out += pool[:n]
    return out


def removal_rates(comp) -> dict[tuple[str, str], Counter]:
    """Per (partition, repo): target files outlined at T; class members (own and inherited, as rule 4 lists them)
    shown, removed by rule 3, by rules 1–2 and by rule 5 (no declaration before T), with their ④(a) lines; ④(b)
    snapshot functions, those shown in their pre-T form (changed since T), and those that rule 5 (changed, and the
    pre-T form cannot be located) and rule 3 (on the text shown) drop."""
    out: dict[tuple[str, str], Counter] = {}
    for r in comp.records.values():
        ctx, c = comp.contexts[r["repo"]], out.setdefault((r["partition"], r["repo"]), Counter())
        if ctx.language == "ruby":
            continue
        t, gone, masked = r["t_vis"], gone_of(comp, r), comp.masked(r)
        for path in dict.fromkeys(x["path"] for x in r["create"]):
            removed: list = []
            shown = t_view(ctx, path, t, gone, masked, removed)
            if shown is None:
                continue
            c["files"] += 1
            for m, _, why in [(x.member, x.origin, "kept") for x in shown] + removed:
                c[why] += 1
                c[why + "_lines"] += len(m.text.splitlines())
        fns, new = comp.catalog[r["repo"]], {x["fn_id"] for x in r["create"] if not x["preexisting"]}
        for fid in snapshot_fns(r, fns, comp.hidden(r)):
            if fid not in new:
                c["fns"] += 1
                pre = pre_t_code(ctx, fid, t)
                c["fns_pre_t"] += bool(pre)
                c["fns_rule5"] += pre == ""
                c["fns_rule3"] += pre != "" and future_ref(ctx, fns[fid], t, masked, pre)
    return out


def unmatched_steps(comp) -> tuple[Counter, Counter, list[str]]:
    """Catalog step functions in an outlined file whose fn_id line is no method's body line, so rule 2 cannot see them
    as steps (only rule 3 by name protects them): (all by repo, cucumber-java8 lambdas by repo, the other fn_ids)."""
    every, lam, other = Counter(), Counter(), []
    for repo, fns in sorted(comp.catalog.items()):
        ctx = comp.contexts.get(repo)
        if ctx is None or ctx.language == "ruby":
            continue
        for fid, fn in sorted(fns.items()):
            o = ctx.outline(fn["path"])
            if o is None or any(m.body_line == int(fid.rsplit(":", 1)[1]) for m in o.members if m.kind == "method"):
                continue
            every[repo] += 1
            if re.match(r"\s*(Given|When|Then|And|But)\s*\(", fn["code"]):
                lam[repo] += 1
            else:
                other.append(fid)
    return every, lam, other


def manual_entry(comp, rid: str, flag: bool) -> str:
    r = comp.records[rid]
    ctx = comp.contexts[r["repo"]]
    lines = [f"### {rid}", "", f"- self_contained: **{flag}**; t_vis {r['t_vis']}", ""]
    for c, used, missing in member_use(ctx, r, gone_of(comp, r), comp.masked(r)):
        new = ctx.is_new_file(c["path"], r["t_vis"])
        lines += [f"- `{c['fn_id']}` (preexisting={c['preexisting']}, new file={new})",
                  f"  - members used (✗ = not in V(r) and not defined by the answer): "
                  f"{', '.join(u + (' ✗' if u in missing else '') for u in sorted(used)) or '—'}",
                  "", TICKS, c["code"], TICKS, ""]
        sk = None if new else skeleton(ctx, c["path"], r["t_vis"], gone_of(comp, r), comp.masked(r))
        if sk:
            lines += ["④(a):", TICKS, sk, TICKS, ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("config/sources.json"))
    ap.add_argument("--phase2", type=Path, default=Path("out/phase2"))
    ap.add_argument("--symtimes", type=Path, default=Path("out/phase3/symtimes.jsonl"))
    ap.add_argument("--full-root", type=Path, default=Path("repos-full"))
    ap.add_argument("--out", type=Path, default=Path("out/phase3"))
    args = ap.parse_args()
    dirty = dirty_code(ROOT)
    if dirty:
        print("refusing to run: uncommitted code or config in", ROOT, file=sys.stderr)
        print(dirty, file=sys.stderr)
        return 2
    comp = load(args.config, args.phase2, args.symtimes, args.full_root)
    f = flags(comp)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "self_contained.jsonl").write_text("".join(json.dumps({"record_id": i, "self_contained": v}) + "\n"
                                                           for i, v in sorted(f.items())), encoding="utf-8")
    tot, sc, newt = Counter(), Counter(), Counter()
    for rid, r in comp.records.items():
        key = (r["partition"], r["repo"] + (" exec_core" if r["exec_core"] else ""))
        tot[key] += 1
        sc[key] += f[rid]
        newt[key] += any(comp.contexts[r["repo"]].is_new_file(c["path"], r["t_vis"]) for c in r["create"])
    inf = comp.infra()
    gate = json.loads((args.symtimes.parent / "PARENTS.json").read_text())
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    md = ["# Phase 3 — trạng thái nhìn thấy V(r)", "",
          f"- v2tools commit: `{commit}`; PHASE2_SHA256SUMS sha256 `{sha256_file(args.phase2 / 'PHASE2_SHA256SUMS')}`; "
          f"symtimes sha256 `{sha256_file(args.symtimes)}`; self_contained.jsonl sha256 "
          f"`{sha256_file(args.out / 'self_contained.jsonl')}`", "",
          "## self_contained theo phần và repo", "", "| phần | repo | n | self_contained | có tệp đích mới |", "|---|---|---|---|---|"]
    for key in sorted(tot):
        md.append(f"| {key[0]} | {key[1]} | {tot[key]} | {sc[key]} ({sc[key] / tot[key]:.0%}) | {newt[key]} |")
    md += ["", "## Thành viên, dòng ④(a) và hàm ④(b) bị bỏ, theo phần và repo (luật 1–3, 5)", "",
           "Thành viên = của lớp đích và kế thừa (luật 4), trên các tệp đích có ở T. Tỷ lệ che = luật 3 / tổng; "
           "luật 5 ở ④(a) = không có khai báo trước T; ở ④(b) = hàm đã đổi kể từ T: hiện dạng trước T, hoặc bị bỏ "
           "khi không định vị được dạng đó.", "",
           "| phần | repo | tệp đích | thành viên hiện | che luật 3 | bỏ luật 1–2 | bỏ luật 5 | tỷ lệ che | "
           "dòng ④(a) che luật 3 | hàm ④(b) snapshot | dạng trước T | bỏ luật 5 | bỏ luật 3 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def pct(a: int, b: int) -> str:
        return f"{a / b:.1%}" if b else "—"
    for (part, repo), c in sorted(removal_rates(comp).items()):
        n = c["kept"] + c["rule3"] + c["rule12"] + c["rule5"]
        lines = sum(c[k + "_lines"] for k in ("kept", "rule3", "rule12", "rule5"))
        md.append(f"| {part} | {repo} | {c['files']} | {c['kept']} | {c['rule3']} | {c['rule12']} | {c['rule5']} | "
                  f"{pct(c['rule3'], n)} | {c['rule3_lines']}/{lines} ({pct(c['rule3_lines'], lines)}) | {c['fns']} | "
                  f"{c['fns_pre_t']} ({pct(c['fns_pre_t'], c['fns'])}) | {c['fns_rule5']} ({pct(c['fns_rule5'], c['fns'])}) | "
                  f"{c['fns_rule3']} ({pct(c['fns_rule3'], c['fns'])}) |")
    md += ["", "## Cổng lớp cha", "", "| repo | resolved | external | unresolved |", "|---|---|---|---|"]
    md += [f"| {repo} | {g['resolved']} | {g['external']} | {g['unresolved']} |" for repo, g in sorted(gate.items())]
    every, lam, other = unmatched_steps(comp)
    md += ["", "## Hàm step không khớp phương thức nào của outline", "",
           "Luật 2 không thấy các hàm này là step; chỉ luật 3 (theo tên) còn che chúng.", "",
           "| repo | không khớp | trong đó lambda cucumber-java8 |", "|---|---|---|"]
    md += [f"| {repo} | {every[repo]} | {lam[repo]} |" for repo in sorted(every)]
    md += ["", f"- khác lambda: {', '.join(f'`{f}`' for f in other) or '—'}"]
    md += ["", "## Hạ tầng đáp án (test và aux_test)", "",
           f"- thành viên: {sum(len(v) for v in inf['members'].values())} "
           f"({', '.join(f'{k} {len(v)}' for k, v in sorted(inf['members'].items()))})",
           f"- bản ghi train/dev dùng tới (bị loại khỏi huấn luyện và pool): {len(inf['users'])}", "",
           "## Mẫu kiểm tay (30 bản ghi test)", ""]
    for rid in sample(comp.records):
        md += [manual_entry(comp, rid, f[rid]), ""]
    (args.out / "STATE_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(f), "self_contained": sum(f.values()), "infra_users": len(inf["users"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
