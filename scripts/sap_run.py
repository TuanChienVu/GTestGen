"""Run the locked SAP on the test score rows (§5 of the design specification at 40bc09a): completeness check,
confirmatory RQ1/RQ2 on CA with both tests and Holm, descriptive analyses, pre-written answers, the blind SemV sheet.
Refuses to run (exit 2) when --lock is not the file of --lock-commit, an ancestor of HEAD on a branch of origin (a fork
does not do), when the sha256 of any locked input or of the 3D analysis code differs from SAP_LOCK.json (other code
files and git are not looked at), or when the pre-T list (--pre-t-flags, the frozen FLAGGED_PRE_T.tsv) does not have the
sha256 config/sap.json names or does not pass sap.pre_t_errors; the robustness view without_pre_t drops the records it
lists for the run's partition (test, or aux_test with --descriptive-only). Also refuses (exit 2) when the lock's own
commit date (--lock-commit, printed as today, taken from git) is not strictly before the scored_utc of a given
SCORES_META, so a lock cannot have been adjusted after the scores it will run on were already known
(sap.lock_precedes_scoring_errors; a documented limit remains: git itself still rests on the local checkout's clock and
history, so the independent time anchor is the lock's push to origin, recorded outside the checkout, not anything --repo
alone can attest). Refuses (exit 3), so that no denominator moves unseen: when the er_complete.json next to an ER
control file is missing, not complete, says running or has a manifest_sha256 that is not the --er-manifest's (the
er_results contract, 0c42865); when the ER control results do not cover exactly the exec_core records of the locked
records/test.jsonl with one gold and one stub row each (a partial ER control file, a record with no row at all, a
duplicated (record_id, condition) row, a row of a test record that is not exec_core; the rows of records outside
records/test.jsonl, the dev records of the same run, are ignored: sap.er_scope), or an ER control row of a test record
is not the result of its --er-manifest line; when a --scores row carries filled_missing, is of a record with no gold row
in the locked controls.jsonl, has a scenario_key, repo or feature_path that is not that of its record_id, has an excl_bv
or a ca_defined that is not its record's in the locked controls run, has no ca although ca_defined or no bv although not
excl_bv, or (full mode) is an anchor-model main row of an ER-eligible record without an er value (sap.score_row_errors);
in full mode, when a design run (anchor model, a design condition, variant main) has a row under a seed the design does
not have, a duplicated row, or no row for a record of the controls run that compose did not drop (usable or not: BV and
ER count the mechanical exclusions) that --accepted-missing does not list (sap.check_complete over sap.design_records),
or when the confirmatory comparisons do not share one n and one two-level n; when a row's sub_* flag differs from the
locked sensitivity_subsets.json, or its self_contained from the locked self_contained.jsonl (the sensitivity analyses of
CA then use these files' membership: subsets (i)-(iii), and (iv) self_contained; sap.subset_flags). An accepted missing
row scores CA = 0 and BV = 0 with its record's locked flags, and has no ER value. CA_prec (D11, the design specification
at 3739dd6, fixed 25/09/2026): CA is unchanged; the pre-registered sensitivity analysis re-runs every confirmatory
comparison on CA_prec = CA × [call_precision ≥ ca_prec_tau] (0.25 in the locked config) with the same procedure and Holm
families, reported next to CA with every difference of conclusion stated (SAP_RESULTS.json ca_prec); the report adds the
controls' CA_prec rates at each pre-registered τ (stub-v3top4 included, D11 (d), the design specification at 4d8bf2a),
ca_ref_v3names and ca_ref_top4 next to v3m and v3all with the CA above each ("CA_prec vượt mức lười top4" for top4:
CA_prec = 1 and ca_ref_top4 = 0, also next to CA_prec for the two runs of every RQ1/RQ2 comparison), the rate of
php_tag_stripped per run (§1.4), and call_precision and n_calls per run and per repo, PrestaShop's rows saying that CA
is easy to reach there through infrastructure calls; a score row without call_precision or n_calls is refused (exit 3).

Generation (fixed 26/09/2026; design specification §5 at 7af1673; exit 3): --manifest and --coverage (the generation
run's manifest.<variant>.jsonl and coverage.<variant>.json, in pairs) must keep their contract (sap.read_manifests:
fields, status, line_sha256, attempts, receipt of the last attempt, evidence_eligible recomputed, never trusted;
coverage hashing its manifest, RECHECK_MISMATCH failing both statuses); in full mode their ok and missing lines of the
test records are exactly the design, every model × condition × seed × record of design_records
(sap.manifest_design_errors); each SCORES_META.json of the scored runs (--scores-meta, by default the one next to each
--scores file) must first sit at the canonical path out/phase3/scores/<partition>/SCORES_META.json under --repo, one per
partition, i.e. exactly one --scores file per run (sap.canonical_meta_errors, added 28/09/2026: an edited scores.jsonl
and a fresh META anchored in a folder of their own passed every other check just as well, moving RQ1 from 0.2711 to
0.5482 at exit 0; any other path is now refused before it is even read; since 28/09/2026, that path is a fixed string
under --repo, no step down to it may be a symlink nor the file a hard link, and --repo must be the top level of its
checkout, closing the symlink and subfolder bypasses). It then names a given manifest and its coverage, the partition
analysed, its own --scores file (scores_sha256, recomputed here over the file's current bytes: a scores.jsonl edited
after scoring is refused even though nothing else moved) and, if any, the --accepted-missing file and its commit
(sap.scores_meta_errors); the SCORES_META files are the ones of --scores-meta-commit, committed once (genuinely: a
--no-ff merge that only carries the file forward does not count as a second write) and pushed to origin right after the
scoring (the only commit of origin that writes each META), a DESCENDANT of --lock-commit (added 28/09/2026; git
merge-base --is-ancestor, not merely a later date, which a commit cut from a branch older than the lock could also
claim; the full sha the lock check verified is the one compared) and dated after the lock's commit and after every
scored_utc (sap.scores_meta_commit_errors: otherwise a scores.jsonl edited together with its META's scores_sha256 would
pass); every score row is the output of an ok line, carries its sha256 (decision_sha256 and gen_code_sha256 required)
and its line_sha256 (it ties the row to the exact ok line, not merely a same-key one a tampered manifest could
substitute).

Missing pairs are accepted only before the scoring, with --accepted-missing (accepted_missing.jsonl) and
--accepted-missing-commit: the file must be that commit's, the commit an ancestor of HEAD, on a branch of origin, the
only commit of origin that genuinely writes the file (written once, never edited; since 28/09/2026, a later --no-ff
merge that only carries the file forward does not itself count as a write), and dated before every SCORES_META
scored_utc that names it; each entry must match a hashed manifest `missing` line (same key, same receipt_sha256) with
evidence (sap.EVIDENCE_RULE: last receipt CRASHED or INCOMPLETE over >= 2 attempts, each with a sha256; REFUSED_* and
absent never), and every missing line must be listed (sap.accepted_missing_errors). The accepted pairs of the design
score CA = BV = 0; SAP_RESULTS.json (accepted_missing) and SAP_REPORT.md record the file, its sha256, its commit with
the --format=fuller dates, every accepted pair and those filled.

Writes SAP_RESULTS.json, SAP_REPORT.md and SHA256SUMS; with --rehearsal (a run on control outputs; added 26/09/2026)
SAP_REPORT.md opens with "**KHÔNG PHẢI KẾT QUẢ — chạy thử trên đối chứng**", the same line heads every table of the
confirmatory and CA_prec sections and the pre-written answers (also on stdout), and SAP_RESULTS.json says "rehearsal":
true (false otherwise). The analyses config/sap.json lists as deferred (§0b of the design specification: learning curve,
ablations, the SemV sheet) run only with --with-deferred; the SemV sheet then adds semv_blind.jsonl and semv_key.jsonl.
--descriptive-only (aux_test: CA and BV per condition and per repo, with the lazy references) reads the locked aux_test
controls and records and compares every locked input and the code with the lock; it still checks er_complete.json, the
ER rows, the score rows (er values aside), the manifests, SCORES_META, its commit and the accepted missing pairs (not
the design of the manifests), the subsets, and the completeness of the aux design (sap.aux_design: the anchor model's 17
runs over the aux_test records; a duplicated, missing or other-model row stops it), and skips the confirmatory tests and
the answers. P22 (fixed 27/09/2026): the main table, each RQ line and each pre-written answer carry the label of
sap.robust_verdicts ("vững" or "chưa vững", with "(ước lượng trái dấu)" for two no_difference verdicts whose estimates
are of opposite signs); since 28/09/2026, conclusion_robust itself carries that same text as "label" next to "robust" in
SAP_RESULTS.json, for every comparison and every RQ entry, not only in the rendered report. P26 (added 06/10/2026,
before any test output): the report gives the expected directions of §2.4 of the design specification (sap.contrasts:
Full-aware − Full-neutral with its CI and its label), the label of §2.4 next to the P11 flag (`opposite`) of each
descriptive metric beside a comparison (two columns; the flag is kept as declared), the rates of duplicate_reuse and
exact per run, and the prompt length groups of the robustness section (sap.len_strata, by the anchor model's Full-aware
prompt) with the records in neither."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2tools.sap import (EXTRACT_STATUSES, REPO_NOTES, SapInputError, above_top4, accepted_missing_errors,  # noqa: E402
                         answers, aux_design, ca_prec_denominator_errors, ca_prec_differences, canonical_meta_errors,
                         check_complete, code_digest, committed_bytes, committed_file_errors, compose_dropped,
                         confirmatory,
                         controls_ca_prec, denominator_rows, descriptive, design_records, differing, digest_inputs,
                         eligibility, er_complete_errors, er_scope, git_head, jsonl_bytes, jsonl_rows,
                         lock_commit_errors,
                         lock_inputs, lock_precedes_scoring_errors, locked_flags, manifest_design_errors,
                         manifest_overlays, overlay_errors, pre_t_errors, pre_t_flags, read_manifests, robust_verdicts,
                         repo_rel, score_row_errors, scores_meta_commit_errors, scores_meta_errors, semv_sample,
                         subset_flags,
                         type_i_rates, usable_records, with_ca_prec)

TICKS = "`" * 3  # a Markdown fence in SAP_REPORT.md; never three backticks in the plan's code


def load(p: Path) -> list[dict]:
    return jsonl_rows(p)  # 3A's rule: "\n" only, a blank line inside the file refused


REHEARSAL = "**KHÔNG PHẢI KẾT QUẢ — chạy thử trên đối chứng**"  # --rehearsal (added 26/09/2026)


def banner(res: dict) -> list[str]:
    """The rehearsal banner and a blank line, for a --rehearsal run; nothing otherwise."""
    return [REHEARSAL, ""] if res["rehearsal"] else []


AUX_RQ3 = ("aux_test chỉ có các lượt của mô hình neo và không có lượt ER (P14), nên hàng của hai mô hình kia và các ô "
           "ER là «–» theo thiết kế (07/10/2026).")
AUX_NOTE = ("aux_test: các bản ghi nằm trước mốc cắt dữ liệu của mô hình; rủi ro nhiễm là giới hạn CHƯA ĐO, vì Min-K% "
            "tạm bỏ (đặc tả §0b).")


def fmt(x) -> str:
    return "–" if x is None or x != x else f"{x:+.3f}"


def fmtr(x) -> str:
    return "–" if x is None else f"{x:.3f}"


def fmtp(x) -> str:
    return "–" if x is None or x != x else f"{x:.4f}"


def ci(xs) -> str:
    return f"[{fmt(xs[0])}, {fmt(xs[1])}]"


def cell(x) -> str:
    """A table cell of text (a run key, a repo, a name): a "|" in it would split the cell, so it is shown as " · " (a
    run key "model|condition|variant[|seed]" reads "model · condition · variant[ · seed]")."""
    return str(x).replace("|", " · ")


def fmtg(x) -> str:
    return "–" if x is None else f"{x:g}"


def precision_cell(s: dict) -> str:
    """call_precision's median [Q1, Q3] (n) (D11 (b))."""
    return f"{fmtr(s['median'])} [{fmtr(s['q1'])}, {fmtr(s['q3'])}] ({s['n']})"


def calls_cell(s: dict) -> str:
    """n_calls' median, p90 (n) (D11 (b))."""
    return f"{fmtg(s['median'])}, p90 {fmtg(s['p90'])} ({s['n']})"


def generation_lines(res: dict) -> list[str]:
    """The manifests the outputs are tied to (design specification §5 at 7af1673), and the accepted missing pairs, with
    their commit and its --format=fuller dates."""
    gen, am = res["generation"], res.get("accepted_missing")
    out = ["Gắn output với lượt sinh (§5): " + "; ".join(f"`{m['manifest']}` (sha256 `{m['sha256']}`, coverage "
                                                            f"`{m['coverage_sha256']}`)" for m in gen["manifests"])
           + ". Mỗi SCORES_META.json của các tệp điểm ghi đúng các sha256 đó; mỗi dòng điểm là output của một dòng "
           "`ok` của manifest, cùng các sha (prompt, adapter, quyết định canary, mã sinh, biên lai chống lộ); các dòng "
           "ok hoặc missing của manifest trùng thiết kế.", ""]
    smc = gen.get("scores_meta_commit")
    if smc:  # the META files, anchored like the lock and accepted_missing
        out += [f"Các SCORES_META.json ({', '.join(f'`{q}`' for q in smc['paths'])}) được commit ngay sau khi chấm, "
                f"ở `{smc['sha']}` (trùng từng byte, commit là tổ tiên của HEAD, đã đẩy lên origin, ngày commit sau "
                "ngày commit của khoá và sau scored_utc); ngày của commit:", "", TICKS, smc["fuller"].rstrip(), TICKS,
                ""]
    if not am:
        return out + ["Manifest không có dòng `missing`: không có cặp thiếu nào được chấp nhận.", ""]
    filled = {tuple(x) for x in am["filled"]}
    out += [f"Dòng thiếu: được chấp nhận trước khi chấm, bằng `{am['file']}` (sha256 `{am['sha256']}`), commit "
            f"`{am['sha']}` (tệp `{am['path']}` trùng từng byte, commit là tổ tiên của HEAD, đã đẩy lên origin, ngày "
            f"commit trước tệp điểm): {len(am['accepted'])} cặp, mỗi cặp khớp một dòng `missing` của manifest đã băm "
            "và có bằng chứng (receipt CRASHED hoặc INCOMPLETE, ít nhất 2 lần thử có sha; REFUSED_* và absent không "
            f"bao giờ được chấp nhận); {len(filled)} cặp của thiết kế tính CA = BV = 0. Ngày của commit:", "", TICKS,
            am["fuller"].rstrip(), TICKS, ""]
    out += [f"- `{rid}` · {m} · {cond} · {variant} · {seed}"
            + (" (thiếu, đã tính CA = BV = 0)" if (rid, m, cond, seed) in filled else "")
            for rid, m, cond, variant, seed in am["accepted"]] + [""]
    return out


def by_repo_lines(res: dict) -> list[str]:
    """The CA and BV levels per repo and run, with the rates of the lazy references of the run in that repo
    (sap.lazy_by_repo: v3m, v3all, v3names, top4; "–" where its rows have none) and the repo's sentence of
    sap.REPO_NOTES (PrestaShop)."""
    out = ["", "## CA và BV theo repo và theo lượt chạy (mô tả)", ""] + ([AUX_NOTE, ""] if res["aux"] else [])
    out += ["Mức lười: tỷ lệ CA của từng đối chứng lười trên các dòng CA của lượt ở repo đó, các seed gộp.", "",
            "| Repo | Lượt chạy | CA | BV | v3m | v3all | v3names | top4 | Ghi chú |",
            "|---|---|---|---|---|---|---|---|---|"]
    lazy = res["descriptive"].get("lazy_by_repo", {})
    for repo, v in res["descriptive"]["by_repo"].items():
        for run, c in v["ca"].items():
            b, z = v["bv"].get(run, {}), lazy.get(repo, {}).get(run, {})
            refs = " | ".join(fmtr(z.get(k, {}).get("rate")) for k in ("ca_ref_v3m", "ca_ref_v3all", "ca_ref_v3names",
                                                                       "ca_ref_top4"))
            out.append(f"| {cell(repo)} | {cell(run)} | {fmt(c['estimate'])} {ci(c['ci95'])} | "
                       f"{fmt(b.get('estimate'))} {ci(b.get('ci95', [None, None]))} | {refs} | "
                       f"{cell(REPO_NOTES.get(repo, ''))} |")
    return out


def calls_by_repo_lines(calls: dict) -> list[str]:
    """D11 (b): call_precision and n_calls per repo and condition (sap.calls_by_repo; §2.2b of the design specification
    at 3739dd6); the rows of a repo of sap.REPO_NOTES (PrestaShop) carry its sentence."""
    out = ["| Repo | Lượt chạy | call_precision trung vị [Q1, Q3] (n) | n_calls trung vị, p90 (n) | Ghi chú |",
           "|---|---|---|---|---|"]
    for repo, by in calls.items():
        for run, v in by.items():
            out.append(f"| {cell(repo)} | {cell(run)} | {precision_cell(v['call_precision'])} | "
                       f"{calls_cell(v['n_calls'])} | {cell(REPO_NOTES.get(repo, ''))} |")
    return out


def ca_prec_lines(res: dict, cfg: dict) -> list[str]:
    """D11 (c): the CA_prec comparisons next to CA's conclusions, the RQ verdicts, every difference of conclusion
    (sap.ca_prec_differences, no selection), and the controls' CA_prec rates at each pre-registered τ."""
    cp = res["ca_prec"]
    out = ["", "## Độ nhạy CA_prec (D11, khai trước)", "",
           f"CA_prec = CA × [call_precision ≥ {cp['tau']:g}] (τ khoá trong cấu hình; call_precision rỗng tính là 0), "
           "cùng các phép so, hai phép thử và họ Holm như CA; CA chính không đổi.", "",
           res["conclusion_robust"]["rule"], ""]
    if cp["confirmatory"]:
        out += ["«CA_prec vượt mức lười top4» (D11 (d), mô tả, không kiểm định): tỷ lệ dòng có CA_prec = 1 và "
                "ca_ref_top4 = 0, trên các dòng CA của mô hình neo có ca_ref_top4, các seed gộp; A · B là hai lượt của "
                "phép so. CA_prec chặn «rải tên» nhưng không chặn «vài lời gọi hạ tầng quen thuộc».", "",
                *banner(res),
                "| Họ | Phép so | n kịch bản / cụm | Ước lượng CA_prec | KTC 95% | p Holm (1) đảo dấu | KTC hai tầng "
                "| p Holm (2) hai tầng | Kết luận CA_prec | Ước lượng CA | Kết luận CA | Vững "
                "| CA_prec vượt mức lười top4 (A · B) |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        top4 = lambda c: (lambda v: f"{fmtr(v['rate'])} ({v['n']})")(cp["above_top4"][c])
        for fam, f in cp["confirmatory"].items():
            for name, v in f["comparisons"].items():
                c, t = v["ca_prec"], v["ca_prec"]["two_level"]
                out.append(f"| {cell(fam)} | {cell(name)} | {c['n_scenarios']} / {c['n_clusters']} | "
                           f"{fmt(c['estimate'])} | {ci(c['ci95'])} | {fmtp(c['p_holm'])} | {ci(t['ci95'])} | "
                           f"{fmtp(t['p_holm'])} | {cell(v['verdict'])} | "
                           f"{fmt(res['confirmatory'][fam]['comparisons'][name]['ca']['estimate'])} | "
                           f"{cell(res['confirmatory'][fam]['comparisons'][name]['verdict'])} | "
                           f"{res['conclusion_robust']['comparisons'][fam][name]['label']} | "
                           + " · ".join(map(top4, name.split("_vs_"))) + " |")
        out += [""] + [f"- {fam}: CA_prec **{f['rq_verdict']}** (CA **{res['confirmatory'][fam]['rq_verdict']}**): "
                       f"**{res['conclusion_robust']['rq'][fam]['label']}**"
                       for fam, f in cp["confirmatory"].items()]
        out += [""] + [f"- {d}" for d in cp["differences"]]
    taus = cfg["ca_prec_control_taus"]
    out += ["", "Đối chứng (mô tả), trên các bản ghi dùng được có CA xác định:", "", *banner(res),
            "| Đối chứng | n | CA | " + " | ".join(f"CA_prec τ = {t:g}" for t in taus) + " |",
            "|---|---|---|" + "---|" * len(taus)]
    for name, v in cp["controls"].items():
        rates = " | ".join(fmtr(v[f"ca_prec@{t:g}"]) for t in taus)
        out.append(f"| {cell(name)} | {v['n']} | {fmtr(v['ca'])} | {rates} |")
    return out


def robustness_lines(res: dict) -> list[str]:
    """The exploratory "độ vững" (robustness) section, n, estimate and CI per view and comparison;
    no p, no multiple-testing adjustment, no verdict, no family label: it never yields a conclusion. It names the
    locked pre-T list and how many records of the run's partition its without_pre_t view drops."""
    pt, ls = res["pre_t"], res["descriptive"]["len_strata"]
    out = ["", "## Độ vững (khám phá)", "",
           "Phần khám phá, không bao giờ cho kết luận chính: mỗi góc nhìn (bốn tập con, tầng theo repo và ngôn ngữ, "
           "nhóm preexisting, bỏ từng repo, bỏ các bản ghi pre-T đã khoá, nhóm độ dài prompt) lặp lại các phép so xác "
           "nhận, không gắn nhãn họ, chỉ với n, ước lượng và KTC 95% bootstrap cụm; không có p, không hiệu chỉnh đa "
           "phép thử, không kết luận.", "",
           f"Bỏ pre-T theo danh sách pre-T đã khoá `{pt['file']}` (sha256 `{pt['sha256']}`): góc nhìn without_pre_t bỏ "
           f"{pt['dropped']} bản ghi của phần {pt['partition']}.", "",
           "Nhóm độ dài prompt (đặc tả §2.4; P26) theo prompt Full-aware của mô hình neo (`len_bucket` của dòng "
           f"Full-aware thật; mọi điều kiện của bản ghi theo nhóm đó): {ls['records']['le8192']} bản ghi ≤ 8.192 token "
           f"(len_bucket=le8192), {ls['records']['gt8192']} bản ghi > 8.192 (len_bucket=gt8192); {ls['unplaced']} bản "
           "ghi không có dòng Full-aware thật nên ra khỏi cả hai nhóm. Các số này gồm cả bản ghi loại máy móc (ngoài "
           "mẫu số CA); nhóm theo bản ghi, nên một kịch bản có thể có bản ghi ở cả hai nhóm.", "",
           "| Góc nhìn | Phép so | n kịch bản / cụm | Ước lượng | KTC 95% |", "|---|---|---|---|---|"]
    for view, comps in res["descriptive"]["robustness"].items():
        for name, c in comps.items():
            out.append(f"| {cell(view)} | {cell(name)} | {c['n_scenarios']} / {c['n_clusters']} | "
                       f"{fmt(c['estimate'])} | {ci(c['ci95'])} |")
    return out


CONTRAST_NAMES = {"hallucinated_rate": "ký hiệu bịa (hallucinated / members_used)",
                  "conv_style": "quy ước: dạng annotation", "conv_pattern": "quy ước: dạng placeholder",
                  "conv_naming": "quy ước: cách đặt tên"}


def contrast_lines(res: dict) -> list[str]:
    """P26 (added 06/10/2026, before any test output): the expected directions of §2.4 of the design specification,
    Full-aware − Full-neutral (sap.contrasts), descriptive."""
    out = ["", "## Chiều kỳ vọng của thước đo mô tả: Full-aware − Full-neutral (mô tả)", "",
           "Đặc tả §2.4 khai trước chiều kỳ vọng: ký hiệu bịa ở F2 < F1, tỷ lệ tuân quy ước ở F2 > F1; với các điều "
           "kiện có adapter, F2 − F1 là Full-aware − Full-neutral (mô hình neo, main), ghép theo seed, KTC 95% "
           "bootstrap cụm, không có p. Ký hiệu bịa tính theo bản ghi (hallucinated / members_used, trên các dòng có "
           "members_used > 0). Cột «Chiều» theo quy ước trình bày của §2.4: «không rõ chiều» khi KTC chứa 0 hay dựa "
           "trên dưới 2 cụm.", "",
           "| Thước đo | Chiều kỳ vọng | n kịch bản / cụm | Ước lượng | KTC 95% | Chiều (quy ước §2.4) |",
           "|---|---|---|---|---|---|"]
    for m, d in res["descriptive"]["contrasts"].items():
        out.append(f"| {CONTRAST_NAMES[m]} | {d['expected']} | {d['n_scenarios']} / {d['n_clusters']} | "
                   f"{fmt(d['estimate'])} | {ci(d['ci95'])} | {d['direction'] or '–'} |")
    out += ["", "Liều–đáp ứng (§2.4, P14; P26 (e)): F2 − F1 = Full-aware − Full-neutral của CA theo số ký hiệu của "
            "đáp án chuẩn có mặt trong ④, mỗi mức một KTC và một nhãn chiều (kỳ vọng > 0). Chiều kỳ vọng «mức lợi tăng "
            "theo liều» chỉ được đọc mô tả từ các KTC này, KHÔNG gắn nhãn xu hướng: các mức là các tập bản ghi rời "
            "nhau.", "",
            "| Mức liều | n kịch bản / cụm | Ước lượng | KTC 95% | Chiều (quy ước §2.4) |", "|---|---|---|---|---|"]
    for b, d in res["descriptive"]["dose"].items():
        out.append(f"| {b} | {d['n_scenarios']} / {d['n_clusters']} | {fmt(d['estimate'])} | {ci(d['ci95'])} | "
                   f"{d['direction'] or '–'} |")
    return out


RQ3_NAMES = {"ca": "CA", "bv": "BV", "er": "ER", "er@self_contained": "ER (self_contained)", "er_trace": "ER-trace",
             "er_trace@self_contained": "ER-trace (self_contained)"}


def rq3_lines(res: dict) -> list[str]:
    """P14 (added 07/10/2026): each model's Full-aware level of CA, BV, ER and ER-trace (sap.rq3_profile), with its 95%
    cluster-bootstrap CI and n scenarios, no p, no label; below the table, each model's records in the design
    (completeness.records_covered, added 07/10/2026: compose drops records per model, so the denominators may differ),
    then the reason of each null cell."""
    prof = res["descriptive"]["rq3_profile"]
    names = list(RQ3_NAMES)
    out = ["", "## RQ3: hồ sơ theo mô hình ở Full-aware (mô tả)", ""] + ([AUX_NOTE, "", AUX_RQ3, ""] if res["aux"] else [])
    out += ["Mức ở Full-aware (variant main), các seed của mô hình gộp, KTC 95% bootstrap cụm, n kịch bản; không p, không "
            "nhãn (H3/RQ3, đặc tả §5; P14). Mẫu số: CA trên bản ghi dùng được, BV trừ `excl_bv`, ER và ER-trace trên "
            "bản ghi đủ điều kiện (P11). Ô «–» không có mức; lý do ghi dưới bảng.", "",
            "| Mô hình | " + " | ".join(RQ3_NAMES[m] for m in names) + " |", "|---|" + "---|" * len(names)]
    why = []
    for model, v in prof.items():
        cells = []
        for m in names:
            d = v[m]
            if d["estimate"] is None:
                cells.append("–")
                why.append(f"- {cell(model)} · {RQ3_NAMES[m]}: {d['reason']}")
            else:
                lo, hi = d["ci95"]
                cells.append(f"{fmtr(d['estimate'])} [{fmtr(lo)}, {fmtr(hi)}] ({d['n_scenarios']})")
        out.append(f"| {cell(model)} | " + " | ".join(cells) + " |")
    # each model's records in the design (added 07/10/2026)
    covered = res.get("completeness", {}).get("records_covered", {})
    out += ["", "Bản ghi trong thiết kế của từng mô hình (P4; compose loại riêng theo mô hình): "
            + ", ".join(f"{cell(m)} {covered[m]}" for m in sorted(covered)) + "."] if covered else []
    return out + ([""] + why if why else [])


def docblock_lines(res: dict) -> list[str]:
    """§2.3 of the design specification (added 26/09/2026): PHP docblock steps next to ER per repo and run
    (sap.docblock_by_repo); descriptive."""
    out = ["", "## Bước docblock PHP cạnh ER (mô tả)", "",
           "Ảnh ER chỉ đăng ký step dạng attribute của PHP 8; step dạng docblock không được định nghĩa trong ER, còn "
           "BV và CA coi hai dạng như nhau (§2.3). ER tính trên các dòng có giá trị er; các cột docblock trên các "
           "output PHP có văn bản, các seed gộp.", "",
           "| Repo | Lượt chạy | ER (n) | Tỷ lệ output có ≥ 1 step docblock (n) | Số step docblock "
           "| Số step attribute |",
           "|---|---|---|---|---|---|"]
    for repo, by in res["descriptive"]["docblock_by_repo"].items():
        for run, v in by.items():
            out.append(f"| {cell(repo)} | {cell(run)} | {fmtr(v['er']['rate'])} ({v['er']['n']}) | "
                       f"{fmtr(v['docblock_rate'])} ({v['n']}) | {v['docblock_steps']} | {v['attribute_steps']} |")
    return out


def kappa_lines(res: dict) -> list[str]:
    """P16: the pooled CA × ER-trace agreement of each eligibility, next to its attribute-only version (outputs with
    php_docblock_steps == 0; sap.kappa_ca_trace)."""
    out = ["", "## Đồng thuận CA × ER-trace (P16, mô tả)", "",
           "Gộp mọi lượt của mô hình neo (P16; 07/10/2026: lượt ER của mô hình khác ở Full-aware chỉ phục vụ RQ3 "
           "và không vào hàng gộp); bản attribute_only chỉ gồm các output có php_docblock_steps = 0 (§2.3).", "",
           "| Tập | n | Tỷ lệ đồng thuận | kappa |", "|---|---|---|---|"]
    for name, v in res["descriptive"]["kappa_ca_trace_action"].items():
        a = v.get("ALL", {"n": 0, "agreement": None, "kappa": None})
        out.append(f"| {cell(name)} | {a['n']} | {fmtr(a['agreement'])} | {fmtr(a['kappa'])} |")
    return out


def denominator_lines(res: dict) -> list[str]:
    """Records and scenarios per repo and partition of the locked records (sap.denominator_rows)."""
    out = ["", "## Mẫu số theo repo và phần dữ liệu", "",
           "| Repo | Phần | Bản ghi | Kịch bản | Bản ghi dùng cho CA | Kịch bản dùng cho CA | Loại máy móc "
           "| Compose loại (mô hình neo) |", "|---|---|---|---|---|---|---|---|"]
    for d in res["denominators"]:
        out.append(f"| {cell(d['repo'])} | {cell(d['partition'])} | {d['records']} | {d['scenarios']} | "
                   f"{d['ca_records']} | {d['ca_scenarios']} | {d['excluded']} | {d['dropped']} |")
    return out


def type_i_lines(res: dict) -> list[str]:
    """The measured rejection rates under H0 of both tests and of "supported", from the locked
    CALIBRATION.json (sap.type_i_rates); the method is unchanged."""
    out = ["", "## Sai lầm loại I đo được của hai phép thử (từ CALIBRATION.json đã khoá)", "",
           "Tỷ lệ bác bỏ khi không có hiệu ứng (delta = 0), trên các bộ dữ liệu mô phỏng theo cấu trúc cụm của test "
           "(sap_calibrate); sai số Monte Carlo trong ngoặc.", "",
           "| Loại phép so | τ | Số bộ H0 | α | (1) đảo dấu | (2) hai tầng | supported |",
           "|---|---|---|---|---|---|---|"]
    rate = lambda x: f"{x['rate']:.4f} ({x['mc_se']:.4f})"
    for t in res["calibration_type_i"]:
        for a, x in t["alphas"].items():
            out.append(f"| {cell(t['type'])} | {t['tau']} | {t['n']} | {a.split('=')[1]} | {rate(x['sign_flip'])} | "
                       f"{rate(x['two_level'])} | {rate(x['supported'])} |")
    return out


def report(res: dict, cfg: dict) -> str:
    covered = ", ".join(f"{m} {n}" for m, n in res["completeness"].get("records_covered", {}).items()) or "–"
    lc = res["lock_commit"]
    lines = [*banner(res), "# Kết quả SAP Giai đoạn 3", "",
             f"Khoá `{res['lock'].get('locked_utc')}`; công cụ `{res['tool_commit']}`; "
             f"cấu hình sha256 `{res['lock']['inputs'].get('config')}`.", "",
             f"Khoá được xác thực bằng commit `{lc['sha']}` (tệp `{lc['path']}` trùng từng byte, commit là tổ tiên của "
             "HEAD và đã đẩy lên origin); ngày của commit:", "", TICKS, lc["fuller"].rstrip(), TICKS, "",
             f"Tính đầy đủ: {res['completeness'].get('usable_records', '–')} bản ghi dùng được cho CA; mỗi lượt "
             + ("của thiết kế aux_test (chỉ mô hình neo) " if res["aux"] else "của thiết kế (mọi mô hình) ")
             + f"có đúng một dòng cho mỗi bản ghi của lượt đối chứng mà compose không loại "
             f"cho mô hình đó ({covered}; nên mẫu số BV và ER cũng được kiểm); "
             f"{res['completeness'].get('missing_filled', 0)} dòng thiếu tính CA = 0 sau khi chạy lại vẫn thất bại.",
             ""] + ([AUX_NOTE, ""] if res["aux"] else [])  # every aux table under the note (added 06/10/2026)
    if not res["dose_checked"]:
        lines += ["dose chưa được đối chiếu với đầu vào đã khoá.", ""]
    lines += generation_lines(res)
    lines += ["## Kiểm định xác nhận (CA, DeepSeek, main)", "", *banner(res),
              "| Họ | Phép so | n kịch bản / cụm | Ước lượng | KTC 95% | p Holm (1) đảo dấu | KTC hai tầng "
              "| p Holm (2) hai tầng | Seed dương | Kết luận | Vững |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    # P22 (fixed 27/09/2026): the robustness label next to every conclusion (sap.robust_verdicts' own "label")
    rb = res["conclusion_robust"]
    for fam, f in res["confirmatory"].items():
        for name, v in f["comparisons"].items():
            c, t = v["ca"], v["ca"]["two_level"]
            lines.append(f"| {cell(fam)} | {cell(name)} | {c['n_scenarios']} / {c['n_clusters']} | "
                         f"{fmt(c['estimate'])} | {ci(c['ci95'])} | {fmtp(c['p_holm'])} | {ci(t['ci95'])} | "
                         f"{fmtp(t['p_holm'])} | {c['seeds_positive']}/{len(c['per_seed'])} | {cell(v['verdict'])} | "
                         f"{rb['comparisons'][fam][name]['label']} |")
    lines += [""] + [f"- {fam}: **{f['rq_verdict']}** ({rb['rq'][fam]['label']})"
                     for fam, f in res["confirmatory"].items()]
    if res["confirmatory"]:
        lines += ["", "Cột «Vững» và nhãn sau mỗi RQ: CA và CA_prec có cùng kết luận hay không (quy tắc P22 ở mục «Độ "
                  "nhạy CA_prec»)."]
    lines += ["", "## Ước lượng từng seed (mô tả)", "", *banner(res), "| Phép so | Seed | Ước lượng | KTC 95% |",
              "|---|---|---|---|"]
    for f in res["confirmatory"].values():
        for name, v in f["comparisons"].items():
            for s, c in v["ca"]["per_seed"].items():
                lines.append(f"| {cell(name)} | {cell(s)} | {fmt(c['estimate'])} | {ci(c['ci95'])} |")
    lines += ["", "## Thước đo mô tả cạnh mỗi phép so", "",
              "Chiều kỳ vọng: A − B > 0. Cột «Chiều» theo quy ước trình bày của đặc tả §2.4: «không rõ chiều» khi KTC "
              "95% chứa 0 hay dựa trên dưới 2 cụm, nên «ngược chiều» chỉ khi KTC loại 0 về phía âm. Cột «Cờ P11» "
              "là cờ khai trước, giữ nguyên: KTC loại 0 về phía âm, hoặc ước lượng âm cạnh một CA supported.", "",
              *banner(res),
              "| Phép so | Thước đo | n kịch bản / cụm | Ước lượng | KTC 95% | Chiều (quy ước §2.4) | Cờ P11 |",
              "|---|---|---|---|---|---|---|"]
    for f in res["confirmatory"].values():
        for name, v in f["comparisons"].items():
            for m, d in v["descriptive"].items():
                lines.append(f"| {cell(name)} | {cell(m)} | {d['n_scenarios']} / {d['n_clusters']} | "
                             f"{fmt(d['estimate'])} | {ci(d['ci95'])} | {d['direction'] or '–'} | "
                             f"{'CÓ' if d['opposite'] else ''} |")
    lines += contrast_lines(res)
    lines += docblock_lines(res) + kappa_lines(res) + rq3_lines(res)
    lines += ["", "## Câu trả lời (viết sẵn trước khi có output)", "", *banner(res)]
    lines += [f"- {a}" for a in res["answers"]]
    lines += ca_prec_lines(res, cfg)
    lines += robustness_lines(res) + denominator_lines(res) + type_i_lines(res)
    if res["semv_items"]:
        lines += ["", "## SemV", "", f"Mẫu mù: {res['semv_items']} mục ({cfg['semv']['n_records']} bản ghi × "
                  f"{len(cfg['semv']['conditions'])} điều kiện) trong `semv_blind.jsonl`; khoá điều kiện nằm riêng ở `semv_key.jsonl`.",
                  "Rubric (khoá cùng cấu hình):", ""] + [f"- {x}" for x in cfg["semv"]["rubric"]]
    lines += by_repo_lines(res)
    lines += ["", "## Trích mã theo lượt chạy (mô tả)", "",
              "Số dòng theo trạng thái trích (trên các dòng có output, trừ bản ghi `excl_bv`, cộng lại bằng n), tỷ lệ "
              "fence_truncated (trên các dòng có văn bản) và tỷ lệ php_tag_stripped (dòng `<?php` đầu hay `?>` cuối "
              "khối PHP được bỏ trước khi chèn, mọi điều kiện; trên các dòng PHP có văn bản; §1.4). Theo đặc "
              "tả §2.2: tỷ lệ output "
              "có ít nhất một dòng trông như khai báo step mà bộ đọc không nhận (tổng số dòng), có pattern Cucumber "
              "expression với ký tự thoát `\\(` `\\{` `\\/`, có kiểu tham số tự định nghĩa (tổng số pattern), trên các "
              "dòng có văn bản; và tỷ lệ er_status ERROR (output làm Behat dừng lúc nạp) trên các dòng có er_status. "
              "Vòng thử dev chỉ đo được PHP; Java, C# và Ruby được báo ở đây trên output test.", "",
              "| Lượt chạy | n | " + " | ".join(EXTRACT_STATUSES) + " | fence_truncated (n) | php_tag_stripped (n) "
              "| step không đọc được (n; tổng) | expression thoát (n; tổng) | kiểu tự định nghĩa (n; tổng) "
              "| ER ERROR (n) |",
              "|---|---|" + "---|" * len(EXTRACT_STATUSES) + "---|---|---|---|---|---|"]
    for run, v in res["descriptive"]["per_run"].items():
        e, ft, tag = v["extract"], v["fence_truncated"], v["php_tag_stripped"]
        counts = " | ".join(f"{fmtr(v[k]['rate'])} ({v[k]['n']}; {v[k]['sum']})"
                            for k in ("unread_step_annotations", "cuke_escaped_patterns", "cuke_custom_types"))
        lines.append(f"| {cell(run)} | {e['n']} | " + " | ".join(str(e[s]) for s in EXTRACT_STATUSES)
                     + f" | {fmtr(ft['rate'])} ({ft['n']}) | {fmtr(tag['rate'])} ({tag['n']}) | {counts} "
                     f"| {fmtr(v['er_error']['rate'])} ({v['er_error']['n']}) |")
    lines += ["", "## Viết trùng và khớp nguyên văn theo lượt chạy (mô tả)", "",
              "Đặc tả §2.4 (P26): tỷ lệ viết trùng (code sinh ra khớp một step REUSE đã có định nghĩa, "
              "`duplicate_reuse`) và tỷ lệ khớp nguyên văn đáp án chuẩn sau chuẩn hoá khoảng trắng (`exact`), trên các "
              "dòng có output, trừ bản ghi `excl_bv` (mẫu số BV, như cột n của bảng trích mã).", "",
              "| Lượt chạy | Viết trùng (n) | Khớp nguyên văn (n) |", "|---|---|---|"]
    for run, v in res["descriptive"]["per_run"].items():
        d, e = v["duplicate_reuse"], v["exact"]
        lines.append(f"| {cell(run)} | {fmtr(d['rate'])} ({d['n']}) | {fmtr(e['rate'])} ({e['n']}) |")
    lines += ["", "## CA, mức lười và lời gọi theo lượt chạy (mô tả)", "",
              "Mỗi mức lười tính trên các dòng CA có mức đó; «vượt» là CA = 1 và mức lười = 0, riêng top4 là «CA_prec "
              "vượt mức lười top4»: CA_prec = 1 và ca_ref_top4 = 0 (D11 (d)). call_precision và n_calls tính trên các "
              "dòng CA có output (D11).", "",
              "| Lượt chạy | CA (n) | v3m | v3all | v3names | top4 | CA vượt v3m | CA vượt v3all | CA vượt v3names "
              "| CA_prec vượt mức lười top4 | call_precision trung vị [Q1, Q3] (n) | n_calls trung vị, p90 (n) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, v in res["descriptive"]["per_run"].items():
        refs = (v.get(k, {}).get("rate") for k in ("ca_ref_v3m", "ca_ref_v3all", "ca_ref_v3names", "ca_ref_top4",
                                                   "ca_above_lazy", "ca_above_v3all", "ca_above_v3names",
                                                   "ca_prec_above_top4"))
        lines.append(f"| {cell(run)} | {fmtr(v['ca']['rate'])} ({v['ca']['n']}) | " + " | ".join(map(fmtr, refs))
                     + f" | {precision_cell(v['call_precision'])} | {calls_cell(v['n_calls'])} |")
    lines += ["", "## call_precision và n_calls theo repo và điều kiện (mô tả)", ""]
    lines += calls_by_repo_lines(res["descriptive"]["calls_by_repo"])
    lines += ["", "Các phân tích mô tả còn lại (RQ3, liều–đáp ứng, các tỷ lệ khác theo lượt chạy, kappa theo lượt)",
              "nằm trong `SAP_RESULTS.json`. "
              + ("Đã chạy cả các phân tích tạm bỏ (--with-deferred)." if res["deferred_run"] else
                 f"Tạm bỏ theo §0b, không chạy: {', '.join(cfg.get('deferred', []))}."), ""]
    return "\n".join(lines)


def generation(args, cfg: dict, part: dict, compose: list[dict], lock_committed: str, lock_sha: str,
               score_bytes: list[bytes]) -> tuple[dict, dict, dict | None, list[str]]:
    """The manifests and coverage files (fixed 26/09/2026; design specification §5 at 7af1673), the SCORES_META.json of
    each score file and the accepted_missing file, checked (sap.canonical_meta_errors, sap.read_manifests,
    manifest_design_errors in full mode, committed_file_errors on origin, accepted_missing_errors, scores_meta_errors),
    and the commit of the SCORES_META files (--scores-meta-commit on origin, after the lock and the scoring, and a
    descendant of --lock-commit, added 28/09/2026: sap.scores_meta_commit_errors; `lock_sha` is --lock-commit as
    lock_commit_errors verified it, the full sha, so that no other commit can pass there; added 28/09/2026). Since
    28/09/2026, before any of that, every SCORES_META.json named (--scores-meta, or by default the one next to each
    --scores file) must sit at the canonical path out/phase3/scores/<partition>/SCORES_META.json under --repo, one per
    partition; any other path is refused right here, closing the bypass of anchoring an edited scores.jsonl and a fresh
    META in a folder of its own. Since 28/09/2026, the content of each SCORES_META.json and of the accepted_missing file
    is read from its commit (sap.committed_bytes), never from the path given, and each --scores file from `score_bytes`,
    the bytes main() read once and parsed. Returns (pair -> manifest line, what SAP_RESULTS records of them, the
    accepted_missing record or None, problems)."""
    aux = args.descriptive_only
    metas = args.scores_meta or [f.parent / "SCORES_META.json" for f in args.scores]
    bad = canonical_meta_errors(metas, args.repo, "aux_test" if aux else "test")  # the canonical path, first of all
    if bad:
        return {}, {}, None, bad
    if len(args.manifest) != len(args.coverage):
        return {}, {}, None, ["--manifest and --coverage go in pairs (one coverage.<variant>.json per manifest)"]
    if len(metas) != len(args.scores):
        return {}, {}, None, ["--scores-meta gives one SCORES_META.json per --scores file"]
    if (args.accepted_missing is None) != (args.accepted_missing_commit is None):
        return {}, {}, None, ["--accepted-missing and --accepted-missing-commit go together"]
    lines, shas, bad = read_manifests(list(zip(args.manifest, args.coverage)))
    ids = {k: {r["id"] for r in recs} for k, (recs, _, _) in part.items()}
    controls = part["test"][1]
    bad += manifest_design_errors(lines, cfg, {m: design_records(controls, compose, m) for m in cfg["seeds"]},
                                  {m: compose_dropped(compose, m) for m in cfg["seeds"]},
                                  set() if aux else ids["test"], ids["test"] | ids["aux_test"], args.with_deferred)
    entries, commit, acc = None, None, None
    if args.accepted_missing is not None:
        problems, commit = committed_file_errors(args.accepted_missing, args.accepted_missing_commit, args.repo,
                                                 "the accepted_missing file", "--accepted-missing-commit", "origin",
                                                 once=True)  # written once (fixed 27/09/2026)
        if problems:
            return {}, {}, None, problems
        data = committed_bytes(args.repo, commit["sha"], commit["path"])  # its content, from the commit
        entries = jsonl_bytes(data, args.accepted_missing)
        acc = hashlib.sha256(data).hexdigest()
    bad += accepted_missing_errors(entries, lines)
    read, scores_shas = [], []
    for p, data in zip(metas, score_bytes):  # each --scores file's own sha256, so a
        rel = repo_rel(p, args.repo)            # scores.jsonl edited after scoring is caught even though nothing else
        blob = None if rel is None else committed_bytes(args.repo, args.scores_meta_commit, rel)  # moved;
        if blob is None:                        # the META as committed, not as on disk
            bad.append(f"{p}: --scores-meta-commit {args.scores_meta_commit} holds no {rel or 'such file'}")
            continue
        try:
            m = json.loads(blob)
        except ValueError as e:
            bad.append(f"{p}: the SCORES_META.json of --scores-meta-commit {args.scores_meta_commit} is not JSON ({e})")
            continue
        read.append((p, m if isinstance(m, dict) else {}))
        scores_shas.append(hashlib.sha256(data).hexdigest())
    bad += scores_meta_errors(read, shas, "aux_test" if aux else "test", acc, commit and commit["sha"],
                              commit and commit["committed"], scores_shas)
    problems, meta_commit = scores_meta_commit_errors(read, args.scores_meta_commit, args.repo, lock_committed,
                                                      lock_sha)  # also a descendant of the lock
    bad += problems
    gen = {"manifests": [{"manifest": str(m), "sha256": hashlib.sha256(m.read_bytes()).hexdigest(), "coverage": str(c),
                          "coverage_sha256": hashlib.sha256(c.read_bytes()).hexdigest()}
                         for m, c in zip(args.manifest, args.coverage)],
           "scores_meta": [{"file": str(p), **{k: m.get(k) for k in ("partition", "manifest_sha256", "scored_utc",
                                                                      "accepted_missing_sha256", "scores_sha256")}}
                          for p, m in read],
           "scores_meta_commit": meta_commit}
    record = None if entries is None else {
        "file": str(args.accepted_missing), "sha256": acc, **commit,
        "accepted": sorted([e["record_id"], e["model"], e["condition"], e.get("variant") or "main", e["seed"]]
                           for e in entries), "filled": []}
    return lines, gen, record, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("config/sap.json"))
    ap.add_argument("--lock", type=Path, required=True, help="SAP_LOCK.json written when the SAP was locked")
    ap.add_argument("--mde", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--dev-selection", type=Path, required=True)
    ap.add_argument("--dev-scores", type=Path, nargs="+", required=True)
    ap.add_argument("--dev-controls-dir", type=Path, required=True)
    ap.add_argument("--controls-dir", type=Path, required=True)
    ap.add_argument("--compose-excluded", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True, help="out/phase2/records/test.jsonl")
    ap.add_argument("--subsets", type=Path, required=True)
    ap.add_argument("--self-contained", type=Path, required=True)
    ap.add_argument("--phase2-sums", type=Path, required=True)
    ap.add_argument("--er-controls", type=Path, nargs="+", required=True)
    ap.add_argument("--er-manifest", type=Path, required=True,
                    help="manifest.jsonl of the ER controls run (write_overlays)")
    ap.add_argument("--aux-controls-dir", type=Path, required=True,
                    help="aux_test controls run (controls.jsonl, exclusions.jsonl, future_methods.jsonl)")
    ap.add_argument("--aux-records", type=Path, required=True, help="out/phase2/records/aux_test.jsonl")
    ap.add_argument("--pre-t-flags", type=Path, required=True,
                    help="the frozen pre-T list (evidence/pret_flags/FLAGGED_PRE_T.tsv)")
    ap.add_argument("--dose", type=Path, help="3C's dose table (score_run --dose-table), when the lock holds one")
    ap.add_argument("--lock-commit", required=True,
                    help="the commit that holds --lock (it must be that commit's file, an ancestor of HEAD, pushed)")
    ap.add_argument("--repo", type=Path, default=ROOT,
                    help="the git checkout that holds --lock, --scores-meta and --accepted-missing "
                         "(default: this code's)")
    ap.add_argument("--scores", type=Path, nargs="+", required=True)
    ap.add_argument("--outputs", type=Path, nargs="*", default=[], help="generation outputs (text) for the SemV sheet")
    ap.add_argument("--manifest", type=Path, nargs="+", required=True,
                    help="the generation run's manifest.<variant>.jsonl of the scored runs")
    ap.add_argument("--coverage", type=Path, nargs="+", required=True,
                    help="the coverage.<variant>.json of each --manifest, in the same order")
    ap.add_argument("--scores-meta", type=Path, nargs="+",
                    help="the SCORES_META.json of each --scores file, in order (default: the one next to it)")
    ap.add_argument("--scores-meta-commit", required=True,
                    help="the commit that holds every --scores-meta, pushed to origin right after the scoring")
    ap.add_argument("--accepted-missing", type=Path,
                    help="the accepted_missing.jsonl, committed and pushed to origin before the scoring")
    ap.add_argument("--accepted-missing-commit", help="the commit that holds --accepted-missing")
    ap.add_argument("--descriptive-only", action="store_true",
                    help="aux_test: the completeness of the aux design (the anchor model's runs), no confirmatory test")
    ap.add_argument("--rehearsal", action="store_true",
                    help="a run on control outputs: every result is bannered as not a result (added 26/09/2026)")
    ap.add_argument("--with-deferred", action="store_true", help="also run the analyses config/sap.json lists as deferred")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    problems, lock_commit = lock_commit_errors(args.lock, args.lock_commit, args.repo)  # the lock and its commit
    if problems:
        print(f"refusing: {'; '.join(problems)}", file=sys.stderr)
        return 2
    lock = json.loads(committed_bytes(args.repo, lock_commit["sha"], lock_commit["path"]))  # as committed
    now = digest_inputs(lock_inputs(args))
    bad = differing(lock["inputs"], now)  # every input, --descriptive-only included
    if bad:
        print(f"refusing: inputs differ from the lock: {bad}", file=sys.stderr)
        return 2
    bad = differing(lock["code"], code_digest(ROOT))
    if bad:
        print(f"refusing: the analysis code differs from the lock: {bad}", file=sys.stderr)
        return 2
    # The lock's own commit date (today, from git; never inferred) must be strictly before
    # every scored_utc a given SCORES_META names, so the lock was not adjusted after the scores were already known.
    # A meta that fails to parse is left to scores_meta_errors later (exit 3); only ones read as a dict here count.
    # Each META as committed at --scores-meta-commit (sap.committed_bytes), not as on disk.
    meta_paths = args.scores_meta or [f.parent / "SCORES_META.json" for f in args.scores]
    meta_pairs = []
    for p in meta_paths:
        rel = repo_rel(p, args.repo)
        blob = None if rel is None else committed_bytes(args.repo, args.scores_meta_commit, rel)
        try:
            m = json.loads(blob if blob is not None else b"")
        except ValueError:
            continue
        if isinstance(m, dict):
            meta_pairs.append((p, m))
    bad = lock_precedes_scoring_errors(meta_pairs, lock_commit["committed"])
    if bad:
        print(f"refusing: {'; '.join(bad)}", file=sys.stderr)
        return 2
    unfinished = er_complete_errors(args.er_controls, now["er_manifest"])
    if unfinished:
        print(f"refusing: {'; '.join(unfinished)}: the ER control results are read from a complete run only",
              file=sys.stderr)
        return 3
    cfg = json.loads(args.config.read_text())
    run_deferred = lambda name: args.with_deferred or name not in cfg.get("deferred", [])
    score_bytes = [f.read_bytes() for f in args.scores]  # read ONCE; rows and sha256 from these bytes
    rows = [r for f, data in zip(args.scores, score_bytes) for r in jsonl_bytes(data, f)]
    compose = load(args.compose_excluded)
    aux = args.descriptive_only  # aux_test: its own locked controls and records
    part = {"test": (load(args.records), args.controls_dir),
            "aux_test": (load(args.aux_records), args.aux_controls_dir)}
    part = {k: (recs, load(d / "controls.jsonl"), load(d / "exclusions.jsonl")) for k, (recs, d) in part.items()}
    # the sha256 config/sap.json names (the list frozen at 30706f2, 26/09/2026), the lock's
    flags = pre_t_flags(args.pre_t_flags)
    bad = pre_t_errors(cfg, args.pre_t_flags, now["pre_t_flags"], flags,
                       {k: {r["id"] for r in recs} for k, (recs, _, _) in part.items()})
    if bad:
        print(f"refusing: {'; '.join(bad)}", file=sys.stderr)
        return 2
    pre_t = flags.get("aux_test" if aux else "test", set())
    records, controls, exclusions = part["aux_test" if aux else "test"]
    dose = None if aux or args.dose is None else {(x["model"], x["record_id"]): x["dose"] for x in load(args.dose)}
    keep = usable_records(controls, exclusions, compose, cfg["anchor_model"])
    dropped = compose_dropped(compose, cfg["anchor_model"])
    sc = {x["record_id"]: x["self_contained"] for x in load(args.self_contained)}
    er_rows, expected = er_scope([x for f in args.er_controls for x in load(f)], part["test"][0])
    try:
        elig = eligibility(er_rows, sc, expected)
        bad = overlay_errors(er_rows, manifest_overlays(args.er_manifest))
    except SapInputError as e:
        print(f"refusing: {e}", file=sys.stderr)
        return 3
    if bad:
        print(f"refusing: {'; '.join(bad)}", file=sys.stderr)
        return 3
    lines, gen, accepted_missing, bad = generation(args, cfg, part, compose, lock_commit["committed"],
                                                   lock_commit["sha"], score_bytes)  # generation manifests, META
    if bad:
        print(f"refusing: {'; '.join(bad)}", file=sys.stderr)
        return 3
    bad = score_row_errors(rows, controls, exclusions, elig, cfg, not aux, {r["id"]: r for r in records},
                           dose, lines)  # the input rows, unfilled
    if bad:
        print(f"refusing: {'; '.join(bad)}", file=sys.stderr)
        return 3
    accepted = {(rid, m, cond, seed) for rid, m, cond, variant, seed in (accepted_missing or {}).get("accepted", [])
                if variant == "main"}
    dcfg = aux_design(cfg) if aux else cfg  # aux_test is checked too (added 27/09/2026)
    try:  # every model of the design, each over its own records; aux: the anchor's 17 runs
        rows, info = check_complete(rows, {m: design_records(controls, compose, m) for m in dcfg["seeds"]}, dcfg,
                                    accepted, locked_flags(controls, exclusions), args.with_deferred)
    except SapInputError as e:
        print(f"refusing: {e}", file=sys.stderr)
        return 3
    completeness = {"usable_records": len(keep), **info}
    if accepted_missing:  # not in the lock, which precedes it: recorded in full
        accepted_missing["filled"] = sorted([r["record_id"], r["model"], r["condition"], r["seed"]] for r in rows
                                            if r.get("filled_missing"))
    excluded = {r["record_id"] for r in rows} - keep  # after the filled rows, whose records may have had none
    try:
        rows = subset_flags(rows, json.loads(args.subsets.read_text()), cfg["subsets"], sc)
    except SapInputError as e:
        print(f"refusing: {e}", file=sys.stderr)
        return 3
    rows = with_ca_prec(rows, cfg["ca_prec_tau"])  # D11 (c): the pre-registered CA_prec sensitivity analysis reads it
    try:
        conf = {} if args.descriptive_only else confirmatory(rows, cfg, excluded, elig, dropped)
        conf_prec = {} if args.descriptive_only else confirmatory(rows, cfg, excluded, elig, dropped, metric="ca_prec")
    except SapInputError as e:
        print(f"refusing: {e}", file=sys.stderr)
        return 3
    bad = ca_prec_denominator_errors(conf, conf_prec)  # CA_prec over CA's records only
    if bad:
        print(f"refusing: {'; '.join(bad)}", file=sys.stderr)
        return 3
    mdes = json.loads(args.mde.read_text())["mde"]
    blind, key = [], []
    if run_deferred("semv"):
        texts = {(o["record_id"], o["condition"], o["seed"]): o["text"] for f in args.outputs for o in load(f)
                 if o.get("model") == cfg["anchor_model"] and (o.get("variant") or "main") == "main"}
        blind, key = semv_sample(rows, cfg, excluded, texts)
    res = {"rehearsal": args.rehearsal, "tool_commit": git_head(ROOT), "lock": lock, "lock_commit": lock_commit,
           "n_rows": len(rows),
           "excluded_records": len(excluded & {r["record_id"] for r in rows}),
           "completeness": completeness, "generation": gen, "accepted_missing": accepted_missing,
           "confirmatory": conf,
           "conclusion_robust": (robust := robust_verdicts(conf, conf_prec)),
           "denominators": [d for recs, ctl, exc in part.values()
                            for d in denominator_rows(recs, ctl, exc, compose, cfg["anchor_model"])],
           "calibration_type_i": type_i_rates(json.loads(args.calibration.read_text())),
           "dose_checked": dose is not None,
           "pre_t": {"file": str(args.pre_t_flags), "sha256": now["pre_t_flags"],
                     "counts": {k: len(v) for k, v in sorted(flags.items())},
                     "partition": "aux_test" if aux else "test", "dropped": len(pre_t)},
           "ca_prec": {"tau": cfg["ca_prec_tau"], "confirmatory": conf_prec,
                       "differences": ca_prec_differences(conf, conf_prec) if conf else [],
                       "above_top4": above_top4(rows, cfg, excluded, dropped) if conf else {},
                       "controls": controls_ca_prec(controls, keep, cfg)},
           "answers": answers(conf, mdes, robust), "mde": mdes,
           "deferred_run": args.with_deferred, "semv_items": len(blind), "aux": any(r.get("partition") == "aux_test" for r in rows),
           "descriptive": descriptive(rows, cfg, excluded, elig, args.with_deferred, dropped, pre_t)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SAP_RESULTS.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    (args.out / "SAP_REPORT.md").write_text(report(res, cfg), encoding="utf-8")
    names = ["SAP_RESULTS.json", "SAP_REPORT.md"]
    if blind:
        for name, xs in (("semv_blind.jsonl", blind), ("semv_key.jsonl", key)):
            (args.out / name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in xs), encoding="utf-8")
            names.append(name)
    (args.out / "SHA256SUMS").write_text("".join(f"{hashlib.sha256((args.out / n).read_bytes()).hexdigest()}  {n}\n" for n in names))
    print("\n".join(([REHEARSAL] if args.rehearsal else []) + res["answers"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SapInputError as e:  # an input that is not JSONL (sap.jsonl_rows), wherever it is read: bad input
        print(f"refusing: {e}", file=sys.stderr)
        raise SystemExit(3)
