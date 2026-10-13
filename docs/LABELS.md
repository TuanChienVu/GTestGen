# Labels in the analysis reports

The analysis reports in A4 (`out/phase3/sap/results/SAP_REPORT.md` for the test set,
`out/phase3/sap/aux_test/SAP_REPORT.md` for the auxiliary test set, the `SAP_RESULTS.json` next to each, and the
`*_META.json` files of the descriptive analyses) are written in Vietnamese and name items of the analysis plan and
sections of the design specification by short labels. The labels are kept as printed, because the published files and the locked code carry them. This file gives
the meaning of each label and, where it shows the order of the pre-registration, the date on which it was fixed. The
analysis plan was locked on 7 October 2026, before any output on the test set existed.

## Labels of the analysis

Labels that start with P number the items of the analysis plan document. Unless the last column names another file,
a label is printed in both `SAP_REPORT.md` files.

| Label | Meaning | Fixed | Printed in |
|---|---|---|---|
| D11 | Safeguard of Call Agreement against outputs that call every visible name: two more lazy controls (`stub-v3names`, `stub-v3top4`), the measures `call_precision` and `n_calls`, and the CA_prec sensitivity analysis | 25/09/2026, before any model output | `SAP_REPORT.md` |
| D11 (d) | The fourth lazy control, `stub-v3top4` (each stub body makes the four most frequent calls), and its rate "CA_prec vượt mức lười top4" (CA_prec above the top-4 lazy level) | 26/09/2026, before any model output | `SAP_REPORT.md` |
| P4 | Usable records and completeness: the records of the controls run, minus mechanical exclusions and the records that prompt building dropped; one row per record and design run | 24/09/2026, in the plan locked on 7 October 2026 | `SAP_REPORT.md` |
| P9 | Verdict of one comparison: "supported" or "contrary" needs both tests significant after Holm; "conditional_only_positive" or "conditional_only_negative" when only the cluster sign-flip test is | 24/09/2026, before the lock | `SAP_REPORT.md`; `SAP_RESULTS.json`, field `conclusion_robust.rule` |
| P11 | Descriptive measures beside each comparison (BV, ER, ER-trace), with their eligibility table and the pre-declared `opposite` flag (the confidence interval excludes 0 on the negative side, or a negative estimate stands beside a supported CA comparison); estimates and confidence intervals, no p-value | 24/09/2026, clarified 06/10/2026, before the lock | `SAP_REPORT.md` |
| P14 | Descriptive analyses outside the Holm families: the RQ3 level of CA per model at Full-aware, and the RQ3 quality profile (CA, BV, ER, ER-trace) | 24/09/2026; the quality profile added 07/10/2026, before any test output | `SAP_REPORT.md` |
| P16 | Agreement of CA with ER-trace (Cohen's kappa), pooled over the runs of the anchor model, with an attribute-only version | 26/09/2026; pooling over the anchor model only fixed 07/10/2026, before any test output | `SAP_REPORT.md` |
| P18 | Rule for the configuration chosen on the development set: keep `dev_pilot_v0` unless another configuration scores at least 3 more (the sum over development scenarios of scenario CA at Full-aware, DeepSeek-Coder); a smaller gap is a tie; among several above, the highest score wins, then the first name in string order | 24/09/2026, before any model output | the table of `tools/d4_table.py`; applied by `select_dev_config` in `v2tools/sap.py` |
| P22 | Robustness label: a confirmatory conclusion is "vững" (robust) only when CA and CA_prec give the same verdict label (P9), per comparison and per research question | 27/09/2026, before any output | `SAP_REPORT.md`; `SAP_RESULTS.json`, field `conclusion_robust.rule` |
| P25 | Manual error reading: 30 test records with CA = 0 at Full-aware on all five DeepSeek-Coder seeds, stratified by repository and read by hand | 06/10/2026, before the lock and before any test output | `out/phase3/sap/errsample_test/ERRSAMPLE_META.json`, field `name` |
| P26 | Descriptive measures of section 2.4 of the design specification, added before the lock: reuse of duplicates, exact match, expected directions, prompt-length groups and dose response | 06/10/2026, before the lock and before any test output | `SAP_REPORT.md` |
| P26 (e) | Dose response: Full-aware minus Full-neutral CA by the number of gold-answer symbols in the project-knowledge block, each level with a confidence interval and a direction label | 06/10/2026, before any test output | `SAP_REPORT.md` |
| H0 | The null hypothesis of no difference (delta = 0), under which the calibration measured the rejection rates of the tests | before the lock (`prereg/CALIBRATION.json`, registered) | `SAP_REPORT.md` |
| H3 | Hypothesis H3 of the project outline: the RQ3 comparison between models is descriptive, with no test and no label | project outline | `SAP_REPORT.md` |
| S-wrap | Descriptive sensitivity analysis of RQ1: the base-model runs scored again after removing a namespace or class shell around the step definitions; outside the Holm families | 05/10/2026, before the lock | `out/phase3/sap/swrap_test/SWRAP.json` (fields `name` and `note`) and `SWRAP_META.json` |

## Sections of the design specification

`§` followed by a number, sometimes written "đặc tả §…" (specification §…), names a section of the design
specification `SPEC_V2_PHASE3_DESIGN.md`. The specification is not published. The registration's `HASHES.txt` gives
the commit of the registered version (`05b428741f70c38d13a987063849a354aea18843`) and the SHA-256 of the file at that
commit (`fe6aaca984d367da918e3117c78d9520d718d92f0bac449166b7353b72089e09`).

| Label | Section | Fixed |
|---|---|---|
| §0b | The reduced scope of the study | 24/09/2026 |
| §1.4 | The output format and the code extractor (handling of PHP tags) | before any model output |
| §2.2 | Binding Validity (BV), and the step-reader counts reported with it | before any model output |
| §2.2b | Call Agreement (CA), the decisive measure: whether the output calls the project components that the developer's solution calls | 24/09/2026, before any model output |
| §2.3 | Executable Rate (ER) on the Sylius records of the execution core (`exec_core`) | before any model output |
| §2.3b | ER-trace, a descriptive measure: the trace of a real run (API requests and database writes) is compared with the gold trace, and in a run where the create steps do nothing, the generated Then steps must fail wherever the gold's fail. With it, before the SAP lock, the hand check of 30 cases that pass ER but fail the trace, to measure its false negatives (`tools/ER_TRACE_FN_SAMPLE.md`) | 24/09/2026, before any model output; the sampling rule of the hand check 06/10/2026, before any new output |
| §2.4 | The descriptive measures: directions, reuse of duplicates, exact match, dose response, prompt length | before any test output |
| §4.5 | The selection rule of the development trial round, declared in advance; P18 applies it in the analysis code | 24/09/2026 |
| §5 | The statistical analysis plan (SAP) | locked on 7 October 2026, before any test output |

## Labels in code names

A few names in the code carry a label. They are kept because many files and tests refer to them.

| Label | Meaning | Where |
|---|---|---|
| D4 | The table of the configuration chosen on the development set: the score of each configuration (CA at Full-aware, DeepSeek-Coder, development set), the choice made, and the choice derived again by P18 | `tools/d4_table.py` |
| D10 | CA with future helpers: an output's method replaces a not-yet-visible method of the same class, and calls into it are opened for CA and for condition 5 of BV (fixed 25/09/2026, before any model output). `d10_controls` and `D10_NAMES` are its four helper controls (`gold-helpers`, `gold-helpers-renamed`, `stub-v3m-h`, `stub-v3all-h`); `D10_LANGUAGES` lists the languages with classes (PHP, Java, C#) | `v2tools/controls.py`, `tests/test_controls.py` |
| step 8 | In the names `test_step_8_*`: the final check of the analysis, `scripts/sap_verify.py`. `sap_run` run again from a fresh clone of the origin repository must give the same `SAP_RESULTS.json` as the working run, and the result row of the pre-registration ledger must match it. A rerun from this repository replaces the ledger comparison by the checks of `prereg/LOCK_EQUIVALENCE.md`, "Provenance checks during a rerun" | `tests/test_sap_cli.py` |

## Probe names in tests

The names below are local to one test each. They name probes, inputs changed on purpose to check that the analysis
refuses them, and are not items of the analysis plan: that P4b and P9 look like the plan's P4 and P9 is a coincidence.

In `tests/test_sap_cli.py`, `test_run_refuses_rows_that_move_the_denominators` feeds each probe to
`scripts/sap_run.py` under a valid lock. Every probe except P6 must be refused.

| Probe | Changed input |
|---|---|
| P1b | The Zero-shot rows added again under seed s42, with CA changed |
| P1c | The Zero-shot rows added again under seed s42, unchanged |
| P2 | An extra Full-aware seed (s5) on 20 usable records |
| P3 | The Full-aware rows of two records removed |
| P4b | Rows of a record that has no gold row in the locked controls |
| P5b | Another `scenario_key` on the Full-aware rows of one scenario |
| P6 | One missing row, accepted beforehand in a committed `accepted_missing.jsonl`; the run goes on and scores the filled row as CA = 0 and BV = 0 |
| P9 | No CA value on rows that count in CA |
| P10 | Input rows that already carry `filled_missing` |

The probe outputs written by `scripts/probe_outputs.py` have names of their own (U…, X…, H1–H7); they are not
the hypotheses H0 and H3 above.
