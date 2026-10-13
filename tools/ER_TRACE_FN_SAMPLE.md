# Hand check of 30 cases (§2.3b): sampling rule (fixed 06/10/2026, before any new output)

The pre-registered design (§2.3b) asks, before the SAP lock, for a hand check of 30 cases that PASS ER but fail the
trace, on the outputs of the dev round, to measure the false negatives of ER-trace. This file records the sampling rule.
The rule was fixed before the outputs of the new generation run (commit `0482a76`) existed, so which cases are checked
does not depend on having seen any result.

## Population P

- P is the **union** of every dev ER run that exists before the SAP lock:
  - `dev_pilot_v0` (earlier outputs);
  - `dev_pilot_v0_0482a76` (new outputs);
  - the ER runs of the grid, named `<grid label>_0482a76`.
- Earlier outputs may be used because the false-negative rate is a property of the ER-trace scorer, not of a model.
  They are valid evidence for this measurement, but they are never used for scores.
- An `er_results` row is in P when all four hold:
  - its condition is a model condition, not `neg`, `gold` or `stub`;
  - `er` = 1 and `act_evaluable` is true;
  - `trace_action` = 0, or `trace_then` = 0;
  - its tamper verdict is not `cheat`.
- Rows that PASS ER with a null trace ("not evaluable") are counted and reported apart, and never sampled.

## Sampling

- |P| ≤ 30: every case is checked.
- |P| > 30: the 30 cases with the smallest `sha256("2.3b|<run>|<record_id>|<condition>")` are checked.
- No extra ER run is made just to reach 30.
- Kind of each case:
  - `action`: `trace_action` = 0;
  - `then`: `trace_action` = 1 and `trace_then` = 0.

## Verdicts and report

- Each case gets one verdict:
  - `FN`: the output behaves correctly, but the trace fails for a reason that is not behaviour. This is a false
    negative of ER-trace.
  - `TP`: the failed trace reflects a real difference.
  - `UNCLEAR`: cannot be decided; the reason is written down.
- How the verdicts are reached:
  - the verdicts are filled in `WORKSHEET.md`;
  - every case is judged a second time, independently, without seeing the first verdicts;
  - the two judgments are compared case by case.
- Report:
  - the actual n;
  - the FN rate = FN / (n − UNCLEAR), with a 95 % Wilson interval;
  - by run, by condition, by kind and by record, since cases may cluster.

## Command

The output directory must be new. Each further run of the development grid in the population adds one more `--source`
with its label, its ER run directory under `out/phase3/er/` and its overlay manifest.

```bash
python3 tools/er_trace_fn_sample.py --selftest
python3 tools/er_trace_fn_sample.py --controls-er out/phase3/er/controls --out out/phase3/er_trace_fn_sample_rerun \
  --source dev_pilot_v0 out/phase3/er/dev_pilot_v0 out/phase3/overlays/dev_pilot_v0/manifest.jsonl \
  --source dev_pilot_v0_0482a76 out/phase3/er/dev_pilot_v0_0482a76 out/phase3/overlays/dev_pilot_v0_0482a76/manifest.jsonl
```

This example draws from two runs only. To repeat the official draw over all 28 runs, see
`packages/A5_er/README.md`.

The tool only reads `out/`, writes only into `--out` (it refuses an existing directory) and does not import the
scoring code. It writes three files:
- `sample.jsonl`: the cases, with the paths of the gold and output traces (normal run, M1), the log, the multiset
  difference and the Then status under M1;
- `SAMPLE_META.json`: the population by run, condition and kind;
- `WORKSHEET.md`: the form for the verdicts.

The A5 package holds every file this command reads for the official draw (28 runs: `dev_pilot_v0`,
`dev_pilot_v0_0482a76` and the 26 grid runs; population 69, 30 checked), so the draw can be repeated from it. The
worksheet text is in Vietnamese, as it was filled in.

Trial run on the earlier `dev_pilot_v0` alone (06/10, only to test the flow): |P| = 8, all of kind `then`, on 5
records of `sorting_catalog_promotions`; plus 11 rows that PASS ER with a trace that cannot be evaluated.
