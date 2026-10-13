# Known issues

This file lists every change of the published code beyond its comments and texts, and what is known not to work,
or to work only under stated conditions, in the published code and data. None of the items below changes a
published number.

## Code changes beyond comments

The published code differs from the code of the study in its comments and docstrings, in some help and message texts,
and in the changes listed below, by file. None of them changes a published number. For the files of the locked
analysis, `prereg/LOCK_EQUIVALENCE.md` explains how they differ from the locked files and how to check it.

### Dataset, prompts and the step reader

- `scripts/fetch_history.sh` (7 lines): each repository is checked out at the commit pinned in the dataset records
  (read with `fetch_full.pins`, the same source as `fetch_full.py`) instead of the head of a branch; the clone no longer
  takes `--branch`, and the `git pull` with `|| true` that could hide a failed step is removed; the script sets
  `-o pipefail` and stops unless `HEAD` equals the pinned commit after checkout.
- `config/phase2.json` (3 lines): three free-text notes are reworded, two under `excluded_tags_evidence` and one under
  `required_tags_any_evidence`; no value changes.
  The configuration copied into `out/phase2/split.json` (A1) carries the same rewording, listed with its original
  and public sha256 in `packages/A1_dataset/SANITIZE_MAP.tsv`.
- `tools/build_feature_provenance.py` is replaced by `tools/build_provenance.py`, and the root `provenance.jsonl` and
  `summary.json` by `data/provenance.jsonl` and `data/summary.json`. The new tool reads the commits pinned in the
  dataset records and lists every feature, step, context and scoping file with its licence; the old files carried
  fields this release does not use, local paths, and an apache/fineract commit other than the pinned one.
- The step reader of `v2tools/stepdefs.py` is the one described under "Step reader of Phase 1 and 2" (one file of
  nhsconnect/gpconnect-provider-testing); no other change to the dataset code.

### Scoring and analysis

- `scripts/score_run.py:83–86`: `A3_ALLOWED` holds a third sha256 of `v2tools/symcache.py`, `f1c41842…`: the published
  file, whose code differs from the study's version only in its docstrings. `score_run` therefore accepts prompt sets
  rebuilt from this repository. `tests/test_score_run.py:645–648` pins the same set. No score changes.
- `scripts/sap_run.py` (four texts printed into `SAP_REPORT.md`) and `scripts/sap_swrap.py` (the `note` field of
  `SWRAP.json`): internal working labels are removed; the dates and the analysis labels (see `docs/LABELS.md`) are
  kept. The published `SAP_REPORT.md` and `SWRAP.json` carry the same wording. `tests/test_sap_cli.py:2216` expects
  the new text of one of the notes. No number changes.
- `v2tools/sap.py`: twelve refusal messages no longer end with a working label; `tests/test_sap.py` and
  `tests/test_sap_cli.py` assert the new texts. The messages are printed only when an input is refused, so no
  published file changes.
- `tests/test_sap.py:1052`: the "edit on main" commit is now dated one second after the branch commit, so the two
  commits are listed in a fixed order (see "A test that depended on commit times"). The locked code is unchanged.
- Test names: 40 test functions are renamed (`tests/test_argcheck.py`: 4, `tests/test_calls.py`: 1,
  `tests/test_controls.py`: 3, `tests/test_describe.py`: 2, `tests/test_extract.py`: 3, `tests/test_final_split.py`:
  1, `tests/test_sap.py`: 2, `tests/test_sap_cli.py`: 1, `tests/test_sap_errsample.py`: 1, `tests/test_sap_swrap.py`:
  3, `tests/test_score_independent.py`: 18, `tests/test_score_run.py`: 1); pytest collects and runs the same tests.
- Test helpers: the temporary git branches of two tests in `tests/test_sap.py` (now `side` and `side2`) and their
  variables, the folder of the symlink probes in `tests/test_sap.py` and `tests/test_sap_cli.py` (now `copied_meta`),
  and the table of contrived output shapes in `tests/test_score_independent.py` (now `SHAPE_CASES`) are renamed; the
  tests check the same behaviour.
- `scripts/sap_swrap.py`: the internal function that reads the output rows and checks each against the generation
  run's manifest is renamed `manifest_rows`; it is called only inside the file, and its logic is unchanged.
- `tools/score_launcher.py`: the launcher used for every scoring call of the study. The published copy differs from it
  only in its docstring and in its three stop messages, which are now in English ("STOP: …"). It runs only on CPython
  3.12.12, the interpreter of the scoring run, and stops on any other version.

### GPU code and operations

- `conftest.py` (new file): when the private run sheets and planning document are absent, the test functions that
  need them are not collected, and the count is printed (see "Tests that need private documents").
- `ops/pod_bootstrap.sh` and `ops/pod_parity_gate.sh`: the test set of the training bootstrap (`KIND=gpu`) is listed
  explicitly, `tests_gpu` and nine `tests/` modules (`bind`, `gherkin_steps`, `history`, `patterns`, `scopes`,
  `split`, `stepdefs`, `wrong_suite_control`, `yield_report`), instead of `tests tests_gpu`, because `tests/` now also
  holds tests of later phases that the training machines did not run (see "Test set of the training bootstrap").
  `tests_gpu/test_ops_scripts.py` pins the new command.
- Default paths: `ops/populate_hf_cache.sh` (`DEFAULT_HOME`), `ops/pod_parity_gate.sh` (the `HF_CACHE` default) and
  `tests_gpu/conftest.py` use `hf-cache/` at the repository root instead of an absolute path on the build machine.
  `tests_gpu/test_ops_scripts.py` pins them.
- `ops/pod_parity_gate.sh`: the default image is `gtestgen-cpu-gate`, built from `ops/gate_images/cpu-test`, instead
  of an internal image that is not published. `PODSAFETY_IMAGE` still overrides it.
- `v2gpu/canary.py`: the key of the training-canary verdict that flags a loss difference is `flag_for_review`. The
  canary reports of the study, which are not published, use an earlier key name. `tests_gpu/test_canary.py` follows.
- Message texts: refusal, error and status messages that named internal working labels are reworded. In `v2gpu/`:
  `audit.py` 1, `engines.py` 3, `examples.py` 2, `gencanary.py` 1, `gencoverage.py` 4, `generate.py` 1, `genprep.py`
  2, `genqueue.py` 4, `inputs.py` 1, `jobqueue.py` 1, `train_probe.py` 1. In `ops/`: `pod_bootstrap.sh` 7,
  `pod_parity_gate.sh` 1, `watch_pod.sh` 1. Only the text changed. When a check fails, some of these messages go into
  the problem list of a coverage, verdict or receipt file, and one goes into the `outcome.json` of a queue refused at
  start. None of them occurs in the published A3 package, whose coverage and receipts report no problem. The tests
  that pin these messages follow (`tests_gpu/test_engines.py`, `test_gencoverage.py`, `test_genqueue.py`,
  `test_ops_scripts.py`, `test_pod_safety.py`).
- `tools/d4_table.py`: one printed message is reworded; no value changed.
- Test names: 134 test functions in `tests_gpu/` are renamed, most of them to drop an internal label at the start of
  the name (`test_engines.py` 5, `test_gencoverage.py` 15, `test_generate.py` 1, `test_genprep.py` 2,
  `test_genqueue.py` 7, `test_jobqueue.py` 14, `test_leakcheck.py` 1, `test_ops_scripts.py` 54, `test_plan_time.py` 5,
  `test_pod_safety.py` 3, `test_train_job.py` 15, `test_train_probe.py` 12). One test helper (`test_ops_scripts.py`)
  and one test constant (`test_genprep.py`) are renamed too. `conftest.py` lists the renamed tests it does not collect
  without the private documents. The test logic is unchanged.
- `tests_gpu/test_ops_scripts.py`: `test_the_bootstrap_stops_with_exit_2_when_the_first_run_lock_has_no_sha256` runs
  the bootstrap with a fake `df` that reports 100 GB free, so that the test does not stop at the disk check on a
  machine with less than 45 GB free, before the step it checks; its assertions are unchanged.
- `config/plan_tokens_scenarioA.json`: the `_note` text, which no code reads, is reworded.

### Execution harness

- `er/batch.py:84`: the default of `--out` (`DEFAULT_OUT`) was an absolute path on the machine that ran the study; it
  is now `out/phase3/er`, relative to the working directory (run from the repository root). No score, status or count
  in `er_results.jsonl` depends on it.
- `er/records.py:9`: the default of `ER_PHASE2`, where the sealed Phase 2 records are read, was an absolute machine
  path; it is now `out/phase2`, relative to the repository root. The records read are the same, and `seal_sha256` still
  identifies the sealed dataset from `PHASE2_SHA256SUMS`.
- `er/er_run.py:31`: `PHASE2` of the single-run command line had the same change (now `out/phase2`). `er.batch` does
  not use this module's default.
- `er/batch.py`, help text of `--root`, and `scripts/write_overlays.py`, help text of `--symtimes`: reworded to name
  `scripts/write_overlays.py` and "the sealed symbol times". Defaults and behaviour are unchanged.
- `tools/er_trace_fn_sample.py:96`, `:107–108`, `:115` and `:128`: the `rule` string written to `SAMPLE_META.json`
  and three strings of `WORKSHEET.md` (the instruction line, a column of the table and the line filled in for each
  case) are reworded; the worksheet now names the two judgments of each case. These strings are exactly those of the
  published `out/phase3/er_trace_fn_sample/` files (rule R5 of `packages/A5_er/SANITIZE_MAP.tsv`). The population, the
  order and `sample.jsonl` are unchanged: the draw re-run from the public tree gives the published files.

## Tests that need private documents

The operational run sheets of the GPU work (`ops/*RUNSHEET*.md`) and one planning document are kept private and are
not published. Twenty test functions (21 test cases) in `tests_gpu/` read those
documents. In this repository the root `conftest.py` does not collect them when the run sheets are absent: it lists
each test with its reason, and both the pytest session header and the terminal summary report how many were not run.
With the run sheets present the tests run as before. Test modules whose only subject was those documents (the
run-sheet tests and the tests of the training-canary evidence folder) are not published.

The study's internal CUDA gate image is not published; the published tests run in the CPU test image defined in
`ops/gate_images/cpu-test`.

## Test set of the training bootstrap

During the study, the bootstrap of a training pod (`KIND=gpu bash ops/pod_bootstrap.sh`, step 5) ran
`pytest tests tests_gpu`, and at the commit that trained the adapters `tests/` held nine modules (49 tests). In this
repository `tests/` also holds the tests of the later phases, among them tests that need Docker (the execution-based
harness) or a git history newer than the one a pod's checkout carries; a training pod never ran them, and the
bootstrap's no-skip check would refuse their skips. `ops/pod_bootstrap.sh` and `ops/pod_parity_gate.sh` therefore list
the test set explicitly: `tests_gpu` and those nine modules (`tests/test_bind.py`, `test_gherkin_steps.py`,
`test_history.py`, `test_patterns.py`, `test_scopes.py`, `test_split.py`, `test_stepdefs.py`,
`test_wrong_suite_control.py`, `test_yield_report.py`). The generation bootstrap (`KIND=gen`) is unchanged.

## Git default branch in the test image

Three tests of the analysis code (`tests/test_sap.py`, `tests/test_sap_cli.py`) create bare git repositories and expect
their first branch to be `main`. Measured: with git 2.39.5 (Debian bookworm) they fail unless `init.defaultBranch` is
set to `main`, and pass when it is; with git 2.50.1 they pass either way. Other git versions were not measured. The
CPU test image (`ops/gate_images/cpu-test/Dockerfile`) sets `git config --system init.defaultBranch main`.

## Running the tests in a container

Three tests of `tests_gpu/test_jobqueue.py` check that a killed job's whole process group, grandchildren included,
is gone. In a container, process 1 must reap orphaned processes, or the killed grandchildren remain as zombies and
those tests fail; run the container with `docker run --init`, as the commands in docs/REPRODUCE.md do.

`ops/pod_parity_gate.sh` creates its work directory with `mktemp -d` in the system temporary directory and keeps it
when the gate fails, so that the logs can be read; remove it by hand afterwards.

## A test that depended on commit times

tests/test_sap.py::test_committed_file_errors_once_survives_a_no_ff_merge expects a refusal message that names two
commits in the order `git log --full-history` lists them. The test makes both commits itself, usually within the same
second, and on git 2.39 the order of two commits with equal dates is not fixed: the original test failed in 27 of 30
runs on git 2.39, while it passed on git 2.50. The published test dates the second commit one second after the first,
so the order is fixed; the checked code (v2tools/sap.py) is unchanged.

## Step reader of Phase 1 and 2

The released v2tools/stepdefs.py is the version that scored the outputs. The dataset (out/phase2, package A1) was
produced by an earlier version of the reader, which the released module keeps as `_extract_sealed`; `extract` uses it
whenever it is called without `imports`, as Phase 1 calls it. On every source file of the six repositories at their
pinned commits (25,047 files), the released reader returns exactly the step definitions of the earlier one, except in
one file: nhsconnect/gpconnect-provider-testing, GPConnect.Provider.AcceptanceTests/Steps/ManualSteps.cs. Its three
catch-all methods each carry `[Given(".*"), When(".*"), Then(".*")]`; the earlier reader took only the first attribute
of each method (3 definitions), the released one takes all three (9). No dataset record refers to this file. A rebuild
of Phase 1 and 2 from the released code was run and compared with the sealed files: the scenario pairs, labels,
splits, forbidden lists, records and sensitivity subsets are byte-identical; only the annotations of these three
catalog rows differ in catalog.jsonl, and the step-definition counts in yield.json (gpconnect-provider-testing: 619 →
625 definitions, 3 → 9 catch-all definitions). The sealed dataset (A1) is the reference.

## Pod disk size

On pods created with a 50 GB container disk, the bootstrap's own test step (`ops/pod_bootstrap.sh`, step 5) failed two
tests that check the room left for the virtual environments and the model weights
(`tests_gpu/test_ops_scripts.py`, `tests_gpu/test_genqueue.py`). Those pods were replaced before any job ran; every
pod of the main training and generation runs used a 100 GB container disk.

## Generation run operations

The generation pods were driven by orchestration scripts that are not part of this repository (creating pods, copying
results home while a queue runs, watching for stalls, the final copy window, deleting pods). The items below were
observed during the main generation run and were recorded rather than fixed, because no further GPU run was planned.
None of them affected an output: all 14 generation jobs and 20 re-check jobs completed, every output was copied home
and checked (coverage PASS, 17,712 output lines), and every re-check, on the same pod and on another pod, was
byte-identical to the original output.

- **Stall check.** The check for a stopped or stalled pod reads the queue's heartbeat only. It would be stronger with
  a process check of the generation job (`pgrep`) and `nvidia-smi --query-compute-apps`, and with the exit code of
  `pgrep` checked, so that "no such process" is not confused with a failed command.
- **Idle alert.** The idle alert names the problem but not the command to attach to the pod by hand; printing that
  command would save a step.
- **Broken `started.json`.** A broken queue marker file (`started.json`) needs a manual cleanup step; the procedure is
  not written down in a published file.
- **Partial copies.** An interrupted `scp` can leave a partial log file at home. Copying to a temporary name and
  renaming after the copy completes avoids this.
- **Stall threshold of the canary.** The stall check of a generation canary pod must allow for the repeat-longest
  phase (at least 10 x 6,144 tokens at the measured speed, plus a margin); a fixed threshold can report a stall that
  is not one.
- **Re-starting a pod's setup.** Re-running the setup of a pod must stop the earlier preparation watchdog by its exact
  process id and confirm that exactly one watchdog is left before the queue starts. Several process ids returned over
  one ssh call can arrive on separate lines and must be parsed as such.
- **Waiting for the final copy window.** The final-copy watcher waits a fixed 12 hours for the queue's copy window
  instead of waiting until the queue's own deadline; on long queues it stopped early and had to be restarted by hand
  (23 restarts over 13 pods in the main run).
- **Re-shipping onto a network volume.** `ops/ship.sh` cannot ship a source file of mode 0444 a second time onto a
  MooseFS network volume (a re-ship to the same pod fails; a fresh pod is not affected). Ship files with mode 0644, or
  skip files that are already present and identical.
- **Early delete.** When a pod is deleted early, after its results are home, the final-copy watcher always ends with
  "outcome missed", which is expected; an explicit "early delete" mark would make the log clearer.
- **Copy-loop timestamps.** The copy loop met occasional ssh timeouts (never enough in a row to stop it); a timestamp
  on each ssh error line would make them easier to match with other logs.
- **Delete times.** The recorded delete time of a pod is the time the deletion was confirmed, up to about 10 seconds
  after the provider's own record; recording the time the delete request was sent as well removes the gap.
