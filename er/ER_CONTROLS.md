# ER harness for Sylius: design, controls and reproduction

The ER harness runs the Behat scenario of a Sylius test record in a fresh container, with the record's ER tree and
the condition's code inserted, and records whether the scenario passes (ER) and how its actions and checks behave
(ER-trace, a descriptive metric). This file documents the environment, the controls, the official runs, the threat
model, the known limits and the commands to reproduce the runs. The result files of the runs are in the A5 package.

All commands below run from the repository root.

## 1. What is run

- **Records:** the 100 Sylius test records of suites `api*` (`exec_core` in `out/phase2/records/test.jsonl`; one
  record per scenario) and, for the dev round, the 49 Sylius dev records of suites `api*`. Suites `api*` run under
  Behat without a browser.
- **One run** = one (record, condition): `vendor/bin/behat --strict --suite=<suite> <feature>:<line>` inside a fresh
  container of the trace image, on exactly one scenario. ER = 1 when that scenario PASSes. `--strict` makes an
  undefined step an error, so the negative control keeps its meaning.
- **ER tree** (`v2tools/ertree.py`): Sylius HEAD minus the record's answer and minus every step function with no
  annotation older than the record's visibility time. The base tree of a record is shared by all its conditions; a
  condition adds only the target files with the code inserted (`scripts/write_overlays.py`, contract in
  `er/manifest.py`).
- **Conditions:** three controls (`gold`: the gold answer inserted; `neg`: nothing inserted; `stub`: one new method
  with an empty body per CREATE annotation) and one condition per model run (`<model>.<seed|base>.<condition>`).
- **ER-trace** (`er/trace.py`): the scenario's action trace (multiset, at the level of base status), a mutation run M1
  (the CREATE action steps made no-ops) and the Then sensitivity. It is reported apart from ER and is descriptive.

## 2. Environment and pins

| Item | Pin |
|---|---|
| Sylius | `c096ca3fc01545ee25078ba3df914bd2069c7fa6`, the commit of every Sylius record (`out/phase2/DATASET.md`) |
| PHP base image | `php:8.4-cli@sha256:2b87a5d0c0465ade44b1bbec46e8f63520448a7bd7ec90a284dd8b8f0b07fd53` |
| PHP extensions | intl, gd, exif, zip, pdo_mysql (docker-php-extension-installer 2.11.33) |
| Composer | 2.10.3 |
| `composer.lock` | generated once (Sylius does not commit one), kept at `er/docker/lock/composer.lock`, sha256 `f71ed33c72a4e30df98db06bf8fa4688e40b3328af9ac982ff1a5bfd86a9e72f` |
| Main packages in the lock | symfony/framework-bundle v7.4.19, api-platform/symfony v4.4.1, doctrine/orm 3.7.2, doctrine/doctrine-bundle 3.3.2, behat/behat v4.0.0-alpha1, friends-of-behat/symfony-extension v2.7.0 |
| Database | `mysql:8.4@sha256:0744ee5ef89ce6ccfa13de3e579fe6b9e27f93dd70da9c06d2c908b1b193fb8d`, data on tmpfs |
| Application | `APP_ENV=test_cached`, `APP_DEBUG=0`; Symfony as required by Sylius' `composer.json` (`^7.4`) |

The configuration follows the full matrix of Sylius' own CI (`ci_e2e-mysql.yaml`: PHP 8.4, Symfony 7.4, MySQL 8.4),
whose "Run API Behat" job runs the `api` and `domain` suites with the tags `~@todo&&~@cli`. The images were built and
run on arm64 (Docker Desktop on Apple silicon); other platforms were not tried.

**Checks made while the image is built** (`er/docker/Dockerfile`):
- Sylius is fetched at the pinned SHA only (`git fetch --depth 1`), and the build stops unless `git rev-parse HEAD`
  equals it;
- the build stops unless `composer.lock` has the pinned sha256, before `composer install`;
- after installing and warming the cache, the build stops if the tracked tree differs from HEAD or holds an
  untracked file outside `public/build/`;
- the trace image adds the harness tracers (`er/docker/trace/`, never part of an overlay), repeats the clean-tree
  check, then removes `.git`, so model code cannot read the history.

An image ID depends on the local build, so a rebuilt image has another `image_id` than the one recorded in the results
(`sha256:9260a267…`). Three independent image builds (three image IDs) gave, for the same 149 records, the same
`er_status`, `er`, `er_trace`, `trace_action`, `trace_then`, `gold_status`, `ref_error` and `skip` on every row, and
every gold reference was `deterministic` in each of them. A fourth build, from the code of this release, reproduced the
447 rows of the controls run and the 300 rows of `test_deepseek_s42_0482a76` on the same fields. The trace files
themselves can differ between runs. Request paths carry values that change from run to run: the random token Sylius
gives each order, the time-based UUIDs of payment requests, and database ids, which keep counting up on a lane until
the lane is recreated and so depend on what ran on it before. The error message of a failing step can carry such ids
and times as well. Scoring reads none of these values: it compares traces only after `norm_path` in `er/trace.py` has
replaced token route parameters with numbered placeholders, ids with `{n}` and UUIDs with `{uuid}`, and it uses the
status of each step, not its error message.

## 3. Database lanes and isolation

`er/docker/init-db.sh` starts the database container (never recreating it) and creates, per lane, a database
`sylius_laneN` with the Sylius migrations (26 migrations, 83 tables) and a user `er_laneN` limited to SELECT, INSERT,
UPDATE and DELETE on its own database, with a random password written to `$ER_SECRETS_DIR/laneN.pw` (mode 600; the
default directory is `.config/gtestgen-er` in the user's home). Lanes 1 and 2 run batches, lane 3 serves the tests.
The data is on tmpfs: rerun the script after every restart of the database container. `ER_LANES=<n> ER_RESET=1
er/docker/init-db.sh` drops and rebuilds a lane. The script refuses to run while a lane is in use.

Isolation, checked on the image used for every official run:
- the network `gtestgen-er_er` is internal: code in a run cannot reach the Internet;
- the trace image has no `.git`;
- root has no network account (`MYSQL_ROOT_HOST=localhost`) and is denied from the `er` network;
- exactly 7 MySQL accounts: `er_lane1..3@%` and the four local system accounts;
- the password files are readable by their owner only.

Every batch checks the accounts, the grants, the 83 tables, the schema fingerprint and the free space before it
starts (`er.db.preflight`), and checks the lane again after every run that does not PASS.

## 4. Running the harness

Sylius needs a clean full checkout at the pinned commit, because the tamper scan compares against it. The images are
built from scratch, both tags, and the database lanes are created. Then, once the overlays are rebuilt (section 5),
comes the controls run, which freezes the gold references, and each output run, scored against the frozen references of
the controls run; `RUN` names the output run, for example `test_deepseek_s42_0482a76`. `er.tamper review` writes
`er_tamper_review.md` and `er_tamper_verdicts.json`.

```bash
git clone https://github.com/Sylius/Sylius.git repos/Sylius
git -C repos/Sylius checkout --detach c096ca3fc01545ee25078ba3df914bd2069c7fa6
docker compose -f er/docker/compose.yaml build --no-cache sylius sylius-trace
er/docker/init-db.sh
python -m er.batch --root out/phase3/rebuilt/overlays --out out/phase3/rebuilt/er --run controls --head-root repos/Sylius --lanes 2
python -m er.report --out out/phase3/rebuilt/er --run controls
RUN=test_deepseek_s42_0482a76
python -m er.batch --root out/phase3/rebuilt/overlays --out out/phase3/rebuilt/er --run "$RUN" --head-root repos/Sylius --ref-run controls --lanes 2
python -m er.tamper review --out out/phase3/rebuilt/er --run "$RUN"
python -m er.tamper apply --out out/phase3/rebuilt/er --run "$RUN"
python -m er.report --out out/phase3/rebuilt/er --run "$RUN"
```

The batches read the overlays rebuilt in section 5 (`--root`) and write each run to `<out>/<run>/`: `er_results.jsonl`
(one line per record and condition), `er_preflight.json`, `er_complete.json`, `er_refs.jsonl` (controls run) and the
traces and logs under `er_runs/`. The default of `--out` is `out/phase3/er`, where A5 holds the published runs, so
every command above gives `--out out/phase3/rebuilt/er`: without it, `er.tamper` and `er.report` would read the
published A5 results. The controls run and the output runs need the same `--out`, because `--ref-run controls` reads
`<out>/controls/er_refs.jsonl`.

**Run the controls first.** A batch records `harness_code`, the sha256 of the files under `er/` (the `*.md` documents
left out). An output run needs the same `harness_code` as its controls run (`--ref-run`). This public tree differs
from the code that produced the published runs only in comments, documentation, default paths and the removed files
of an early experiment (shown by the AST equivalence check), but those bytes are part of `harness_code`. The published controls run therefore cannot serve as
`--ref-run` for a new output run: run the controls first, then the output runs.

A batch is resumable (`er_complete.json` lists what is missing), holds one lock per run directory and stops on an
infrastructure failure instead of recording it. The harness needs Python 3.12 with `requirements.txt` (defusedxml), git and
the Docker CLI; the tests need pytest. Four tests (`tests/test_er_runner.py`, `test_real_run_*`) make real Docker runs on
lane 3 and are skipped when Docker, the trace image or `repos/Sylius` is missing; every other test uses a fake Docker
and runs without it.

## 5. Rebuilding the overlays

The overlay trees are not published (about 3.6 GB); their manifests are in A5. Build them again from the dataset
(A1), the generation outputs (A3) and the symbol times of A2 (`out/phase3/symtimes.jsonl`), with full bare clones of the
source repositories at their pinned commits (`python scripts/fetch_full.py` writes `repos-full/<repo>.git`).

A5 stays unpacked at the root, where the development run below reads its id list. The overlays are rebuilt into
`out/phase3/rebuilt/overlays/`, because A5 already holds `out/phase3/overlays/<run>/` and `write_overlays.py` never
writes over a run directory (see `docs/REPRODUCE.md`, section 5).

`common` holds the arguments shared by every run; it is an array, so that bash and zsh pass them as separate words. A
test run uses the 100 records of the execution core (`exec_core`) and the outputs of its generation work directory; a
development run uses the 49 development ids and every DeepSeek work directory of its label, in glob order:

```bash
common=(--phase2 out/phase2 --full-root repos-full --symtimes out/phase3/symtimes.jsonl --out out/phase3/rebuilt/overlays)
python scripts/write_overlays.py "${common[@]}" --run controls --controls
python -c "import json; [print(r['id']) for r in sorted(map(json.loads, open('out/phase2/records/test.jsonl')), key=lambda r: r['id']) if r['exec_core']]" > exec_core_ids.txt
python scripts/write_overlays.py "${common[@]}" --run test_deepseek_s42_0482a76 --ids exec_core_ids.txt \
  --outputs A3/gen-main-0482a76-20261008/results/gen-main-deepseek-s42/outputs.jsonl
python scripts/write_overlays.py "${common[@]}" --run dev_grid_k8_n2_r067_0482a76 --ids out/phase3/overlays/dev_sylius_api_ids.txt \
  --outputs A3/gen-dev-0482a76-20261006/results/gen-dev_grid_k8_n2_r067-deepseek-*/outputs.jsonl
```

- `--outputs` takes the lines of the run's conditions only, in the order of the file. The DeepSeek work directories
  hold exactly the run's conditions; for the CodeLlama and Mistral runs, keep the `full_aware` lines of
  `outputs.jsonl` and give that file.
- The manifest row order follows the order of the output lines, so a reordered input gives the same rows in another
  order and another `MANIFEST_SHA256`.
- Check: `shasum -a 256 out/phase3/rebuilt/overlays/<run>/manifest.jsonl` must equal the hash in
  `out/phase3/overlays/<run>/MANIFEST_SHA256` (A5).
  Every overlay tree of the controls run and of the 12 test runs was rebuilt this way and was identical file for file
  to the tree the official runs used.

## 6. Controls: the official run of 05/10/2026

The controls run freezes the gold reference of every record for all later output runs; its results are hashed into
the pre-registered SAP lock.

- **Manifest:** 149 records (the 100 test `exec_core` records and the 49 dev `api*` records) × 3 conditions = 447
  lines, `MANIFEST_SHA256` `031237a844f60f2eafe90c003fdb788f29d72180db89b81c4a9360703774fc72`.
- **Identity:** `harness_code` `5d823e5f37cf2a0c478d516a13e3cc7d298c808301e4aced183eeda0616ce4c4` (clean tree),
  `image_id` `sha256:9260a267d3fc42268f0cbaf011668a950e4bcc612359ae7dd9689ec2117864ed`, schema fingerprint
  `22ec61e5a5e8…`, dataset seal (`PHASE2_SHA256SUMS`) `f6303582…`.
- **Run:** 447 runs, 752.6 s with 2 lanes (mean 1.24 s per run, p95 1.87 s, no cold run); `er_complete.json`:
  complete, 447 rows, none missing, no duplicate.
- **Results:**

  | Condition | ER status | ER | ER-trace | trace_action | trace_then |
  |---|---|---|---|---|---|
  | gold | PASS 149 | 149/149 | 130/130 | 128/128 | 92/92 |
  | neg | UNDEFINED 149 | 0/149 | | | |
  | stub | FAIL 47, PASS 102 | 102/149 | | | |

  ER-trace cannot be evaluated on 19 records (neither trace part is evaluable). No tamper flag, no reference error, no
  infrastructure failure.
- **Records that pass all three controls** (gold PASS, neg and stub not PASS): **47/149**; test 36/100, dev 11/49.
  A stub PASSes when the scenario's checks hold with empty step bodies (no REUSE Then step, or a negative or generic
  check), so such a record cannot tell empty code from correct code and leaves the ER denominator (declared before any
  output).
- **Determinism:** all 447 lines equal those of an earlier full run from scratch (made as two runs: 300 lines of the
  test records, 147 of the dev records) on `er_status`, `er`, `trace_action`, `trace_then`, `er_trace`, `gold_status`
  and `ref_error`; every reference reproduces itself (`deterministic`).
- **Gate declared before any output:** with fewer than 60 test records passing all three controls, ER becomes a
  descriptive metric. Measured 36/100 < 60, so ER and ER-trace are descriptive, and CA is the decisive metric
  (decided 24/09/2026, before any output).

## 7. Output runs on the test set: 11/10/2026

- 12 runs (DeepSeek 5 seeds and the base model, CodeLlama 3 seeds, Mistral 3 seeds), 23 (run, condition) pairs, 100
  records each: 2,300 rows, 2,205 not SKIP; every run complete, none rerun; 15:39–16:51 (UTC+7), one series.
- ER 280/2,300. Statuses: PASS 280, FAIL 1,531, ERROR 165, UNDEFINED 229, SKIP 95, TIMEOUT 0, INFRA 0. Every ERROR is
  "not exactly one scenario ran", an error of the output, not of the infrastructure (section 10 shows one such cause).
- SKIP 95: rows whose output gave no usable block, or collided (a clash with a method of the file, or the same
  method key twice in the output); the manifest row carries `skip` and the runner scores it 0.
- Tamper: 7 rows flagged `dynamic_call`, all with er = 0, so no row needed a manual verdict; every
  `er_tamper_verdicts.json` is `{}`.
- Identity of all 12 runs: the same `harness_code`, `image_id` and schema as the controls run, `ref_run` = controls.

## 8. Threat model and the manual check

Model code is **not** adversarial, but it runs in the same PHP process as the harness. The tamper scan
(`er.manifest.tamper_flags`) therefore only flags; it never refuses an output, and it is a heuristic under that
declared model, not a barrier.

- The scan reads only the lines an overlay ADDS to its base. `dynamic_call` catches calls through `($f)(…)` and names
  built with `implode`, `sprintf`, `str_replace`, `strrev`, `base64_decode` or `chr(` and called at once; a built
  string that is not called is not flagged.
- Rule declared before any output, enforced by `er.tamper`: every flagged output (not a control) with `er = 1`,
  `trace_action = 1` or `trace_then = 1` is checked by hand before results are opened (`python -m er.tamper
  review|apply`). `cheat` sets `er` and the evaluable trace parts to 0 (the earlier values are kept in `pre_tamper`);
  a verdict is bound to the output's `overlay_sha256`, so a regenerated output is checked again.
- `er.report` refuses to run while a flagged output lacks a verdict, or when a row does not carry the verdict and the
  sha256 of the current `er_tamper_verdicts.json`.
- **A reader of `er_results.jsonl`** must use only runs whose `er_complete.json` says `complete: true` (not
  `running`) with the same `manifest_sha256`, and must refuse a row that needs a verdict and has none.

## 9. Known limits

- The trace carries keys, not values: an output that sends the right keys with wrong values still gets
  `trace_action = 1` (errs towards a false positive).
- Token numbering (`{tokK}`, `{n}`) drops entity identity: swapping two carts gives the same trace. On the measured
  records an action trace holds at most one token.
- An output that sets up data through the API instead of a factory may count as a trace mismatch (errs towards a
  false negative). The hand check of §2.3b (`tools/ER_TRACE_FN_SAMPLE.md`) measures ER-trace's false negatives.
- An output whose run shows a database-like error twice on a healthy lane and host is recorded as ERROR (score 0);
  a transient fault repeated exactly twice would be scored the same way.
- A transient TIMEOUT in the gold M1 run marks the record `nondeterministic` for ER-trace (conservative: the record
  leaves ER-trace, it is not scored wrongly).
- The equivalence of soft and strict mode (148/148) is shown on the controls. In an output, steps after the first
  failure still run, so a loop or `exit` there may turn FAIL into TIMEOUT or ERROR: ER stays 0, `trace_action` may
  change.
- `dynamic_call` still misses: a call through another function called at once (`strtolower(...)()`), an array
  callable (`[$o, 'm']()`), a built name passed to `array_map`, and a name built over several lines.
- Do not change code under `er/` while a run is unfinished; a change after the controls run needs a new controls run.
- After `er.tamper review`, always run `er.tamper apply` again before `er.report`.

The tests cover, among other things, the journal and the crash windows, Ctrl-C handling, the pinning of tamper
verdicts, the extended `dynamic_call` patterns and the scan of added lines; the behaviour each test pins is written in
its docstring.

## 10. Behat 4 reads PHP 8 attributes only

- Behat v4.0.0-alpha1 in the image reads step definitions from the attributes `#[Given]`, `#[When]` and `#[Then]`
  only (`DefinitionAttributeReader`); a docblock is read for the description, never for a pattern.
- A step declared only by a docblock (`/** @Then … */`) is UNDEFINED, silently. When a method carries both forms, the
  attribute alone counts. Sylius at `c096ca3` uses attributes throughout (`src/Sylius/Behat/Context`: 5,898 uses; no
  context registers a step by docblock).
- ER does not normalise outputs: no docblock step is rewritten to an attribute (decided before any output).
- A malformed step attribute (`#[Given()]`, `#[Given]`, `#[Given(text: '…')]`, a lower-case class name) makes Behat
  exit 255 with a PHP fatal error while it loads the context. The runner records that as the output's ERROR ("not
  exactly one scenario ran"), not as an infrastructure failure: no retry, the lane stays clean, and the record's gold
  reference is untouched.

## 11. The A5 package

A5 holds, per run, `er_results.jsonl`, `er_preflight.json`, `er_complete.json`, `er_tamper_verdicts.json` (output
runs), the run's own `SHA256SUMS.<run>` and the overlay `manifest.jsonl` and `MANIFEST_SHA256`, laid out as
`out/phase3/...` so that `A5_er.tar.gz` unpacks at the root of the checkout. Runs: the controls run, the 12 test runs,
the 27 dev runs of the dev round and the earlier `dev_pilot_v0` run (part of the §2.3b population), plus the traces
the official §2.3b draw reads, the gold references of its records, and the draw with its verdicts in
`out/phase3/er_trace_fn_sample/`. Machine paths in `er_results.jsonl` and `er_preflight.json` (`results_path`,
`head_root`, `manifest_root`) were replaced by a fixed prefix rule (`harness/` stands for the harness checkout). The
package's own files are in `packages/A5_er/`: `README.md`, `SANITIZE_MAP.tsv` (every changed file with its original
and public sha256) and `SHA256SUMS` (check from the checkout root: `shasum -a 256 -c packages/A5_er/SHA256SUMS`).
