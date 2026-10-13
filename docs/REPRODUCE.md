# Reproducing the results

This file gives the commands for the three levels described in the README. Every command runs from the root of a git
checkout of this repository. In zsh, the shell of macOS, the blocks can be pasted as they are; if you add comments,
run `setopt interactivecomments` first.

Each release asset `<asset>.tar.gz` (`A1_dataset`, `A2_prompts`, `A3_generations`, `A4_scores_sap`, `A5_er`) unpacks at
the root of the checkout:

- the data land at the paths the code uses: `out/…` for A1, A2, A4 and A5, and `A3/…` for A3. For example, A1 puts the
  dataset in `out/phase2/` and the Phase 1 pairs in `out/hist/`, and A2 puts the prompts in `out/phase3/prompts/<set>/`
  with the symbol times in `out/phase3/symtimes.jsonl` and `out/phase3/PARENTS.json`;
- the files of the asset itself land in `packages/<asset>/`: `SHA256SUMS` in every asset, `SANITIZE_MAP.tsv` in A1, A3,
  A4 and A5 (no file of A2 was changed), and `README.md` in A3 and A5.

Unpack every asset and check it against its hash list, whose paths are relative to the checkout root:

```bash
for a in A1_dataset A2_prompts A3_generations A4_scores_sap A5_er; do
  tar -xzf "$a.tar.gz" && shasum -a 256 -c "packages/$a/SHA256SUMS"
done
```

The model outputs used for the reported results are in `A3/gen-main-0482a76-20261008/results`, which holds one
`gen-main-<job>/` directory per generation job, with `manifest.main.jsonl` and `coverage.main.json`. The commands below
refer to it as `$GEN`:

```bash
GEN=A3/gen-main-0482a76-20261008/results
```

## 0. Environment and tests

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt -r requirements-phase3.txt
python -m pytest -q tests
```

This local environment runs the commands of section 1 and the tests of `tests/`: 1,318 pass, and the 4 that need
Docker are skipped. The full suite, which adds `tests_gpu/`, runs in the CPU test image described below.

Three tests (`tests/test_sap.py::test_the_history_git_reads_is_the_committed_one`,
`tests/test_sap_cli.py::test_a_fresh_clone_catches_a_forged_remote_tracking_ref` and
`tests/test_sap_cli.py::test_step_8_verifies_the_fresh_clone_run_against_the_registry`) create a bare origin with
`git init --bare` and push to its `main` branch. They pass with git 2.50.1. With git 2.39.5 they fail unless
`init.defaultBranch` is set to `main`, for example with `git config --global init.defaultBranch main`; with that
setting they pass. Versions between 2.39.5 and 2.50.1 were not tested. The CPU test image in
`ops/gate_images/cpu-test/Dockerfile` sets it.

A container image for the CPU tests and for the commands of sections 2 and 3 is defined in
`ops/gate_images/cpu-test/Dockerfile`: `python:3.12.12-slim-bookworm` pinned by digest, every Python package pinned,
git with `init.defaultBranch main`. CPython 3.12.12 exactly is needed: the scoring runs through
`tools/score_launcher.py`, which accepts only the interpreter version its limits were measured on. Two steps need the
network: fetching the tokenizer files of the three pinned models, and building the image. Run them once from the
checkout root. The first two commands make a small virtual environment, `.venv-gpu`, because
`ops/populate_hf_cache.sh` runs `.venv-gpu/bin/python`; the script then puts the tokenizer files of the three pinned
models into `hf-cache/`, and `docker build` fetches the base image and the Python packages:

```bash
python3 -m venv .venv-gpu
.venv-gpu/bin/pip install huggingface-hub==1.33.0
bash ops/populate_hf_cache.sh
docker build -t gtestgen-cpu-gate ops/gate_images/cpu-test
```

Every later command runs without network. The first command below runs the test suite; the second runs the
bootstrap's own test step, as on a GPU machine, in the image `gtestgen-cpu-gate`, the default of
`ops/pod_parity_gate.sh`:

```bash
docker run --rm --init --network none -v "$PWD":/repo:ro gtestgen-cpu-gate bash -c '
  cp -R /repo /work && useradd -m tester && chown -R tester /work && cd /work &&
  runuser -u tester -- env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/home/tester LANG=C.UTF-8 HF_HOME=/work/hf-cache \
    HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -rs -p no:cacheprovider --basetemp /tmp/bt'
bash ops/pod_parity_gate.sh both
```

Expected result: on the published tree, the full suite collects 2528 tests: 2524 pass and 4 that need Docker and the
execution harness image are skipped; 21 tests that read internal run sheets or planning documents, which are not
published, are not collected. These counts come from a run in the CPU test image, as an unprivileged user without
network access. The four skipped tests (`tests/test_er_runner.py`, `test_real_run_*`) make real Docker
runs on lane 3 and are skipped when Docker, the trace image or `repos/Sylius` is missing; every other test uses a fake
Docker and runs without it.

`--init` is required: process 1 in the container must reap orphaned processes, or three tests of
`tests_gpu/test_jobqueue.py` see killed grandchildren as still alive. The `docker run` command of the test suite
gives the counts above.

The commands of sections 2 and 3 run in the same image, with the checkout mounted read-write and the files written as
your own user. Define this shell function once (bash and zsh):

```bash
in_image() { docker run --rm --init --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/repo \
  -e HF_HOME=/repo/hf-cache -e HF_HUB_OFFLINE=1 -v "$PWD":/repo -w /repo gtestgen-cpu-gate "$@"; }
```

The prompt builder and the scorer read the commit of the checkout with git. On some Docker set-ups (Docker Desktop on
macOS, for example) the mount point `/repo` appears owned by root inside the container, and git then refuses the
repository ("detected dubious ownership"); the three `GIT_CONFIG_*` variables mark `/repo`, and only it, as safe.

## 1. Dataset (A1)

After unpacking A1, check it against the package list and against the sealed list:

```bash
shasum -a 256 -c packages/A1_dataset/SHA256SUMS
shasum -a 256 -c out/phase2/PHASE2_SHA256SUMS
```

- All 14 files of the package list match.
- Against the sealed list, 9 files match. `split.json` and `DATASET.md` differ: three free-text notes in `split.json`
  were reworded, with no value changed, and the Phase 2 report `DATASET.md` was replaced by an English dataset card.
  `data/SANITIZE_MAP.tsv` gives their original and published SHA-256 hashes and the rule applied.
- The four dataset inputs of the analysis lock (`records/test.jsonl`, `records/aux_test.jsonl`,
  `sensitivity_subsets.json` and `PHASE2_SHA256SUMS`) are byte-identical to the locked versions.
- `out/hist/pairs.jsonl` is the output of the Phase 1 history mining (10,440 pairs), which the scorer reads through
  `--pairs`. It is byte-identical to the sealed Phase 1 file, whose SHA-256 is recorded in `DATASET.md`:

  ```bash
  echo "fd59a6511c2bcd14ce9bb29320cdaf7f3eef6fce97c555dd63734e77e5d0d59e  out/hist/pairs.jsonl" | shasum -a 256 -c
  ```

### Fetching the source projects at the pinned commits

```bash
python scripts/fetch_full.py
python scripts/fetch_full.py --verify-only
zsh scripts/fetch_history.sh
```

- `fetch_full.py` writes full bare clones to `repos-full/`, at the pinned commits read from `out/phase2/records`.
  Prompt building and scoring need these clones. With `--verify-only` it only checks them, and exits with 1 if a
  pinned commit is missing or a clone is partial.
- `fetch_history.sh` writes partial clones (`blob:none`, sparse) to `repos-hist/` for Phase 1, each checked out at its
  pinned commit from the same source as `fetch_full.py`. It needs zsh; set `PHASE2=<directory>` if the records are
  elsewhere. The script stops at the first failing step, and checks at the end that every `HEAD` is the pinned
  commit.

### Rebuilding the dataset

The rebuild needs `repos-hist/` and the records of A1, which hold the pinned commits. The commands write to `out/hist/`
and `out/phase2/` by default, so unpack the A1 copy used for the comparison elsewhere, for example with
`mkdir sealed && tar -xzf A1_dataset.tar.gz -C sealed`, and check it from the root of that copy, since the paths in
the hash list are repository paths: `(cd sealed && shasum -a 256 -c packages/A1_dataset/SHA256SUMS)`.

Phase 1:

```bash
python -m v2tools.yield_report --config config/sources.json --root repos-hist --out out/nohist --no-history
python -m v2tools.yield_report --config config/sources.json --root repos-hist --out out/hist --cutoff 2024-01-01T00:00:00Z
python -m v2tools.split --pairs out/hist/pairs.jsonl --cutoff 2024-01-01T00:00:00Z --out out/split.json
python scripts/check_split.py --pairs out/hist/pairs.jsonl --split out/split.json
python scripts/wrong_suite_control.py --config config/sources.json --root repos-hist --repo Sylius/Sylius \
  --out out/wrong_suite_control.json
shasum -a 256 out/hist/pairs.jsonl out/hist/yield.json out/split.json out/split_members.jsonl \
  out/nohist/yield.json out/nohist/pairs.jsonl out/wrong_suite_control.json > out/PHASE1_SHA256SUMS
```

Phase 2, with the default paths `repos-hist/`, `out/hist/pairs.jsonl` and `out/phase2/`:

```bash
python -m v2tools.labels
python -m v2tools.final_split
python -m v2tools.records
python -m v2tools.sensitivity
python scripts/check_dataset.py
python scripts/phase2_report.py
```

`phase2_report.py` runs only on a clean tree and writes the report in its original format; the published
`DATASET.md` is an English description and is not produced by this command. Phase 1 takes about 4 minutes and
Phase 2 about 5 minutes on a 12-core Apple M-series machine.

Expected result:

- byte-identical to the sealed files: `out/hist/pairs.jsonl`, `out/nohist/pairs.jsonl`, `out/split.json`,
  `out/split_members.jsonl`, `out/wrong_suite_control.json`, and in `out/phase2/` the files `labels.jsonl`,
  `split_members.jsonl`, `forbidden.json`, `records/{train,dev,test,aux_test}.jsonl` and
  `sensitivity_subsets.json`;
- `split.json` is byte-identical to the published copy in A1, because `config/phase2.json` carries the same three
  cleaned strings;
- three files differ for one known reason (see "Step reader of Phase 1 and 2" in `docs/KNOWN_ISSUES.md`):
  - `out/phase2/catalog.jsonl`, only in the `annotations` field of 3 rows of gpconnect (`ManualSteps.cs`, lines 11,
    16 and 21): the public reader reads all three attributes `[Given(".*"), When(".*"), Then(".*")]`, the reader used
    for the sealed dataset read only the first;
  - `out/hist/yield.json` and `out/nohist/yield.json`, in 4 counts: gpconnect step definitions 619 to 625,
    catch-all definitions 3 to 9, and the two totals;
  - no record, label or split changes;
- `DATASET.md` differs in the line of the tool commit and in the line of the inputs, whose hash follows `yield.json`;
- `check_split.py` and `check_dataset.py` pass.

## 2. Prompts (A2)

A2 holds 30 prompt sets: `test_v0` (the test set and the auxiliary test set, for the three models), `v0`,
`v0-codellama` and `v0-mistral` (the training and development sets, one set per model), and 26 sets of the
development grid for DeepSeek-Coder-6.7B-Instruct. Prompts are built ahead of generation and stored with their
SHA-256 hashes; training and generation read only these files.

Rebuilding the prompts needs `repos-full/`, A1 under `out/phase2/`, `config/special_tokens.json` and the tokenizers
of the three models, taken from the Hugging Face cache at the revisions pinned in `v2gpu/models.py`, with
`HF_HUB_OFFLINE=1`.

A2 stays unpacked at the root, where sections 3 and 4 read it. The rebuild writes to `out/phase3/rebuilt/`, which
mirrors the layout of `out/phase3/`. The first command writes `out/phase3/rebuilt/symtimes.jsonl` and `PARENTS.json`
next to it. The second builds the test set and the auxiliary test set for the three models. The third builds the
training and development sets of DeepSeek-Coder; run it again with `--models codellama` and `--out
out/phase3/rebuilt/prompts/v0-codellama`, and with `--models mistral` and `--out
out/phase3/rebuilt/prompts/v0-mistral`. The last builds one of the 26 sets of the development grid,
`dev_grid_k{4,8,12}_n{1,2,3}_r{033,050,067}` without `k8_n3_r050`, which is the same as `v0`:

```bash
in_image python3 -m v2tools.symcache --out out/phase3/rebuilt/symtimes.jsonl
in_image python3 scripts/build_prompts.py --variant v0 --k 8 --n-examples 3 --ratio 0.5 --mode full \
  --models deepseek,codellama,mistral --parts test,aux_test \
  --symtimes out/phase3/rebuilt/symtimes.jsonl --out out/phase3/rebuilt/prompts/test_v0
in_image python3 scripts/build_prompts.py --variant v0 --k 8 --n-examples 3 --ratio 0.5 --mode full \
  --models deepseek --parts train,dev --symtimes out/phase3/rebuilt/symtimes.jsonl --out out/phase3/rebuilt/prompts/v0
in_image python3 scripts/build_prompts.py --variant dev_grid_k4_n1_r033 --k 4 --n-examples 1 \
  --ratio 0.3333333333333333 --mode full --models deepseek --parts dev \
  --symtimes out/phase3/rebuilt/symtimes.jsonl --out out/phase3/rebuilt/prompts/dev_grid_k4_n1_r033
```

Compare with A2, and repeat the `diff -rq` for every set:

```bash
cmp out/phase3/rebuilt/symtimes.jsonl out/phase3/symtimes.jsonl
cmp out/phase3/rebuilt/PARENTS.json out/phase3/PARENTS.json
diff -rq out/phase3/rebuilt/prompts/test_v0 out/phase3/prompts/test_v0
```

Expected result: the two `cmp` print nothing. For each set, `diff -rq` lists only `PROMPTS_META.json` (the field
`v2tools_commit`) and `PROMPTS_SHA256SUMS` (the line of `PROMPTS_META.json`).

The parameters of every set are in its `PROMPTS_META.json`. A full rebuild of the 30 sets gave 264 files: the 204
prompt, excluded, infra and pool files are byte-identical to A2, and the other 60 are the `PROMPTS_META.json` and
`PROMPTS_SHA256SUMS` of each set. The 30 sets take about 39 minutes with three commands in parallel.

The leakage check reads no `v2tools` module:

```bash
python3 scripts/scan_prompts.py --prompts out/phase3/prompts/test_v0 --phase2 out/phase2 --report dropped.json
```

It checks that blocks ③ and ④ of each prompt hold no annotation line, step pattern, method name or the last 200
characters of the ground truth; that each line is one user turn; that no test or auxiliary test line carries the
answer; and that `PROMPTS_SHA256SUMS` covers every file. `--report` writes the number of dropped items by part, kind
and rule. On the 30 sets of A2, with the public A1, every run exits with 0 and reports `{"findings": 0}`.

## 3. Level 1: scoring and analysis from the released outputs

The scoring of the study ran on CPython 3.12.12, and `tools/score_launcher.py`, through which every scoring command of
this section runs, accepts only that version, because the recursion limit it sets was measured on it. One way to get
it without changing the system Python is uv:

```bash
uv python install --no-bin 3.12.12
uv venv --python 3.12.12 .venv-score
uv pip install --python .venv-score/bin/python -r requirements.txt -r requirements-phase3.txt
. .venv-score/bin/activate
```

The commands below then run unchanged. The CPU test image of section 0 also runs CPython 3.12.12, and the re-scoring
commands can run in it instead, as shown below.

### Re-scoring the generation outputs

The published prompt sets (A2) keep their original `PROMPTS_META.json`, whose `v2tools_commit` names a commit of the
private research repository. `scripts/score_run.py` checks that the files that built the prompts
(`v2tools/state.py`, `compose.py`, `records.py`, `outline.py`, `inherit.py`, `symtime.py` and `symcache.py`) are the
same at that commit as in the tree that scores. For that reason it cannot score A3 against the A2 prompt sets as
published. To re-score:

1. Work in a git checkout of this repository. If you start from the release archive, run `git init` and commit the
   tree.
2. Rebuild the prompt sets from A1 with `scripts/build_prompts.py` in that checkout. The prompt files come out
   byte-identical to A2; only `v2tools_commit` in `PROMPTS_META.json` differs, and it now names your commit.
3. Score the controls with this checkout first (`score_run.py --controls`, for the test set and the auxiliary test
   set). `score_run.py --outputs` refuses a controls directory scored by other code, and the public code differs
   from the original code in its comments, so the control scores published in A4 cannot be used here. The
   re-scored controls equal A4 byte for byte in every file except `SCORES_META.json` and its line in `SHA256SUMS`.
4. Run `scripts/score_run.py` through `tools/score_launcher.py`. The launcher raises Python's recursion limit, because
   one output needs a parse tree deeper than the default limit, and it runs only on CPython 3.12.12, on which that
   limit was measured; on another version it stops with a message ("STOP: …"). Pass `--prompts-meta` of the rebuilt
   set, `--controls-dir` of the controls you scored in step 3, and `--a3-repo` pointing at your checkout.

The `scores.jsonl` files of the test set and of the auxiliary test set then equal the published ones byte for byte.
Each new `SCORES_META.json` differs from the published one in 6 of its 16 fields: `tool_commit`, `code_sha256`,
`prompts_v2tools_commit`, `scored_utc`, `a3_sha256` (in all seven of its entries, whose published files differ from the
study's files in comments and docstrings) and `inputs`, which names each input by the path given on the command line,
so its keys differ wherever the rerun reads a file from another directory than the published run (for example the
rebuilt prompt set and the new controls), and whose values differ only for `PROMPTS_META.json`. `SHA256SUMS` differs
from the published one only in the line of `SCORES_META.json`.

The source projects must be available as full clones in `repos-full/`, written by `python scripts/fetch_full.py`, and
the test_v0 prompt set must first be rebuilt in this checkout (section 2). The launcher needs the absolute path of the
script, hence `"$PWD/…"`. `OUTS` is an array, so that bash and zsh both pass the 14 output files as separate
arguments. `P` is the rebuilt test_v0 prompt set of section 2; `CT` and `NEW` are new directories for the re-scored
controls and outputs, which must not exist yet. The first command scores the controls with this checkout (do the same
for the auxiliary test set and the development set); the second scores the outputs of the test set:

```bash
OUTS=()
for j in deepseek-base deepseek-s42 deepseek-s11 deepseek-s23 codellama-base codellama-s42 codellama-s11 \
         codellama-s23 mistral-base mistral-s42 mistral-s11 mistral-s23 deepseek-s7 deepseek-s99; do
  OUTS+=("$GEN/gen-main-$j/outputs.jsonl")
done
P=out/phase3/rebuilt/prompts/test_v0
CT=out/phase3/rescored/controls_test
NEW=out/phase3/rescored/test
python3 tools/score_launcher.py 20000 "$PWD/scripts/score_run.py" --partition test --controls --out "$CT"
python3 tools/score_launcher.py 20000 "$PWD/scripts/score_run.py" --partition test --outputs "${OUTS[@]}" --require-clean \
  --controls-dir "$CT" --prompts-meta $P/PROMPTS_META.json --a3-repo . \
  --er-results out/phase3/er/test_*_0482a76/er_results.jsonl --er-manifest out/phase3/overlays/test_*_0482a76/manifest.jsonl \
  --prompts-f2 deepseek=$P/deepseek.all.test.F2.jsonl codellama=$P/codellama.all.test.F2.jsonl mistral=$P/mistral.all.test.F2.jsonl \
  --coverage $GEN/coverage.main.json --manifest $GEN/manifest.main.jsonl --out "$NEW"
```

To score in the CPU test image instead, put `in_image` (section 0) in front of both commands and give the launcher the
path of the script inside the container, `/repo/scripts/score_run.py`: `"$PWD/…"` is a path on the host, which the
container does not have. Define `OUTS`, `P`, `CT` and `NEW` as above first; `CT` and `NEW` must not exist yet:

```bash
in_image python3 tools/score_launcher.py 20000 /repo/scripts/score_run.py --partition test --controls --out "$CT"
in_image python3 tools/score_launcher.py 20000 /repo/scripts/score_run.py --partition test --outputs "${OUTS[@]}" \
  --require-clean --controls-dir "$CT" --prompts-meta $P/PROMPTS_META.json --a3-repo . \
  --er-results out/phase3/er/test_*_0482a76/er_results.jsonl --er-manifest out/phase3/overlays/test_*_0482a76/manifest.jsonl \
  --prompts-f2 deepseek=$P/deepseek.all.test.F2.jsonl codellama=$P/codellama.all.test.F2.jsonl mistral=$P/mistral.all.test.F2.jsonl \
  --coverage $GEN/coverage.main.json --manifest $GEN/manifest.main.jsonl --out "$NEW"
```

This gives the same files as the commands above, except the time in `scored_utc`, and so the line of
`SCORES_META.json` in `SHA256SUMS`. On a 12-core Apple M-series machine, the controls take about 8 minutes in the image
and the outputs of the test set about 42 minutes.

For the auxiliary test set, score its controls with `--partition aux_test` into a new directory, then run the output
command with `--partition aux_test`, `--controls-dir` set to those controls and another new `--out`, without
`--er-results`, `--er-manifest` and `--prompts-f2`.

The second scorer recomputes Binding Validity and the static measures without reusing the code of the main scorer.
The repository holds two versions of it.

- `scripts/score_independent.py` is the version pinned before any test output existed. On the test set it exits with
  code 1 and reports 43 differing rows of the 16,692 test outputs (38 records, all written in Java: 28 rows of
  fineract and 15 of ph-ee), because it misses member accesses of the form `this.x` when a parameter or a local
  variable has the same name as the member. All 43 rows differ in `unresolved`; 7 of them also differ in
  `bv_resolved`, and 2 also in `open_resolution`. Binding Validity, Call Agreement and every other column agree on
  every row. On the auxiliary test set and on the controls it reports no difference.
- `tools/c11fix/score_independent.py` fixes that one point. It agrees with the main scorer on every output: 16,692 test
  outputs, 1,020 auxiliary test outputs, 5,913 control outputs of the test set and 628 control outputs of the
  auxiliary test set, and reports no difference (`{}`).

The main scorer is the same in both cases, so no reported number depends on the second scorer. The fixed version
imports `v2tools`, hence `PYTHONPATH`. Its command for the test set:

```bash
PYTHONPATH="$PWD" python3 tools/score_launcher.py 20000 "$PWD/tools/c11fix/score_independent.py" --partition test \
  --outputs "${OUTS[@]}" --scores out/phase3/scores/test/scores.jsonl --symtimes out/phase3/symtimes.jsonl \
  --infra out/phase3/prompts/test_v0/infra.json
```

For the auxiliary test set, use `--partition aux_test` and `--scores out/phase3/scores/aux_test/scores.jsonl`. For the
controls, leave out `--partition` and use `--outputs out/phase3/scores/controls_test/control_outputs.jsonl` and
`--scores out/phase3/scores/controls_test/controls.jsonl` (`controls_aux_test` for the auxiliary test set). The pinned
version runs with the same commands, with `"$PWD/scripts/score_independent.py"` and without `PYTHONPATH`.

With the fixed version, the four reports are identical byte for byte to the reports of the original run: the test set
exits with 0 and reports `"differences": {}` on 16,692 rows, and the auxiliary test set (1,020), the test controls
(5,913) and the auxiliary controls (628) report `{}` as well.

### Rerunning the analysis

`tools/rerun_sap.py` reruns `scripts/sap_run.py` for both parts, and `scripts/sap_dev_select.py`, from this
repository and the data packages.

- `sap_run` checks the code and the 22 inputs against `prereg/SAP_LOCK.public.json`. That file differs from the
  registered `prereg/SAP_LOCK.json` only in the values listed in `prereg/LOCK_PUBLIC_MAP.tsv`;
  `prereg/LOCK_EQUIVALENCE.md` explains why.
- `sap_run` also reads the lock and the score manifests as committed in git. Because this repository has a new
  history, the script builds a temporary replay repository that holds those files, with commits dated as the
  original ones. Its date and ancestry checks are therefore a replay; the independent time anchor is the OSF
  registration.

```bash
python3 tools/rerun_sap.py --data . --a3 A3 --work ../rerun_work
```

`--data` is the directory that holds the unpacked assets at the paths of the repository, here the checkout root; the
script reads only the data under `out/` there, and reads `prereg/`, `config/` and `evidence/` from its own tree.
`--a3` is the `A3` directory itself: the script adds `gen-main-0482a76-20261008/results`, because the published
results name their inputs by that path. `--work` is a new or empty directory, preferably outside the checkout: the
script writes everything there, including the temporary replay repository.

The script exits with 0 when every check passes and writes its report to `<work>/RERUN_REPORT.json`; it takes about a
minute. It makes three checks:

- every numeric field of the rerun's `SAP_RESULTS.json` equals the published `SAP_RESULTS.json`;
- after a fixed substitution, `SAP_RESULTS.json` and `SAP_REPORT.md` are identical byte for byte to the published
  files. The substitution replaces the three commit identifiers of the replay repository with the original ones
  (`63fd681`, `8380158` and `ce20632`) and the tool commit with its published value;
- `scripts/sap_dev_select.py` reproduces `prereg/DEV_SELECTION.json` byte for byte.

The reports are written in Vietnamese and name items of the analysis plan and sections of the design specification
by short labels; `docs/LABELS.md` explains them.

## 4. Level 2: regenerating outputs on a GPU

Generation used vLLM 0.27.1 with NF4 quantisation, greedy decoding, one sequence per request, eager mode, a GPU memory
fraction of 0.95 and batch-invariant kernels, on NVIDIA RTX A4000 GPUs with 16 GB of memory.

### Environment

Generation ran on NVIDIA RTX A4000 GPUs (16 GB) and training on NVIDIA RTX A6000 GPUs (48 GB), with drivers that support
CUDA 13.0, Python 3.12, the container image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` and a 100 GB container
disk; with a 50 GB disk the check step of the bootstrap fails (see `docs/KNOWN_ISSUES.md`). On a generation machine,
`KIND=gen` builds a virtual environment from `requirements-gen.lock`, downloads the three models at their pinned
revisions and runs the generation test files, whose run must end with "N passed". On a training machine, `KIND=gpu`
builds a virtual environment from `requirements-gpu.lock`, downloads the named model or models and runs `tests_gpu`
and the nine `tests/` modules the training machines ran. The bootstrap puts the models in the Hugging Face cache
`$V2GPU_LOCAL/hf`, by default `/opt/v2/hf`.

```bash
git clone https://github.com/TuanChienVu/GTestGen repo && cd repo
KIND=gen bash ops/pod_bootstrap.sh
KIND=gpu bash ops/pod_bootstrap.sh deepseek
```

The models are downloaded from Hugging Face at the pinned revisions (`THIRD_PARTY_NOTICES.md`); a model whose terms
must be accepted on Hugging Face needs an access token in the environment (`HF_TOKEN`). The test step of the bootstrap
passes only if every collected test passes and none is skipped.

### Generating the base-model conditions

A generation job accepts a leakage-check receipt only if the receipt was made by the code of its own checkout: the
receipt records the SHA-256 of the modules that decide the check (`code_sha256`). The receipts in A3
(`A3/gen-main-0482a76-20261008/stage/receipts/`) record the study's code, which differs from this repository in
comments, so a job refuses them ("leak-audit receipt was made by other code than this checkout"). Rerun the check
first. It needs no GPU, but it needs the tokenizers of the three models: on a generation machine run the block as it
is after the bootstrap; on another machine put `in_image` in front of `python` (under a minute per model).

```bash
P=out/phase3/prompts/test_v0
GEN=A3/gen-main-0482a76-20261008/results
mkdir -p ship
for m in deepseek codellama mistral; do
  python -m v2gpu.leakcheck --model $m --subset all \
    --prompts $P/$m.all.test.F0.jsonl $P/$m.all.test.F1.jsonl $P/$m.all.test.F2.jsonl \
              $P/$m.all.aux_test.F0.jsonl $P/$m.all.aux_test.F1.jsonl $P/$m.all.aux_test.F2.jsonl \
    --prompts-sums $P/PROMPTS_SHA256SUMS \
    --records out/phase2/records/test.jsonl out/phase2/records/aux_test.jsonl \
    --train-records out/phase2/records/train.jsonl --catalog out/phase2/catalog.jsonl \
    --forbidden out/phase2/forbidden.json --phase2-sums out/phase2/PHASE2_SHA256SUMS \
    --out ship/leakcheck-receipt.$m.json
done
```

Each receipt must show `"status": "PASS"` and `"n_problems": 0`, with 1,464 prompt lines checked.

Each generation job runs one model with one source of weights and every condition of that source: the base weights
(`--weights base`) give the Zero-shot and RAG-only conditions. The command of the DeepSeek-Coder base job, as in its
receipt (`$GEN/gen-main-deepseek-base/receipt-generate.json`) but with the new leakage-check receipt and the Hugging
Face cache that the bootstrap filled:

```bash
R=ship/leakcheck-receipt.deepseek.json
env HF_HUB_OFFLINE=1 HF_HOME=/opt/v2/hf python -m v2gpu.generate --model deepseek --weights base --variant main \
  --partitions test aux_test \
  --prompts $P/deepseek.all.test.F0.jsonl $P/deepseek.all.test.F1.jsonl $P/deepseek.all.test.F2.jsonl \
            $P/deepseek.all.aux_test.F0.jsonl $P/deepseek.all.aux_test.F1.jsonl $P/deepseek.all.aux_test.F2.jsonl \
  --prompts-sums $P/PROMPTS_SHA256SUMS \
  --leakcheck-receipt $R --leakcheck-receipt-sha256 "$(shasum -a 256 $R | cut -d' ' -f1)" \
  --gpu-name "NVIDIA RTX A4000" --pin-set main --gpu-memory-utilization 0.95 --enforce-eager \
  --decision-sha256 "$(shasum -a 256 $GEN/decision.main.json | cut -d' ' -f1)" --workdir runs/gen-main-deepseek-base
```

The CodeLlama and Mistral base jobs of the main run read the test set only: `--partitions test` and the three
`$m.all.test.F*.jsonl` files. Before using a GPU, check a job with `--preflight-only` in place of `--workdir`: it needs
no GPU and must exit with 0 and print `"problems": []`. Its `ids_sha256` equals the `--ids-sha256` value in the job's
`receipt-generate.json` when the prompts are tokenized as on the generation machines. In the CPU test image, write
`in_image python -m v2gpu.generate …` without `env HF_HUB_OFFLINE=1 HF_HOME=/opt/v2/hf`: `in_image` sets both
variables, with the cache in `hf-cache/`.

Compare the regenerated outputs with A3 field by field, except `leakcheck_receipt_sha256`, which names the new
leakage-check receipt:

```bash
python - runs/gen-main-deepseek-base/outputs.jsonl "$GEN/gen-main-deepseek-base/outputs.jsonl" <<'PY'
import json, sys
def rows(path):
    out, n = {}, 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line); n += 1
        r.pop("leakcheck_receipt_sha256", None)
        out[(r["record_id"], r["condition"], r["fmt"], r["seed"], r["variant"])] = r
    if len(out) != n:
        sys.exit(f"{path}: {n - len(out)} duplicate row key(s)")
    return out
a, b = rows(sys.argv[1]), rows(sys.argv[2])
diff = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
print(f"{len(a)} and {len(b)} rows; {len(diff)} differ in a field other than leakcheck_receipt_sha256")
for k in diff[:10]:
    print("differs:", k)
sys.exit(1 if diff else 0)
PY
```

- With `--pin-set main`, `--ids-sha256` is optional, because the main pin set uses the same transformers and tokenizers
  versions as training. An adapter job also needs `--adapter`, `--adapter-sha256` and `--adapter-config-sha256`.
  `python -m v2gpu.genqueue` can build a `QUEUE.json` for a whole machine, run with `python -m v2gpu.jobqueue QUEUE.json`.
- Expected result: on an RTX A4000 with the generation lock and these settings, the comparison reports 0 rows that
  differ. This expectation rests on the determinism gate of the study: in the main run, 14 recheck jobs on the same
  machine and 6 on another machine with the same GPU type were byte-identical, and the re-run for the operating figures
  regenerated 468 of 468 outputs byte for byte. Generation from this repository has not been run on a GPU; the leakage
  check and the preflight of the three base jobs were run on a CPU. Other GPU types were not tested.

### Training an adapter

The training inputs come from A1 and A2, unpacked at the checkout root. The one input that is not in A1 is
`catalog.train.jsonl`: the sealed catalog (`out/phase2/catalog.jsonl`) without the `code` of every ground-truth
function of the test and auxiliary test sets. `strip-catalog` writes it, together with
`catalog.train.jsonl.receipt.json`, which the training job reads; on the published A1 it gives the same SHA-256 as the
file used by the training runs, `7e8a37a2e1a39a5d76bce9771cce54c7a6cbb31826994351e17179998fd3f38a`. The directory
`ship/` must exist first. `P` is the training prompt set of the model: `out/phase3/prompts/v0` for DeepSeek-Coder,
`v0-codellama` for CodeLlama and `v0-mistral` for Mistral. The command of the DeepSeek-Coder adapter with seed 42:

```bash
mkdir -p ship
python -m v2gpu.inputs strip-catalog --phase2-sums out/phase2/PHASE2_SHA256SUMS --out ship/catalog.train.jsonl
P=out/phase3/prompts/v0
python -m v2gpu.train_job --model deepseek --seed 42 --subset all \
  --records out/phase2/records/train.jsonl --dev-records out/phase2/records/dev.jsonl \
  --catalog ship/catalog.train.jsonl --forbidden out/phase2/forbidden.json \
  --train-prompts $P/deepseek.all.train.F0.jsonl $P/deepseek.all.train.F1.jsonl $P/deepseek.all.train.F2.jsonl \
  --dev-prompts $P/deepseek.all.dev.F0.jsonl $P/deepseek.all.dev.F1.jsonl $P/deepseek.all.dev.F2.jsonl \
  --prompts-sums $P/PROMPTS_SHA256SUMS --phase2-sums out/phase2/PHASE2_SHA256SUMS \
  --catalog-sha256 "$(shasum -a 256 ship/catalog.train.jsonl | cut -d' ' -f1)" --upcast-fp32 \
  --workdir runs/deepseek-s42-all
```

- The inputs of this command were checked with `--preflight-only` in the CPU test image
  (`in_image python -m v2gpu.inputs …`, `in_image python -m v2gpu.train_job …`), without network, on the published A1
  and A2: it exits with 0 and reports no problem over 9,075 examples. Training itself needs a GPU; the training jobs
  ran on RTX A6000 GPUs (`KIND=gpu`), and the selected adapter is recorded as `selected_adapter` in the receipt.
- The job mode (the default) locks every training parameter to its value declared in advance. The allowed seeds are
  42, 11 and 23 for every model, plus 7 and 99 for DeepSeek-Coder on the full training set.
- Retraining an adapter is not bit-for-bit deterministic: in a 50-step training trial on DeepSeek-Coder before the
  main runs, two bf16 runs of the same configuration had the same loss at step 0 and then differed by 0.0019 on
  average over the last 10 steps. Adapters are not published; the receipts record the SHA-256 of each adapter tensor
  that produced the published outputs.

## 5. Level 3: execution on Sylius (ER and ER-trace)

The harness runs Behat on Sylius at the pinned commit, in Docker images built from `er/docker/`. The database runs in
a separate MySQL container pinned by digest. `er/ER_CONTROLS.md` describes the design, the pins, the isolation of the
database lanes and the controls in full. The images were built and run on arm64 (Docker Desktop on Apple silicon);
other platforms were not tried. The harness needs Python 3.12, git and the Docker CLI.

### Images and database

Sylius needs a clean full checkout at the pinned commit, because the tamper scan compares against it. Then build the
images and create the database lanes:

```bash
git clone https://github.com/Sylius/Sylius.git repos/Sylius
git -C repos/Sylius checkout --detach c096ca3fc01545ee25078ba3df914bd2069c7fa6
docker compose -f er/docker/compose.yaml build --no-cache sylius sylius-trace
er/docker/init-db.sh
```

- The image build stops if Sylius is not at the pinned commit, if `composer.lock` does not have the pinned SHA-256,
  or if the source tree is not clean after installation. The trace image has no `.git`.
- An image ID depends on the machine that builds it, so a rebuilt image has another `image_id` than the one recorded in
  the results. Three independent image builds (three image IDs) gave, for the same 149 records, the same `er_status`,
  `er`, `er_trace`, `trace_action`, `trace_then`, `gold_status`, `ref_error` and `skip` on every row, and every gold
  reference was `deterministic` in each of them. A fourth build, from the code of this release, reproduced the 447 rows
  of the controls run and the 300 rows of `test_deepseek_s42_0482a76` on the same fields. The trace files themselves
  can differ between runs. Request paths carry values that change from run to run: the random token Sylius gives each
  order, the time-based UUIDs of payment requests, and database ids, which keep counting up on a lane until the lane is
  recreated and so depend on what ran on it before. The error message of a failing step can carry such ids and times as
  well. Scoring reads none of these values: it compares traces only after `norm_path` in `er/trace.py` has replaced
  token route parameters with numbered placeholders, ids with `{n}` and UUIDs with `{uuid}`, and it uses the status of
  each step, not its error message.
- `er/docker/init-db.sh` creates the databases `sylius_lane1` to `sylius_lane3` (26 migrations, 83 tables) and, per
  lane, a user limited to SELECT, INSERT, UPDATE and DELETE, with a random password written to
  `$ER_SECRETS_DIR/laneN.pw`. The data is on tmpfs, so rerun the script after every restart of the database
  container. The script refuses to run while a lane is in use.
- The database password in `er/docker/compose.yaml` is for a local, single-use database.

### Rebuilding the overlays

The overlay trees are not published; their manifests are in A5. Build them again from the dataset (A1), the generation
outputs (A3) and the symbol times of A2 (`out/phase3/symtimes.jsonl`). `repos-full/` holds full bare clones of the source
repositories at their pinned commits, written by `python scripts/fetch_full.py`.

A5 stays unpacked at the root, where section 3 reads it and where the development run below reads its id list. The
rebuilt overlays and the new runs go to `out/phase3/rebuilt/`, as in section 2, because A5 already occupies
`out/phase3/overlays/` and `out/phase3/er/`: `write_overlays.py` refuses an existing run directory, and `er.report`
without `--out` reads the published A5 results.

`common` holds the arguments shared by every run; it is an array, so that bash and zsh pass them as separate words. A
test run uses the 100 records of the execution core (`exec_core`) and the outputs of its generation work directory; a
development run uses the 49 development ids and every DeepSeek work directory of its label, in glob order:

```bash
common=(--phase2 out/phase2 --full-root repos-full --symtimes out/phase3/symtimes.jsonl --out out/phase3/rebuilt/overlays)
python scripts/write_overlays.py "${common[@]}" --run controls --controls
python -c "import json; [print(r['id']) for r in sorted(map(json.loads, open('out/phase2/records/test.jsonl')), key=lambda r: r['id']) if r['exec_core']]" > exec_core_ids.txt
python scripts/write_overlays.py "${common[@]}" --run test_deepseek_s42_0482a76 --ids exec_core_ids.txt \
  --outputs "$GEN/gen-main-deepseek-s42/outputs.jsonl"
python scripts/write_overlays.py "${common[@]}" --run dev_grid_k8_n2_r067_0482a76 --ids out/phase3/overlays/dev_sylius_api_ids.txt \
  --outputs A3/gen-dev-0482a76-20261006/results/gen-dev_grid_k8_n2_r067-deepseek-*/outputs.jsonl
```

- `--outputs` takes the lines of the run's conditions only, in the order of the file. For the CodeLlama and Mistral
  runs, keep the `full_aware` lines of `outputs.jsonl` and give that file.
- Check: `shasum -a 256 out/phase3/rebuilt/overlays/<run>/manifest.jsonl` must equal the hash in
  `out/phase3/overlays/<run>/MANIFEST_SHA256` (A5).

### Running the harness

First the controls run, which freezes the gold references; then each output run, scored against the frozen references
of the controls run. `RUN` names the output run, for example:

```bash
python -m er.batch --root out/phase3/rebuilt/overlays --out out/phase3/rebuilt/er --run controls --head-root repos/Sylius --lanes 2
python -m er.report --out out/phase3/rebuilt/er --run controls
RUN=test_deepseek_s42_0482a76
python -m er.batch --root out/phase3/rebuilt/overlays --out out/phase3/rebuilt/er --run "$RUN" --head-root repos/Sylius --ref-run controls --lanes 2
python -m er.tamper review --out out/phase3/rebuilt/er --run "$RUN"
python -m er.tamper apply --out out/phase3/rebuilt/er --run "$RUN"
python -m er.report --out out/phase3/rebuilt/er --run "$RUN"
```

- Run the controls first. A batch records `harness_code`, a hash of the harness files, and an output run needs the
  same `harness_code` as its controls run. This public tree differs from the code that produced the published runs
  in comments, documentation and default paths, so the published controls run cannot serve as `--ref-run`.
- Give the controls run and the output runs the same `--out`: `--ref-run controls` reads the frozen references from
  `<out>/controls/er_refs.jsonl`.
- Compare `out/phase3/rebuilt/er/<run>/er_results.jsonl` with `out/phase3/er/<run>/er_results.jsonl` (A5) by
  `(record_id, condition)` on the fields `er_status`, `er`, `er_trace`, `trace_action`, `trace_then`, `gold_status`,
  `ref_error` and `skip`. The fields `harness_code`, `harness_commit`, `image_id` and the paths differ by construction
  and are not result differences; the fields ending in `trace_sha256`, `log_sha256` and the measured times and memory
  (`seconds`, `behat_seconds`, `mem_peak_bytes`) can differ as well (see "Images and database").
- The official controls run is deterministic: on `er_status`, `er`, `trace_action`, `trace_then`, `er_trace`,
  `gold_status` and `ref_error`, its 447 rows equal those of an earlier full run of the same 149 records, made as two
  runs (300 rows for the 100 test records and 147 for the 49 development records), and all 149 references reproduce
  themselves (`deterministic`).

## 6. Known issues

See `docs/KNOWN_ISSUES.md`.
