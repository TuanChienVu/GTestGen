# GTestGen dataset (Phase 2)

- Cutoff: `2024-01-01T00:00:00Z` · seed `20260923` · tool commit (research repository) `8473872f457f1d9b22af17b0051c292e1a30851e`
- Inputs (Phase 1 outputs): `out/PHASE1_SHA256SUMS` sha256 `c0df08b39a9812bf2825c3960191b07a365a6e386e4f1c99f8f8a93174e86a3d` (not part of this release) · `out/hist/pairs.jsonl` sha256 `fd59a6511c2bcd14ce9bb29320cdaf7f3eef6fce97c555dd63734e77e5d0d59e` (in this package, byte-identical)
- Tools: git version 2.50.1 (Apple Git-155) · Python 3.12.12
- Verdict against the minimum of 600/150/150 unique scenarios (train/dev/test): **GO**

## Files of this package

The package unpacks at the root of a checkout of the code repository. Its data files land in `out/phase2/`
(names below are relative to it) and `out/hist/`; the package's own files, `SANITIZE_MAP.tsv` and `SHA256SUMS`, are in
`packages/A1_dataset/`. Every package of the release has its `SHA256SUMS` in `packages/<package>/`; only A3 and A5 add
a `README.md`, and A1, A3, A4 and A5 add a `SANITIZE_MAP.tsv`.

| File | Content |
|---|---|
| `records/{train,dev,test,aux_test}.jsonl` | one record per (scenario, suite) pair: the scenario text, its steps, the step functions to generate (`create`) and the fields listed below |
| `catalog.jsonl` | every step function of the six repositories at the pinned commits, with its annotations, their first-seen times and flags |
| `labels.jsonl` | per BOUND (scenario, suite) pair of Phase 1, set-aside pairs included (10,439): the step definitions each step binds to, each step's kind (create or reuse), the CREATE and REUSE functions |
| `split.json`, `split_members.jsonl` | the split report (counts, sets aside, configuration) and, for each of the 10,439 pairs, its partition (or `set_aside:<reason>`), learning-curve subsets and `exec_core` flag |
| `forbidden.json` | per repository and partition, the functions hidden from other partitions (test answers and their twins, near-duplicates and containers) |
| `sensitivity_subsets.json` | the three test subsets of the sensitivity analysis, declared before any model was scored |
| `PHASE2_SHA256SUMS` | the sealed list of the files above, as locked in the pre-registration |
| `DATASET.md` | this description |
| `out/hist/pairs.jsonl` | the Phase 1 history-mining output: the 10,440 (scenario, suite) pairs of the six repositories (10,439 BOUND, 1 PARTIAL) with their steps, the step functions they bind to, the post-cutoff targets and the times; the input of the Phase 2 steps and the `--pairs` input of the scoring scripts (`score_run.py`, `score_independent.py`); byte-identical to the sealed Phase 1 file |
| `packages/A1_dataset/SANITIZE_MAP.tsv`, `packages/A1_dataset/SHA256SUMS` | the cleaned file with its original and public sha256; the sha256 of every file of the package (`shasum -a 256 -c packages/A1_dataset/SHA256SUMS` from the checkout root) |

All files are byte-identical to the sealed dataset except two. In `split.json`, three free-text notes of the copied
configuration were reworded (no value changed); `packages/A1_dataset/SANITIZE_MAP.tsv` gives its original and public sha256 and the rule.
`DATASET.md` is this English description, which replaces the original report of the sealed dataset (original sha256
`92f9bb1dd84f1b76ece4047d01ce77bc4adedf948dbe0785b69a5ef986fb3cc0`, kept privately). `shasum -a 256 -c out/phase2/PHASE2_SHA256SUMS`,
run from the checkout root with the package unpacked there, therefore reports 9 files OK and these two as different.

## Sources and licences

Every record names its source repository in the field `repo`, its pinned commit in `commit` and its feature file in
`feature_path`; record ids (`id`, and `record_id` in the prompt and generation packages) start with the repository name,
followed by `|`. Catalog rows name `repo` and `path`. `data/provenance.jsonl` of the code repository lists every source
file the dataset draws on (sha256, git blob, URL at the pinned commit, licence). Text taken from these repositories
(scenarios, steps, step function code) keeps the licence of its repository:

| Repository | Licence (SPDX) | Licence file at the pinned commit | Language | Pinned commit | train | dev | test | test, executed (`exec_core`) |
|---|---|---|---|---|---|---|---|---|
| Sylius/Sylius | MIT | `LICENSE` | php | `c096ca3fc015` | 963 | 80 | 100 | 100 |
| PrestaShop/PrestaShop | OSL-3.0 | `LICENSE.md` | php | `1f83d7b44599` | 826 | 69 | 100 | 0 |
| apache/fineract | Apache-2.0 | `LICENSE_SOURCE`, `NOTICE_SOURCE` | java | `95d444751144` | 86 | 7 | 100 | 0 |
| airslie/renalware | MIT | `MIT-LICENSE` | ruby | `7105621abe1c` | 71 | 6 | 0 | 0 |
| nhsconnect/gpconnect-provider-testing | Apache-2.0 | `LICENSE` | csharp | `c59258277e7f` | 209 | 0 | 0 | 0 |
| openMF/ph-ee-integration-test | MPL-2.0 | `LICENSE` | java | `4a833d5e36fe` | 24 | 1 | 43 | 0 |

The counts are unique scenarios; a scenario that runs in several suites gives several records (pairs). The 60 records of
the auxiliary test set (`aux_test`, 30 renalware and 30 gpconnect-provider-testing scenarios) are counted below.

## Partitions and subsets

| Partition | Scenarios | Pairs |
|---|---|---|
| dev | 163 | 206 |
| test | 343 | 429 |
| train | 2179 | 2860 |
| lc1200 (train subset) | 1200 | 1566 |
| lc600 (train subset) | 600 | 773 |
| test, executed (exec_core) | 100 | 100 |

## Subsets for the secondary reports of Phase 3 (records)

| Partition | Records | Every CREATE item is post-cutoff code (`post_cutoff_code`) | Every CREATE pattern within 30 days of the scenario (`t_vis − scenario_time`) | Has a `preexisting` CREATE item |
|---|---|---|---|---|
| train | 2860 | 81 | 756 | 816 |
| dev | 206 | 22 | 146 | 21 |
| test | 429 | 357 | 339 | 58 |
| aux_test | 60 | 0 | 44 | 0 |

## Set aside (in no partition)

| Reason | Scenarios | Pairs |
|---|---|---|
| ambiguous | 29 | 29 |
| duplicate_structure | 812 | 917 |
| excluded_tag | 47 | 47 |
| no_create | 1758 | 1803 |
| same_file_as_test | 1606 | 1757 |
| test_sibling | 1 | 1 |
| uses_aux_answer | 95 | 95 |
| uses_dev_answer | 33 | 37 |
| uses_test_answer | 1569 | 2198 |

## Auxiliary healthcare test set (aux_test): before the cutoff, EXPLORATORY, never pooled with the main numbers

Per repository, the K scenarios with the fewest dependants are chosen greedily. Selection bias, compared with every record
of the repository (medians):

| Repository | aux scenarios | Steps (aux / repository) | CREATE steps (aux / repository) | Snapshot functions (aux / repository) |
|---|---|---|---|---|
| airslie/renalware | 30 | 5.0 / 6 | 3.0 / 3 | 192.0 / 112 |
| nhsconnect/gpconnect-provider-testing | 30 | 7.0 / 9 | 1.0 / 4 | 498.5 / 52 |

## Twin functions of an answer (same `body_key` or `body_key_nosig`, other fn_id)

`body_key` keeps the method signature; `body_key_nosig` drops the declaration, so it also catches a renamed copy. The time
of a function is the earliest of its dated annotations, i.e. the time of the step PATTERN, not the blame of its body.
Post-cutoff twins of a test answer are joined to the test answer (hidden from train/dev; a scenario that uses one is set
aside): 6. Pre-cutoff twins are legitimate past code and are NOT removed; the table only describes the records whose
answer (for test: the CREATE functions and the pair's `new_target_fns`) has a twin that reaches a learning partition
(as target, bound step or snapshot): train and dev for test and aux_test answers, train only for dev answers:

| Answers of | Records | Twins reaching a learning partition | of which post-cutoff | of which not older than the record's scenario | Records only through an idiom (key group ≥ 5 functions) |
|---|---|---|---|---|---|
| test | 39 | 31 | 0 | 0 | 16 |
| dev | 26 | 16 | 0 | 8 | 10 |
| aux_test | 2 | 3 | 0 | 1 | 0 |

## Near-duplicates of a test answer

A post-cutoff function (same repository, not a test answer) with `difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() ≥ 0.90`,
where a = the `body_key_nosig` string of the FUNCTION and b = that of a test ANSWER (both ≥ 60 characters; the ratio is not
symmetric), is joined to the test answer like a twin: 150 functions, 6 of them also exact twins. Ratios rounded to 4
digits, against the nearest answer:

| Repository | Near-duplicates | 0.90–<0.95 | 0.95–<0.98 | 0.98–<1 | = 1 |
|---|---|---|---|---|---|
| PrestaShop/PrestaShop | 30 | 21 | 6 | 2 | 1 |
| Sylius/Sylius | 36 | 15 | 17 | 0 | 4 |
| apache/fineract | 83 | 43 | 24 | 15 | 1 |
| openMF/ph-ee-integration-test | 1 | 0 | 0 | 1 | 0 |

## Functions that contain a test answer

A post-cutoff function (same repository, not a test answer) that contains ≥ 90% of a test answer of ≥ 200 characters:
the total size of the matching blocks of `difflib.SequenceMatcher(None, a, b, autojunk=False)` ≥ 0.90 × len(a), with
a = the `body_key_nosig` string of the ANSWER and b = that of the FUNCTION. The rule catches an answer body inside a longer
function, where the ratio is pulled down by length. 168 functions are joined to the test answer, 65 of them also
near-duplicates and 1 a twin. Containment rounded to 4 digits:

| Repository | Containing functions | 0.90–<0.95 | 0.95–<0.98 | 0.98–<1 | = 1 |
|---|---|---|---|---|---|
| PrestaShop/PrestaShop | 10 | 6 | 1 | 2 | 1 |
| apache/fineract | 155 | 38 | 34 | 60 | 23 |
| openMF/ph-ee-integration-test | 3 | 2 | 0 | 1 | 0 |

## Dead functions in the catalog

A DEAD function: every annotation line of it is a line comment (`//`, or `#` except `#[`) or lies inside a `/* … */`
block that is not a `/**` docblock (Behat docblocks are live code); or it has no body line (every other line is blank, a
comment, or an annotation of a definition in the file: the Phase 1 extractor splits the first of two stacked Java
annotations into a separate "function"). The catalog keeps them, so the step labels do not change, but flags them `dead`;
a function whose range contains another function of the same file is flagged `contains_fn`; `snapshot_fns` drops both.
Dead functions: fineract 3, gpconnect-provider-testing 2; functions containing another: 3. Phase 1 had 29 BOUND pairs
bound to dead code; all were set aside (0 records use them).

## Test subsets for the sensitivity analysis (declared before scoring)

`sensitivity_subsets.json`: `post_cutoff_all` (every CREATE item is post-cutoff code), `time_coherent_30d` (every CREATE
pattern within 30 days of the scenario) and `residual_clean` (no answer of the record, CREATE functions and
`new_target_fns`, has a residual relation with a function that train/dev reach: a post-cutoff function at ratio
0.85–<0.90, or a function not after the cutoff at ratio ≥ 0.90; a = the function, b = the answer). Phase 3 reports CA on
all three.

| Subset | Test records |
|---|---|
| post_cutoff_all | 357 |
| residual_clean | 259 |
| time_coherent_30d | 339 |

## Fields for Phase 3

- `create[].preexisting`: the function already existed (older step pattern); the scenario only adds a new pattern.
- `create[].post_cutoff_code`: the function is in the pair's `new_target_fns` (Phase 1 classed its code as post-cutoff).
- `t_vis`: the latest time of the record's CREATE patterns (the visible-code time of Phase 3).
- `create_def_ids`: the record's CREATE def_ids, sorted (from the labels).
- `neighbours` (train only, null elsewhere): ids of the OTHER train records of the same .feature file or sharing ≥ 1
  CREATE function, direct neighbours, not transitive; Phase 3 removes them from the record's example pool.
- `snapshot_size`: the number of functions of `snapshot_fns(rec, fns, hidden_for(rec['partition'], forbidden.get(rec['repo'], {})))`.
- `catalog.jsonl`: flags `dead` (dead code) and `contains_fn` (range contains another function); `snapshot_fns` drops
  flagged functions, as it drops `catch_all`.

## Record shape (medians per partition)

| Partition | Steps per record | Share of new steps | Functions to generate | Answer characters | Visible functions (snapshot) |
|---|---|---|---|---|---|
| train | 11.0 | 44% | 4.0 | 1598.0 | 174.0 |
| dev | 7.0 | 36% | 2.5 | 877.0 | 201.0 |
| test | 11 | 32% | 2 | 946 | 398 |

Every column is a median per record except the share of new steps, which is pooled over the partition (CREATE steps
of all records over all their steps).

## Limitations

- The healthcare domain (airslie/renalware, nhsconnect/gpconnect-provider-testing) is not in the main test set, only in
  train/dev and the auxiliary set aux_test: about 450 healthcare repositories were screened, the git trees of 82 checked,
  2 cloned and measured; none met the criteria (licence, step language, new scenarios after the cutoff).
- aux_test lies BEFORE the cutoff (renalware scenarios up to 2019, its aux scenarios up to 2017-06-01; gpconnect up to
  2020), so it carries a contamination risk: the base models may have seen this code. This risk is an unmeasured
  limitation. The set is exploratory only, reported separately and never pooled with the main numbers.
- aux_test greedily picks the scenarios with the fewest dependants, so it leans towards "leaf" scenarios written late in
  the repository's history (fewer CREATE steps, larger snapshot than the whole repository); see the bias table above.
- The snapshot of a test record may hold a function that is the answer of another test record, if that function existed
  (with an older step pattern) before the scenario's time. This is by design: each record is scored on its own and no
  model is trained on test prompts.
- `create[].preexisting = true`: the function existed before with an older step pattern and the scenario only adds a new
  pattern, so the old body may appear in the snapshot. Phase 3 scores this kind separately from wholly new functions.
- `create[].post_cutoff_code = false`: Phase 1 did not class the function's code as post-cutoff (not in the pair's
  `new_target_fns`). The test set has 137 such items in 72 records; Phase 3 also reports the test subset in which every
  CREATE item is post-cutoff code (subset table).
- NEW steps are counted against the scenario's OLDEST line (`scenario_time`), not against when each step was written.
  Records with at least one CREATE pattern dated more than 30 days after the scenario: train 2104/2860, dev 60/206, test
  90/429, aux_test 16/60; share of (record, CREATE pattern) pairs more than 30 days / more than 1 year later: train 52% /
  25%, dev 18% / 10%, test 10% / 1%, aux_test 16% / 2% (mostly steps or Backgrounds edited or added later). No answer
  leaks, but a CREATE function is not necessarily "code written for this scenario" and the share of new steps is
  inflated. `t_vis` = the latest time of the CREATE patterns; Phase 3 also reports the subset
  `t_vis − scenario_time ≤ 30 days` (subset table).
- The time of a step pattern is the first appearance of the (normalised) pattern string anywhere in the repository's
  history, not per function. If it is wrong, a step is labelled reused instead of new (an easier task); no answer leaks.
  The twin, near-duplicate and containment rules use this time too (the earliest of the dated patterns). Compared with
  the blame of the function's oldest line (as Phase 1's `def_time`): the twin rule has no differing case; the
  near-duplicate rule has a few (a check found 11/267, 10 of them because blame caught an old `{`/`}` line or a rewritten
  file), so the pattern time is kept.
- `render` drops annotation lines that did not exist yet; in Ruby the annotation line also opens the block. This has no
  effect now, since every Ruby function in the data has one annotation.
- A `#` comment line right above a scenario's tags or title is not part of `scenario` (Gherkin comments carry no meaning).
- The Phase 1 `java_annotation` extractor mis-splits two stacked Java annotations: the first is assigned to an empty
  "function", so the real function carries only the second. The whole catalog has 1 case
  (WorkingCapitalNearBreachActionStepDef.java:79); 0 records and 0 BOUND pairs use it; the empty item is flagged `dead`,
  so snapshots drop it. Phase 1 is sealed, so its code is not changed.
- The `code` field of `catalog.jsonl` is the code at HEAD, with annotations unfiltered. Phase 3 builds the context from the
  state "HEAD minus the answer": it drops the answer functions and the steps written after the scenario, then masks by
  SYMBOL (a name first seen in the file at or after the scenario's time). The gold answer and the executable-rate run
  also use HEAD. Every record also hides answers by partition through `hidden_for(partition, forbidden)`: train hides
  the test, dev and aux answers; dev hides test and aux; test and aux_test hide nothing more. `forbidden` is
  conservative: its test part includes the reused functions that post-cutoff code calls (`new_target_fns` of the test
  pair) and the post-cutoff twins, near-duplicates and containers of the test answers.
- Near-duplicates below the 0.90 threshold or with a comparison string shorter than 60 characters, and functions that
  contain less than 90% of an answer (or an answer shorter than 200 characters), are not joined. Non-step members (helper
  functions, properties) are not in `forbidden`; Phase 3 masks the "test answer infrastructure" when it composes prompts.
- Residual relations, described and not removed (functions that train/dev reach: target, bound step or snapshot): test
  records whose answer relates to a post-cutoff function at ratio 0.85–<0.90: 108; to a function not after the cutoff
  at ratio ≥ 0.90: 76; dev records whose answer relates to a function that train reaches at ratio ≥ 0.90: 70; aux_test
  records with a function that train/dev reach at ratio ≥ 0.90: 20. Cost of the joining rules against the run just
  before them: near-duplicates 14 scenarios / 15 train/dev pairs, containment 2 dev scenarios; a single scenario is lost
  through an idiom: Sylius `browsing_currencies.feature:12` (2 pairs, ratio exactly 0.9000).
- Share of same-repository train records that `neighbours` removes from a train record's example pool (median /
  maximum): PrestaShop 11% / 32%, Sylius 2% / 33%, renalware 9% / 30%, fineract 6% / 13%, gpconnect-provider-testing
  61% / 75%, ph-ee-integration-test 13% / 26%; the CREATE function shared by the most train records: PrestaShop 177,
  Sylius 251, renalware 12, fineract 7, gpconnect-provider-testing 123, ph-ee-integration-test 4 records.
- The split is NOT by time: train has 96 post-cutoff records, 92 of them later than the earliest test scenario of the
  same repository. `forbidden` hides step functions only, not helper functions or properties added with a test answer.
- `check_dataset.py` is an independent re-check of the RULES on top of the catalog: it recomputes every rule but trusts
  the annotation times, suites, scopes and catch-all flags (`catalog.jsonl`) and the step bindings (`labels.jsonl`). The
  evidence for those times themselves is a full pipeline re-run that gave every byte again, and manual checks of the
  catalog and labelling steps. Some checks are NAMED PROOFS that follow from other checks, not independent detectors:
  `test_answer_twin_in_training`, `near_duplicate_in_training` and `test_answer_container_in_training` report together
  with `forbidden_file_mismatch`; `neighbours_asymmetric` implies `neighbours_wrong`; `train_neighbour_of_other_partition`
  follows from the `file_shared_*` and `*_answer` checks.
- The two `check_dataset` checks `train_dev_uses_forbidden` and `train_dev_answer_code_equals_test` apply to every
  partition other than test, aux_test included (the check names are narrower than their scope).

## Configuration

The configuration used (`config/phase2.json`; also copied into `split.json` under `config`):

```json
{
  "aux_holdout": {
    "airslie/renalware": 30,
    "nhsconnect/gpconnect-provider-testing": 30
  },
  "cutoff_utc": "2024-01-01T00:00:00Z",
  "dev_total": 180,
  "excluded_tags": {
    "PrestaShop/PrestaShop": [],
    "Sylius/Sylius": [
      "@todo",
      "@cli"
    ],
    "airslie/renalware": [
      "@wip",
      "@pending"
    ],
    "apache/fineract": [
      "@ignore",
      "@Skip"
    ],
    "nhsconnect/gpconnect-provider-testing": [],
    "openMF/ph-ee-integration-test": []
  },
  "excluded_tags_evidence": {
    "PrestaShop/PrestaShop": "No global tag filter found. Read tests/Integration/Behaviour/behat.yml (809 lines): only per-suite routing filters exist, e.g. line 561 tags: \"~@legacy-product-type&&~@stock-movement-history\" for the 'product' suite — these only route scenarios to legacy_product/stock_movement_history suites so they aren't double-run, they don't remove scenarios from CI. Read composer.json:297 — \"@php -d date.timezone=UTC ./vendor/bin/behat -c tests/Integration/Behaviour/behat.yml --format progress --no-snippets --strict\" (script 'integration-behaviour-tests', no --tags flag). Read .github/workflows/behaviour.yml — step 'Run integration-behaviour-tests' runs 'composer run-script integration-behaviour-tests --timeout=0' with no extra flags. No exclusion filter is used by CI.",
    "Sylius/Sylius": "behat.dist.php:35 — ->withGherkinOptions((new GherkinOptions())->withFilter(new TagFilter('~@todo&&~@cli'))) [default profile]; confirmed by .github/workflows/ci_e2e-mysql.yaml:123 — $BEHAT_BASE_CMD --tags=\"~@todo&&~@cli\" --suite-tags=\"@api,@domain\" [job 'Run API Behat', matches exec_suite_prefix 'api']",
    "airslie/renalware": "config/cucumber.yml:7 — std_opts = \"... --tags 'not @wip' --tags 'not @legacy' --tags 'not @pending' --strict\" (default profile, line 11: default: <%= std_opts %>), used by bin/gh/cucumber/domain — 'bundle exec cucumber --format progress' (no -p flag => default profile), which is CI job 'Cucumber domain tests' in .github/workflows/verify.yml:183-184. config/cucumber.yml:9 — web_opts = \"... --tags '@web or @legacy' --tags 'not @wip' --tags 'not @pending' --strict\" (rake_web profile, line 16: rake_web: <%= web_opts %>), used by bin/gh/cucumber/web — 'TEST_DEPTH=web bundle exec cucumber -p rake_web --format progress', which is CI job 'Cucumber web tests' in .github/workflows/verify.yml:187-188. Both jobs run in the same CI workflow (verify.yml); the domain job excludes @wip/@legacy/@pending while the web job requires (@web OR @legacy) and excludes @wip/@pending. Taking the union of what CI actually executes across both jobs, @legacy-tagged scenarios DO run (via the web job) so @legacy is not truly excluded from CI overall; only @wip and @pending never run in either job.",
    "apache/fineract": "fineract-provider/build.gradle:380 — tags = 'not @ignore' (cucumber{} block, plugin 'se.thinkcode.cucumber-runner' applied at build.gradle:29); this task runs via .github/workflows/build-cucumber.yml:72 — './gradlew --no-daemon cucumber -x :fineract-e2e-tests-runner:cucumber ...' (only the e2e-tests-runner cucumber task is excluded, :fineract-provider:cucumber still runs, see check.dependsOn('cucumber') at fineract-provider/build.gradle:55). Separately, fineract-e2e-tests-runner/build.gradle:112 — tags = 'not @Skip' (default, overridable via -Pcucumber.tags, not overridden in CI); this task runs via .github/workflows/build-e2e-tests.yml:156 — ':fineract-e2e-tests-runner:cucumber' step (sharded per feature file, no -Pcucumber.tags passed). Both workflows are invoked by .github/workflows/full-build-ci.yml (jobs 'run-cucumber-tests' -> build-cucumber.yml, 'run-e2e-tests' -> build-e2e-tests.yml), so CI as a whole excludes both @ignore and @Skip. No suite scoping is defined for apache/fineract in config/sources.json, so both step-definition sets are in scope.",
    "nhsconnect/gpconnect-provider-testing": "No CI configuration exists in this repository at all: `git ls-tree -r --name-only HEAD` (263 tracked files) contains no .github/workflows directory, no *.runsettings file, no specflow.json, and no azure-pipelines.yml/appveyor.yml/.travis.yml/Jenkinsfile. Read GPConnect.Provider.AcceptanceTests/App.config, App.Debug.config, App.Develop.config, App.Release.config and packages.config — none contain a SpecFlow tag-filter/category-filter section. Read README.md — documents running tests manually via 'nunit3-console \"GPConnect.Provider.AcceptanceTests\\\\bin\\\\Debug\\\\GPConnect.Provider.AcceptanceTests.dll\" --result=TestResult.xml;format=nunit2' with no --where/category filter. README.md:74 mentions scenarios tagged @ignore only in the context of the optional external JSON reporting feature not sending a report for them — this is unrelated to test selection/exclusion. Recorded as [], the rule for repositories with no filter.",
    "openMF/ph-ee-integration-test": "No exclusion filter exists. src/test/java/org/mifos/integrationtest/TestRunner.java — @CucumberOptions has no 'tags' attribute; build.gradle:194 — systemProperty \"cucumber.filter.tags\", System.getProperty(\"cucumber.filter.tags\") (pass-through, no default); .circleci/config.yml (only CI config; no test-running GitHub Actions workflow) uses INCLUSION filters only — see required_tags_any_evidence. No 'not @x' / '~@x' pattern anywhere in the repo's .gradle/.yml/.java/.properties files. The one @Ignore-tagged scenario (src/test/java/resources/voucherManagementTest.feature:98) carries none of @gov/@e2e/@amsIntegration, so required_tags_any sets it aside. Recorded as [] (fixed 23/09)."
  },
  "exec_suite_prefix": {
    "Sylius/Sylius": "api"
  },
  "learning_curve": [
    600,
    1200
  ],
  "minimum_unique_scenarios": {
    "dev": 150,
    "test": 150,
    "train": 600
  },
  "required_tags_any": {
    "openMF/ph-ee-integration-test": [
      "@gov",
      "@e2e",
      "@amsIntegration"
    ]
  },
  "required_tags_any_evidence": {
    "openMF/ph-ee-integration-test": ".circleci/config.yml:230 — ./gradlew test -Dcucumber.filter.tags=\"@gov\" [job test-chart-gov]; .circleci/config.yml:268 — ./gradlew test -Dcucumber.filter.tags=\"@e2e\" [job test-chart-e2e]; .circleci/config.yml:306 — ./gradlew test -Dcucumber.filter.tags=\"@amsIntegration\" [job test-chart-ams]. CI selects scenarios by these INCLUSION filters only (build.gradle:194 passes cucumber.filter.tags through; TestRunner.java @CucumberOptions has no tags), so a scenario carrying none of them never runs in CI (fixed 23/09: set aside as excluded_tag)."
  },
  "seed": 20260923,
  "test_quota": {
    "PrestaShop/PrestaShop": 100,
    "Sylius/Sylius": 100,
    "apache/fineract": 100,
    "openMF/ph-ee-integration-test": 47
  }
}
```
