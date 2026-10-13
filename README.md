# GTestGen

GTestGen generates step definitions for Gherkin scenarios with large language models (LLMs), using domain knowledge
taken from the source code of the project under test. This repository holds the code, the pre-registered analysis
files and the documentation of the study *Context-Adaptive Test Case Generation from Gherkin with Domain-Augmented
Large Language Models* (master's project report, Ho Chi Minh City University of Industry and Trade, 2026). The data,
prompts, model outputs, scores and execution results are attached to the GitHub Release (see
[Release assets](#release-assets)).

## Overview

In Behaviour-Driven Development, a scenario written in Gherkin runs only when each of its steps is bound to a step
definition: test code that calls the components of the project. Developers still write these functions by hand. The
task studied here is to generate the missing step definitions of a new scenario in a project that is already under
development. The input follows a dry-run assumption: the model is told which steps need a definition, and in which
file and class.

The full configuration, Full Hybrid, combines three parts:

- retrieval of similar scenario and step-definition pairs from the same project with BM25;
- fine-tuning with a QLoRA adapter, so that the model learns how step definitions are written;
- the Domain Context Adapter, which selects project knowledge (class skeletons and related step functions) from the
  version of the code just before the missing steps were written. It has no trained weights.

One prompt builder serves five conditions: Zero-shot, RAG-only, FT-only, Full-neutral and Full-aware. Full-aware and
Full-neutral use the same adapter and differ only in the project-knowledge block, so their comparison isolates the
contribution of the Domain Context Adapter.

The decisive measure is Call Agreement (CA): whether the generated code calls the project components that the
developer's own solution calls. Binding Validity (BV) checks that the output is syntactically valid, binds the right
step and uses components that exist in the project. Executable Rate (ER) and ER-trace come from running the generated
code in the Sylius project. The measures were checked against controls before any model output on the test set
existed.

The dataset holds 3,495 records mined from the git history of six open-source projects, plus 60 records of an
auxiliary test set. Its 429 test records come from PHP and Java scenarios written after 1 January 2024, after the
release of all three evaluated models: CodeLlama-7B-Instruct, DeepSeek-Coder-6.7B-Instruct and
Mistral-7B-Instruct-v0.2.

On DeepSeek-Coder-6.7B-Instruct with 5 training seeds, Full Hybrid exceeds the three single-component conditions in CA
by +0.065 to +0.238, and the project-knowledge block alone adds +0.050. Both confirmatory research questions are
supported, and both conclusions hold when CA is replaced by precision-thresholded CA. At the full configuration the
three models reach a CA of 0.221 to 0.252, with median latencies of 17.5 to 22.8 s per output on a 16 GB GPU.

## Repository layout

| Path | Contents |
|---|---|
| `config/` | Source projects, dataset and analysis configuration, special tokens, token plans |
| `v2tools/` | Dataset construction, prompt building, scoring, statistics and the analysis pipeline |
| `scripts/` | Command-line entry points: fetching sources, building prompts, checks, scoring, the locked analysis |
| `v2gpu/` | Training and generation on GPU (QLoRA training, vLLM generation, determinism checks) |
| `er/` | Execution harness for Executable Rate and ER-trace on Sylius, with its Docker setup |
| `ops/` | Scripts used on the rented GPU machines |
| `tools/` | Helpers for provenance, tables and the ER-trace sample |
| `tests/`, `tests_gpu/` | Test suites (CPU; GPU code tested on CPU) |
| `prereg/` | Files of the registered analysis plan, byte-identical to the OSF registration; the public lock file and its map; the equivalence proof |
| `data/` | `provenance.jsonl`, `summary.json`, `DATASET.md`, `SANITIZE_MAP.tsv` and `SANITIZE_RULES.md` |
| `docs/` | `REPRODUCE.md` (commands), `KNOWN_ISSUES.md` and `LABELS.md` (labels in the analysis reports) |
| `evidence/` | `pret_flags/FLAGGED_PRE_T.tsv`: the locked list of records that the `without_pre_t` view of the analysis leaves out |

The module names `v2tools` and `v2gpu` are kept as they were when the analysis was locked, because the locked hashes,
the equivalence check and the tests refer to them.

Comments, file names and reports refer to the stages of the study. Phase 1 mines the pairs of scenarios and step
definitions from the git history of the source projects, and Phase 2 builds the dataset from them. Phase 3 has four
parts: 3A builds the prompts (`v2tools/compose.py`), 3B trains the adapters and generates the outputs (`v2gpu/`), 3C
scores the outputs (`v2tools/evaluate.py`, `scripts/score_run.py`, `scripts/score_independent.py`) and 3D runs the
pre-registered analysis (`scripts/sap_*.py`).

## Release assets

Each asset of Release v1.0.0 is a `.tar.gz` file that unpacks at the root of a checkout. The data land at the paths the
code uses (`out/…`, and `A3/…` for A3). The files of the asset itself land in `packages/<asset>/`: `SHA256SUMS` in
every asset, `SANITIZE_MAP.tsv` in A1, A3, A4 and A5 (no file of A2 was changed), and `README.md` in A3 and A5. Check
an asset with `shasum -a 256 -c packages/<asset>/SHA256SUMS`.

| Asset | Contents | Used for |
|---|---|---|
| A1 | Dataset: records, splits, labels, the sealed hash list, and the Phase 1 pairs read by the scorer | Every level below |
| A2 | The 30 prompt sets (test, auxiliary test, training, development and the development grid), symbol times | Scoring, regeneration, prompt checks |
| A3 | Model outputs of the main run and of the development rounds, with manifests, coverage files, configuration decisions and receipts | Scoring, analysis |
| A4 | Scores, analysis results, controls, dose table, error sample | Checking the reported numbers |
| A5 | Execution results (ER and ER-trace), preflight, overlay manifest | Execution level |

Trained adapters and the generated overlay trees of the execution harness are not released. The overlay trees can be
rebuilt from A3 with `scripts/write_overlays.py` (see `docs/REPRODUCE.md`). Sizes of the compressed archives:
A1 7.6 MB, A2 38.1 MB, A3 11.3 MB, A4 10.1 MB, A5 7.2 MB.

## Installation

The CPU code was run with CPython 3.12.12 on macOS.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt -r requirements-phase3.txt
```

- GPU work uses pinned lock files, installed by `ops/pod_bootstrap.sh`: `requirements-gen.lock` for generation
  (vLLM 0.27.1, torch 2.13.0) and `requirements-gpu.lock` for training (torch 2.14.0, a CUDA 13.0 build, with
  transformers 5.17.0, peft 0.21.0 and bitsandbytes 0.50.2). The fallback pin sets `requirements-gen-fallback-*.in`
  were declared in advance and not used.
- The execution harness needs Python 3.12, git and the Docker CLI. The images are built from `er/docker/` with pinned
  base images and a pinned `composer.lock`; they were built and run on arm64.

## Reproducing the results

`docs/REPRODUCE.md` lists the exact commands. There are three levels.

1. **Scoring and analysis from the released outputs (CPU).** Rebuild the prompt sets from A1 in a git checkout of
   this repository, score the outputs of A3, then rerun the locked analysis. The scores equal the published ones byte
   for byte, and every number of the rerun's analysis results equals the published results in A4.
2. **Regenerating outputs on a GPU.** The main run used a deterministic generation configuration: a gate before the
   main run gave byte-identical outputs on two physical GPUs, and in the main run every regenerated sample (237 on the
   same machine, 93 on another machine) matched its original byte for byte. A generation job accepts only a
   leakage-check receipt made by the code of its own checkout, so a regeneration from this repository first reruns the
   leakage check of the prompts (CPU, under a minute per model). Its outputs then differ from A3 in one field,
   `leakcheck_receipt_sha256`; the other fields are expected to match A3 for the two conditions that use the base model
   (Zero-shot, RAG-only) on the same GPU type (NVIDIA RTX A4000, 16 GB), according to the determinism gate of the
   study. Generation from this repository has not been run on a GPU; the leakage check and the preflight of the jobs
   were run on a CPU. The three adapter conditions need adapters, which are not released. They can be retrained with
   the released code, data and seeds, but retraining is not bit-for-bit deterministic, so these conditions are
   reproduced statistically, not byte for byte.
3. **Execution on Sylius (ER and ER-trace).** Build the Docker images, rebuild the overlays from A3 and check them
   against the overlay manifests in A5, run the controls and then the output runs, and compare the results with A5.

## Pre-registration

The analysis plan was locked on 7 October 2026, before any output was generated on the test set, and registered on
the Open Science Framework the same day: <https://osf.io/2qpvs> (Open-Ended Registration).

- The lock commit is `63fd681` of the private research repository. That commit and the earlier tag of the design
  decisions are not part of this repository, whose history starts from a clean export.
- `prereg/SAP_LOCK.json`, `prereg/MDE.json`, `prereg/CALIBRATION.json` and `prereg/DEV_SELECTION.json` are
  byte-identical to the files in the registration. Their SHA-256 hashes:

  | File | SHA-256 |
  |---|---|
  | `SAP_LOCK.json` | `a7557c21b79e547821814d0aa5e075e39e0fa58d8c72d43c82d242653ca64c0b` |
  | `MDE.json` | `991a548203645d9efd482105acd2a419e35a148aeab5be0954e989fb3672c84b` |
  | `CALIBRATION.json` | `6d13a9cde39826ded0f95bebda5485b8132025735bbc5d189337fcf925d7e258` |
  | `DEV_SELECTION.json` | `74141e917f4fd69a6a0322fed03708e2c672f64ce49d8f384edd00c11c28c898` |
- The analysis code checks its inputs against `prereg/SAP_LOCK.public.json`, which differs from the registered lock
  file only in the values listed in `prereg/LOCK_PUBLIC_MAP.tsv`. Because this repository has a new history, the
  date and ancestry checks of the analysis run on a replay repository built by the rerun script; the independent time
  anchor is the OSF registration.
- The code in this repository is a cleaned copy of the code of the study. It differs in comments, docstrings, help
  and printed texts, and in the changes listed under "Code changes beyond comments" in `docs/KNOWN_ISSUES.md`.
  `prereg/LOCK_EQUIVALENCE.md` and `prereg/ast_equiv.json` show that the syntax trees of the locked analysis files
  match after comments and docstrings are removed, and list every remaining difference with its reason. Rerunning
  the analysis from this repository and the release assets gives every number of the locked analysis results.
- The analysis reports in A4 are written in Vietnamese and name items of the analysis plan and sections of the design
  specification by short labels, such as P9, D11 or §2.4; `docs/LABELS.md` gives the meaning of each label and,
  where it matters, the date on which it was fixed.
- The design specification, the analysis plan document and the pre-registration ledger are kept private; the
  registration lists their commits and SHA-256 hashes, so a copy can be checked against it on request. The run sheets
  used to carry out the steps are also private and are not listed in the registration.
- Published data files that differ from the files the study produced, and configuration or build files of this
  repository that were changed line by line, are listed in `data/SANITIZE_MAP.tsv` with the original SHA-256, the
  published SHA-256 and the rule applied; `data/SANITIZE_RULES.md` explains each rule. For `DATASET.md`, the Phase 2
  report was replaced by an English dataset card written for the release; its numbers are unchanged.

## Licences

- **Code:** MIT, see `LICENSE`.
- **Data created in this work** (labels, splits, scores, analysis results, execution results): CC BY 4.0, see
  `LICENSE-DATA.md`.
- **Files and excerpts from the six source projects** keep their original licences (MIT, OSL-3.0, Apache-2.0,
  MPL-2.0). `data/provenance.jsonl` records the licence and the pinned commit of each file; `THIRD_PARTY_NOTICES.md`
  summarises them.
- **Models:** no model weights are redistributed. The three models are used under their own licences, listed in
  `THIRD_PARTY_NOTICES.md`.

## Citation

Please cite this work with the metadata in `CITATION.cff`.
