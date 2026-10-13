# The registered analysis lock and the code in this repository

The statistical analysis plan (SAP) was locked before any test output existed. The lock is the file
`prereg/SAP_LOCK.json`. Its sha256 is part of the OSF registration (<https://osf.io/2qpvs>, files uploaded on
7 October 2026 and registered at 13:25 UTC that day). This note explains how the code published here relates to
that lock, and how to check the relation yourself.

## What is in `prereg/`

| File | What it is |
|---|---|
| `SAP_LOCK.json` | the registered lock, byte for byte (sha256 `a7557c21b79e547821814d0aa5e075e39e0fa58d8c72d43c82d242653ca64c0b`) |
| `MDE.json`, `CALIBRATION.json`, `DEV_SELECTION.json` | the three files registered with the lock, byte for byte |
| `SAP_LOCK.public.json` | the lock with 11 values pointed at the published files (below) |
| `LOCK_PUBLIC_MAP.tsv` | one row per value of `SAP_LOCK.public.json` that differs (11 rows): name, file, locked sha256, public sha256, proof |
| `ast_equiv.json` | the report of the equivalence check between the published code and the code that produced the results |

The sha256 of the four registered files:

    a7557c21b79e547821814d0aa5e075e39e0fa58d8c72d43c82d242653ca64c0b  SAP_LOCK.json
    991a548203645d9efd482105acd2a419e35a148aeab5be0954e989fb3672c84b  MDE.json
    6d13a9cde39826ded0f95bebda5485b8132025735bbc5d189337fcf925d7e258  CALIBRATION.json
    74141e917f4fd69a6a0322fed03708e2c672f64ce49d8f384edd00c11c28c898  DEV_SELECTION.json

## What the lock covers

`SAP_LOCK.json` names, by sha256:

- `code`: the ten files of the analysis code (`v2tools/stats.py`, `v2tools/sap.py` and the eight `scripts/sap_*.py`);
- `inputs`: the 22 inputs the analysis reads (the configuration, the three registered files above, the dev scores
  that chose the configuration, the controls, the records, the ER control results, and so on).

`scripts/sap_run.py` refuses to run when the sha256 of any of these files differs from the lock.

## Why the published lock differs in 11 values

**The ten code files.** The published code is the code that produced the results, with the working notes in
comments, docstrings and help texts rewritten, five texts printed into the reports reworded to drop internal names
and working labels (their dates and analysis labels are kept, since the published reports carry them), working
labels removed from twelve refusal messages, and one internal function of `scripts/sap_swrap.py` renamed. Logic,
constants, key names and number formats are unchanged.
Rewriting a comment changes a file's sha256, so `code` in `SAP_LOCK.public.json` holds the sha256 of the published
files.

The proof that nothing but those texts changed is `ast_equiv.json`. The check parses each published file and the
file of the commit that produced the results, removes docstrings and help texts, applies the listed exceptions to the
original, and compares the two syntax trees. Every remaining difference fails the check.

**One input.** The ER control results (`out/phase3/er/controls/er_results.jsonl`, `inputs.er_controls[0]`) contained
absolute paths of the machine that ran them. The published copy replaces the two path prefixes by a relative path
and by `harness/`, a reversible literal replacement; nothing else in the file changes. Its rule and both sha256 are in
the data package's `SANITIZE_MAP.tsv` and in `LOCK_PUBLIC_MAP.tsv`.

The other 21 inputs are published byte for byte; their sha256 in `SAP_LOCK.public.json` is the locked one.

Apart from these 11 values, `SAP_LOCK.public.json` equals `SAP_LOCK.json`, key for key and in the same layout:
replacing each locked sha256 of `LOCK_PUBLIC_MAP.tsv` by its public sha256 in the text of `SAP_LOCK.json` gives
`SAP_LOCK.public.json` exactly.

## How to read `ast_equiv.json`

The report is written by the equivalence tool; its keys are short Vietnamese words.

- `tong` (totals): `commit_goc` is the commit compared against, `tep` the number of files, `dat` the files that
  pass, `truot` the files that fail. A complete check has `truot` = 0.
- `tep` (files): one entry per file, with `ket_qua` (`ĐẠT` = pass, `TRƯỢT` = fail), `ngoai_le` (the number of
  accepted exceptions for that file) and, for a failure, the strings or names that differ.
- An exception is an accepted difference that is not a comment or docstring. Each one names the file, the kind (`str`
  for a string literal, `name` for an identifier, `line_code` for a code line), the old and new text, the exact
  number of places, and the reason it changes no number. The old texts contain the internal notes that were removed,
  so the exceptions file is kept with the original repository and compared on request, like the original files.

For the ten locked files there are seventeen exceptions, sixteen for strings and one for a name:

- four texts of `scripts/sap_run.py` printed into `SAP_REPORT.md`;
- the `note` field of `scripts/sap_swrap.py` written into `SWRAP.json`;
- eleven texts of `v2tools/sap.py`, which make twelve refusal messages (one text ends two of them). They are printed
  only when an input is refused, so they appear in no result file;
- the name of one function of `scripts/sap_swrap.py` (`manifest_rows`), which reads the generation manifest.

The published `SAP_REPORT.md` and `SWRAP.json` carry the same reworded notes.

## How to check

1. Compare the sha256 of `prereg/SAP_LOCK.json` with the one in the OSF registration.
2. Read `ast_equiv.json`, or rerun the equivalence tool against the original commit when the original repository is
   made available (see below).
3. Rerun the analysis from this repository and the data packages (see `docs/REPRODUCE.md`). `sap_run` accepts
   `SAP_LOCK.public.json` because its 22 inputs and its code match the published files. Every number of the rerun's
   `SAP_RESULTS.json` must equal the published `SAP_RESULTS.json`. After a fixed substitution of commit identifiers
   and file paths (listed in the rerun script), the two files must be equal byte for byte.

## Provenance checks during a rerun

`sap_run` does more than compare sha256 values. It reads the lock and each scores manifest (`SCORES_META.json`) as
committed in git, and it checks three things:

- the commits are on a branch of `origin` and are ancestors of `HEAD`;
- the scores commit descends from the lock commit;
- the lock's commit date is earlier than every scoring time.

This repository starts from a new, clean history, so those commits are not in it.

The rerun script therefore builds a temporary replay repository with a local `origin`. One commit holds
`SAP_LOCK.public.json`, dated as the original lock commit. A later commit holds each published `SCORES_META.json`,
dated as its original commit. `sap_run` then runs with `--repo` pointing at the replay repository. No line of the
locked code is changed for this.

In a rerun, the date and ancestry checks are therefore a replay, not independent evidence. The independent evidence
that the lock came before the test outputs is the OSF registration time (13:25 UTC on 7 October 2026), which is
earlier than the first test scoring (16:16 UTC on 11 October 2026).

The original run also ended with `scripts/sap_verify.py`. That script compares a run from a fresh clone of the
original repository with the run in the working copy, and checks a row of a private registration ledger. In a rerun
from this repository, that ledger check is replaced by three checks:

- (a) `prereg/SAP_LOCK.json` equals the registered file byte for byte;
- (b) `sap_verify` was run on a fresh clone of the original repository on 12 October 2026 and exited 0 for both the
  test and the auxiliary test parts;
- (c) the rerun's `SAP_RESULTS.json` equals the published one: every numeric field, then every byte after the fixed
  substitution above.

## The original files

The original repository and the original files, including the ten code files and the ER control results with their
locked sha256, are kept and can be compared by hash on request, as stated in the OSF registration.
