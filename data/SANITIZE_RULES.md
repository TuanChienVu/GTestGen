# Rules applied to the published copies

`data/SANITIZE_MAP.tsv` lists every published data file that differs from the file the study produced, and every file
of this repository outside the code that was changed line by line. Each row gives the asset (or `repository`), the
path, the SHA-256 of the original file, the SHA-256 of the published file, the rule and a note. Assets A1, A3, A4 and
A5 also ship their own rows in `packages/<asset>/SANITIZE_MAP.tsv`; no file of A2 was changed. Code files (`.py`,
`.sh`) are not listed here; how the code was cleaned and checked is described in the README and, for the locked
analysis code, in `prereg/LOCK_EQUIVALENCE.md`.

| Rule | What changed |
|---|---|
| `R1` | Absolute paths of the build machine to the code repository became paths relative to the repository root. |
| `R2` | Absolute paths of the build machine to the checkout of the execution harness became `harness/`. |
| `R3` | Absolute paths of the build machine to the generation run records became `A3/`, where the A3 asset unpacks. |
| `R4` | The author's personal e-mail address in recorded git headers became the author's GitHub no-reply address. |
| `R5` | Internal working labels in notes became neutral words. In files written by the published code, the new text is exactly the text that code writes. |
| `R6` | A SHA-256 recorded for a file that another rule changed became the SHA-256 of the published file. |
| `A3_IDS` | Cloud infrastructure identifiers became labels: pod ids became the machine labels of the study's run log, GPU UUIDs became `gpu-card-NN`, private network addresses became `pod-local-ip-NN`. |
| `A3_TEXT` | A few fixed phrases in two method scripts of A3 and in the READMEs of its RQ4 and run-integrity tables (internal working labels, a machine path) were replaced. |
| `CONFIG-NOTE-1` | Three free-text notes of the dataset configuration were reworded to remove internal working labels; no value changed. |
| `DOC-REWRITE` | The Phase 2 report was replaced by an English dataset card written for the release; its numbers are unchanged. |
| `LINE` | One or two lines of a configuration or build file changed, as the note says; no value read by the code changed. |

Every replacement of `R1`–`R6`, `A3_IDS`, `A3_TEXT` and `CONFIG-NOTE-1` is literal and can be undone: no target string
occurs in the original file, and applying the rules to the original gives the published file byte for byte. This was
checked for every asset row. The `repository` rows were checked line by line by the equivalence check of the
repository. Model outputs (`outputs.jsonl`), run manifests and the result tables were not changed.
