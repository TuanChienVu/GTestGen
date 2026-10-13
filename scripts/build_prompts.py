"""Build the pre-rendered prompt files of one compose variant with the three pinned tokenizers. Needs the v2gpu package
for the tokenizer pins and prompt_ids/answer_ids, so that budgets and length rules count exactly the tokens the GPU jobs
will see. Refuses to run when the code or config differs from the commit it records (modified, staged or untracked files
under v2tools/, scripts/ or config/)."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2tools.compose import Params  # noqa: E402
from v2tools.phase3io import dirty_code, load, pools, sha256_file  # noqa: E402
from v2tools.prompts import build  # noqa: E402


def compose_params(variant: str, k: int | None, n_examples: int | None, ratio: float | None,
                   mode: str | None) -> Params:
    """The variant label names its compose parameters: "v0" means exactly the default Params, given or omitted; any
    other label must give --k, --n-examples, --ratio and --mode, so that a new label never silently carries the v0
    parameters. The two budgets stay at their defaults."""
    given = {"k": k, "n_examples": n_examples, "ratio": ratio, "mode": mode}
    v0 = Params()
    if variant == "v0":
        other = {f: v for f, v in given.items() if v is not None and v != getattr(v0, f)}
        if other:
            raise ValueError(f"variant v0 means the v0 parameters; got {other}")
        return v0
    missing = [f for f, v in given.items() if v is None]
    if missing:
        raise ValueError(f"variant {variant!r} needs explicit {missing}")
    return Params(k, n_examples, ratio, v0.budget_examples, v0.budget_knowledge, mode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--k", type=int, help="required unless --variant v0 (8)")
    ap.add_argument("--n-examples", type=int, help="required unless --variant v0 (3)")
    ap.add_argument("--ratio", type=float, help="required unless --variant v0 (0.5)")
    ap.add_argument("--mode", choices=["full", "4a", "4b", "placebo"], help="required unless --variant v0 (full)")
    ap.add_argument("--models", default="deepseek,codellama,mistral")
    ap.add_argument("--parts", default="train,dev")
    ap.add_argument("--pools", default="all")
    ap.add_argument("--config", type=Path, default=Path("config/sources.json"))
    ap.add_argument("--phase2", type=Path, default=Path("out/phase2"))
    ap.add_argument("--symtimes", type=Path, default=Path("out/phase3/symtimes.jsonl"))
    ap.add_argument("--full-root", type=Path, default=Path("repos-full"))
    ap.add_argument("--special-tokens", type=Path, default=Path("config/special_tokens.json"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    try:
        params = compose_params(args.variant, args.k, args.n_examples, args.ratio, args.mode)
    except ValueError as e:
        print("refusing to run:", e, file=sys.stderr)
        return 2
    dirty = dirty_code(ROOT)
    if dirty:
        print("refusing to run: uncommitted code or config in", ROOT, file=sys.stderr)
        print(dirty, file=sys.stderr)
        return 2
    from v2gpu.models import MODELS, load_tokenizer
    from v2gpu.serialize import answer_ids, prompt_ids

    models = args.models.split(",")
    toks = {m: load_tokenizer(MODELS[m]) for m in MODELS}
    counters = {m: (lambda t: lambda text: len(t(text, add_special_tokens=False)["input_ids"]))(toks[m]) for m in MODELS}
    lengths = {m: (lambda t: lambda msgs, gold: len(prompt_ids(t, msgs)) + len(answer_ids(t, msgs, gold)))(toks[m])
               for m in MODELS}
    special = json.loads(args.special_tokens.read_text())  # v2gpu schema: {"models": {m: {"control_tokens": [{content, id}]}}}
    control = {m: [t["content"] for t in special["models"][m]["control_tokens"]] for m in MODELS}
    comp = load(args.config, args.phase2, args.symtimes, args.full_root, params)
    all_pools = pools(comp.records)
    chosen = {p: all_pools[p] for p in args.pools.split(",")}
    parts = args.parts.split(",")
    meta = {"v2tools_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                             text=True, check=True).stdout.strip(),
            "phase2_sums_sha256": sha256_file(args.phase2 / "PHASE2_SHA256SUMS"),
            "symtimes_sha256": sha256_file(args.symtimes),
            "special_tokens_sha256": sha256_file(args.special_tokens),
            "tokenizer_revisions": {m: MODELS[m].revision for m in MODELS}}
    res = build(comp, args.out, args.variant, models, parts, chosen, counters, lengths, control, meta)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
