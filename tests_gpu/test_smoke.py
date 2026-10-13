import hashlib
import json

from tests_gpu.tiny import make_tiny_llama
from v2gpu import smoke
from v2gpu.examples import Example, build, write_jsonl
from v2gpu.modeling import LoadCfg, load_for_training

DEV = [{"id": f"Repo{i % 3}/x|f{i}.feature|{i}|s", "partition": "dev", "scenario": f"Given dev step {i}",
        "create": [{"code": f"function d{i}() {{}}"}]} for i in range(9)]


def run_(ids, eos):
    return {"new_ids": ids, "stopped_at_eos": eos}


def test_verdict_codepath_needs_identical_runs_only():
    a = [run_([1, 2], False)] * 5
    assert smoke.verdict(a, a, "codepath")["pass"]
    assert not smoke.verdict(a, [run_([1, 3], False)] + a[1:], "codepath")["pass"]


def test_verdict_gate_needs_four_of_five_eos():
    four = [run_([1], True)] * 4 + [run_([1], False)]
    three = [run_([1], True)] * 3 + [run_([1], False)] * 2
    assert smoke.verdict(four, four, "gate")["pass"]
    assert not smoke.verdict(three, three, "gate")["pass"]


def test_sample_spreads_over_repos():
    exs = [Example(r["id"], "F2", "m", "dev", "p", [], "a", {}) for r in DEV]
    picked = smoke.sample(exs, 3)
    assert sorted(e.record_id.split("|")[0] for e in picked) == ["Repo0/x", "Repo1/x", "Repo2/x"]
    assert len(smoke.sample(exs, 5)) == 5 and smoke.sample(exs, 5) == smoke.sample(list(reversed(exs)), 5)


def test_smoke_end_to_end_on_cpu(tmp_path, tokenizers):
    tok = tokenizers["codellama"]
    tiny = make_tiny_llama(tmp_path / "tiny", len(tok))
    model, _ = load_for_training(LoadCfg(tiny, None, quant="none", attn="sdpa", dtype="float32",
                                         gradient_checkpointing=False), seed=42)
    model.save_pretrained(tmp_path / "adapter")
    P = tmp_path / "prompts"
    P.mkdir()
    digest = write_jsonl(P / "dev.F2.jsonl", build(DEV, "codellama", tok, [],
                                                   lambda r, f, t, p: ([{"role": "user", "content": r["scenario"]}], {}),
                                                   lambda r: r["create"][0]["code"], formats=("F2",)))
    (P / "PROMPTS_META.json").write_text("{}")
    meta = hashlib.sha256(b"{}").hexdigest()
    (P / "PROMPTS_SHA256SUMS").write_text(f"{digest}  dev.F2.jsonl\n{meta}  PROMPTS_META.json\n")
    args = ["--model", "codellama", "--adapter", str(tmp_path / "adapter"), "--dev-prompts", str(P / "dev.F2.jsonl"),
            "--prompts-sums", str(P / "PROMPTS_SHA256SUMS"), "--out", str(tmp_path / "smoke.json"), "--check",
            "codepath", "--n", "2", "--cap", "8", "--model-path", tiny, "--quant", "none", "--dtype", "float32"]
    code = smoke.main(args)
    r = json.loads((tmp_path / "smoke.json").read_text())
    assert code == 0 and r["verdict"]["identical"] == [True, True] and r["verdict"]["n"] == 2
    assert all(o["budget"] == 8 for o in r["outputs"]) and len({e.split("|")[0] for e in r["examples"]}) == 2
    assert r["prompts"]["sums_sha256"] == hashlib.sha256((P / "PROMPTS_SHA256SUMS").read_bytes()).hexdigest()
    with (P / "dev.F2.jsonl").open("a") as f:                     # prompts changed after the sums were written
        f.write("\n")
    assert smoke.main(args) == 2
    r = json.loads((tmp_path / "smoke.json").read_text())
    assert r["verdict"]["pass"] is False and any("sha256 differs" in p for p in r["problems"])
