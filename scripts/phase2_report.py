"""Write out/phase2/DATASET.md: provenance, licences and counts of the Phase 2 dataset, every number read from files.
Refuses to run on a working tree with uncommitted changes, so the tool commit it records is the code that ran."""
import argparse
import datetime as dt
import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the repo whose code runs: v2tools is imported from it, git is asked about it
sys.path.insert(0, str(ROOT))
from v2tools.body import body_key, body_key_nosig, nosig_text  # noqa: E402
from v2tools.records import hidden_for, snapshot_fns  # noqa: E402
from v2tools.sensitivity import touched_answers  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--phase2", type=Path, default=Path("out/phase2"))
ap.add_argument("--sources", type=Path, default=Path("config/sources.json"))
ap.add_argument("--pairs", type=Path, default=Path("out/hist/pairs.jsonl"))
ap.add_argument("--phase1-sums", type=Path, default=Path("out/PHASE1_SHA256SUMS"))
args = ap.parse_args()
git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=True).stdout.strip()
if git("status", "--porcelain", "--untracked-files=no"):
    raise SystemExit("phase2_report: uncommitted changes in the working tree; commit first so the tool commit is exact")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
key = lambda r: (r["repo"], r["feature_path"], r["scenario_line"], r["suite"])
split = json.loads((args.phase2 / "split.json").read_text())
recs = {p: load(args.phase2 / "records" / f"{p}.jsonl") for p in ("train", "dev", "test")}
aux = load(args.phase2 / "records" / "aux_test.jsonl") if (args.phase2 / "records" / "aux_test.jsonl").exists() else []
every = {**recs, "aux_test": aux}
sources = json.loads(args.sources.read_text())
cat: dict[str, dict[str, dict]] = defaultdict(dict)
for f in load(args.phase2 / "catalog.jsonl"):
    cat[f["repo"]][f["fn_id"]] = f
labels = {key(l): l for l in load(args.phase2 / "labels.jsonl")}
forbidden = {r: {k: set(v) for k, v in parts.items()} for r, parts in json.loads((args.phase2 / "forbidden.json").read_text()).items()}
ann_time = {(r, a["def_id"]): a["time"] for r, fs in cat.items() for f in fs.values() for a in f["annotations"]}
minimum = split["config"]["minimum_unique_scenarios"]
cut = int(dt.datetime.fromisoformat(split["cutoff_utc"].replace("Z", "+00:00")).timestamp())
DAY = 86400
med = lambda xs: statistics.median(xs) if xs else 0

out = [f"# GTestGen V2 — bộ dữ liệu Giai đoạn 2", "",
       f"- Mốc cắt: `{split['cutoff_utc']}` · seed `{split['seed']}` · commit công cụ `{git('rev-parse', 'HEAD')}`",
       f"- Đầu vào: `{args.phase1_sums}` sha256 `{sha(args.phase1_sums)}` · `{args.pairs}` sha256 `{sha(args.pairs)}`",
       f"- Công cụ: {git('--version')} · Python {platform.python_version()}",
       f"- Kết luận so với mức tối thiểu {minimum['train']}/{minimum['dev']}/{minimum['test']}: **{split['verdict']}**", "",
       "## Nguồn", "", "| Repo | Giấy phép | Ngôn ngữ | Commit nguồn | train | dev | test | test chạy thật |", "|---|---|---|---|---|---|---|---|"]
for s in sources:
    rs = {p: [r for r in recs[p] if r["repo"] == s["repo"]] for p in recs}
    commit = next((r["commit"][:12] for p in recs for r in rs[p]), "")
    uniq = {p: len({(r["feature_path"], r["scenario_line"]) for r in rs[p]}) for p in rs}
    out.append(f"| {s['repo']} | {s['license_spdx']} | {s['language']} | `{commit}` | {uniq['train']} | {uniq['dev']} | "
               f"{uniq['test']} | {sum(r['exec_core'] for r in rs['test'])} |")
out += ["", "Số trong bảng là kịch bản duy nhất; một kịch bản chạy trên nhiều suite cho nhiều bản ghi (cặp).", "",
        "## Phần và tập con", "", "| Phần | Kịch bản | Cặp |", "|---|---|---|"]
out += [f"| {p} | {v['unique_scenarios']} | {v['pairs']} |" for p, v in split["counts"].items()]
out += [f"| {p} (tập con train) | {v['unique_scenarios']} | {v['pairs']} |" for p, v in split["subsets"].items()]
out += [f"| test chạy thật (exec_core) | {split['exec_core']['unique_scenarios']} | {split['exec_core']['pairs']} |", "",
        "## Tập con để báo cáo phụ ở Giai đoạn 3 (số bản ghi)", "",
        "| Phần | Bản ghi | Mọi mục CREATE là mã sau mốc cắt (`post_cutoff_code`) | Mọi mẫu CREATE ≤ 30 ngày sau kịch bản "
        "(`t_vis − scenario_time`) | Có mục CREATE `preexisting` |", "|---|---|---|---|---|"]
for p, rs in every.items():
    out.append(f"| {p} | {len(rs)} | {sum(all(x['post_cutoff_code'] for x in r['create']) for r in rs)} | "
               f"{sum(r['t_vis'] - r['scenario_time'] <= 30 * DAY for r in rs)} | {sum(any(x['preexisting'] for x in r['create']) for r in rs)} |")
out += ["", "## Để riêng (không vào phần nào)", "", "| Lý do | Kịch bản | Cặp |", "|---|---|---|"]
out += [f"| {k} | {v['unique_scenarios']} | {v['pairs']} |" for k, v in split["set_aside"].items()]
if aux:
    out += ["", "## Tập test y tế phụ (aux_test) — trước mốc cắt, THĂM DÒ, không bao giờ gộp vào số chính", "",
            "Chọn tham lam K kịch bản ít bị phụ thuộc nhất mỗi repo (ruling 4). Độ lệch chọn mẫu, so với mọi bản ghi của repo:", "",
            "| Repo | Kịch bản aux | Step (aux / repo) | Step CREATE (aux / repo) | Hàm snapshot (aux / repo) |", "|---|---|---|---|---|"]
    for repo in sorted({r["repo"] for r in aux}):
        a = [r for r in aux if r["repo"] == repo]
        pop = a + [r for p in ("train", "dev") for r in recs[p] if r["repo"] == repo]
        ncreate = lambda r: sum(k == "create" for k in r["step_kinds"])
        out.append(f"| {repo} | {len({(r['feature_path'], r['scenario_line']) for r in a})} | {med([len(r['steps']) for r in a])} / "
                   f"{med([len(r['steps']) for r in pop])} | {med([ncreate(r) for r in a])} / {med([ncreate(r) for r in pop])} | "
                   f"{med([r['snapshot_size'] for r in a])} / {med([r['snapshot_size'] for r in pop])} |")

# Twins: answers sharing body_key or body_key_nosig with another function that the learning parts can
# reach, as a target, a bound step or a snapshot function. Post-cutoff twins of test answers already joined the test
# answer (split.json). Test answers are CREATE functions and the pair's new_target_fns, as in final_split.
languages = {s["repo"]: s["language"] for s in sources}
new_fns = {key(r): r["new_target_fns"] for r in load(args.pairs)}
keys: dict[tuple, set] = defaultdict(set)
by_key = defaultdict(set)
for r, fs in cat.items():
    for f, fn in fs.items():
        for k in (("sig", body_key(fn)), ("nosig", body_key_nosig(fn, languages[r]))):
            if k[1] is not None:
                keys[(r, f)].add(k)
                by_key[(r, k)].add(f)
reach = {p: defaultdict(set) for p in ("train", "dev")}
for p in ("train", "dev"):
    for r in recs[p]:
        seen = {x["fn_id"] for x in r["create"]} | set(r["step_fns"])
        seen |= set(snapshot_fns(r, cat[r["repo"]], hidden_for(p, forbidden.get(r["repo"], {}))))
        for f in seen:
            reach[p][(r["repo"], f)].add(r["id"])


def dated_after_cutoff(repo: str, f: str) -> bool:
    times = [a["time"] for a in cat[repo][f]["annotations"] if a["time"] is not None]
    return bool(times) and min(times) > cut


out += ["", "## Hàm sinh đôi của đáp án (cùng `body_key` hoặc `body_key_nosig`, fn_id khác)", "",
        "`body_key` giữ chữ ký phương thức; `body_key_nosig` bỏ phần khai báo, nên bắt cả bản sao đổi tên. Thời điểm của một "
        "hàm là mốc sớm nhất trong các annotation có thời điểm, tức theo thời điểm của MẪU bước, không theo blame của thân hàm. "
        f"Hàm sinh đôi sau mốc cắt của đáp án test đã gộp vào đáp án test (ẩn khỏi train/dev, kịch bản dùng nó bị để riêng): "
        f"{len(split['test_answer_twins'])}. Hàm sinh đôi trước mốc cắt là code quá khứ hợp lệ nên KHÔNG loại; bảng dưới chỉ mô tả "
        "bản ghi có đáp án (với test: hàm CREATE và `new_target_fns` của cặp) mà hàm sinh đôi của nó đi tới phần học "
        "(đích, step được gắn hoặc snapshot) — train và dev với đáp án test và aux_test, chỉ train với đáp án dev:", "",
        "| Đáp án của | Bản ghi | Hàm sinh đôi đi tới phần học | trong đó sau mốc cắt | trong đó không cũ hơn kịch bản "
        "của bản ghi | Bản ghi chỉ qua idiom (nhóm khoá ≥ 5 hàm) |", "|---|---|---|---|---|---|"]
for hp in ("test", "dev", "aux_test"):
    parts = ("train",) if hp == "dev" else ("train", "dev")
    hit, idiom_only, twins, not_older = 0, 0, set(), set()
    for r in every[hp]:
        answer = {x["fn_id"] for x in r["create"]} | (set(new_fns[key(r)]) if hp == "test" else set())
        group_sizes = []
        for f in answer:
            for k in keys.get((r["repo"], f), ()):
                for g in by_key[(r["repo"], k)] - {f}:
                    if any(reach[p].get((r["repo"], g)) for p in parts):
                        group_sizes.append(len(by_key[(r["repo"], k)]))
                        twins.add((r["repo"], g))
                        dated = [a["time"] for a in cat[r["repo"]][g]["annotations"] if a["time"] is not None]
                        if dated and min(dated) >= r["scenario_time"]:
                            not_older.add((r["repo"], g))
        hit += bool(group_sizes)
        idiom_only += bool(group_sizes) and min(group_sizes) >= 5
    out.append(f"| {hp} | {hit} | {len(twins)} | {sum(dated_after_cutoff(*t) for t in twins)} | {len(not_older)} | {idiom_only} |")
near_dups = split["test_answer_near_duplicates"]
twin_set = {tuple(t) for t in split["test_answer_twins"]}
out += ["", "## Hàm gần trùng với đáp án test (ruling 5c)", "",
        "Hàm sau mốc cắt (cùng repo, không phải đáp án test) có `difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() "
        "≥ 0,90`, với a = chuỗi `body_key_nosig` của HÀM và b = của một ĐÁP ÁN test (cả hai ≥ 60 ký tự; ratio không đối xứng), "
        "được gộp vào đáp án test như hàm sinh đôi: "
        f"{len(near_dups)} hàm, trong đó {sum((x[0], x[1]) in twin_set for x in near_dups)} cũng là hàm sinh đôi chính xác. "
        "ratio làm tròn 4 chữ số, theo đáp án gần nhất:", "",
        "| Repo | Hàm gần trùng | 0,90–<0,95 | 0,95–<0,98 | 0,98–<1 | = 1 |", "|---|---|---|---|---|---|"]
for repo in sorted({x[0] for x in near_dups}):
    ratios = [x[3] for x in near_dups if x[0] == repo]
    out.append(f"| {repo} | {len(ratios)} | " + " | ".join(str(sum(lo <= v < hi for v in ratios))
                                                         for lo, hi in ((0.90, 0.95), (0.95, 0.98), (0.98, 1.0)))
               + f" | {sum(v == 1.0 for v in ratios)} |")
containers = split["test_answer_containers"]
near_set = {(x[0], x[1]) for x in near_dups}
out += ["", "## Hàm chứa trọn đáp án test (ruling 5d)", "",
        "Hàm sau mốc cắt (cùng repo, không phải đáp án test) chứa ≥ 90% một đáp án test dài ≥ 200 ký tự: tổng kích thước các "
        "khối khớp của `difflib.SequenceMatcher(None, a, b, autojunk=False)` ≥ 0,90 × len(a), với a = chuỗi `body_key_nosig` của "
        "ĐÁP ÁN và b = của HÀM. Luật này bắt thân đáp án nằm trong một hàm dài hơn, nơi ratio bị độ dài kéo xuống. "
        f"{len(containers)} hàm được gộp vào đáp án test, trong đó {sum((x[0], x[1]) in near_set for x in containers)} cũng là "
        f"hàm gần trùng và {sum((x[0], x[1]) in twin_set for x in containers)} là hàm sinh đôi. Tỷ lệ chứa làm tròn 4 chữ số:", "",
        "| Repo | Hàm chứa trọn | 0,90–<0,95 | 0,95–<0,98 | 0,98–<1 | = 1 |", "|---|---|---|---|---|---|"]
for repo in sorted({x[0] for x in containers}):
    shares = [x[3] for x in containers if x[0] == repo]
    out.append(f"| {repo} | {len(shares)} | " + " | ".join(str(sum(lo <= v < hi for v in shares))
                                                         for lo, hi in ((0.90, 0.95), (0.95, 0.98), (0.98, 1.0)))
               + f" | {sum(v == 1.0 for v in shares)} |")
dead = {(r, f) for r, fs in cat.items() for f, fn in fs.items() if fn["dead"]}
swallowing = sum(fn["contains_fn"] for fs in cat.values() for fn in fs.values())
dead_pairs = {"|".join(map(str, key(x))) for x in load(args.pairs)
              if x["status"] == "BOUND" and any((x["repo"], f) in dead for f in x["target_fns"])}
out += ["", "## Hàm chết trong catalog (ruling 5d)", "",
        "Hàm CHẾT: mọi dòng annotation của nó là chú thích dòng (`//`, hoặc `#` trừ `#[`) hoặc nằm trong một khối `/* … */` "
        "không phải docblock `/**` (docblock của Behat là mã sống); hoặc nó không có dòng thân nào (mọi dòng khác trống, là chú "
        "thích, hoặc là annotation của một định nghĩa trong tệp: bộ trích xuất Giai đoạn 1 tách annotation đầu của hai annotation "
        "Java xếp chồng thành một 'hàm' riêng). Catalog giữ chúng, nên nhãn của Task 3 không đổi, nhưng gắn "
        "cờ `dead`; hàm có phạm vi chứa trọn một hàm khác cùng tệp gắn cờ `contains_fn`; `snapshot_fns` bỏ cả hai. Hàm chết: "
        + ", ".join(f"{repo.split('/')[1]} {n}" for repo, n in sorted(Counter(r for r, _ in dead).items()))
        + f"; hàm chứa hàm khác: {swallowing}. Giai đoạn 1 có {len(dead_pairs)} cặp BOUND gắn vào mã chết; tất cả đã bị để riêng "
        f"({sum(r['id'] in dead_pairs for rs in every.values() for r in rs)} bản ghi dùng chúng)."]
def_lines = defaultdict(set)  # every definition's annotation line, per file
for r, fs in cat.items():
    for fn in fs.values():
        def_lines[(r, fn["path"])] |= {a["line"] for a in fn["annotations"]}
comment_line = re.compile(r"^\s*(/\*\*|\*|\*/|//|#(?!\[))")
bodiless = sorted((r, f) for r, fs in cat.items() for f, fn in fs.items()
                  if all(i in def_lines[(r, fn["path"])] or not t.strip() or comment_line.match(t)
                         for i, t in enumerate(fn["code"].splitlines(), start=fn["first_line"])))
bodiless_pairs = sum(x["status"] == "BOUND" and any((x["repo"], f) in set(bodiless) for f in x["target_fns"]) for x in load(args.pairs))
bodiless_records = sum(any((r["repo"], f) in set(bodiless) for f in {x["fn_id"] for x in r["create"]} | set(r["step_fns"]))
                       for rs in every.values() for r in rs)
subsets = json.loads((args.phase2 / "sensitivity_subsets.json").read_text())
out += ["", "## Tập con test cho phân tích độ nhạy (ruling 5d, khai trước)", "",
        "`sensitivity_subsets.json`: `post_cutoff_all` (mọi mục CREATE là mã sau mốc cắt), `time_coherent_30d` (mọi mẫu CREATE "
        "≤ 30 ngày sau kịch bản) và `residual_clean` (không đáp án nào của bản ghi, gồm hàm CREATE và `new_target_fns`, có quan hệ "
        "còn sót với một hàm mà train/dev chạm tới: hàm sau mốc cắt có ratio 0,85–<0,90, hoặc hàm không sau mốc cắt có ratio "
        "≥ 0,90; a = hàm, b = đáp án). Giai đoạn 3 báo CA trên cả ba tập.", "",
        "| Tập | Bản ghi test |", "|---|---|"] + [f"| {n} | {len(v)} |" for n, v in sorted(subsets.items())]
languages = {s["repo"]: s["language"] for s in sources}
touched_post, touched_pre, touched_dev, touched_aux = set(), set(), set(), set()
for repo, fns in sorted(cat.items()):
    text = lambda f: nosig_text(fns[f], languages[repo])
    by_train = {f for (rr, f) in reach["train"] if rr == repo}
    by_learning = by_train | {f for (rr, f) in reach["dev"] if rr == repo}
    test_ans = {f: text(f) for r in recs["test"] if r["repo"] == repo
                for f in {x["fn_id"] for x in r["create"]} | set(new_fns[key(r)]) if f in fns}
    touched_post |= {(repo, f) for f in touched_answers(test_ans, {g: (text(g), True) for g in by_learning if dated_after_cutoff(repo, g)})}
    touched_pre |= {(repo, f) for f in touched_answers(test_ans, {g: (text(g), False) for g in by_learning if not dated_after_cutoff(repo, g)})}
    dev_ans = {x["fn_id"]: text(x["fn_id"]) for r in recs["dev"] if r["repo"] == repo for x in r["create"]}
    touched_dev |= {(repo, f) for f in touched_answers(dev_ans, {g: (text(g), False) for g in by_train})}
    aux_ans = {x["fn_id"]: text(x["fn_id"]) for r in aux if r["repo"] == repo for x in r["create"]}
    touched_aux |= {(repo, f) for f in touched_answers(aux_ans, {g: (text(g), False) for g in by_learning})}
records_touching = lambda rs, touched, with_new: sum(
    any((r["repo"], f) in touched for f in {x["fn_id"] for x in r["create"]} | (set(new_fns[key(r)]) if with_new else set())) for r in rs)
n_train = Counter(r["repo"] for r in recs["train"])
pool_share, hub = defaultdict(list), defaultdict(Counter)
for r in recs["train"]:
    if n_train[r["repo"]] > 1:
        pool_share[r["repo"]].append(len(r["neighbours"]) / (n_train[r["repo"]] - 1))
    hub[r["repo"]].update(x["fn_id"] for x in r["create"])
out += ["", "## Trường dành cho Giai đoạn 3", "",
        "- `create[].preexisting`: hàm đã có từ trước (mẫu bước cũ hơn), kịch bản chỉ thêm mẫu mới.",
        "- `create[].post_cutoff_code`: hàm thuộc `new_target_fns` của cặp (Giai đoạn 1 xếp code sau mốc cắt).",
        "- `t_vis`: thời điểm muộn nhất của các mẫu CREATE của bản ghi (mốc code nhìn thấy trong đặc tả Giai đoạn 3).",
        "- `create_def_ids`: các def_id CREATE của bản ghi, đã sắp xếp (lấy từ labels).",
        "- `neighbours` (chỉ train, phần khác là null): id các bản ghi train KHÁC cùng tệp .feature hoặc chung ≥ 1 hàm CREATE, "
        "láng giềng trực tiếp, không bắc cầu; Giai đoạn 3 loại chúng khỏi pool ví dụ của bản ghi.",
        "- `snapshot_size`: số hàm của `snapshot_fns(rec, fns, hidden_for(rec['partition'], forbidden.get(rec['repo'], {})))`.",
        "- `catalog.jsonl`: cờ `dead` (mã chết) và `contains_fn` (phạm vi chứa hàm khác); `snapshot_fns` bỏ hàm mang cờ, như "
        "`catch_all`."]

late = {p: [(r, ann_time[(r["repo"], d)] - r["scenario_time"]) for r in rs for d in labels[key(r)]["create_def_ids"]]
        for p, rs in every.items()}
late_rec = {p: sum(any(ann_time[(r["repo"], d)] - r["scenario_time"] > 30 * DAY for d in labels[key(r)]["create_def_ids"]) for r in rs)
            for p, rs in every.items()}
first_test = defaultdict(lambda: float("inf"))
for r in recs["test"]:
    first_test[r["repo"]] = min(first_test[r["repo"]], r["scenario_time"])
train_post = [r for r in recs["train"] if r["scenario_time"] > cut]
out += ["", "## Hình dạng bản ghi (trung vị theo phần)", "",
        "| Phần | Step/bản ghi | Tỷ lệ step viết mới | Hàm phải sinh | Ký tự đáp án | Hàm được thấy (snapshot) |", "|---|---|---|---|---|---|"]
for p, rs in recs.items():
    share = sum(k == "create" for r in rs for k in r["step_kinds"]) / max(1, sum(len(r["step_kinds"]) for r in rs))
    out.append(f"| {p} | {med([len(r['steps']) for r in rs])} | {share:.0%} | {med([len(r['create']) for r in rs])} | "
               f"{med([sum(len(c['code']) for c in r['create']) for r in rs])} | {med([r['snapshot_size'] for r in rs])} |")
out += ["", "## Giới hạn", "",
        "- Miền y tế (airslie/renalware, nhsconnect/gpconnect-provider-testing) không có trong test chính, chỉ có trong train/dev "
        "và tập phụ aux_test: đã rà khoảng 450 repo y tế, kiểm cây git 82 repo, clone và đo 2 repo; không repo nào đạt "
        "(giấy phép, ngôn ngữ step, kịch bản mới sau mốc cắt) — xem out/healthcare_scout/REPORT.md.",
        "- aux_test nằm TRƯỚC mốc cắt (repo renalware có kịch bản đến 2019, aux của renalware đến 2017-06-01; gpconnect đến 2020) "
        "nên có rủi ro nhiễm: mô hình nền có thể đã thấy mã này. Tập này chỉ để thăm dò, báo riêng, không bao giờ gộp vào số "
        "chính. Giai đoạn 3 đo nhiễm bằng Min-K% Prob, đối chứng bằng mã ứng dụng renalware trước và sau 2024; gpconnect không "
        "có mã C# sau mốc cắt nên gắn nhãn 'chưa hiệu chỉnh được mức nhiễm'.",
        "- aux_test chọn tham lam các kịch bản ít bị phụ thuộc nhất, nên thiên về kịch bản 'lá' viết muộn trong lịch sử repo "
        "(ít step CREATE hơn, snapshot lớn hơn toàn repo); xem bảng độ lệch ở mục aux_test.",
        "- Snapshot của một bản ghi test có thể chứa hàm là đáp án của một bản ghi test khác, nếu hàm đó đã tồn tại "
        "(có mẫu bước cũ hơn) trước thời điểm kịch bản. Đây là thiết kế: mỗi bản ghi được chấm độc lập và không huấn luyện "
        "trên prompt test.",
        "- `create[].preexisting = true`: hàm đã có từ trước với mẫu bước cũ hơn, kịch bản chỉ thêm mẫu bước mới, nên thân "
        "cũ của hàm có thể xuất hiện trong snapshot. Giai đoạn 3 chấm riêng loại này với loại hàm hoàn toàn mới.",
        "- `create[].post_cutoff_code = false`: Giai đoạn 1 không xếp code của hàm vào sau mốc cắt (không thuộc `new_target_fns` "
        f"của cặp). Test có {sum(not x['post_cutoff_code'] for r in recs['test'] for x in r['create'])} mục như vậy trong "
        f"{sum(not all(x['post_cutoff_code'] for x in r['create']) for r in recs['test'])} bản ghi; Giai đoạn 3 báo thêm tập con "
        "test mà mọi mục CREATE là mã sau mốc cắt (bảng tập con).",
        "- VIẾT MỚI tính theo dòng CŨ NHẤT của kịch bản (`scenario_time`), không theo lúc từng step được viết. Bản ghi có ít "
        "nhất một mẫu CREATE ghi ngày hơn 30 ngày sau kịch bản: "
        + ", ".join(f"{p} {late_rec[p]}/{len(every[p])}" for p in every) + "; tỷ lệ cặp (bản ghi, mẫu CREATE) hơn 30 ngày / hơn "
        "1 năm: " + ", ".join(f"{p} {sum(d > 30 * DAY for _, d in late[p]) / max(1, len(late[p])):.0%} / "
                               f"{sum(d > 365 * DAY for _, d in late[p]) / max(1, len(late[p])):.0%}" for p in every)
        + " (phần lớn do step hoặc Background được sửa/thêm sau). Không lộ đáp án, nhưng CREATE không chắc là 'code viết cho kịch bản này' và tỷ lệ step "
        "viết mới bị thổi phồng. `t_vis` = thời điểm muộn nhất của các mẫu CREATE; Giai đoạn 3 báo thêm tập con "
        "`t_vis − scenario_time ≤ 30 ngày` (bảng tập con).",
        "- Thời điểm của một mẫu bước là lần đầu chuỗi mẫu (đã chuẩn hoá) xuất hiện ở bất kỳ đâu trong lịch sử repo, không "
        "tính riêng theo hàm. Nếu sai, step bị gán dùng lại thay vì viết mới (bài dễ hơn); đáp án không bị lộ. Luật hàm sinh "
        "đôi, gần trùng và chứa trọn cũng dùng thời điểm này (mốc sớm nhất của các mẫu có thời điểm). So với blame dòng cũ nhất "
        "của hàm (như `def_time` của Giai đoạn 1): luật sinh đôi không có ca nào khác; luật gần trùng có một số ít hàm khác (review "
        "5c: 11/267, 10 ca do blame bắt một dòng `{`/`}` cũ hoặc tệp được viết lại), nên thời điểm của mẫu được giữ.",
        "- `render` bỏ các dòng annotation chưa tồn tại; với Ruby, dòng annotation cũng là dòng mở khối. Hiện không ảnh "
        "hưởng vì mọi hàm Ruby trong dữ liệu chỉ có một annotation.",
        "- Dòng chú thích `#` ngay trên tag hoặc tiêu đề kịch bản không nằm trong `scenario` (chú thích Gherkin không mang "
        "ngữ nghĩa).",
        "- Bộ trích xuất `java_annotation` của Giai đoạn 1 tách sai khi hai annotation Java xếp chồng: annotation đầu bị gán cho "
        "một 'hàm' rỗng, nên hàm thật chỉ mang annotation thứ hai. Toàn catalog có "
        f"{len(bodiless)} ca ({', '.join(f.split('/')[-1] for _, f in bodiless)}); {bodiless_records} bản ghi và {bodiless_pairs} "
        "cặp BOUND dùng tới; mục rỗng mang cờ `dead` nên snapshot bỏ nó. Giai đoạn 1 đã niêm phong nên mã của nó không sửa.",
        "- Trường `code` trong `catalog.jsonl` là mã ở HEAD, chưa lọc annotation. Giai đoạn 3 dựng ngữ cảnh theo trạng thái "
        "'HEAD trừ đáp án': bỏ hàm đáp án và các step viết sau kịch bản, rồi che theo KÝ HIỆU (tên xuất hiện lần đầu trong tệp "
        "≥ thời điểm kịch bản). Đáp án chuẩn và ER cũng dùng HEAD. Xem SPEC_V2_PHASE3_DESIGN.md. Ngoài ra mọi bản ghi còn "
        "ẩn đáp án theo phần qua `hidden_for(partition, forbidden)`: train ẩn đáp án test, dev và aux; dev ẩn test và aux; "
        "test và aux_test không ẩn thêm. `forbidden` bảo thủ: phần test của nó gồm cả các hàm dùng lại mà code sau mốc cắt "
        "(`new_target_fns` của cặp test), các hàm sinh đôi, gần trùng và chứa trọn sau mốc cắt của đáp án test.",
        "- Hàm gần trùng dưới ngưỡng 0,90 hoặc có chuỗi so sánh ngắn hơn 60 ký tự, và hàm chứa dưới 90% một đáp án (hoặc đáp "
        "án ngắn hơn 200 ký tự), không bị gộp. Thành viên không phải step (hàm phụ trợ, thuộc tính) không nằm trong `forbidden`; "
        "Giai đoạn 3 ẩn 'hạ tầng đáp án test' ở tầng compose.",
        "- Phần còn sót, chỉ mô tả, không loại (hàm mà train/dev chạm tới: đích, step được gắn hoặc snapshot): bản ghi test có "
        f"đáp án với hàm sau mốc cắt ratio 0,85–<0,90: {records_touching(recs['test'], touched_post, True)}; với hàm không sau mốc "
        f"cắt ratio ≥ 0,90: {records_touching(recs['test'], touched_pre, True)}; bản ghi dev có đáp án với hàm mà train chạm tới "
        f"ratio ≥ 0,90: {records_touching(recs['dev'], touched_dev, False)}; bản ghi aux_test với hàm train/dev chạm tới ratio ≥ "
        f"0,90: {records_touching(aux, touched_aux, False)}. Chi phí các luật gộp so với bản ngay trước (theo lượt chạy): gần trùng "
        "(5c) 14 kịch bản / 15 cặp train/dev, chứa trọn (5d) 2 kịch bản dev; duy nhất một kịch bản mất vì idiom: Sylius "
        "`browsing_currencies.feature:12` (2 cặp, ratio đúng 0,9000).",
        "- Tỷ lệ train cùng repo mà `neighbours` loại khỏi pool ví dụ của một bản ghi train (trung vị / tối đa): "
        + ", ".join(f"{repo.split('/')[1]} {statistics.median(v):.0%} / {max(v):.0%}" for repo, v in sorted(pool_share.items()))
        + "; hàm CREATE được nhiều bản ghi train cùng CREATE nhất: "
        + ", ".join(f"{repo.split('/')[1]} {hub[repo].most_common(1)[0][1]}" for repo in sorted(hub)) + " bản ghi.",
        f"- Split KHÔNG theo thời gian (phương án C): train có {len(train_post)} bản ghi sau mốc cắt, "
        f"{sum(r['scenario_time'] > first_test[r['repo']] for r in train_post)} bản muộn hơn kịch bản test sớm nhất cùng repo. "
        "`forbidden` chỉ ẩn hàm step, không ẩn hàm phụ trợ hay thuộc tính được thêm cùng đáp án test.",
        "- `check_dataset.py` là kiểm lại độc lập về LUẬT, dựa trên catalog: nó tự tính lại mọi quy tắc nhưng tin thời điểm "
        "annotation, suite, scope, catch-all (`catalog.jsonl`) và binding step (`labels.jsonl`). Bằng chứng cho chính các thời "
        "điểm là lượt chạy lại toàn pipeline ra đúng từng byte và review Task 2–3. Một số kiểm là CHỨNG MINH CÓ TÊN, suy ra "
        "được từ kiểm khác, không phải bộ phát hiện độc lập: `test_answer_twin_in_training`, `near_duplicate_in_training` và "
        "`test_answer_container_in_training` báo cùng `forbidden_file_mismatch`; `neighbours_asymmetric` kéo theo `neighbours_wrong`; "
        "`train_neighbour_of_other_partition` suy ra từ các kiểm `file_shared_*` và `*_answer`.",
        "- Hai kiểm `train_dev_uses_forbidden` và `train_dev_answer_code_equals_test` của check_dataset áp cho mọi phần khác "
        "test, kể cả aux_test (tên kiểm nói hẹp hơn phạm vi thật)."]
fence = "`" * 3  # never write three backticks literally: the plan embeds this file in a Markdown code fence
out += ["", "## Cấu hình", "", fence + "json", json.dumps(split["config"], indent=2, ensure_ascii=False), fence, ""]
(args.phase2 / "DATASET.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
