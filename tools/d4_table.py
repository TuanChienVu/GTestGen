"""D4 (3D, P18; v2tools/sap.select_dev_config): the dev configuration table, Markdown in Vietnamese.

Rule: keep the baseline dev_pilot_v0 unless another configuration's score is at least margin (3) above it; a smaller
difference is a tie and keeps the baseline; among several above, the highest score wins, then the first name in STRING
order (dev_grid_k12_… before dev_grid_k4_…), as the locked code does. A score is the sum over dev scenarios of scenario
CA at Full-aware, DeepSeek (records excluded from CA are left out; the mean over seeds, one seed here).

  d4_table.py                         the empty table (27 rows: dev_pilot_v0 and the 26 grid labels)
  d4_table.py DEV_SELECTION.json      the filled table; re-derives the choice and refuses (exit 3) if it differs"""
import json
import sys

GRID = [f"dev_grid_k{k}_n{n}_r{r}" for k in (4, 8, 12) for n in (1, 2, 3) for r in ("033", "050", "067")
        if (k, n, r) != (8, 3, "050")]  # k8_n3_r050 is the v0 configuration itself (k 8, 3 examples, ratio 0.5)
BASE = "dev_pilot_v0"


def params(v: str) -> tuple[str, str, str]:
    if v == BASE:
        return "8", "3", "0,50 (v0)"
    k, n, r = v.removeprefix("dev_grid_").split("_")
    return k[1:], n[1:], {"r033": "0,33", "r050": "0,50", "r067": "0,67"}[r]


def choose(scores: dict, base: str, margin: float) -> str:
    over = [v for v in scores if scores[v] >= scores[base] + margin]
    return min(over, key=lambda v: (-scores[v], v)) if over else base


def main() -> int:
    sel = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else None
    scores = sel["scores"] if sel else {}
    margin = sel["margin"] if sel else 3
    base = scores.get(BASE)
    lines = ["# D4: chọn cấu hình compose ở dev (P18, §4.5)", "",
             "Điểm = tổng CA theo kịch bản ở Full-aware (DeepSeek, seed 42) trên dev, bỏ bản ghi bị loại khỏi CA. "
             f"Giữ `{BASE}` trừ khi một cấu hình hơn nó ít nhất {margin} điểm; chênh nhỏ hơn là hoà. Nhiều cấu hình "
             "cùng vượt: lấy điểm cao nhất, rồi theo tên, tức thứ tự CHUỖI của nhãn (`dev_grid_k12_…` đứng trước "
             "`dev_grid_k4_…`), đúng như mã đã khoá (`select_dev_config`, P18). Nguồn: "
             "`out/phase3/sap/DEV_SELECTION.json` (`scripts/sap_dev_select.py`).", "",
             "| cấu hình | k | n | tỷ lệ | điểm | chênh so với v0 | ≥ ngưỡng | thứ hạng |", "|---|---|---|---|---|---|---|---|"]
    order = sorted(scores, key=lambda v: (-scores[v], v)) if sel else []
    for v in [BASE, *GRID]:
        k, n, r = params(v)
        if sel and v in scores:
            d = scores[v] - base
            lines.append(f"| `{v}` | {k} | {n} | {r} | {scores[v]:g} | {d:+g} | {'có' if v != BASE and d >= margin else '–'} "
                         f"| {order.index(v) + 1} |")
        else:
            lines.append(f"| `{v}` | {k} | {n} | {r} | {'(không có dòng)' if sel else ''} |  |  |  |")
    if sel:
        extra = sorted(set(scores) - {BASE, *GRID})
        mine = choose(scores, BASE, margin)
        lines += ["", f"Lựa chọn của `sap_dev_select`: **`{sel['choice']}`**; suy lại theo P18: `{mine}`"
                  f" ({'khớp' if mine == sel['choice'] else 'LỆCH'}).", f"Biến thể ngoài danh sách 27: {extra or 'không có'}."]
        print("\n".join(lines))
        return 0 if mine == sel["choice"] else 3
    lines += ["", "Lựa chọn: … (điền từ DEV_SELECTION.json)"]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
