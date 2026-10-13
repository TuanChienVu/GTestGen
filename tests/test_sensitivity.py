from v2tools.sensitivity import touched_answers


def test_touched_answers_flag_below_threshold_post_cutoff_and_old_copies():
    answer = "".join(chr(0x4E00 + i) for i in range(100))                  # 100 distinct characters

    def swap(k):  # replaces k spread characters (k <= 20): ratio(swap(k), answer) = (100 - k) / 100
        return "".join(chr(0x5E00 + i) if i % 5 == 2 and i // 5 < k else c for i, c in enumerate(answer))

    def touched(fn_id, k, post):
        return touched_answers({"x:1": answer}, {fn_id: (swap(k), post)}) == {"x:1"}

    assert touched("p85", 15, True) and touched("p89", 11, True)                 # post-cutoff 0.85 <= ratio < 0.90
    assert not touched("p84", 16, True) and not touched("p90", 10, True)         # 0.84: too far; 0.90: already joined
    assert touched("o90", 10, False) and not touched("o89", 11, False)          # before the cutoff: ratio >= 0.90
    assert touched_answers({"x:1": answer}, {"x:1": (answer, False)}) == set()    # the answer itself is no residual
