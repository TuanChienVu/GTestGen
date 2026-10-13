from collections import Counter

import pytest

from v2gpu.schedule import FORMATS, epoch_items

IDS = [f"repo|f{i}.feature|{i}|default" for i in range(3000)]


def test_each_record_sees_each_format_once_over_three_epochs():
    seen = {r: [] for r in IDS}
    for e in (1, 2, 3):
        for it in epoch_items(IDS, 42, e):
            seen[it.record_id].append(it.fmt)
    assert all(sorted(v) == sorted(FORMATS) for v in seen.values())


def test_extra_epochs_continue_the_rotation():
    e1 = {it.record_id: it.fmt for it in epoch_items(IDS, 42, 1)}
    e4 = {it.record_id: it.fmt for it in epoch_items(IDS, 42, 4)}
    assert e1 == e4


def test_deterministic_and_seed_dependent():
    assert epoch_items(IDS, 42, 1) == epoch_items(list(reversed(IDS)), 42, 1)
    assert [i.record_id for i in epoch_items(IDS, 42, 1)] != [i.record_id for i in epoch_items(IDS, 11, 1)]
    assert [i.record_id for i in epoch_items(IDS, 42, 1)] != [i.record_id for i in epoch_items(IDS, 42, 2)]


def test_formats_are_balanced_within_an_epoch():
    counts = Counter(it.fmt for it in epoch_items(IDS, 23, 1))
    assert all(abs(c - 1000) < 100 for c in counts.values())


def test_duplicate_ids_refused():
    with pytest.raises(ValueError):
        epoch_items(IDS[:2] + IDS[:1], 42, 1)
