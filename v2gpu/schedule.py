"""Scenario A format rotation: every record is seen in exactly one prompt format per epoch, and in all three formats over
epochs 1-3 (F0 ft_only, F1 full_neutral, F2 full_aware). The start format and the order within an epoch come from a
sha256 of (seed, record id), so a schedule is reproducible from its seed alone and never depends on file order."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

FORMATS = ("F0", "F1", "F2")


@dataclass(frozen=True)
class Item:
    epoch: int
    record_id: str
    fmt: str


def _h(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def epoch_items(record_ids: list[str], seed: int, epoch: int) -> list[Item]:
    """Items of one epoch (epochs are 1-based; epochs 4 and 5 continue the rotation)."""
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("duplicate record ids")
    items = [Item(epoch, r, FORMATS[(_h(seed, "fmt", r) + epoch - 1) % len(FORMATS)]) for r in record_ids]
    items.sort(key=lambda it: (_h(seed, "order", epoch, it.record_id), it.record_id))
    return items
