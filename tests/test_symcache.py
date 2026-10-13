import json

import pytest

from tests.phase3_fixture import SHOP, catalog, make_repo, record
from v2tools.state import RepoContext
from v2tools.symcache import build as build_symtimes, load as load_symtimes, parent_gate


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    return make_repo(tmp_path_factory.mktemp("repo"))


def test_symcache_rows_round_trip_and_the_parent_gate_counts_every_parent(repo, tmp_path):
    bare, head = repo
    cat = catalog()
    (tmp_path / "Shop.git").symlink_to(bare)
    rows = build_symtimes([{"repo": "Acme/Shop", "dir": "Shop", "language": "php"}], tmp_path, {"Acme/Shop": head},
                          list(cat.values()), [record(cat)], gate := {})
    assert {r["path"] for r in rows} == {SHOP, "src/Behat/Context/BaseContext.php"}  # the parent file is included
    assert gate["Acme/Shop"]["resolved"] == 1 and gate["Acme/Shop"]["unresolved"] == 0
    (tmp_path / "s.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    loaded = load_symtimes(tmp_path / "s.jsonl")["Acme/Shop"]
    assert loaded[SHOP]["names"]["checkout"] == next(r for r in rows if r["path"] == SHOP)["names"]["checkout"]
    ctx = RepoContext("Acme/Shop", "php", bare, head, cat)
    assert parent_gate(ctx, {SHOP})["resolved"] == 1
