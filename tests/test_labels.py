import pytest

from v2tools.catalog import build_catalog
from v2tools.labels import label_pair
from tests.test_catalog import NEW, OLD, commit, ts

def make_catalog(tmp_path):
    repo = tmp_path / "demo"
    (repo / "src").mkdir(parents=True)
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "src" / "BrowseContext.php").write_text(OLD)
    commit(repo, "2022-01-01T00:00:00Z")
    (repo / "src" / "BrowseContext.php").write_text(NEW)
    commit(repo, "2025-01-01T00:00:00Z")
    cat = build_catalog({"repo": "x/demo", "styles": ["php_attribute"], "step_globs": ["src/*.php"]}, repo)
    return cat, {a.def_id: a.time for f in cat.fns.values() for a in f.annotations}

def row(steps, targets, when="2025-01-01"):
    return {"repo": "x/demo", "feature_path": "f.feature", "scenario_line": 2, "suite": "default", "tags": [],
            "steps": steps, "scenario_min_time": ts(when), "target_ids": sorted(targets)}

def test_step_is_reuse_only_if_its_pattern_existed_before_the_scenario(tmp_path):
    cat, times = make_catalog(tmp_path)
    r = row(["I want to browse", "I am browsing", "I see it"],
            ["src/BrowseContext.php:4", "src/BrowseContext.php:3", "src/BrowseContext.php:12"])
    lab = label_pair(r, cat, times)
    assert lab["step_kinds"] == ["reuse", "create", "create"]
    assert lab["create_def_ids"] == ["src/BrowseContext.php:12", "src/BrowseContext.php:3"]
    assert lab["create_fns"] == ["src/BrowseContext.php:14", "src/BrowseContext.php:6"]
    assert lab["reuse_fns"] == []          # browse() gets a new pattern here, so it counts as CREATE, not reuse
    older = label_pair(row(["I want to browse"], ["src/BrowseContext.php:4"], when="2023-06-01"), cat, times)
    assert older["step_kinds"] == ["reuse"] and older["reuse_fns"] == ["src/BrowseContext.php:6"]

def test_label_pair_refuses_binding_drift_and_missing_history(tmp_path):
    cat, times = make_catalog(tmp_path)
    with pytest.raises(ValueError):
        label_pair(row(["I see it"], ["src/BrowseContext.php:3"]), cat, times)
    bad = row(["I see it"], ["src/BrowseContext.php:12"])
    bad["scenario_min_time"] = None
    with pytest.raises(ValueError):
        label_pair(bad, cat, times)
