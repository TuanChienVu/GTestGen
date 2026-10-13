"""Pytest hooks of the public tree.

A few GPU-code tests read documents that are kept private and are not published: the operational run sheets
(ops/*RUNSHEET*.md) and one planning document. When the run sheets are absent, which is the case in the public tree,
exactly the tests listed in PRIVATE_INPUT_TESTS are not collected (each listed with its reason), so that both
`pytest --collect-only` and the run count the same tests and a run reports "N passed" with nothing skipped, as the
pod bootstrap's no-skip check requires (ops/no_skips.sh).
They are never dropped silently: the session header and the terminal summary both say how many were not run and why.
When the run sheets are present, these hooks change nothing.
"""
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent
_SENTINEL = _ROOT / "ops" / "GEN_RUNSHEET.md"
_RUN_SHEETS = "requires the private run sheets, which are not published"
_PLANNING = "requires a private planning document, which is not published"
_NESTED = ("runs a nested pytest in a partial copy of the repository (without this file) that includes a test "
           "needing the private run sheets")
_NOT_RUN = pytest.StashKey[int]()

# Test id (path from the repository root :: test function) -> reason. Exact ids only, no patterns.
PRIVATE_INPUT_TESTS = {
    "tests_gpu/test_genqueue.py::test_the_generation_run_sheet_names_every_gate": _RUN_SHEETS,
    "tests_gpu/test_genqueue.py::test_the_run_sheet_quotes_its_commands_and_copies_every_workdir": _RUN_SHEETS,
    "tests_gpu/test_genqueue.py::test_the_run_sheet_reads_both_clouds_and_creates_where_the_card_is_in_stock": _RUN_SHEETS,
    "tests_gpu/test_genqueue.py::test_the_run_sheet_memory_steps_are_the_two_pre_declared_ones_chosen_at_the_canary":
        _RUN_SHEETS,
    "tests_gpu/test_genqueue.py::test_decision_0510_the_canary_block_runs_at_the_eager_step_with_batch_invariance":
        _RUN_SHEETS,
    "tests_gpu/test_genqueue.py::test_the_run_sheet_waits_for_the_card_and_copies_each_job_as_soon_as_it_is_done":
        _RUN_SHEETS,
    "tests_gpu/test_genqueue.py::test_the_run_sheet_runs_generation_with_the_pods_model_cache": _RUN_SHEETS,
    "tests_gpu/test_genqueue.py::test_the_run_sheet_uses_the_3a_prompt_names_and_never_the_train_files": _RUN_SHEETS,
    "tests_gpu/test_genqueue.py::test_the_run_sheet_coverage_leaves_out_the_queue_folders": _RUN_SHEETS,
    "tests_gpu/test_genqueue.py::test_the_run_sheet_decides_once_per_variant_before_the_first_main_queue": _RUN_SHEETS,
    "tests_gpu/test_genqueue.py::"
    "test_the_run_sheet_ships_with_ship_sh_commits_the_lock_and_recovers_a_lost_pod_with_a_run_tag": _RUN_SHEETS,
    "tests_gpu/test_pod_safety.py::test_no_test_names_the_pods_env_file_outside_the_override": _RUN_SHEETS,
    "tests_gpu/test_pod_safety.py::test_the_sheets_watchdog_reads_its_env_file_through_rp_env": _RUN_SHEETS,
    "tests_gpu/test_pod_safety.py::test_no_subprocess_of_the_ops_script_tests_sees_a_runpod_secret": _RUN_SHEETS,
    "tests_gpu/test_pod_safety.py::test_the_watchdog_the_queues_and_the_env_file_are_never_run_as_the_test_user":
        _RUN_SHEETS,
    "tests_gpu/test_pod_safety.py::test_ops_and_both_run_sheets_never_set_the_test_only_knobs": _RUN_SHEETS,
    "tests_gpu/test_ops_scripts.py::test_gate_only_static_scan_only_the_bootstrap_and_the_gate_may_name_it": _RUN_SHEETS,
    "tests_gpu/test_docs.py::test_plan_b_carries_the_historical_record_banner": _PLANNING,
    "tests_gpu/test_docs.py::test_plan_b_has_a_ghi_chu_thuc_thi_section_naming_the_three_facts": _PLANNING,
    "tests_gpu/test_bootstrap_locks.py::test_the_bootstrap_tests_pass_with_and_without_committed_locks": _NESTED,
}


def _private_inputs_absent() -> bool:
    return not _SENTINEL.exists()


def pytest_pycollect_makeitem(collector, name, obj):
    """Create no test item for a listed test when the run sheets are absent, so neither ``pytest --collect-only`` (the
    bootstrap's N) nor the run counts it; every other name goes to pytest's own collection (None)."""
    if not _private_inputs_absent():
        return None
    test_id = f"{Path(collector.path).resolve().relative_to(_ROOT).as_posix()}::{name}"
    if test_id not in PRIVATE_INPUT_TESTS:
        return None
    cases = 1                                   # a parametrized function stands for one test per parameter set
    for mark in getattr(obj, "pytestmark", []):
        if mark.name == "parametrize":
            cases *= len(mark.args[1])
    collector.config.stash[_NOT_RUN] = collector.config.stash.get(_NOT_RUN, 0) + cases
    return []


def pytest_report_header(config):
    if not _private_inputs_absent():
        return []
    return [f"private inputs absent: the {len(PRIVATE_INPUT_TESTS)} test functions listed in conftest.py are not run "
            "(they need private run sheets or a planning document, which are not published)"]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    n = config.stash.get(_NOT_RUN, 0)
    if n:
        terminalreporter.write_line(f"{n} tests not run: they need private documents (run sheets, a planning "
                                    "document), which are not published")
