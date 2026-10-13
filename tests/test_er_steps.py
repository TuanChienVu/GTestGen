import pytest

from er.steps import effective_keywords, step_plan


def test_and_but_inherit_per_block():
    bg = "Background:\n    Given a\n    And b\n"
    sc = "@api\nScenario: s\n    When c\n    But d\n    Then e\n    And f\n"
    assert effective_keywords(bg, sc) == ["Given", "Given", "When", "When", "Then", "Then"]


def test_doc_strings_and_tables_are_not_steps():
    sc = ('Scenario: s\n    Given a doc:\n      """\n      Given not a step\n      """\n'
          "    When I have\n      | Then | x |\n    Then ok\n")
    assert effective_keywords(None, sc) == ["Given", "When", "Then"]


def test_leading_and_is_rejected():
    with pytest.raises(ValueError):
        effective_keywords(None, "Scenario: s\n    And a\n")


def test_step_plan_indices():
    rec = {"id": "r", "background": "Background:\n    Given a\n",
           "scenario": "Scenario: s\n    When b\n    And c\n    Then d\n    And e\n",
           "steps": ["a", "b", "c", "d", "e"], "step_kinds": ["reuse", "create", "reuse", "create", "reuse"]}
    p = step_plan(rec)
    assert p.act_create == (1,) and p.then_create == (3,) and p.actions == (0, 1, 2)


def test_step_plan_rejects_misaligned_record():
    rec = {"id": "r", "background": None, "scenario": "Scenario: s\n    Given a\n",
           "steps": ["a", "b"], "step_kinds": ["reuse", "create"]}
    with pytest.raises(ValueError):
        step_plan(rec)
