import pytest

from v2tools.scenario_text import scenario_block

FEATURE = '''@shop
Feature: Cart
  Background:
    Given a store

  @api
  Scenario: add with table
    When I add:
      | name | qty |
      | Mug  | 2   |
    Then I see
      """
      Scenario: not a header
      """

  @ui
  Scenario Outline: outline <x>
    Then x is <x>

    @first
    Examples:
      | x |
      | 1 |

  Rule: discounts
    Background:
      Given a promo

    Scenario: in rule
      Then ok
'''

def test_scenario_block_keeps_tags_tables_docstrings_and_examples():
    b = scenario_block(FEATURE, 7)
    assert b["scenario"].splitlines()[0] == "@api"
    assert "| Mug  | 2   |" in b["scenario"] and "Scenario: not a header" in b["scenario"]
    assert not b["scenario"].rstrip().endswith("@ui")
    assert b["background"] == "Background:\n  Given a store"
    outline = scenario_block(FEATURE, 17)
    assert "@first" in outline["scenario"] and outline["scenario"].endswith("| 1 |")

def test_rule_background_follows_feature_background():
    r = scenario_block(FEATURE, 29)
    assert r["background"] == "Background:\n  Given a store\n\nBackground:\n  Given a promo"
    assert r["scenario"] == "Scenario: in rule\n  Then ok"

def test_rejects_a_line_that_is_not_a_scenario_header():
    with pytest.raises(ValueError):
        scenario_block(FEATURE, 2)
