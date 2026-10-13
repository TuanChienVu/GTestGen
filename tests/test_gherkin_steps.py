from v2tools.gherkin_steps import parse_feature

FEATURE = '''@tag
Feature: Cart
  Background:
    Given the store operates on a single channel

  Scenario: Adding a product
    Given there is a product "T-Shirt" priced at "$10.00"
    And I added it to the cart
    When I check out
    Then I should see "Thank you"
    But I should not see an error
    """
    free text with Given inside
    """

  Scenario Outline: Discount <rate>
    Given a promotion of <rate> percent
    Then total is <total>
    Examples:
      | rate | total |
      | 10   | 90    |
      | 20   | 80    |
'''

def test_background_and_keyword_resolution():
    s = parse_feature(FEATURE)
    assert [x.name for x in s] == ["Adding a product", "Discount <rate>"]
    first = s[0]
    assert [st.keyword for st in first.steps] == ["Given", "Given", "Given", "When", "Then", "Then"]
    assert first.steps[0].from_background is True and first.steps[1].from_background is False
    assert first.steps[2].text == "I added it to the cart"

def test_docstring_is_not_a_step():
    assert all("free text" not in st.text for st in parse_feature(FEATURE)[0].steps)

def test_outline_expanded_with_first_example_row():
    outline = parse_feature(FEATURE)[1]
    assert outline.is_outline and outline.expanded
    assert [st.text for st in outline.steps if not st.from_background] == ["a promotion of 10 percent", "total is 90"]

TAGGED = '''@managing_countries
Feature: Countries
  Background:
    Given I am logged in

  @api @ui
  Scenario: Adding country
    When I add it

  Scenario Outline: Outline <x>
    Then x is <x>

    @examples_tag
    Examples:
      | x |
      | 1 |

    @other_tag
    Examples:
      | x |
      | 2 |

  @rule_tag
  Rule: some rule
    @rule_scn
    Scenario: In rule
      Then ok
'''

def test_tags_are_inherited_from_feature_rule_scenario_and_first_examples():
    s = parse_feature(TAGGED)
    assert [x.tags for x in s] == [["@managing_countries", "@api", "@ui"],
                                   ["@managing_countries", "@examples_tag"],
                                   ["@managing_countries", "@rule_tag", "@rule_scn"]]
    assert s[1].examples_tags == [["@examples_tag"], ["@other_tag"]] and s[0].examples_tags == []

COMMENTED_TAGS = '''Feature: F
  @api #@ui @x TODO: later
  Scenario: S
    Given ok
'''

def test_tag_line_comment_is_not_parsed_as_tags():
    s = parse_feature(COMMENTED_TAGS)
    assert s[0].tags == ["@api"]
