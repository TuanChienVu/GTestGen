from v2tools.patterns import compile_pattern

def match(kind, pattern, text):
    compiled = compile_pattern(kind, pattern)
    assert compiled is not None
    regex, mode = compiled
    return bool(regex.fullmatch(text) if mode == "fullmatch" else regex.search(text))

def test_turnip_placeholder_optional_and_alternation():
    p = "I set the position of :productVariant to :position"
    assert match("turnip", p, 'I set the position of "Blue T-Shirt" to 3')
    assert not match("turnip", p, 'I set the position of "Blue" to')
    assert match("turnip", "there is/are :count product(s)", "there are 5 products")
    assert match("turnip", "there is/are :count product(s)", "there is 1 product")

def test_cucumber_expression_parameters():
    p = "The journal entry table contains {int} row(s) for GL code {string}"
    assert match("cucumber", p, 'The journal entry table contains 2 rows for GL code "3100"')
    assert not match("cucumber", p, 'The journal entry table contains two rows for GL code "3100"')

def test_ruby_regex_is_searched_and_named_groups_converted():
    assert match("regex", r"Patty has a recorded clinic visit", "Patty has a recorded clinic visit today")
    assert match("regex", r"^(?<name>\w+) logs in$", "Clyde logs in")

def test_csharp_full_regex():
    assert match("regex_full", r'I set the Parameter name "(.*)" to "(.*)"', 'I set the Parameter name "a" to "b"')

def test_broken_regex_returns_none():
    assert compile_pattern("regex", "(unclosed") is None

def test_turnip_follows_behat_rules():
    # Behat compiles Turnip patterns with /iu, lets "(word)" swallow its neighbouring space, limits bare tokens to [\w.,-]
    assert match("turnip", "The default category of product :p has no group reduction",
                 'the default category of product "Mug" has no group reduction')
    assert match("turnip", "there is (also) a promotion :name", 'there is a promotion "Holiday"')
    assert match("turnip", "there is (also) a promotion :name", 'there is also a promotion "Holiday"')
    assert match("turnip", "the store is( also) available in :localeCode", 'the store is available in "English (United States)"')
    assert match("turnip", "I pay :amount", "I pay 10.50")
    assert not match("turnip", "I pay :amount", "I pay $10")
