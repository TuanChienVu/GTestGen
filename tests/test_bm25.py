from v2tools.bm25 import BM25, tokens


def test_bm25_tokenises_camel_case_and_breaks_ties_by_id():
    assert tokens("iFilterByCode_x2") == ["i", "filter", "by", "code", "x", "2"]
    bm = BM25([("b", tokens("filter by code")), ("a", tokens("filter by code")), ("c", tokens("open the shop"))])
    assert [d for d, _ in bm.rank(tokens("filter code"))] == ["a", "b", "c"]
