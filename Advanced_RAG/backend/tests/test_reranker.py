import unittest
from src.reranker import Reranker

class TestReranker(unittest.TestCase):
    def test_rerank_default(self):
        reranker = Reranker("monT5")
        docs = ["Doc1", "Doc2", "Doc3"]
        reranked = reranker.rerank(docs)
        self.assertEqual(reranked, ["Doc1", "Doc2", "Doc3"])

    def test_rerank_with_method(self):
        reranker = Reranker("someMethod")
        docs = ["Doc1", "Doc2", "Doc3"]
        reranked = reranker.rerank(docs)
        self.assertEqual(reranked, ["Doc1", "Doc2", "Doc3"])
