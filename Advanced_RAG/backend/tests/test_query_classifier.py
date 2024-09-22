import unittest
from unittest.mock import patch, MagicMock
from src.query_classifier import QueryClassifier, QueryRewritingError

class TestQueryClassifier(unittest.TestCase):
    @patch('src.query_classifier.pipeline')
    def test_decompose_success(self, mock_pipeline):
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = [{'generated_text': 'Sub-query 1'}, {'generated_text': 'Sub-query 2'}]
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = QueryClassifier()
        sub_queries = classifier.decompose("Complex query about AI advancements and healthcare.")
        self.assertEqual(sub_queries, ['Sub-query 1', 'Sub-query 2'])

    @patch('src.query_classifier.pipeline')
    def test_decompose_failure(self, mock_pipeline):
        mock_pipeline.side_effect = Exception("Model error")
        classifier = QueryClassifier()
        with self.assertRaises(QueryRewritingError):
            classifier.decompose("Another complex query.")

    def test_classify(self):
        classifier = QueryClassifier()
        classification = classifier.classify("What is the capital of France?")
        self.assertEqual(classification, "informational")
