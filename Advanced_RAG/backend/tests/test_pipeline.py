import unittest
from unittest.mock import patch, MagicMock
from src.main import main

class TestRAGPipeline(unittest.TestCase):
    @patch('src.main.QueryClassifier')
    @patch('src.main.Retriever')
    @patch('src.main.Reranker')
    @patch('src.main.Summarizer')
    @patch('src.main.load_and_validate_config')
    def test_pipeline_execution(self, mock_load_config, mock_summarizer, mock_reranker, mock_retriever, mock_classifier):
        mock_load_config.return_value = MagicMock(reranking="monT5")
        mock_classifier_instance = MagicMock()
        mock_classifier_instance.classify.return_value = "default"
        mock_classifier.return_value = mock_classifier_instance

        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve.return_value = ["Doc1", "Doc2"]
        mock_retriever.return_value = mock_retriever_instance

        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = ["Doc1", "Doc2"]
        mock_reranker.return_value = mock_reranker_instance

        mock_summarizer_instance = MagicMock()
        mock_summarizer_instance.summarize.return_value = "Summary text."
        mock_summarizer.return_value = mock_summarizer_instance

        with patch('builtins.print') as mock_print:
            main()
            mock_print.assert_any_call("Query classified as: default")
            mock_print.assert_any_call("Retrieved Documents: ['Doc1', 'Doc2']")
            mock_print.assert_any_call("Reranked Documents: ['Doc1', 'Doc2']")
            mock_print.assert_any_call("Summary: Summary text.")
