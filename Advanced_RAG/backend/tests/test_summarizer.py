import unittest
from unittest.mock import patch, MagicMock
from src.summarizer import Summarizer, SummarizationError

class TestSummarizer(unittest.TestCase):
    @patch('src.summarizer.pipeline')
    def test_abstractive_summarization_success(self, mock_pipeline):
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = [{'summary_text': 'Summary text.'}]
        mock_pipeline.return_value = mock_pipeline_instance
        summarizer = Summarizer('backend/config/config.yaml')
        summary = summarizer.summarize(["Document 1", "Document 2"])
        self.assertEqual(summary, "Summary text.")

    @patch('src.summarizer.openai.Completion.create')
    def test_abstractive_summarization_success_openai(self, mock_create):
        mock_create.return_value = {'choices': [{'text': ' Summary text.'}]}
        summarizer = Summarizer('backend/config/config.yaml')  # Assuming config set to openai
        summary = summarizer.summarize(["Document 1", "Document 2"])
        self.assertEqual(summary, "Summary text.")

    @patch('src.summarizer.openai.Completion.create')
    def test_abstractive_summarization_failure_openai(self, mock_create):
        mock_create.side_effect = Exception("API Error")
        summarizer = Summarizer('backend/config/config.yaml')  # Assuming config set to openai
        with self.assertRaises(SummarizationError):
            summarizer.summarize(["Document 1", "Document 2"])

    def test_extractive_summarization(self):
        summarizer = Summarizer('backend/config/config.yaml')
        summarizer.method = 'extractive'
        summarizer.extractive_summarizer = MagicMock()
        summarizer.extractive_summarizer.summarize.return_value = "Extracted summary."
        summary = summarizer.summarize(["Doc1", "Doc2", "Doc3", "Doc4"])
        self.assertEqual(summary, "Extracted summary.")
