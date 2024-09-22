import unittest
from unittest.mock import patch, MagicMock
from src.retriever import Retriever, RetrievalError
from src.query_classifier import QueryClassifier
import json

class TestRetriever(unittest.TestCase):
    @patch('src.retriever.openai.Completion.create')
    @patch('src.retriever.Embedder')
    @patch('src.retriever.VectorDB')
    @patch('src.retriever.redis.Redis')
    @patch.object(QueryClassifier, 'decompose', return_value=["Test sub-query"])
    def test_retrieve_with_hyde(self, mock_decompose, mock_redis, mock_vector_db, mock_embedder, mock_openai):
        # Mock Redis cache miss
        mock_cache = MagicMock()
        mock_redis.return_value = mock_cache
        mock_cache.get.return_value = None

        # Mock HyDE document generation
        mock_openai.return_value = MagicMock(choices=[MagicMock(text="Generated HyDE document.")])

        # Mock embedding
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_embedder_instance.embed.return_value = [0.1, 0.2, 0.3]

        # Mock vector DB search
        mock_search_result = [MagicMock(entity=MagicMock(get=MagicMock(return_value="Retrieved Document")))]
        mock_vector_db_instance = MagicMock()
        mock_vector_db_instance.search.return_value = mock_search_result
        mock_vector_db.return_value = mock_vector_db_instance

        retriever = Retriever('backend/config/config.yaml')
        documents = retriever.retrieve("Test query")
        self.assertEqual(documents, ["Retrieved Document"])
        mock_cache.set.assert_called_with("Test query", json.dumps(["Retrieved Document"]), ex=3600)

    @patch('src.retriever.openai.Completion.create')
    def test_generate_hyde_document_success(self, mock_openai):
        mock_openai.return_value = MagicMock(choices=[MagicMock(text="Generated HyDE document.")])
        retriever = Retriever('backend/config/config.yaml')
        hyde_doc = retriever.generate_hyde_document("Test query")
        self.assertEqual(hyde_doc, "Generated HyDE document.")

    @patch('src.retriever.openai.Completion.create')
    def test_generate_hyde_document_failure(self, mock_openai):
        mock_openai.side_effect = Exception("API Error")
        retriever = Retriever('backend/config/config.yaml')
        with self.assertRaises(RetrievalError):
            retriever.generate_hyde_document("Test query")
