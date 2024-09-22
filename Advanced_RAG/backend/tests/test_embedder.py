import unittest
from unittest.mock import patch, MagicMock
from src.embedder import Embedder, EmbeddingError

class TestEmbedder(unittest.TestCase):
    @patch('src.embedder.openai.Embedding.create')
    def test_openai_embedder_success(self, mock_create):
        mock_create.return_value = {'data': [{'embedding': [0.1, 0.2, 0.3]}]}
        embedder = Embedder('backend/config/embeddings.yaml')
        embedding = embedder.embed("Test text")
        self.assertEqual(embedding, [0.1, 0.2, 0.3])

    @patch('src.embedder.openai.Embedding.create')
    def test_openai_embedder_failure(self, mock_create):
        mock_create.side_effect = Exception("API Error")
        embedder = Embedder('backend/config/embeddings.yaml')
        with self.assertRaises(EmbeddingError):
            embedder.embed("Test text")

    @patch('src.embedder.cohere.Client')
    def test_cohere_embedder_success(self, mock_cohere_client):
        mock_instance = MagicMock()
        mock_instance.embed.return_value = MagicMock(embeddings=[[0.4, 0.5, 0.6]])
        mock_cohere_client.return_value = mock_instance
        embedder = Embedder('backend/config/embeddings.yaml')  # Assuming config set to cohere_embedder
        embedding = embedder.embed("Test text")
        self.assertEqual(embedding, [0.4, 0.5, 0.6])

    @patch('src.embedder.cohere.Client')
    def test_cohere_embedder_failure(self, mock_cohere_client):
        mock_cohere_client.return_value.embed.side_effect = Exception("Cohere Error")
        embedder = Embedder('backend/config/embeddings.yaml')  # Assuming config set to cohere_embedder
        with self.assertRaises(EmbeddingError):
            embedder.embed("Test text")
