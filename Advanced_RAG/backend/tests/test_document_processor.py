import unittest
from unittest.mock import patch, MagicMock
from src.document_processor import DocumentProcessor
from src.document_indexer import DocumentIndexer
from src.embedder import Embedder
from src.vector_db import VectorDB

class TestDocumentProcessor(unittest.TestCase):
    @patch('src.document_processor.DocumentIndexer')
    @patch('src.document_processor.Embedder')
    @patch('src.document_processor.VectorDB')
    @patch('src.document_processor.setup_logging')
    def test_process_and_index(self, mock_logging, mock_vector_db, mock_embedder, mock_indexer):
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_embedder_instance.embed.return_value = [0.1, 0.2, 0.3]

        mock_vector_db_instance = MagicMock()
        mock_vector_db.return_value = mock_vector_db_instance

        mock_indexer_instance = MagicMock()
        mock_indexer.return_value = mock_indexer_instance

        mock_logger = MagicMock()
        mock_logging.return_value = mock_logger

        processor = DocumentProcessor('backend/config/config.yaml')
        processor.process_and_index("This is a test document for chunking and indexing.")

        # Check if embed was called
        self.assertTrue(mock_embedder_instance.embed.called)

        # Check if insert was called in vector DB
        self.assertTrue(mock_vector_db_instance.collection.insert.called)

        # Check if document was indexed in Elasticsearch
        self.assertTrue(mock_indexer_instance.index_document.called)

        # Check logging
        mock_logger.info.assert_any_call(f"Processing document ID: {unittest.mock.ANY}")
        mock_logger.info.assert_any_call(f"Document ID: {unittest.mock.ANY} indexed successfully.")
