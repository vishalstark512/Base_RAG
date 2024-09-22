import unittest
from unittest.mock import patch, MagicMock
from src.vector_db import VectorDB, DatabaseError

class TestVectorDB(unittest.TestCase):
    @patch('src.vector_db.Collection')
    @patch('src.vector_db.connections.connect')
    def test_milvus_connection_success(self, mock_connect, mock_collection):
        db = VectorDB('backend/config/database.yaml')
        mock_connect.assert_called_with(host="milvus-service", port="19530")
        mock_collection.assert_called_with("rag_collection")

    @patch('src.vector_db.Collection')
    @patch('src.vector_db.connections.connect')
    def test_milvus_connection_failure(self, mock_connect, mock_collection):
        mock_connect.side_effect = Exception("Connection failed")
        with self.assertRaises(DatabaseError):
            VectorDB('backend/config/database.yaml')

    @patch('src.vector_db.Collection.search')
    def test_search_milvus_success(self, mock_search):
        mock_search.return_value = [MagicMock(entity=MagicMock(get=MagicMock(return_value="Document 1")))]
        db = VectorDB('backend/config/database.yaml')
        results = db.search([0.1, 0.2, 0.3])
        self.assertEqual(results[0][0].entity.get('text'), "Document 1")

    @patch('src.vector_db.Collection.search')
    def test_search_milvus_failure(self, mock_search):
        mock_search.side_effect = Exception("Search error")
        db = VectorDB('backend/config/database.yaml')
        with self.assertRaises(DatabaseError):
            db.search([0.1, 0.2, 0.3])

    @patch('src.vector_db.Elasticsearch')
    @patch('src.vector_db.Collection.search')
    def test_search_hybrid_success(self, mock_search, mock_elasticsearch):
        mock_es = MagicMock()
        mock_es.search.return_value = {
            'hits': {
                'hits': [
                    {'_id': 'doc1', '_source': {'text': 'Document 1'}},
                    {'_id': 'doc2', '_source': {'text': 'Document 2'}}
                ]
            }
        }
        mock_elasticsearch.return_value = mock_es
        mock_search.return_value = None  # Not used in hybrid

        db = VectorDB('backend/config/database.yaml')
        results = db.search([0.1, 0.2, 0.3], "Test query")
        self.assertEqual(results, ['Document 1', 'Document 2'])
