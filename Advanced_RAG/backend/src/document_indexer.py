from elasticsearch import Elasticsearch, helpers
from src.utils import load_config, IndexingError, setup_logging
from typing import List, Dict, Any
import time
from prometheus_client import Counter, Histogram

logger = setup_logging()

# Prometheus metrics
INDEXING_COUNTER = Counter('documents_indexed_total', 'Total number of documents indexed')
INDEXING_ERRORS = Counter('document_indexing_errors_total', 'Total number of document indexing errors')
INDEXING_LATENCY = Histogram('document_indexing_latency_seconds', 'Document indexing latency in seconds')

class DocumentIndexer:
    def __init__(self, config: Dict[str, Any]):
        try:
            self.es = Elasticsearch([{'host': config['elasticsearch']['host'], 'port': config['elasticsearch']['port']}])
            self.index_name = config['elasticsearch']['index_name']
            self.batch_size = config['elasticsearch'].get('batch_size', 100)
            self._create_index_if_not_exists()
            logger.info(f"DocumentIndexer initialized with index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error initializing DocumentIndexer: {str(e)}", exc_info=True)
            raise IndexingError(f"Failed to initialize DocumentIndexer: {str(e)}")

    def _create_index_if_not_exists(self):
        if not self.es.indices.exists(index=self.index_name):
            try:
                self.es.indices.create(index=self.index_name, body={
                    "settings": {
                        "number_of_shards": 3,
                        "number_of_replicas": 1
                    },
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "embedding": {"type": "dense_vector", "dims": 768},  # Adjust dims based on your embedding size
                            "metadata": {"type": "object"}
                        }
                    }
                })
                logger.info(f"Created new index: {self.index_name}")
            except Exception as e:
                logger.error(f"Failed to create index {self.index_name}: {str(e)}", exc_info=True)
                raise IndexingError(f"Failed to create index: {str(e)}")

    def index_document(self, doc_id: str, text: str, embedding: List[float], metadata: Dict[str, Any] = None):
        try:
            with INDEXING_LATENCY.time():
                self.es.index(index=self.index_name, id=doc_id, body={
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata or {}
                })
            INDEXING_COUNTER.inc()
            logger.info(f"Indexed document {doc_id}")
        except Exception as e:
            INDEXING_ERRORS.inc()
            logger.error(f"Failed to index document {doc_id}: {str(e)}", exc_info=True)
            raise IndexingError(f"Failed to index document {doc_id}: {str(e)}")

    def bulk_index_documents(self, documents: List[Dict[str, Any]]) -> Tuple[int, int]:
        try:
            start_time = time.time()
            actions = [
                {
                    "_index": self.index_name,
                    "_id": doc['id'],
                    "_source": {
                        "text": doc['text'],
                        "embedding": doc['embedding'],
                        "metadata": doc.get('metadata', {})
                    }
                }
                for doc in documents
            ]
            
            success, failed = helpers.bulk(self.es, actions, chunk_size=self.batch_size, stats_only=True)
            
            end_time = time.time()
            indexing_time = end_time - start_time
            
            INDEXING_COUNTER.inc(success)
            INDEXING_ERRORS.inc(failed)
            INDEXING_LATENCY.observe(indexing_time)
            
            logger.info(f"Bulk indexed {success} documents, {failed} failed, took {indexing_time:.2f} seconds")
            
            if failed > 0:
                logger.warning(f"{failed} documents failed to index in bulk operation")
            
            return success, failed
        except Exception as e:
            logger.error(f"Bulk indexing operation failed: {str(e)}", exc_info=True)
            raise IndexingError(f"Bulk indexing operation failed: {str(e)}")

    def update_document(self, doc_id: str, update_fields: Dict[str, Any]):
        try:
            with INDEXING_LATENCY.time():
                self.es.update(index=self.index_name, id=doc_id, body={"doc": update_fields})
            logger.info(f"Updated document {doc_id}")
        except Exception as e:
            INDEXING_ERRORS.inc()
            logger.error(f"Failed to update document {doc_id}: {str(e)}", exc_info=True)
            raise IndexingError(f"Failed to update document {doc_id}: {str(e)}")

    def delete_document(self, doc_id: str):
        try:
            self.es.delete(index=self.index_name, id=doc_id)
            logger.info(f"Deleted document {doc_id}")
        except Exception as e:
            INDEXING_ERRORS.inc()
            logger.error(f"Failed to delete document {doc_id}: {str(e)}", exc_info=True)
            raise IndexingError(f"Failed to delete document {doc_id}: {str(e)}")

    def search_documents(self, query: str, size: int = 10) -> List[Dict[str, Any]]:
        try:
            response = self.es.search(index=self.index_name, body={
                "query": {
                    "match": {
                        "text": query
                    }
                },
                "size": size
            })
            return [hit['_source'] for hit in response['hits']['hits']]
        except Exception as e:
            logger.error(f"Failed to search documents: {str(e)}", exc_info=True)
            raise IndexingError(f"Failed to search documents: {str(e)}")

if __name__ == "__main__":
    # Test the DocumentIndexer
    config = {
        'elasticsearch': {
            'host': 'localhost',
            'port': 9200,
            'index_name': 'test_index'
        }
    }
    indexer = DocumentIndexer(config)
    
    # Test single document indexing
    indexer.index_document("test1", "This is a test document", [0.1, 0.2, 0.3], {"source": "test"})
    
    # Test bulk indexing
    test_docs = [
        {"id": "test2", "text": "Another test document", "embedding": [0.4, 0.5, 0.6], "metadata": {"source": "test"}},
        {"id": "test3", "text": "Yet another test document", "embedding": [0.7, 0.8, 0.9], "metadata": {"source": "test"}}
    ]
    success, failed = indexer.bulk_index_documents(test_docs)
    print(f"Bulk indexed: {success} succeeded, {failed} failed")
    
    # Test document update
    indexer.update_document("test1", {"text": "This is an updated test document"})
    
    # Test document search
    results = indexer.search_documents("test document")
    print(f"Search results: {results}")
    
    # Test document deletion
    indexer.delete_document("test1")