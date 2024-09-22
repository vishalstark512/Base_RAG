import qdrant_client
from qdrant_client.http import models as rest
from qdrant_client.http.models import Distance, VectorParams
from src.utils import setup_logging
import numpy as np
from typing import List, Dict, Any, Union
from prometheus_client import Counter, Histogram
import time
from functools import wraps

logger = setup_logging()

# Prometheus metrics
DB_OPERATIONS = Counter('vector_db_operations_total', 'Total number of vector database operations', ['operation'])
DB_ERRORS = Counter('vector_db_errors_total', 'Total number of vector database errors', ['operation'])
DB_LATENCY = Histogram('vector_db_latency_seconds', 'Vector database operation latency in seconds', ['operation'])

def db_operation(operation_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                DB_OPERATIONS.labels(operation=operation_name).inc()
                return result
            except Exception as e:
                DB_ERRORS.labels(operation=operation_name).inc()
                raise
            finally:
                DB_LATENCY.labels(operation=operation_name).observe(time.time() - start_time)
        return wrapper
    return decorator

class VectorDB:
    def __init__(self, config: Dict[str, Any]):
        self.client = qdrant_client.QdrantClient(
            url=config['url'],
            api_key=config.get('api_key')
        )
        self.collection_name = config['collection_name']
        self.dimension = config['dimension']
        self._create_collection_if_not_exists()
        logger.info(f"VectorDB initialized with collection: {self.collection_name}")

    def _create_collection_if_not_exists(self):
        try:
            collections = self.client.get_collections().collections
            if not any(collection.name == self.collection_name for collection in collections):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
                )
                logger.info(f"Created new collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}", exc_info=True)
            raise

    @db_operation("insert")
    def insert(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> int:
        try:
            total_inserted = 0
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                points = [
                    rest.PointStruct(
                        id=str(doc['id']),
                        vector=doc['embedding'],
                        payload={k: v for k, v in doc.items() if k != 'embedding'}
                    )
                    for doc in batch
                ]
                operation_info = self.client.upsert(
                    collection_name=self.collection_name,
                    wait=True,
                    points=points
                )
                total_inserted += len(batch)
                logger.info(f"Inserted batch of {len(batch)} documents. Total inserted: {total_inserted}")
            return total_inserted
        except Exception as e:
            logger.error(f"Error inserting documents: {str(e)}", exc_info=True)
            raise

    @db_operation("search")
    def search(self, query_vector: Union[List[float], np.ndarray], top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            return [
                {
                    'id': result.id,
                    'score': result.score,
                    **result.payload
                }
                for result in search_result
            ]
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}", exc_info=True)
            raise

    @db_operation("delete")
    def delete(self, document_ids: List[str]):
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest.PointIdsList(
                    points=document_ids
                )
            )
            logger.info(f"Deleted {len(document_ids)} documents")
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}", exc_info=True)
            raise

    @db_operation("search_with_filter")
    def search_with_filter(self, query_vector: Union[List[float], np.ndarray], filter_condition: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=rest.Filter(**filter_condition),
                limit=top_k
            )
            return [
                {
                    'id': result.id,
                    'score': result.score,
                    **result.payload
                }
                for result in search_result
            ]
        except Exception as e:
            logger.error(f"Error performing filtered search: {str(e)}", exc_info=True)
            raise

    @db_operation("get_collection_info")
    def get_collection_info(self) -> Dict[str, Any]:
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.name,
                "vector_size": collection_info.vectors_config.size,
                "points_count": collection_info.points_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    # Test the VectorDB
    config = {
        'url': 'http://localhost:6333',
        'collection_name': 'test_collection',
        'dimension': 384  # Dimension of 'all-MiniLM-L6-v2' embeddings
    }
    db = VectorDB(config)
    
    # Test insertion
    test_docs = [
        {
            'id': f'doc_{i}',
            'embedding': np.random.rand(384).tolist(),
            'content': f'Test document {i}',
            'metadata': {'source': 'test'}
        }
        for i in range(10)
    ]
    inserted = db.insert(test_docs)
    print(f"Inserted {inserted} documents")
    
    # Test search
    search_results = db.search(np.random.rand(384).tolist(), top_k=3)
    print(f"Search results: {search_results}")
    
    # Test filtered search
    filter_condition = {"must": [{"key": "metadata.source", "match": {"value": "test"}}]}
    filtered_results = db.search_with_filter(np.random.rand(384).tolist(), filter_condition, top_k=3)
    print(f"Filtered search results: {filtered_results}")
    
    # Test deletion
    db.delete([f'doc_{i}' for i in range(5)])
    
    # Get collection info
    collection_info = db.get_collection_info()
    print(f"Collection info: {collection_info}")