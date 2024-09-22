from src.vector_db import VectorDB
from sentence_transformers import SentenceTransformer
from src.utils import setup_logging
from typing import List, Dict, Any
from prometheus_client import Counter, Histogram
import time
import numpy as np
from functools import lru_cache

logger = setup_logging()

# Prometheus metrics
RETRIEVAL_COUNTER = Counter('documents_retrieved_total', 'Total number of documents retrieved')
RETRIEVAL_ERRORS = Counter('retrieval_errors_total', 'Total number of retrieval errors')
RETRIEVAL_LATENCY = Histogram('retrieval_latency_seconds', 'Retrieval latency in seconds')

class Retriever:
    def __init__(self, config: Dict[str, Any]):
        self.vector_db = VectorDB(config['vector_db'])
        self.model = SentenceTransformer(config.get('model_name', 'all-MiniLM-L6-v2'))
        self.top_k = config.get('top_k', 10)
        self.use_hybrid = config.get('use_hybrid', False)
        self.alpha = config.get('alpha', 0.5)  # Weight for balancing vector and keyword search
        logger.info(f"Retriever initialized with model: {self.model.get_model_name()}")

    @lru_cache(maxsize=1000)
    def _encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(query)

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        start_time = time.time()
        try:
            top_k = top_k or self.top_k
            results = self.hybrid_retrieve(query, top_k) if self.use_hybrid else self.vector_retrieve(query, top_k)
            
            logger.info(f"Retrieved {len(results)} documents for query: {query}")
            RETRIEVAL_COUNTER.inc(len(results))
            return results
        except Exception as e:
            logger.error(f"Error during document retrieval: {str(e)}", exc_info=True)
            RETRIEVAL_ERRORS.inc()
            return []
        finally:
            RETRIEVAL_LATENCY.observe(time.time() - start_time)

    def vector_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_vector = self._encode_query(query)
        return self.vector_db.search(query_vector, top_k)

    def hybrid_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_vector = self._encode_query(query)
        vector_results = self.vector_db.search(query_vector, top_k)
        keyword_results = self.vector_db.keyword_search(query, top_k)
        combined_results = self._combine_results(vector_results, keyword_results)
        logger.info(f"Hybrid retrieval returned {len(combined_results)} documents for query: {query}")
        return combined_results[:top_k]

    def _combine_results(self, vector_results: List[Dict[str, Any]], keyword_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        combined = {}
        for result in vector_results:
            combined[result['id']] = {'score': self.alpha * result['score'], 'content': result['content']}
        
        for result in keyword_results:
            if result['id'] in combined:
                combined[result['id']]['score'] += (1 - self.alpha) * result['score']
            else:
                combined[result['id']] = {'score': (1 - self.alpha) * result['score'], 'content': result['content']}
        
        return sorted(
            [{'id': k, 'content': v['content'], 'score': v['score']} for k, v in combined.items()],
            key=lambda x: x['score'],
            reverse=True
        )

    def retrieve_with_facets(self, query: str, facets: Dict[str, Any], top_k: int = None) -> List[Dict[str, Any]]:
        start_time = time.time()
        try:
            top_k = top_k or self.top_k
            query_vector = self._encode_query(query)
            results = self.vector_db.search_with_facets(query_vector, facets, top_k)
            logger.info(f"Retrieved {len(results)} documents for query: {query} with facets: {facets}")
            RETRIEVAL_COUNTER.inc(len(results))
            return results
        except Exception as e:
            logger.error(f"Error during faceted retrieval: {str(e)}", exc_info=True)
            RETRIEVAL_ERRORS.inc()
            return []
        finally:
            RETRIEVAL_LATENCY.observe(time.time() - start_time)

if __name__ == "__main__":
    # Test the Retriever
    config = {
        'vector_db': {
            'url': 'http://localhost:6333',
            'collection_name': 'documents',
            'dimension': 384
        },
        'model_name': 'all-MiniLM-L6-v2',
        'top_k': 5,
        'use_hybrid': True,
        'alpha': 0.7
    }
    retriever = Retriever(config)
    
    test_query = "What is machine learning?"
    results = retriever.retrieve(test_query)
    print("Retrieval results:")
    for result in results:
        print(f"ID: {result['id']}, Score: {result['score']:.4f}")

    faceted_results = retriever.retrieve_with_facets(test_query, {'category': 'technology'})
    print("\nFaceted search results:")
    for result in faceted_results:
        print(f"ID: {result['id']}, Score: {result['score']:.4f}")
