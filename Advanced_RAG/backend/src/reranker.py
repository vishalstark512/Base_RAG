from sentence_transformers import SentenceTransformer, util
import torch
from src.utils import setup_logging
from typing import List, Dict, Any
from prometheus_client import Counter, Histogram
import time
import numpy as np

logger = setup_logging()

# Prometheus metrics
RERANK_COUNTER = Counter('documents_reranked_total', 'Total number of documents reranked')
RERANK_ERRORS = Counter('reranking_errors_total', 'Total number of reranking errors')
RERANK_LATENCY = Histogram('reranking_latency_seconds', 'Reranking latency in seconds')

class Reranker:
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get('model_name', 'msmarco-distilbert-base-v4')
        self.model = SentenceTransformer(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.cache = {}
        logger.info(f"Reranker initialized with model: {self.model_name} on device: {self.device}")

    def rerank(self, documents: List[Dict[str, Any]], query: str, query_type: str, top_k: int = 10) -> List[Dict[str, Any]]:
        start_time = time.time()
        try:
            if not documents:
                logger.warning("No documents to rerank.")
                return []

            query_embedding = self._get_embedding(query)
            doc_embeddings = self._get_embeddings([doc['content'] for doc in documents])

            similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
            top_k = min(top_k, len(documents))  # Ensure top_k doesn't exceed number of documents
            top_results = torch.topk(similarities, k=top_k)
            
            reranked_docs = [
                {**documents[idx.item()], 'score': score.item()}
                for score, idx in zip(top_results.values, top_results.indices)
            ]

            logger.info(f"Reranked {len(reranked_docs)} documents for query type: {query_type}")
            RERANK_COUNTER.inc()
            return reranked_docs
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}", exc_info=True)
            RERANK_ERRORS.inc()
            return documents  # Return original documents if reranking fails
        finally:
            RERANK_LATENCY.observe(time.time() - start_time)

    def rerank_with_diversity(self, documents: List[Dict[str, Any]], query: str, query_type: str, top_k: int = 10, diversity_factor: float = 0.5) -> List[Dict[str, Any]]:
        start_time = time.time()
        try:
            initial_ranking = self.rerank(documents, query, query_type, top_k=len(documents))
            selected_docs = []
            remaining_docs = initial_ranking.copy()

            while len(selected_docs) < top_k and remaining_docs:
                if not selected_docs:
                    selected_docs.append(remaining_docs.pop(0))
                else:
                    diversity_scores = self._calculate_diversity_scores(selected_docs, remaining_docs)
                    combined_scores = [
                        (1 - diversity_factor) * doc['score'] + diversity_factor * div_score
                        for doc, div_score in zip(remaining_docs, diversity_scores)
                    ]
                    best_index = combined_scores.index(max(combined_scores))
                    selected_docs.append(remaining_docs.pop(best_index))

            logger.info(f"Reranked {len(selected_docs)} documents with diversity for query type: {query_type}")
            RERANK_COUNTER.inc()
            return selected_docs
        except Exception as e:
            logger.error(f"Error during diversity-aware reranking: {str(e)}", exc_info=True)
            RERANK_ERRORS.inc()
            return self.rerank(documents, query, query_type, top_k)  # Fallback to standard reranking
        finally:
            RERANK_LATENCY.observe(time.time() - start_time)

    def _calculate_diversity_scores(self, selected_docs: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> np.ndarray:
        selected_embeddings = self._get_embeddings([doc['content'] for doc in selected_docs])
        candidate_embeddings = self._get_embeddings([doc['content'] for doc in candidates])
        
        similarities = util.pytorch_cos_sim(candidate_embeddings, selected_embeddings)
        diversity_scores = 1 - similarities.max(dim=1).values
        
        return diversity_scores.cpu().numpy()

    def _get_embedding(self, text: str) -> torch.Tensor:
        if text in self.cache:
            return self.cache[text]
        embedding = self.model.encode(text, convert_to_tensor=True, device=self.device)
        self.cache[text] = embedding
        return embedding

    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        embeddings = []
        texts_to_encode = []
        indices_to_encode = []

        for i, text in enumerate(texts):
            if text in self.cache:
                embeddings.append(self.cache[text])
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)

        if texts_to_encode:
            new_embeddings = self.model.encode(texts_to_encode, convert_to_tensor=True, device=self.device)
            for text, embedding in zip(texts_to_encode, new_embeddings):
                self.cache[text] = embedding

            for i, embedding in zip(indices_to_encode, new_embeddings):
                embeddings.insert(i, embedding)

        return torch.stack(embeddings)

    def clear_cache(self):
        self.cache.clear()
        logger.info("Reranker cache cleared")

if __name__ == '__main__':
    # Test the Reranker
    config = {'model_name': 'msmarco-distilbert-base-v4'}
    reranker = Reranker(config)
    test_docs = [
        {'id': 1, 'content': 'This is a test document about AI.'},
        {'id': 2, 'content': 'Another document discussing machine learning.'},
        {'id': 3, 'content': 'A document about natural language processing.'},
        {'id': 4, 'content': 'This document is about computer vision.'},
        {'id': 5, 'content': 'A document discussing robotics and automation.'}
    ]
    test_query = "What is AI and machine learning?"
    
    print("Standard Reranking:")
    reranked = reranker.rerank(test_docs, test_query, 'informational')
    for doc in reranked:
        print(f"ID: {doc['id']}, Score: {doc['score']:.4f}, Content: {doc['content']}")

    print("\nDiversity-aware Reranking:")
    diverse_reranked = reranker.rerank_with_diversity(test_docs, test_query, 'informational')
    for doc in diverse_reranked:
        print(f"ID: {doc['id']}, Score: {doc['score']:.4f}, Content: {doc['content']}")

    reranker.clear_cache()