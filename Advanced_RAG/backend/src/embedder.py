import openai
import cohere
from src.utils import load_config, EmbeddingError, setup_logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import List, Union, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from prometheus_client import Counter, Histogram
import functools
import time

logger = setup_logging()

# Prometheus metrics
EMBEDDING_COUNTER = Counter('embeddings_generated_total', 'Total number of embeddings generated', ['model'])
EMBEDDING_ERRORS = Counter('embedding_errors_total', 'Total number of embedding errors', ['model'])
EMBEDDING_LATENCY = Histogram('embedding_latency_seconds', 'Embedding generation latency in seconds', ['model'])

def timeit(method):
    @functools.wraps(method)
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        duration = te - ts
        EMBEDDING_LATENCY.labels(model=args[0].type).observe(duration)
        return result
    return timed

class Embedder:
    def __init__(self, config: Dict[str, Any]):
        try:
            self.type = config['type']
            if self.type == 'openai':
                openai.api_key = config['api_key']
                self.model = config['model']
            elif self.type == 'cohere':
                self.cohere_client = cohere.Client(config['api_key'])
                self.model = config['model']
            elif self.type == 'sentence_transformer':
                self.model = SentenceTransformer(config['model'])
            else:
                raise EmbeddingError(f"Embedder type {self.type} not implemented.")
            
            self.cache = {}  # Simple in-memory cache
            self.cache_size = config.get('cache_size', 10000)
            logger.info(f"Embedder initialized with type: {self.type}")
        except KeyError as e:
            logger.error(f"Missing configuration key: {e}")
            raise EmbeddingError(f"Missing configuration key: {e}")
        except Exception as e:
            logger.error(f"Error initializing Embedder: {str(e)}", exc_info=True)
            raise EmbeddingError(f"Error initializing Embedder: {str(e)}")

    @retry(stop=stop_after_attempt(5), 
           wait=wait_exponential(multiplier=1, min=4, max=10),
           retry=retry_if_exception_type(EmbeddingError))
    @timeit
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        try:
            # Check cache first
            if isinstance(text, str) and text in self.cache:
                return self.cache[text]
            
            if self.type == 'openai':
                response = openai.Embedding.create(input=text, model=self.model)
                embeddings = [data['embedding'] for data in response['data']]
            elif self.type == 'cohere':
                response = self.cohere_client.embed(texts=[text] if isinstance(text, str) else text, model=self.model)
                embeddings = response.embeddings
            elif self.type == 'sentence_transformer':
                embeddings = self.model.encode(text).tolist()
            else:
                raise EmbeddingError(f"Embedder type {self.type} not implemented.")
            
            EMBEDDING_COUNTER.labels(model=self.type).inc(len(embeddings) if isinstance(embeddings[0], list) else 1)
            
            # Cache the result if it's a single string
            if isinstance(text, str):
                self._add_to_cache(text, embeddings[0])
                return embeddings[0]
            return embeddings
        except (openai.error.OpenAIError, cohere.error.CohereError) as e:
            logger.error(f"Embedding generation error: {e}", exc_info=True)
            EMBEDDING_ERRORS.labels(model=self.type).inc()
            raise EmbeddingError(f"Embedding generation error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in embed method: {str(e)}", exc_info=True)
            EMBEDDING_ERRORS.labels(model=self.type).inc()
            raise EmbeddingError(f"Unexpected error in embed method: {str(e)}")

    def batch_embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embed(batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def _add_to_cache(self, text: str, embedding: List[float]):
        if len(self.cache) >= self.cache_size:
            # Remove the oldest item if cache is full
            self.cache.pop(next(iter(self.cache)))
        self.cache[text] = embedding

    def clear_cache(self):
        self.cache.clear()
        logger.info("Embedder cache cleared")

if __name__ == "__main__":
    # Test the Embedder
    config = {
        'type': 'sentence_transformer',
        'model': 'all-MiniLM-L6-v2',
        'cache_size': 1000
    }
    
    embedder = Embedder(config)
    
    # Test single string embedding
    test_text = "This is a test sentence for embedding."
    embedding = embedder.embed(test_text)
    print(f"Single embedding shape: {np.array(embedding).shape}")
    
    # Test batch embedding
    test_texts = ["This is the first test sentence.", "This is the second test sentence.", "And this is the third."]
    embeddings = embedder.batch_embed(test_texts)
    print(f"Batch embeddings shape: {np.array(embeddings).shape}")
    
    # Test caching
    cached_embedding = embedder.embed(test_text)
    print(f"Cached embedding shape: {np.array(cached_embedding).shape}")
    
    embedder.clear_cache()