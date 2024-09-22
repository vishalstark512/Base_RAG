from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from src.utils import setup_logging, QueryRewritingError
import torch
from typing import List, Dict, Any
from prometheus_client import Counter, Histogram
import time

logger = setup_logging()

# Prometheus metrics
QUERY_CLASSIFICATIONS = Counter('query_classifications_total', 'Total number of query classifications')
QUERY_CLASSIFICATION_ERRORS = Counter('query_classification_errors_total', 'Total number of query classification errors')
QUERY_CLASSIFICATION_LATENCY = Histogram('query_classification_latency_seconds', 'Query classification latency in seconds')

class QueryClassifier:
    def __init__(self, config: Dict[str, Any]):
        try:
            model_name = config.get('model_name', "distilbert-base-uncased-finetuned-sst-2-english")
            self.classifier = pipeline("text-classification", model=model_name)
            self.rewriter = pipeline("text2text-generation", model="t5-base")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logger.info(f"Query classifier initialized successfully with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize query classifier: {str(e)}", exc_info=True)
            raise QueryRewritingError(f"Failed to initialize query components: {e}")

    def classify(self, query: str) -> str:
        start_time = time.time()
        try:
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            
            class_mapping = {0: "informational", 1: "navigational", 2: "transactional"}
            classification = class_mapping.get(predicted_class, "unknown")
            
            logger.info(f"Query classified as: {classification}")
            QUERY_CLASSIFICATIONS.inc()
            return classification
        except Exception as e:
            logger.error(f"Error classifying query: {str(e)}", exc_info=True)
            QUERY_CLASSIFICATION_ERRORS.inc()
            return "unknown"
        finally:
            QUERY_CLASSIFICATION_LATENCY.observe(time.time() - start_time)

    def decompose(self, query: str) -> List[str]:
        try:
            prompt = f"Decompose the following query into simpler sub-queries:\n\n{query}"
            decomposition = self.rewriter(prompt, max_length=512, min_length=10, num_return_sequences=3, num_beams=4)
            sub_queries = [item['generated_text'].strip() for item in decomposition]
            logger.info(f"Query decomposed into {len(sub_queries)} sub-queries")
            return sub_queries
        except Exception as e:
            logger.error(f"Error decomposing query: {str(e)}", exc_info=True)
            return [query]  # Return original query if decomposition fails

    def get_query_sentiment(self, query: str) -> Dict[str, Any]:
        try:
            result = self.classifier(query)[0]
            sentiment = result['label']
            confidence = result['score']
            logger.info(f"Query sentiment: {sentiment} (confidence: {confidence:.2f})")
            return {"sentiment": sentiment, "confidence": confidence}
        except Exception as e:
            logger.error(f"Error getting query sentiment: {str(e)}", exc_info=True)
            return {"sentiment": "neutral", "confidence": 0.0}

    def process_query(self, query: str) -> Dict[str, Any]:
        try:
            classification = self.classify(query)
            decomposed_queries = self.decompose(query)
            sentiment = self.get_query_sentiment(query)
            
            return {
                "original_query": query,
                "classification": classification,
                "decomposed_queries": decomposed_queries,
                "sentiment": sentiment
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "original_query": query,
                "classification": "unknown",
                "decomposed_queries": [query],
                "sentiment": {"sentiment": "neutral", "confidence": 0.0}
            }

if __name__ == "__main__":
    # Test the QueryClassifier
    config = {
        'model_name': "distilbert-base-uncased-finetuned-sst-2-english"
    }
    classifier = QueryClassifier(config)
    test_queries = [
        "What are the best restaurants in New York City?",
        "How do I reset my password?",
        "Buy the latest iPhone model"
    ]
    
    for query in test_queries:
        result = classifier.process_query(query)
        print(f"Query: {query}")
        print(f"Classification: {result['classification']}")
        print(f"Decomposed queries: {result['decomposed_queries']}")
        print(f"Sentiment: {result['sentiment']}")
        print("---")