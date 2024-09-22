from celery import Celery
from src.query_classifier import QueryClassifier
from src.retriever import Retriever
from src.reranker import Reranker
from src.summarizer import Summarizer
from src.prompt_rewriter import PromptRewriter
from src.utils import load_and_validate_config, setup_logging
from src.document_processor import DocumentProcessor
from src.vector_db import VectorDB
from celery.signals import worker_process_init
import time
from typing import Dict, Any, List
from prometheus_client import Counter, Histogram

# Initialize Celery
app = Celery('tasks', broker='redis://redis-service:6379/0', backend='redis://redis-service:6379/0')

# Load and validate configuration
config = load_and_validate_config('backend/config/config.yaml')

# Setup logging
logger = setup_logging()

# Prometheus metrics
TASK_COUNTER = Counter('celery_tasks_total', 'Total number of Celery tasks', ['task_name', 'status'])
TASK_LATENCY = Histogram('celery_task_latency_seconds', 'Task latency in seconds', ['task_name'])

# Initialize components
query_classifier = None
retriever = None
reranker = None
summarizer = None
prompt_rewriter = None
document_processor = None
vector_db = None

@worker_process_init.connect
def init_worker(**kwargs):
    global query_classifier, retriever, reranker, summarizer, prompt_rewriter, document_processor, vector_db
    query_classifier = QueryClassifier()
    retriever = Retriever(config['retriever'])
    reranker = Reranker(config['reranker'])
    summarizer = Summarizer(config['summarizer'])
    prompt_rewriter = PromptRewriter()
    document_processor = DocumentProcessor(config['document_processor'])
    vector_db = VectorDB(config['vector_db'])
    logger.info("Worker initialized with all components")

@app.task(bind=True, max_retries=3, retry_backoff=True)
def process_query_task(self, query: str) -> Dict[str, Any]:
    """
    Celery task to process a user query: classify, retrieve, rerank, and summarize.
    """
    start_time = time.time()
    TASK_COUNTER.labels(task_name='process_query', status='started').inc()
    
    try:
        logger.info(f"Processing query: {query}")
        
        query_type = query_classifier.classify(query)
        expanded_query = prompt_rewriter.expand(query)
        retrieved_docs = retriever.retrieve(expanded_query)
        reranked_docs = reranker.rerank(retrieved_docs, expanded_query, query_type)
        summary = summarizer.summarize(reranked_docs, query_type)
        
        process_time = time.time() - start_time
        logger.info(f"Query processed in {process_time:.2f} seconds")
        
        TASK_COUNTER.labels(task_name='process_query', status='completed').inc()
        TASK_LATENCY.labels(task_name='process_query').observe(process_time)
        
        return {
            "summary": summary,
            "query_type": query_type,
            "expanded_query": expanded_query,
            "process_time": process_time
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        TASK_COUNTER.labels(task_name='process_query', status='failed').inc()
        raise self.retry(exc=e)

@app.task(bind=True, max_retries=3, retry_backoff=True)
def index_document_task(self, document: Dict[str, Any]) -> bool:
    """
    Celery task to index a single document asynchronously.
    """
    start_time = time.time()
    TASK_COUNTER.labels(task_name='index_document', status='started').inc()
    
    try:
        logger.info(f"Indexing document: {document.get('id', 'Unknown ID')}")
        
        # Process the document
        processed_doc = document_processor.process_document(document['content'])
        
        # Generate embedding
        embedding = document_processor.generate_embedding(processed_doc)
        
        # Index the document in the vector database
        vector_db.insert([{
            "id": document['id'],
            "content": processed_doc,
            "embedding": embedding,
            "metadata": document.get('metadata', {})
        }])
        
        process_time = time.time() - start_time
        logger.info(f"Document indexed in {process_time:.2f} seconds")
        
        TASK_COUNTER.labels(task_name='index_document', status='completed').inc()
        TASK_LATENCY.labels(task_name='index_document').observe(process_time)
        
        return True
    except Exception as e:
        logger.error(f"Error indexing document: {str(e)}", exc_info=True)
        TASK_COUNTER.labels(task_name='index_document', status='failed').inc()
        raise self.retry(exc=e)

@app.task(bind=True, max_retries=3, retry_backoff=True)
def batch_index_documents_task(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Celery task to index a batch of documents asynchronously.
    """
    start_time = time.time()
    TASK_COUNTER.labels(task_name='batch_index_documents', status='started').inc()
    
    try:
        logger.info(f"Batch indexing {len(documents)} documents")
        
        processed_docs = []
        for doc in documents:
            processed_content = document_processor.process_document(doc['content'])
            embedding = document_processor.generate_embedding(processed_content)
            processed_docs.append({
                "id": doc['id'],
                "content": processed_content,
                "embedding": embedding,
                "metadata": doc.get('metadata', {})
            })
        
        # Batch insert into vector database
        vector_db.insert(processed_docs)
        
        process_time = time.time() - start_time
        logger.info(f"Batch indexed {len(documents)} documents in {process_time:.2f} seconds")
        
        TASK_COUNTER.labels(task_name='batch_index_documents', status='completed').inc()
        TASK_LATENCY.labels(task_name='batch_index_documents').observe(process_time)
        
        return {"indexed_count": len(documents), "process_time": process_time}
    except Exception as e:
        logger.error(f"Error batch indexing documents: {str(e)}", exc_info=True)
        TASK_COUNTER.labels(task_name='batch_index_documents', status='failed').inc()
        raise self.retry(exc=e)

if __name__ == '__main__':
    app.start()