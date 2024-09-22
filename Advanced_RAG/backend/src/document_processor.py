import ray
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredEmailLoader
)
from sentence_transformers import SentenceTransformer
from src.vector_db import VectorDB
from src.utils import setup_logging
import os
from typing import List, Dict, Any
from prometheus_client import Counter, Histogram
import hashlib

logger = setup_logging()

# Prometheus metrics
DOCUMENTS_PROCESSED = Counter('documents_processed_total', 'Total number of documents processed')
PROCESSING_ERRORS = Counter('document_processing_errors_total', 'Total number of document processing errors')
PROCESSING_TIME = Histogram('document_processing_time_seconds', 'Time taken to process a document')

@ray.remote
class DocumentProcessor:
    def __init__(self, vector_db_config: Dict[str, Any]):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_db = VectorDB(vector_db_config)
        logger.info("DocumentProcessor initialized")

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            loader_class = self._get_loader_class(file_extension)
            loader = loader_class(file_path)
            documents = loader.load()
            texts = self.text_splitter.split_documents(documents)
            logger.info(f"Processed {len(texts)} text chunks from {file_path}")
            return texts
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            PROCESSING_ERRORS.inc()
            return []

    def embed_and_index(self, texts: List[Dict[str, Any]], metadata: Dict[str, Any]) -> int:
        try:
            embeddings = self.embedder.encode([text.page_content for text in texts])
            documents = [
                {
                    "id": self._generate_chunk_id(metadata['file_name'], i, text.page_content),
                    "content": text.page_content,
                    "metadata": {**text.metadata, **metadata},
                    "embedding": embedding.tolist()
                }
                for i, (text, embedding) in enumerate(zip(texts, embeddings))
            ]
            self.vector_db.insert(documents)
            logger.info(f"Indexed {len(documents)} documents")
            return len(documents)
        except Exception as e:
            logger.error(f"Error embedding and indexing documents: {str(e)}", exc_info=True)
            PROCESSING_ERRORS.inc()
            return 0

    def process_and_index(self, file_path: str) -> int:
        with PROCESSING_TIME.time():
            texts = self.process_file(file_path)
            if not texts:
                return 0
            metadata = {
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "file_type": os.path.splitext(file_path)[1].lower()
            }
            indexed_count = self.embed_and_index(texts, metadata)
            DOCUMENTS_PROCESSED.inc()
            return indexed_count

    @staticmethod
    def _generate_chunk_id(file_name: str, chunk_index: int, content: str) -> str:
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{file_name}_{chunk_index}_{content_hash}"

    @staticmethod
    def _get_loader_class(file_extension: str):
        loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.md': TextLoader,
            '.ppt': UnstructuredPowerPointLoader,
            '.pptx': UnstructuredPowerPointLoader,
            '.eml': UnstructuredEmailLoader,
            '.msg': UnstructuredEmailLoader
        }
        loader_class = loaders.get(file_extension)
        if not loader_class:
            raise ValueError(f"Unsupported file type: {file_extension}")
        return loader_class

@ray.remote
def process_file_task(file_path: str, vector_db_config: Dict[str, Any]) -> int:
    processor = DocumentProcessor(vector_db_config)
    return processor.process_and_index(file_path)

def batch_process_files(file_paths: List[str], vector_db_config: Dict[str, Any], batch_size: int = 10) -> int:
    ray.init(ignore_reinit_error=True)
    total_processed = 0
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]
        futures = [process_file_task.remote(file_path, vector_db_config) for file_path in batch]
        results = ray.get(futures)
        total_processed += sum(results)
        logger.info(f"Processed batch {i//batch_size + 1}, total documents indexed: {total_processed}")
    ray.shutdown()
    return total_processed

if __name__ == "__main__":
    # Test the DocumentProcessor
    test_files = ["path/to/test1.pdf", "path/to/test2.docx", "path/to/test3.txt", "path/to/test4.pptx", "path/to/test5.eml"]
    vector_db_config = {
        "host": "localhost",
        "port": 6333,
        "collection_name": "documents"
    }
    total_indexed = batch_process_files(test_files, vector_db_config)
    print(f"Total documents indexed: {total_indexed}")