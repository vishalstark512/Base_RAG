import yaml
import os
import logging
from logging.handlers import RotatingFileHandler
from pydantic import BaseModel, ValidationError
from typing import Dict, Any, List
import hashlib
from prometheus_client import Counter

# Prometheus metrics
CONFIG_ERRORS = Counter('config_errors_total', 'Total number of configuration errors')

class ConfigError(Exception):
    pass

class EmbeddingError(Exception):
    pass

class DatabaseError(Exception):
    pass

class SummarizationError(Exception):
    pass

class QueryRewritingError(Exception):
    pass

class IndexingError(Exception):
    pass

class LLMConfig(BaseModel):
    type: str
    api_key: str
    model: str

class EmbeddingsConfig(BaseModel):
    type: str
    api_key: str = None
    model: str

class DatabaseConfig(BaseModel):
    type: str
    host: str
    port: int
    collection_name: str

class ChunkingConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int

class ConfigSchema(BaseModel):
    llm: LLMConfig
    embeddings: EmbeddingsConfig
    database: DatabaseConfig
    chunking: ChunkingConfig
    reranking: str
    summarization: str

def load_config(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Replace placeholders with environment variables
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str) and sub_value.startswith("${") and sub_value.endswith("}"):
                        env_var = sub_value[2:-1]
                        config[key][sub_key] = os.getenv(env_var)
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                config[key] = os.getenv(env_var)
        
        return config
    except Exception as e:
        CONFIG_ERRORS.inc()
        raise ConfigError(f"Error loading configuration: {str(e)}")

def load_and_validate_config(file_path: str) -> ConfigSchema:
    try:
        config = load_config(file_path)
        validated_config = ConfigSchema(**config)
        return validated_config
    except ValidationError as e:
        CONFIG_ERRORS.inc()
        raise ConfigError(f"Invalid configuration: {e}")

def setup_logging(log_file: str = 'rag_system.log', log_level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler with rotation
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    fh = RotatingFileHandler(os.path.join(log_dir, log_file), maxBytes=10*1024*1024, backupCount=5)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
    return chunks

def generate_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def sanitize_filename(filename: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def truncate_text(text: str, max_length: int, suffix: str = '...') -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def format_summary_with_citations(summary: str, citations: List[Dict[str, Any]]) -> str:
    formatted_summary = summary
    offset = 0
    for citation in sorted(citations, key=lambda x: x['start']):
        citation_marker = f"[{citation['document_id']}]"
        insert_position = citation['end'] + offset
        formatted_summary = formatted_summary[:insert_position] + citation_marker + formatted_summary[insert_position:]
        offset += len(citation_marker)
    return formatted_summary

if __name__ == "__main__":
    # Test the utility functions
    config = load_and_validate_config('config.yaml')
    print("Configuration loaded and validated successfully.")
    
    logger = setup_logging()
    logger.info("Logging set up successfully.")
    
    test_text = "This is a test text that will be chunked into smaller pieces."
    chunks = chunk_text(test_text, chunk_size=20, chunk_overlap=5)
    print("Text chunks:", chunks)
    
    hash_value = generate_hash(test_text)
    print("Hash value:", hash_value)
    
    sanitized_filename = sanitize_filename("invalid:file*name.txt")
    print("Sanitized filename:", sanitized_filename)
    
    truncated_text = truncate_text("This is a very long text that needs to be truncated.", 20)
    print("Truncated text:", truncated_text)

    test_summary = "This is a summary. It contains information from multiple sources."
    test_citations = [
        {"start": 5, "end": 15, "document_id": "doc1"},
        {"start": 40, "end": 60, "document_id": "doc2"}
    ]
    formatted_summary = format_summary_with_citations(test_summary, test_citations)
    print("Formatted summary with citations:", formatted_summary)