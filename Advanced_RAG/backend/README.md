# RAG System Backend

## Overview

The backend component of the Retrieval Augmented Generation (RAG) system handles query processing, document retrieval, reranking, summarization, and provides RESTful API endpoints for interaction.

## Features

- **Query Classification:** Determines the nature and complexity of user queries.
- **Document Retrieval:** Fetches relevant documents using vector databases.
- **Reranking:** Orders retrieved documents based on relevance.
- **Summarization:** Generates concise summaries from documents.
- **Asynchronous Processing:** Utilizes Celery for background tasks.
- **Caching:** Implements Redis caching for efficient retrieval.
- **API Integration:** Provides RESTful APIs using FastAPI.
- **Logging and Monitoring:** Comprehensive logging and integration with monitoring tools.
- **HyDE Integration:** Enhances retrieval with Hypothetical Document Embeddings.
- **Hybrid Search:** Combines keyword-based and vector-based search methods.

## Setup

### Prerequisites

- **Python 3.9+**
- **Docker** (optional, for containerization)
- **Redis**, **Milvus**, and **Elasticsearch** services (can be deployed via Kubernetes)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/rag-system.git
   cd rag-system/backend
