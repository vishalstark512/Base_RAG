from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, constr, Field
from src.query_classifier import QueryClassifier
from src.retriever import Retriever
from src.reranker import Reranker
from src.summarizer import Summarizer
from src.document_processor import batch_process_files
from src.prompt_rewriter import PromptRewriter
from src.utils import load_and_validate_config, ConfigError, setup_logging
from src.celery_worker import process_query_task
from src.feedback_processor import process_feedback
from celery.result import AsyncResult
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import time
import os
import redis
import json
from typing import List
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

app = FastAPI(
    title="RAG System API",
    description="API for the Retrieval Augmented Generation system.",
    version="1.0.0"
)

# Setup logging
logger = setup_logging()

# Add middlewares
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["yourdomain.com", "localhost", "127.0.0.1"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

# API Key authentication
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ConfigError("API_KEY environment variable not set")
api_key_header = APIKeyHeader(name="X-API-Key")

# Redis configuration
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), 
                           port=int(os.getenv("REDIS_PORT", 6379)), 
                           db=int(os.getenv("REDIS_DB", 0)))

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Pydantic models
class QueryRequest(BaseModel):
    query: constr(min_length=1, max_length=1000)

class TaskStatus(BaseModel):
    task_id: str

class TaskResult(BaseModel):
    task_id: str
    status: str
    summary: str = None
    rewritten_query: str = None
    original_query: str = None

class FeedbackRequest(BaseModel):
    task_id: str
    rating: int = Field(..., ge=1, le=5)
    comments: str = Field(None, max_length=1000)

# Initialize components
config = load_and_validate_config('backend/config/config.yaml')
query_classifier = QueryClassifier()
retriever = Retriever(config['retriever'])
reranker = Reranker(config['reranker'])
summarizer = Summarizer(config['summarizer'])
prompt_rewriter = PromptRewriter()

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.post("/summarize", response_model=TaskStatus, summary="Submit a query for summarization", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
async def summarize(query_request: QueryRequest, background_tasks: BackgroundTasks, request: Request):
    try:
        rewritten_query = prompt_rewriter.rewrite(query_request.query)[0]
        task = process_query_task.delay(rewritten_query)
        background_tasks.add_task(store_query_info, task.id, query_request.query, rewritten_query)
        return TaskStatus(task_id=task.id)
    except Exception as e:
        logger.error(f"Error in summarize endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.get("/summarize/{task_id}", response_model=TaskResult, summary="Retrieve the summary for a given task ID", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def get_summary(task_id: str, request: Request):
    try:
        task_result = AsyncResult(task_id)
        query_info = retrieve_query_info(task_id)
        
        if task_result.state == 'PENDING':
            return TaskResult(task_id=task_id, status='PENDING', **query_info)
        elif task_result.state != 'FAILURE':
            return TaskResult(task_id=task_id, status=task_result.state, summary=task_result.result, **query_info)
        else:
            return TaskResult(task_id=task_id, status='FAILURE', summary=str(task_result.info), **query_info)
    except Exception as e:
        logger.error(f"Error in get_summary endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while retrieving the summary.")

@app.post("/upload", summary="Upload documents for indexing", dependencies=[Depends(verify_api_key)])
@limiter.limit("2/minute")
async def upload_docs(request: Request, files: List[UploadFile] = File(...)):
    try:
        file_paths = []
        for file in files:
            if file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain"]:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
            
            file_path = f"temp_uploads/{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                if len(content) > 10 * 1024 * 1024:  # 10 MB limit
                    raise HTTPException(status_code=400, detail="File size exceeds 10 MB limit")
                buffer.write(content)
            file_paths.append(file_path)
        
        background_tasks = BackgroundTasks()
        background_tasks.add_task(process_uploaded_files, file_paths)
        
        return JSONResponse(content={"status": "Files uploaded and queued for processing", "file_count": len(file_paths)})
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during file upload: {str(e)}")

@app.post("/feedback", summary="Submit feedback for a summary", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def submit_feedback(feedback: FeedbackRequest, background_tasks: BackgroundTasks, request: Request):
    try:
        background_tasks.add_task(process_feedback, feedback.dict())
        return {"status": "Feedback submitted successfully"}
    except Exception as e:
        logger.error(f"Error in submit_feedback endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while submitting feedback.")

@app.get("/health", summary="Health check endpoint")
async def health():
    return {"status": "healthy"}

def store_query_info(task_id: str, original_query: str, rewritten_query: str):
    try:
        query_info = {
            "original_query": original_query,
            "rewritten_query": rewritten_query
        }
        redis_client.set(f"query_info:{task_id}", json.dumps(query_info), ex=3600)  # Expire after 1 hour
    except Exception as e:
        logger.error(f"Error storing query info: {str(e)}")

def retrieve_query_info(task_id: str) -> dict:
    try:
        query_info = redis_client.get(f"query_info:{task_id}")
        return json.loads(query_info) if query_info else {"original_query": None, "rewritten_query": None}
    except Exception as e:
        logger.error(f"Error retrieving query info: {str(e)}")
        return {"original_query": None, "rewritten_query": None}

def process_uploaded_files(file_paths: List[str]):
    try:
        total_indexed = batch_process_files(file_paths, config['vector_db'])
        logger.info(f"Indexed {total_indexed} documents from {len(file_paths)} files")
        
        # Clean up temporary files
        for file_path in file_paths:
            os.remove(file_path)
        
        return total_indexed
    except Exception as e:
        logger.error(f"Error processing uploaded files: {str(e)}")
        return 0

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)