from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn
from contextlib import asynccontextmanager
from rag_pipeline import medbot_rag
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and clean up resources"""
    try:
        logger.info("Initializing RAG pipeline...")
        medbot_rag.initialize()
        
        # Test the pipeline with a warmup query
        test_response = medbot_rag.query("System warmup query")
        if isinstance(test_response, dict) and "answer" in test_response:
            logger.info("Service startup complete - Warmup successful")
        else:
            logger.warning("Service started with potential initialization issues")
        
        yield
        
        logger.info("Shutting down...")
        
    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}")
        raise RuntimeError(f"Service initialization failed: {str(e)}")

app = FastAPI(
    title="NthanziLanga+ AI API",
    description="Medical AI Assistant with RAG capabilities",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class HealthCheck(BaseModel):
    status: str = Field(..., example="operational")
    model: str = Field(..., example="gemini-1.5-flash")
    db_status: str = Field(..., example="connected")

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500,
                         example="What are symptoms of diabetes?")
    user_id: Optional[str] = Field(None, example="user123")
    medical_context: Optional[str] = Field(None, max_length=1000,
                                         example="Patient is 65 years old")

class QueryResponse(BaseModel):
    question: str = Field(..., example="What are symptoms of diabetes?")
    answer: str = Field(..., example="Common symptoms include frequent urination...")
    status: str = Field("success", example="success")
    processed_at: str = Field(..., example="2024-03-16T10:30:00Z")

# API Endpoints
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "NthanziLanga+ Medical AI Service"}

@app.get("/health", response_model=HealthCheck)
async def health_check():
    return {
        "status": "operational",
        "model": "gemini-1.5-flash",
        "db_status": "connected" if hasattr(medbot_rag, 'retriever') and medbot_rag.retriever else "disconnected"
    }

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        start_time = datetime.now()
        # Process the query through the enhanced RAG pipeline
        result = medbot_rag.query(
            question=request.question,
            medical_context=request.medical_context
        )
        
        # Handle both string and dictionary responses
        if isinstance(result, dict):
            answer = result.get("answer", "No answer could be generated")
            if "error" in result:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=result.get("fallback_answer", "Service unavailable")
                )
        else:
            answer = str(result)
        
        logger.info(f"Processed query in {(datetime.now() - start_time).total_seconds():.2f}s")
        
        return {
            "question": request.question,
            "answer": answer,
            "status": "success",
            "processed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Medical knowledge service unavailable"
        )

# Error Handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception at {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "Internal server error"},
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
        reload=os.getenv("RELOAD", "false").lower() == "true",
        timeout_keep_alive=120
    )