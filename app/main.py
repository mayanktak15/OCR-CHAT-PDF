"""
PDF OCR + RAG Chat API
FastAPI backend for uploading PDFs, extracting text via OCR, and chatting with the content.
"""

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .ocr_utils import extract_text_from_pdf
from .rag_utils import build_index, query_index

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="PDF OCR + RAG Chat",
    description="Upload PDFs, extract text with OCR, and chat with the content",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory session store
sessions: dict[str, dict[str, Any]] = {}


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main HTML page."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and extract text using OCR.
    Returns a session_id for subsequent chat requests.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    try:
        pdf_bytes = await file.read()
        if len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        logger.info(f"Processing PDF: {file.filename} ({len(pdf_bytes)} bytes)")

        # Extract text from PDF
        text = extract_text_from_pdf(pdf_bytes)

        if not text or len(text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF. Ensure it contains readable text or images with text.",
            )

        # Build search index
        index = build_index(text)

        # Store session
        session_id = str(uuid4())
        sessions[session_id] = {
            "index": index,
            "filename": file.filename,
            "text_length": len(text),
        }

        logger.info(f"Session created: {session_id} with {len(text)} chars")

        return {
            "session_id": session_id,
            "filename": file.filename,
            "text_length": len(text),
            "message": "PDF processed successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing PDF")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/api/chat")
async def chat(
    session_id: str = Body(..., embed=False),
    question: str = Body(..., embed=False),
):
    """
    Chat with the uploaded PDF content.
    Requires session_id from upload and a question.
    """
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid or expired session")

    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        session = sessions[session_id]
        index = session["index"]

        answer = query_index(index, question)

        return {"answer": answer}

    except Exception as e:
        logger.exception("Error in chat")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session to free memory."""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")
