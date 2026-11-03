"""RAGBench FastAPI app. Run from project root: uvicorn backend.main:app --reload"""

import os
import json
import logging
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import rag_engine as rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragbench.main")

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSION = ".pdf"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

app = FastAPI(title="RAGBench API", version="2.0.0")


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str
    history: List[HistoryItem] = Field(default_factory=list)
    sources: Optional[List[str]] = None


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"success": False, "detail": "Unable to contact the RAG backend."},
    )


def _safe_name(filename: str) -> str:
    return os.path.basename((filename or "").replace("\\", "/")).strip()


def _check_ready(question: str):
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")
    if not rag.is_api_configured():
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not configured on the server.")
    if not rag.is_index_ready():
        raise HTTPException(
            status_code=400, detail="Please process your documents before asking a question."
        )


# STATUS / DOCUMENTS
@app.get("/api/status")
def get_status():
    return {
        "api_configured": rag.is_api_configured(),
        "faiss_ready": rag.is_index_ready(),
        "embedding_model": rag.EMBEDDING_MODEL,
        "chat_model": rag.CHAT_MODEL,
        "indexed_documents": rag.list_indexed(),
    }


@app.get("/api/documents")
def list_documents():
    indexed = rag.list_indexed()
    return {
        "documents": [
            {
                "name": p.name,
                "size": p.stat().st_size,
                "indexed": p.name in indexed,
                "chunks": indexed.get(p.name, 0),
            }
            for p in sorted(UPLOAD_DIR.glob("*.pdf"))
        ]
    }


@app.delete("/api/documents/{name}")
def delete_document(name: str):
    safe = _safe_name(name)
    path = UPLOAD_DIR / safe
    if not safe or not path.is_file():
        raise HTTPException(status_code=404, detail=f"'{name}' was not found.")
    try:
        rag.remove_from_index(safe)
        path.unlink()
    except rag.RAGEngineError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"success": True, "deleted": safe}


@app.post("/api/reset")
def reset_index():
    rag.clear_index()
    return {"success": True}


# UPLOAD
@app.post("/api/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No PDF files selected.")

    uploaded = []
    for f in files:
        safe_name = _safe_name(f.filename)
        if not safe_name.lower().endswith(ALLOWED_EXTENSION):
            raise HTTPException(
                status_code=400,
                detail=f"'{f.filename}' is not a PDF file. Only .pdf files are accepted.",
            )

        try:
            content = await f.read()
        except Exception:
            logger.exception("Failed reading upload '%s'", safe_name)
            raise HTTPException(status_code=400, detail=f"Could not read '{safe_name}'.")

        if not content:
            raise HTTPException(status_code=400, detail=f"'{safe_name}' is empty.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail=f"'{safe_name}' exceeds the 50 MB upload limit.")
        if not content.lstrip()[:5] == b"%PDF-":
            raise HTTPException(status_code=400, detail=f"'{safe_name}' is not a valid PDF.")

        dest = UPLOAD_DIR / safe_name
        try:
            dest.write_bytes(content)
        except Exception:
            logger.exception("Failed saving upload '%s'", safe_name)
            raise HTTPException(status_code=500, detail=f"Could not save '{safe_name}'.")

        uploaded.append({"name": safe_name, "size": dest.stat().st_size})

    return {"success": True, "files": uploaded}


# PROCESS (incremental)
@app.post("/api/process")
def process_documents():
    pdf_files = sorted(UPLOAD_DIR.glob("*.pdf"))
    if not pdf_files:
        raise HTTPException(
            status_code=400, detail="No PDF files found. Please upload documents first."
        )

    try:
        stats = rag.index_documents([str(p) for p in pdf_files])
    except rag.RAGEngineError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if stats["chunks"] == 0:
        detail = "Could not extract text from the uploaded PDF(s)."
        if stats["warnings"]:
            detail += " " + " ".join(stats["warnings"])
        raise HTTPException(status_code=422, detail=detail)

    return {"success": True, **stats}


# ASK
@app.post("/api/ask")
def ask(payload: AskRequest):
    question = (payload.question or "").strip()
    _check_ready(question)
    try:
        result = rag.answer_question(
            question,
            history=[h.model_dump() for h in payload.history],
            sources=payload.sources,
        )
    except rag.RAGEngineError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"success": True, **result}


@app.post("/api/ask/stream")
def ask_stream(payload: AskRequest):
    question = (payload.question or "").strip()
    _check_ready(question)
    history = [h.model_dump() for h in payload.history]

    def events():
        try:
            for event in rag.stream_answer(question, history=history, sources=payload.sources):
                yield json.dumps(event) + "\n"
        except rag.RAGEngineError as e:
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


# STATIC FRONTEND
app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")


@app.get("/")
def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))