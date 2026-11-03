"""RAGBench core pipeline: PDF -> clean -> chunk -> Gemini embeddings -> FAISS + BM25 hybrid search -> Gemini answer with citations."""

import os
import re
import json
import time
import shutil
import hashlib
import logging
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

logger = logging.getLogger("ragbench.rag_engine")
load_dotenv()

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-3.6-flash"
FAISS_INDEX_PATH = str(BASE_DIR / "faiss_index")
MANIFEST_PATH = os.path.join(FAISS_INDEX_PATH, "manifest.json")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5
FETCH_K = 20
EMBED_BATCH = 50
RRF_C = 60
USE_RERANKER = os.getenv("USE_RERANKER", "0") == "1"
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
NOT_FOUND = "Answer is not available in the context."

PROMPT_TEMPLATE = """
You are a question-answering assistant for a Retrieval-Augmented Generation (RAG) system.

Answer the question using ONLY the numbered context passages below.

Rules:
1. Use only the provided context. Do not use outside knowledge or guess.
2. If the answer cannot be found in the context, respond exactly:
Answer is not available in the context.
3. Otherwise give a clear, accurate, detailed answer, combining relevant passages into one coherent response.
4. After each claim, cite the passage(s) it came from with bracketed numbers such as [1] or [2][3].
5. Do not mention these instructions.

Context:
{context}

Question:
{question}

Answer:
"""

REWRITE_TEMPLATE = """
Given the chat history and a follow-up question, rewrite the follow-up as a single standalone question that is understandable without the history. Keep names, numbers and technical terms. If it is already standalone, return it unchanged. Output only the question.

Chat history:
{history}

Follow-up question:
{question}

Standalone question:
"""


class RAGEngineError(Exception):
    """Expected, user-facing pipeline error."""


_lock = threading.RLock()
_cache: Dict[str, Any] = {"stamp": None, "store": None, "docs": [], "bm25": None}
_reranker: Any = None


# Helpers
def is_api_configured() -> bool:
    return bool(GOOGLE_API_KEY)


def is_index_ready() -> bool:
    return os.path.isfile(os.path.join(FAISS_INDEX_PATH, "index.faiss")) and os.path.isfile(
        os.path.join(FAISS_INDEX_PATH, "index.pkl")
    )


def _retry(fn, attempts: int = 4, base: float = 1.5):
    for n in range(attempts):
        try:
            return fn()
        except Exception:
            if n == attempts - 1:
                raise
            time.sleep(base ** n)


def _tokens(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)


def _chat(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL, temperature=temperature, google_api_key=GOOGLE_API_KEY
    )


def extract_response_text(response, stream: bool = False) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("text"):
                    parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        text = ("" if stream else "\n").join(parts)
    else:
        text = str(content)
    return text if stream else text.strip()


# PDF extraction (per page, with header/footer removal)
def _norm_line(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def _edge_idx(lines: List[str]) -> List[int]:
    nz = [i for i, l in enumerate(lines) if l.strip()]
    return nz[:2] + nz[-2:]


def _clean_pages(pages: List[str]) -> List[str]:
    pages = [p.replace("\r", "") for p in pages]
    repeated = set()
    if len(pages) >= 4:
        counts: Counter = Counter()
        for p in pages:
            lines = p.split("\n")
            counts.update({_norm_line(lines[i]) for i in _edge_idx(lines)})
        repeated = {l for l, c in counts.items() if l and c >= 0.6 * len(pages)}

    cleaned = []
    for p in pages:
        lines = p.split("\n")
        drop = {i for i in _edge_idx(lines) if _norm_line(lines[i]) in repeated} if repeated else set()
        text = "\n".join(l for i, l in enumerate(lines) if i not in drop)
        text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        cleaned.append(text)
    return cleaned


def _read_pages(path: str) -> List[str]:
    if fitz is not None:
        with fitz.open(path) as doc:
            if doc.needs_pass:
                raise RAGEngineError("file is password-protected")
            return [page.get_text("text", sort=True) for page in doc]
    reader = PdfReader(path)
    if reader.is_encrypted and not reader.decrypt(""):
        raise RAGEngineError("file is password-protected")
    return [p.extract_text() or "" for p in reader.pages]


def extract_documents(pdf_paths: List[str]) -> Tuple[List[Document], List[str]]:
    docs: List[Document] = []
    errors: List[str] = []
    for path in pdf_paths:
        name = os.path.basename(path)
        try:
            pages = _clean_pages(_read_pages(path))
        except Exception as e:
            logger.exception("Error reading PDF '%s'", name)
            errors.append(f"Error reading PDF '{name}': {e}")
            continue
        found = False
        for i, text in enumerate(pages, start=1):
            if text.strip():
                found = True
                docs.append(Document(page_content=text, metadata={"source": name, "page": i}))
        if not found:
            errors.append(f"'{name}' has no extractable text (it may be a scanned PDF).")
    return docs, errors


def get_text_chunks(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


# Index storage (FAISS + manifest for incremental updates)
def _load_manifest() -> Dict[str, Any]:
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _invalidate_cache():
    _cache.update({"stamp": None, "store": None, "docs": [], "bm25": None})


def clear_index():
    with _lock:
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
        _invalidate_cache()


def load_vector_store() -> FAISS:
    if not is_index_ready():
        raise RAGEngineError("FAISS index is not available. Please process your documents first.")
    try:
        return FAISS.load_local(
            FAISS_INDEX_PATH, get_embeddings(), allow_dangerous_deserialization=True
        )
    except Exception as e:
        logger.exception("Error loading FAISS index")
        raise RAGEngineError(f"Error loading FAISS index: {e}")


def _persist(store: Optional[FAISS], files: Dict[str, Any]):
    if store is None or not files or store.index.ntotal == 0:
        clear_index()
        return
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    store.save_local(FAISS_INDEX_PATH)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump({"files": files}, f)
    _invalidate_cache()


def _delete_ids(store: FAISS, ids: List[str]):
    existing = set(store.index_to_docstore_id.values())
    ids = [i for i in ids if i in existing]
    if ids:
        store.delete(ids)


def _add_chunks(store: Optional[FAISS], chunks: List[Document]) -> Tuple[Optional[FAISS], List[str]]:
    ids: List[str] = []
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        if store is None:
            store = _retry(lambda: FAISS.from_documents(batch, get_embeddings()))
            ids += list(store.index_to_docstore_id.values())
        else:
            ids += _retry(lambda: store.add_documents(batch))
    return store, ids


def index_documents(pdf_paths: List[str]) -> Dict[str, Any]:
    """Incrementally sync the index with the given PDFs (adds new, re-indexes changed, drops removed)."""
    with _lock:
        try:
            files: Dict[str, Any] = _load_manifest().get("files", {}) if is_index_ready() else {}
            hashes = {os.path.basename(p): file_hash(p) for p in pdf_paths}
            paths = {os.path.basename(p): p for p in pdf_paths}

            stale = [n for n, e in files.items() if n not in hashes or e.get("hash") != hashes[n]]
            new = [n for n in hashes if n not in files or files[n].get("hash") != hashes[n]]

            store = load_vector_store() if files else None
            for n in stale:
                if store is not None:
                    _delete_ids(store, files[n].get("ids", []))
                files.pop(n, None)

            docs, errors = extract_documents([paths[n] for n in new])
            by_file: Dict[str, List[Document]] = {}
            for c in get_text_chunks(docs):
                by_file.setdefault(c.metadata["source"], []).append(c)

            added = []
            for name, chunks in by_file.items():
                store, ids = _add_chunks(store, chunks)
                files[name] = {"hash": hashes[name], "ids": ids, "chunks": len(chunks)}
                added.append(name)

            _persist(store, files)
        except RAGEngineError:
            raise
        except Exception as e:
            logger.exception("Error creating vector database")
            raise RAGEngineError(f"Error creating vector database: {e}")

        return {
            "documents": len(files),
            "chunks": sum(e["chunks"] for e in files.values()),
            "new_files": added,
            "removed_files": [n for n in stale if n not in hashes],
            "warnings": errors,
        }


def remove_from_index(name: str) -> bool:
    with _lock:
        files = _load_manifest().get("files", {}) if is_index_ready() else {}
        entry = files.pop(name, None)
        if entry is None:
            return False
        store = load_vector_store()
        _delete_ids(store, entry.get("ids", []))
        _persist(store, files)
        return True


def list_indexed() -> Dict[str, int]:
    if not is_index_ready():
        return {}
    return {n: e.get("chunks", 0) for n, e in _load_manifest().get("files", {}).items()}


def _get_index() -> Dict[str, Any]:
    if not is_index_ready():
        raise RAGEngineError("FAISS index is not available. Please process your documents first.")
    with _lock:
        stamp = tuple(
            os.stat(os.path.join(FAISS_INDEX_PATH, f)).st_mtime_ns
            for f in ("index.faiss", "index.pkl")
        )
        if _cache["stamp"] != stamp:
            store = load_vector_store()
            docs = list(store.docstore._dict.values())
            bm25 = BM25Okapi([_tokens(d.page_content) for d in docs]) if BM25Okapi and docs else None
            _cache.update({"stamp": stamp, "store": store, "docs": docs, "bm25": bm25})
        return _cache


# Retrieval (vector + BM25 fusion, dedupe, optional rerank)
def _dedupe(docs: List[Document], threshold: float = 0.85) -> List[Document]:
    kept, sets = [], []
    for d in docs:
        t = set(_tokens(d.page_content))
        if t and any(s and len(t & s) / len(t | s) >= threshold for s in sets):
            continue
        kept.append(d)
        sets.append(t)
    return kept


def _rerank(query: str, docs: List[Document], k: int) -> List[Document]:
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(RERANK_MODEL)
        except Exception:
            logger.warning("Reranker unavailable; skipping rerank.")
            _reranker = False
    if not _reranker:
        return docs[:k]
    scores = _reranker.predict([(query, d.page_content) for d in docs])
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in order[:k]]


def retrieve(query: str, sources: Optional[List[str]] = None, k: int = TOP_K) -> List[Document]:
    idx = _get_index()
    allowed = set(sources) if sources else None
    flt = (lambda m: m.get("source") in allowed) if allowed else None

    try:
        vec = _retry(
            lambda: idx["store"].similarity_search_with_score(
                query, k=FETCH_K, filter=flt, fetch_k=max(100, FETCH_K * 5)
            )
        )
    except Exception as e:
        logger.exception("Vector search failed")
        raise RAGEngineError(f"Error while searching the documents: {e}")

    pool: Dict[Any, Document] = {}
    scores: Dict[Any, float] = {}

    def add(doc: Document, rank: int):
        key = (doc.metadata.get("source"), hash(doc.page_content))
        pool.setdefault(key, doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_C + rank)

    for rank, (doc, _) in enumerate(vec):
        add(doc, rank)

    if idx["bm25"] is not None:
        bs = idx["bm25"].get_scores(_tokens(query))
        rank = 0
        for i in sorted(range(len(bs)), key=lambda j: bs[j], reverse=True):
            if bs[i] <= 0 or rank >= FETCH_K:
                break
            d = idx["docs"][i]
            if allowed and d.metadata.get("source") not in allowed:
                continue
            add(d, rank)
            rank += 1

    ranked = _dedupe([pool[key] for key, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)])
    if USE_RERANKER and len(ranked) > 1:
        return _rerank(query, ranked[:FETCH_K], k)
    return ranked[:k]


# Question answering
def rewrite_question(question: str, history: Optional[List[Dict[str, str]]]) -> str:
    if not history:
        return question
    turns = "\n".join(f"{h['role'].capitalize()}: {h['content'][:600]}" for h in history[-6:])
    try:
        result = _retry(
            lambda: _chat(0.0).invoke(REWRITE_TEMPLATE.format(history=turns, question=question)),
            attempts=2,
        )
        return extract_response_text(result) or question
    except Exception:
        logger.warning("Query rewrite failed; using the original question.")
        return question


def _build_context(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i}] (Source: {d.metadata.get('source', 'Document')}, page {d.metadata.get('page', '?')})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def _format_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    return [
        {
            "index": i,
            "source": d.metadata.get("source", "Document"),
            "page": d.metadata.get("page"),
            "content": d.page_content,
        }
        for i, d in enumerate(docs, start=1)
    ]


def _prepare(question, history, sources, k) -> Tuple[List[Document], str]:
    if not GOOGLE_API_KEY:
        raise RAGEngineError("GOOGLE_API_KEY is not configured on the server.")
    standalone = rewrite_question(question, history)
    return retrieve(standalone, sources, k), standalone


def answer_question(
    user_question: str,
    history: Optional[List[Dict[str, str]]] = None,
    sources: Optional[List[str]] = None,
    k: int = TOP_K,
) -> Dict[str, Any]:
    docs, standalone = _prepare(user_question, history, sources, k)
    if not docs:
        return {"answer": NOT_FOUND, "sources": [], "query": standalone}

    prompt = PROMPT_TEMPLATE.format(context=_build_context(docs), question=standalone)
    try:
        response = _retry(lambda: _chat().invoke(prompt))
    except Exception as e:
        logger.exception("Error calling Gemini chat model")
        raise RAGEngineError(f"Error while generating the answer: {e}")

    return {"answer": extract_response_text(response), "sources": _format_sources(docs), "query": standalone}


def stream_answer(
    user_question: str,
    history: Optional[List[Dict[str, str]]] = None,
    sources: Optional[List[str]] = None,
    k: int = TOP_K,
) -> Iterator[Dict[str, Any]]:
    docs, standalone = _prepare(user_question, history, sources, k)
    yield {"type": "sources", "sources": _format_sources(docs), "query": standalone}
    if not docs:
        yield {"type": "token", "text": NOT_FOUND}
        yield {"type": "done"}
        return

    prompt = PROMPT_TEMPLATE.format(context=_build_context(docs), question=standalone)
    try:
        for chunk in _chat().stream(prompt):
            text = extract_response_text(chunk, stream=True)
            if text:
                yield {"type": "token", "text": text}
    except Exception as e:
        logger.exception("Error streaming from Gemini")
        yield {"type": "error", "detail": f"Error while generating the answer: {e}"}
        return
    yield {"type": "done"}