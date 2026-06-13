"""
AI Recruiter — FastAPI Backend
==============================
Security:      Rate Limit → Validate → Sanitize → Mask PII
Cache:         Semantic cache (0.90 threshold, 300s TTL, 500 max entries)
Pipeline:      Retrieve → Compress → Budget → Generate
Observability: LangSmith tracing on all pipeline spans

Run with:
    uvicorn api:app --reload --port 8000
"""

import os
import re
import time
import math
import logging

import numpy as np
from cachetools import TTLCache

# ── LangSmith — load before any LangChain import ──
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = "ai-recruiter-rag"

import tiktoken

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable

from src.chains.rag_chain import ask, format_chunks, compress_chunks
from src.retrieval.retriever import ResumeRetriever
from src.config.settings import QDRANT_COLLECTION, OPENAI_API_KEY


# =========================
# Structured Logging
# JSON format — easy to query in log aggregators
# =========================

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}'
)
logger = logging.getLogger(__name__)


# =========================
# Rate Limiter
# =========================

limiter = Limiter(key_func=get_remote_address)


# =========================
# App Setup
# =========================

app = FastAPI(
    title="AI Recruiter API",
    description="Search 231 resumes using natural language",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Core components — initialised once at startup ──
retriever = ResumeRetriever()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0
)

# Embedding model for semantic cache
# Same model as retriever — consistent vector space
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

prompt = ChatPromptTemplate.from_template("""
You are an AI recruitment assistant.
Answer ONLY using the provided resume context.
For each matching candidate include:
- Name
- Years of experience
- Relevant technologies
- Short reason they match

Context:
{context}

Recruiter Query:
{question}

Answer:
""")

chain = prompt | llm


# =========================
# Security Config
# All limits in one place — change here, applies everywhere
# =========================

RATE_LIMIT      = "10/minute"
MIN_QUERY_CHARS = 10
MAX_QUERY_CHARS = 500

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"you\s+are\s+now\s+a",
    r"forget\s+(everything|all)",
    r"reveal\s+(all|every|the\s+system)",
    r"show\s+me\s+all\s+(data|emails?|passwords?)",
    r"act\s+as\s+(a\s+)?(different|new|another)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"disregard\s+(all\s+)?(previous|prior)",
    r"override\s+(your\s+)?(instructions?|rules?)",
    r"system\s*prompt",
    r"jailbreak",
    r"prompt\s*injection",
]

# DESIGN DECISION (Recruiter App):
#   User-entered PII in queries  → MASKED before pipeline + traces
#   Candidate CV data (emails)   → REVEALED to recruiter in response
PII_PATTERNS = {
    "EMAIL": r"\b[\w.+-]+@[\w-]+\.\w{2,}\b",
    "PHONE": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "SSN":   r"\b\d{3}-\d{2}-\d{4}\b",
}


# =========================
# Security Functions
# =========================

def validate_query(query: str) -> None:
    """STEP 2 — Check length before any processing."""
    clean = query.strip()
    if len(clean) < MIN_QUERY_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Query too short. Minimum {MIN_QUERY_CHARS} characters required."
        )
    if len(clean) > MAX_QUERY_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Query too long. Maximum {MAX_QUERY_CHARS} characters allowed."
        )


def sanitize_query(query: str) -> None:
    """STEP 3 — Hard block prompt injection. Log and reject."""
    lower = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower):
            logger.warning(
                f"SECURITY_BLOCK type=prompt_injection "
                f"preview={query[:80]!r}"
            )
            raise HTTPException(
                status_code=400,
                detail="Query contains disallowed content."
            )


def mask_pii(text: str) -> str:
    """STEP 4 — Replace PII in user query with placeholders."""
    for label, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[{label}]", text, flags=re.IGNORECASE)
    return text


# =========================
# Semantic Cache
# ─────────────────────────
# How it works:
#   CHECK  → embed query → compare against stored embeddings
#            if similarity > threshold → return cached result (skip pipeline)
#   STORE  → after pipeline runs → save embedding + response
#
# TTLCache handles two problems automatically:
#   1. Max size (500) → oldest entry dropped when full
#   2. TTL (300s)     → expired entries never compared against
# =========================

CACHE_SIMILARITY_THRESHOLD = 0.90
CACHE_TTL                  = 300    # seconds
CACHE_MAX_SIZE             = 500    # max entries

semantic_cache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL)
_cache_counter = 0    # unique key for each cache entry


def _cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two vectors. Returns 0-1."""
    va, vb = np.array(a), np.array(b)
    denom  = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def check_semantic_cache(query_embedding, min_experience, top_n):
    """
    Compare query embedding against all valid (non-expired) cache entries.
    Only matches entries with the same filters (min_experience, top_n).

    Returns: (cached_response, score) or (None, 0.0)
    """
    best_score  = 0.0
    best_result = None

    # list() snapshot — safe to iterate while TTLCache may expire entries
    for entry in list(semantic_cache.values()):
        stored_embedding, stored_min_exp, stored_top_n, stored_response = entry

        # Different filters = different search = different cache entry
        if stored_min_exp != min_experience or stored_top_n != top_n:
            continue

        score = _cosine_similarity(query_embedding, stored_embedding)
        if score > best_score:
            best_score  = score
            best_result = stored_response

    if best_score >= CACHE_SIMILARITY_THRESHOLD:
        logger.info(
            f"cache_hit score={best_score:.3f} "
            f"entries={len(semantic_cache)}"
        )
        return best_result, best_score

    logger.info(
        f"cache_miss best_score={best_score:.3f} "
        f"entries={len(semantic_cache)}"
    )
    return None, best_score


def store_in_cache(query_embedding, min_experience, top_n, response):
    """Store query embedding + response in semantic cache."""
    global _cache_counter
    _cache_counter += 1
    semantic_cache[_cache_counter] = (
        query_embedding,
        min_experience,
        top_n,
        response
    )
    logger.info(
        f"cache_store key={_cache_counter} "
        f"entries={len(semantic_cache)}/{CACHE_MAX_SIZE}"
    )


# =========================
# Token Budgeting
# =========================

MAX_CONTEXT_TOKENS = 4000
encoder = tiktoken.encoding_for_model("gpt-4o-mini")


def count_tokens(text: str) -> int:
    return len(encoder.encode(text))


def _drop_least_relevant(trimmed: dict) -> bool:
    longest = max(trimmed, key=lambda n: len(trimmed[n]))
    if not trimmed[longest]:
        return False
    trimmed[longest].pop()
    return True


@traceable(name="budget-context")
def budget_context(candidate_chunks, max_tokens: int = MAX_CONTEXT_TOKENS):
    context  = format_chunks(candidate_chunks)
    original = count_tokens(context)

    if original <= max_tokens:
        return context, {
            "status":          "within_budget",
            "original_tokens": original,
            "final_tokens":    original,
            "chunks_dropped":  0,
        }

    trimmed       = {name: list(chunks) for name, chunks in candidate_chunks.items()}
    total_chunks  = sum(len(v) for v in trimmed.values())
    avg_per_chunk = max(original / total_chunks, 1)
    overflow      = original - max_tokens
    estimate_drop = math.ceil(overflow / avg_per_chunk)

    for _ in range(estimate_drop):
        if not _drop_least_relevant(trimmed):
            break

    context = format_chunks(trimmed)

    while count_tokens(context) > max_tokens and any(trimmed.values()):
        if not _drop_least_relevant(trimmed):
            break
        context = format_chunks(trimmed)

    final_tokens = count_tokens(context)
    dropped      = total_chunks - sum(len(v) for v in trimmed.values())

    return context, {
        "status":          "trimmed",
        "original_tokens": original,
        "final_tokens":    final_tokens,
        "chunks_dropped":  dropped,
    }


# =========================
# Request / Response Models
# Pydantic validates types automatically — security layer item
# =========================

class SearchRequest(BaseModel):
    query: str = Field(..., example="Find Java engineers with AWS and Docker")
    min_experience: Optional[int] = Field(None, example=5)
    top_n: int = Field(default=5, ge=1, le=10)

class CandidateResult(BaseModel):
    candidate_name:   str
    years_experience: Optional[int]
    skills:           list[str]
    email:            Optional[str]
    relevant_chunks:  int

class SearchResponse(BaseModel):
    query:       str
    answer:      str
    candidates:  list[CandidateResult]
    total_found: int
    time_taken:  float


# =========================
# Traceable pipeline functions
# =========================

@traceable(name="retrieve-candidates")
def retrieve_candidates(query, min_experience, top_n):
    return retriever.retrieve(
        query=query,
        min_experience=min_experience,
        top_n=top_n
    )

@traceable(name="generate-answer")
def generate_answer(context, query):
    response = chain.invoke({
        "context":  context,
        "question": query
    })
    return response.content


# =========================
# POST /search
# ─────────────────────────
# Full request order:
#   1. Rate limit   (decorator)
#   2. Validate     (length)
#   3. Sanitize     (injection block)
#   4. Mask input   (PII in query)
#   5. Cache check  (skip pipeline if hit)
#   6. Pipeline     (retrieve → compress → budget → generate)
#   7. Cache store  (save result for future hits)
#   8. Respond      (real candidate data to recruiter)
# =========================

@app.post("/search", response_model=SearchResponse)
@limiter.limit(RATE_LIMIT)                           # ← STEP 1
async def search(request: Request, body: SearchRequest):

    # ── STEP 2: Validate ─────────────────────────────────────
    validate_query(body.query)

    # ── STEP 3: Sanitize ─────────────────────────────────────
    sanitize_query(body.query)

    # ── STEP 4: Mask input PII ───────────────────────────────
    safe_query = mask_pii(body.query)

    start = time.time()

    try:
        # ── STEP 5: Cache check ──────────────────────────────
        # Embed query once — used for both cache lookup and retrieval
        query_embedding = embeddings_model.embed_query(safe_query)

        cached_result, cache_score = check_semantic_cache(
            query_embedding,
            body.min_experience,
            body.top_n
        )

        if cached_result is not None:
            # Cache hit — skip the entire pipeline
            # Return cached result with fresh time_taken
            return SearchResponse(
                query=cached_result.query,
                answer=cached_result.answer,
                candidates=cached_result.candidates,
                total_found=cached_result.total_found,
                time_taken=round(time.time() - start, 2)
            )

        # ── STEP 6: Pipeline (cache miss only) ───────────────
        candidate_chunks = retrieve_candidates(
            query=safe_query,
            min_experience=body.min_experience,
            top_n=body.top_n
        )

        if not candidate_chunks:
            return SearchResponse(
                query=safe_query,
                answer="No matching candidates found.",
                candidates=[],
                total_found=0,
                time_taken=round(time.time() - start, 2)
            )

        candidate_chunks = compress_chunks(candidate_chunks, safe_query)
        context, budget_info = budget_context(candidate_chunks)

        logger.info(
            f"token_budget status={budget_info['status']} "
            f"original={budget_info['original_tokens']} "
            f"final={budget_info['final_tokens']} "
            f"dropped={budget_info['chunks_dropped']}"
        )

        answer = generate_answer(context, safe_query)

        # Build response object
        candidates = []
        for name, payloads in candidate_chunks.items():
            first  = payloads[0] if payloads else {}
            skills = first.get("skills", [])
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(",")]

            candidates.append(CandidateResult(
                candidate_name=name,
                years_experience=first.get("years_experience"),
                skills=skills,
                email=first.get("email"),
                relevant_chunks=len(payloads)
            ))

        response_obj = SearchResponse(
            query=safe_query,
            answer=answer,
            candidates=candidates,
            total_found=len(candidates),
            time_taken=round(time.time() - start, 2)
        )

        # ── STEP 7: Cache store ───────────────────────────────
        store_in_cache(
            query_embedding,
            body.min_experience,
            body.top_n,
            response_obj
        )

        # ── STEP 8: Respond ───────────────────────────────────
        return response_obj

    except HTTPException:
        raise    # pass our own 400s through unchanged

    except Exception as e:
        logger.error(f"search_error type={type(e).__name__} detail={str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again."
        )


# =========================
# GET /cache/stats
# ─────────────────────────
# Performance Checklist item: "Cache statistics endpoint"
# Shows cache health without exposing cached data
# =========================

@app.get("/cache/stats")
async def cache_stats():
    return {
        "current_entries":        len(semantic_cache),
        "max_size":               CACHE_MAX_SIZE,
        "ttl_seconds":            CACHE_TTL,
        "similarity_threshold":   CACHE_SIMILARITY_THRESHOLD,
        "total_stored_ever":      _cache_counter,
    }


# =========================
# GET /health
# ─────────────────────────
# No security — intentionally public.
# Hosting platforms call this every 30s.
# =========================

@app.get("/health")
async def health():
    try:
        from src.vectordb.qdrant_client import client
        info = client.get_collection(QDRANT_COLLECTION)
        return {
            "status":       "healthy",
            "collection":   QDRANT_COLLECTION,
            "total_chunks": info.points_count,
        }
    except Exception as e:
        logger.error(f"health_check_error type={type(e).__name__}")
        raise HTTPException(status_code=503, detail="Service unavailable.")


# =========================
# GET /candidates
# ─────────────────────────
# Security: rate limit + secure errors
# Output: real candidate data revealed to recruiter (by design)
# =========================

@app.get("/candidates")
@limiter.limit(RATE_LIMIT)
async def list_candidates(request: Request):
    try:
        from src.vectordb.qdrant_client import client
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        result, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=1000,
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(
                must=[FieldCondition(
                    key="section",
                    match=MatchValue(value="summary_index")
                )]
            )
        )

        candidates = []
        for point in result:
            p      = point.payload
            skills = p.get("skills", [])
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(",")]

            candidates.append({
                "name":             p.get("candidate_name", "Unknown"),
                "years_experience": p.get("years_experience"),
                "skills":           skills[:5],
                "email":            p.get("email"),
            })

        candidates.sort(key=lambda x: x["name"])
        return {"total": len(candidates), "candidates": candidates}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"candidates_error type={type(e).__name__}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred."
        )