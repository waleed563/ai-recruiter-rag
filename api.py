"""
AI Recruiter — FastAPI Backend
==============================
Run with:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /search        → main recruiter search
    GET  /health        → health check
    GET  /candidates    → list all candidates in DB
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time

from src.chains.rag_chain import ask, format_chunks
from src.retrieval.retriever import ResumeRetriever
from src.config.settings import QDRANT_COLLECTION


# =========================
# App Setup
# =========================

app = FastAPI(
    title="AI Recruiter API",
    description="Search 231 resumes using natural language",
    version="1.0.0"
)

# Allow frontend to call this API
# (needed for browser-based clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retriever once at startup
# not on every request — saves time
retriever = ResumeRetriever()


# =========================
# Request / Response Models
# =========================

class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        example="Find Java engineers with AWS and Docker",
        description="Natural language recruiter query"
    )
    min_experience: Optional[int] = Field(
        None,
        example=5,
        description="Minimum years of experience filter"
    )
    top_n: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of candidates to return (1-10)"
    )


class CandidateResult(BaseModel):
    candidate_name:   str
    years_experience: Optional[int]
    skills:           list[str]
    email:            Optional[str]
    relevant_chunks:  int


class SearchResponse(BaseModel):
    query:        str
    answer:       str
    candidates:   list[CandidateResult]
    total_found:  int
    time_taken:   float


# =========================
# POST /search
# Main recruiter search endpoint
# =========================

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    start = time.time()

    try:
        # Stage 1 + 2: retrieve candidates and chunks
        candidate_chunks = retriever.retrieve(
            query=request.query,
            min_experience=request.min_experience,
            top_n=request.top_n
        )

        if not candidate_chunks:
            return SearchResponse(
                query=request.query,
                answer="No matching candidates found in the provided resumes.",
                candidates=[],
                total_found=0,
                time_taken=round(time.time() - start, 2)
            )

        # Build context and get LLM answer
        context = format_chunks(candidate_chunks)

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        from src.config.settings import OPENAI_API_KEY

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            temperature=0
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
        chain    = prompt | llm
        response = chain.invoke({
            "context":  context,
            "question": request.query
        })
        answer = response.content

        # Build structured candidate list
        candidates = []
        for name, payloads in candidate_chunks.items():
            first = payloads[0] if payloads else {}
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

        return SearchResponse(
            query=request.query,
            answer=answer,
            candidates=candidates,
            total_found=len(candidates),
            time_taken=round(time.time() - start, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# GET /health
# Health check — confirms API
# and Qdrant are both alive
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
        raise HTTPException(status_code=503, detail=str(e))


# =========================
# GET /candidates
# List all unique candidates
# stored in Qdrant
# =========================

@app.get("/candidates")
async def list_candidates():
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
            p = point.payload
            skills = p.get("skills", [])
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(",")]

            candidates.append({
                "name":            p.get("candidate_name", "Unknown"),
                "years_experience": p.get("years_experience"),
                "skills":          skills[:5],   # top 5 skills preview
                "email":           p.get("email"),
            })

        # Sort by name
        candidates.sort(key=lambda x: x["name"])

        return {
            "total":      len(candidates),
            "candidates": candidates
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))