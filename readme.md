# AI Recruiter — RAG-Powered Resume Search

> Search 231 resumes using natural language. Built with LangChain, Qdrant, FastAPI, and Streamlit. Production-hardened with security, semantic caching, observability, and cost controls.

**Live API:** https://ai-recruiter-rag.onrender.com/docs

---

## What It Does

A recruiter types a plain English query like:

> *"Find Java backend engineers with AWS, Docker, and microservices experience, 5+ years"*

The system searches 231 resumes and returns the top matching candidates with their skills, experience, and a specific reason they match.

---

## Production Results

Real numbers from LangSmith traces — not estimates:

| Metric | Before | After |
|---|---|---|
| Retrieval latency | 14–31s | ~6s |
| Cache hit latency | — | ~0.3s |
| Tokens per query | 4,593 | 3,541 |
| Cost per query | unknown | $0.0008 |
| Chunks dropped by budget | 9 | 0 |

---

## Architecture

```
User Query
     ↓
Security Layer
  ├── Rate limit        (10 requests/min per IP)
  ├── Validate          (min 10, max 500 characters)
  ├── Sanitize          (block prompt injection — 12 patterns)
  └── Mask PII          (emails, phones, SSNs in queries)
     ↓
Semantic Cache
  └── Check similarity against stored results (threshold: 0.90)
      └── HIT  → return in ~0.3s (skip pipeline entirely)
      └── MISS → continue to pipeline
     ↓
RAG Pipeline
  ├── Embed query       (OpenAI text-embedding-3-small)
  ├── Stage 1 retrieval (cosine similarity on 231 summary chunks)
  ├── Stage 2 retrieval (top 10 chunks per candidate from Qdrant)
  ├── Compress chunks   (15 parallel LLM calls — extract relevant parts only)
  ├── Token budget      (cap at 4,000 tokens, trim if needed)
  └── Generate answer   (GPT-4o-mini with structured recruiter prompt)
     ↓
Cache Store
  └── Save result (TTL: 300s, max: 500 entries)
     ↓
Response
  └── Candidates with name, experience, skills, email, match reason
```

---

## Retrieval Strategy

**Stage 1 — Candidate Discovery (Cosine Similarity)**
- Searches only `summary_index` chunks — one per candidate
- Fast scan across all 231 candidates
- Returns top 5 by semantic similarity score

**Stage 2 — Deep Dive**
- For each top candidate, fetches 10 chunks directly from Qdrant
- Qdrant returns chunks in similarity order — most relevant first
- Query embedded once and reused across both stages (no double API call)

**Contextual Compression**
- Top 3 chunks per candidate sent to a compression LLM
- Compression prompt: "extract only parts relevant to this query"
- 15 compression calls fire in parallel — adds only ~1.5s
- Chunks shrink from ~1,700 chars to ~250 chars
- "Not relevant" chunks dropped entirely before LLM sees them

---

## Observability

Every request is fully traced in LangSmith:

```
retrieve-candidates   → Qdrant latency, candidates found
compress-chunks       → parallel compression calls, tokens
budget-context        → original vs final tokens, chunks dropped
generate-answer       → LLM latency, token count, cost
```

Structured JSON logging on every event. Cache stats available at `/cache/stats`.

---

## Evaluation

Evaluated with RAGAS-style LLM-as-judge scoring across 30 recruiter queries:

| Metric | Score |
|---|---|
| Faithfulness | 0.933 |
| Answer Relevancy | 1.000 |
| Context Precision | 0.958 |
| Context Recall | 0.967 |
| **Overall** | **0.965** |

---

## Security

| Protection | Implementation |
|---|---|
| Prompt injection | 12 regex patterns, hard block, logged |
| PII masking | Email, phone, SSN masked in queries and logs |
| Rate limiting | 10 requests/min per IP (slowapi) |
| Input validation | Min/max length enforced before pipeline |
| Secure errors | Internal logging only, generic message to client |
| Pydantic models | Automatic type validation on all requests |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector DB | Qdrant Cloud |
| LLM | GPT-4o-mini |
| Orchestration | LangChain |
| Backend | FastAPI |
| Frontend | Streamlit |
| Observability | LangSmith |
| Caching | TTLCache (cachetools) + cosine similarity |
| Security | slowapi, regex sanitization, tiktoken |
| Deployment | Docker + Render |
| Language | Python 3.13 |

---

## Project Structure

```
ai-recruiter-rag/
├── api.py                          # FastAPI backend — security + cache + pipeline
├── app.py                          # Streamlit recruiter interface
├── main.py                         # Ingestion pipeline runner
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── ingestion/
│   │   ├── loader.py               # Loads .docx files
│   │   ├── parser.py               # Sections raw text
│   │   ├── metadata_extractor.py   # Extracts name, skills, years
│   │   └── chunker.py              # Splits + summary chunk per resume
│   ├── retrieval/
│   │   └── retriever.py            # Two-stage retrieval, embed-once optimisation
│   ├── chains/
│   │   └── rag_chain.py            # Compression + answer generation
│   ├── vectordb/
│   │   └── qdrant_client.py        # Qdrant connection + upload
│   └── config/
│       └── settings.py             # Environment config
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/waleed563/ai-recruiter-rag.git
cd ai-recruiter-rag
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
```

Fill in your credentials:
```env
OPENAI_API_KEY=sk-...
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-key
QDRANT_COLLECTION=resume_collection
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your-langsmith-key
LANGCHAIN_PROJECT=ai-recruiter-rag
```

**5. Add resumes and run ingestion**
```bash
mkdir -p data/raw_resumes
# Copy .docx resume files into data/raw_resumes/
python main.py
```

**6. Start the API**
```bash
uvicorn api:app --reload --port 8000
```

**7. Start the UI**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/search` | Search resumes with natural language |
| GET | `/cache/stats` | Cache health — entries, TTL, threshold |
| GET | `/health` | API + Qdrant health check |
| GET | `/candidates` | List all candidates in database |

**Example search request:**
```json
POST /search
{
  "query": "Find Java engineers with AWS and Docker",
  "min_experience": 5,
  "top_n": 5
}
```

**Example search response:**
```json
{
  "query": "Find Java engineers with AWS and Docker",
  "answer": "- Name: John Smith\n  Years: 7\n  Technologies: Java, AWS, Docker...",
  "candidates": [...],
  "total_found": 5,
  "time_taken": 0.31
}
```

---

## Author

Built by Waleed — AI Engineer specialising in custom RAG systems for SaaS teams.

LinkedIn: linkedin.com/in/waleed-ahmed-ai