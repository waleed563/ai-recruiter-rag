# AI Recruiter — RAG-Powered Resume Search

> Search 231 resumes using natural language. Built with LangChain, Qdrant, FastAPI, and Streamlit.

---

## What It Does

A recruiter types a plain English query like:

> *"Find Java backend engineers with AWS, Docker, and microservices experience, 5+ years"*

The system searches 231 resumes and returns the top matching candidates with their skills, experience, and a specific reason they match — in under 5 seconds.

---

## Architecture

```
231 .docx resumes
        ↓
loader.py         → extracts raw text from each file
        ↓
parser.py         → organizes text into sections (skills, experience, education...)
        ↓
metadata_extractor.py → pulls name, email, phone, skills, years of experience
        ↓
chunker.py        → splits sections into 500-char chunks + one summary chunk per resume
        ↓
OpenAI Embeddings → translates each chunk into 1536-dimensional vectors
        ↓
Qdrant            → stores 10,277 chunks in a vector database
        ↓
retriever.py      → two-stage search: cosine similarity + MMR reranking
        ↓
rag_chain.py      → GPT-4o-mini generates structured recruiter answer
        ↓
api.py            → FastAPI REST backend
        ↓
app.py            → Streamlit recruiter interface
```

---

## Retrieval Strategy

**Stage 1 — Candidate Discovery (Cosine Similarity)**
- Searches only `summary_index` chunks — one per candidate
- Fast scan across all 231 candidates
- Returns top 5 by semantic similarity score

**Stage 2 — Deep Dive (MMR — Maximal Marginal Relevance)**
- For each top candidate, fetches 30 chunks
- MMR selects 10 diverse chunks (lambda=0.7)
- Avoids repetitive context — each chunk covers a different aspect

**Why two stages?**
One-stage search with k=50 returns repetitive chunks from the same candidate. Two-stage search gives diverse, complete context per candidate.

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

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector DB | Qdrant Cloud |
| LLM | GPT-4o-mini |
| Orchestration | LangChain |
| Backend | FastAPI |
| Frontend | Streamlit |
| Language | Python 3.13 |

---

## Project Structure

```
ai-recruiter-rag/
├── api.py                          # FastAPI backend (3 endpoints)
├── app.py                          # Streamlit frontend
├── main.py                         # Ingestion pipeline runner
├── requirements.txt
├── .env.example
├── src/
│   ├── ingestion/
│   │   ├── loader.py               # Loads .docx files
│   │   ├── parser.py               # Sections raw text
│   │   ├── metadata_extractor.py   # Extracts name, skills, years
│   │   └── chunker.py              # Splits + summary chunk
│   ├── retrieval/
│   │   └── retriever.py            # Two-stage MMR retrieval
│   ├── chains/
│   │   └── rag_chain.py            # LLM answer generation
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
# Fill in your OpenAI and Qdrant credentials
```

**5. Add resumes**
```bash
mkdir -p data/raw_resumes
# Copy your .docx resume files into data/raw_resumes/
```

**6. Run ingestion pipeline**
```bash
python main.py
```

**7. Start the API**
```bash
python -m uvicorn api:app --port 8000
```

**8. Start the UI**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/search` | Search resumes with natural language |
| GET | `/health` | Health check + chunk count |
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

---

## Author

Built by Waleed — AI/Automation Engineer