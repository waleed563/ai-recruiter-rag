from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from src.retrieval.retriever import ResumeRetriever
from src.config.settings import OPENAI_API_KEY


# =========================
# Initialize Components
# =========================

retriever_system = ResumeRetriever()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0,
    streaming=True
)


# =========================
# Compression Components
# -------------------------
# Separate LLM instance for compression:
# - max_tokens=200  → keeps output short (we want ~2-3 sentences)
# - streaming=False → we need the full response before moving on
# =========================

compress_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0,
    max_tokens=200
)

compress_prompt = ChatPromptTemplate.from_template("""
You are a resume analyst.
Extract ONLY the parts of this resume section directly relevant to the recruiter query.

Recruiter Query: {query}

Resume Section:
{text}

Rules:
- Return only the relevant extracted text, nothing else
- Keep specific skills, years, and project names that match the query
- If nothing in this section is relevant, return exactly: Not relevant
- Maximum 2-3 sentences
""")

compress_chain = compress_prompt | compress_llm | StrOutputParser()


# =========================
# Compression Config
# -------------------------
# TOP_CHUNKS_TO_COMPRESS: only compress the most relevant N chunks
# per candidate. The rest pass through untouched (they're less
# relevant anyway and contribute little to the answer).
#
# MIN_CHARS_TO_COMPRESS: skip chunks already short enough —
# no point compressing a 150-char snippet.
# =========================

TOP_CHUNKS_TO_COMPRESS = 3     # 5 candidates × 3 = 15 LLM calls max
MIN_CHARS_TO_COMPRESS  = 300   # skip if already concise


def _compress_one(payload: dict, query: str):
    """
    Compress a single chunk payload.
    Returns a new payload dict with compressed page_content,
    or None if the LLM says the chunk isn't relevant.
    """
    text = payload.get("page_content", "")

    # Already concise — pass through unchanged
    if len(text) < MIN_CHARS_TO_COMPRESS:
        return payload

    compressed_text = compress_chain.invoke({
        "query": query,
        "text":  text
    })

    # LLM flagged this chunk as irrelevant — drop it
    if compressed_text.strip().lower() == "not relevant":
        return None

    # Return a new dict — never mutate the original
    return {**payload, "page_content": compressed_text}


@traceable(name="compress-chunks")
def compress_chunks(candidate_chunks: dict, query: str) -> dict:
    """
    For each candidate, compress the top N chunks in parallel —
    all LLM calls fire at once instead of one at a time.

    Pipeline position: AFTER retrieval, BEFORE format_chunks.

    Why parallel: 15 sequential LLM calls ≈ 15s added latency.
                  15 parallel LLM calls ≈ 1-2s (time of one call).
    """
    # ── Separate chunks that need compression from pass-throughs ──
    to_compress   = []    # (name, index, payload)
    pass_throughs = {}    # (name, index) → payload

    for name, payloads in candidate_chunks.items():
        for i, payload in enumerate(payloads):
            if i < TOP_CHUNKS_TO_COMPRESS:
                to_compress.append((name, i, payload))
            else:
                pass_throughs[(name, i)] = payload

    # ── Fire all compression calls in parallel ──
    compressed_results = {}   # (name, index) → payload or None

    def run_one(name, i, payload):
        return (name, i, _compress_one(payload, query))

    with ThreadPoolExecutor(max_workers=len(to_compress) or 1) as ex:
        futures = {
            ex.submit(run_one, name, i, payload): (name, i)
            for name, i, payload in to_compress
        }
        for future in as_completed(futures):
            name, i, result = future.result()
            compressed_results[(name, i)] = result

    # ── Merge compressed + pass-through, preserving order ──
    final = {}

    for name, payloads in candidate_chunks.items():
        new_payloads = []
        for i in range(len(payloads)):
            if i < TOP_CHUNKS_TO_COMPRESS:
                result = compressed_results.get((name, i))
                if result is not None:           # None = LLM said not relevant
                    new_payloads.append(result)
            else:
                new_payloads.append(pass_throughs[(name, i)])

        if new_payloads:
            final[name] = new_payloads

    return final


# =========================
# Prompt
# =========================

prompt = ChatPromptTemplate.from_template("""
You are an AI recruitment assistant.
Answer ONLY using the provided resume context below.

STRICT RULES:
- Do NOT use outside knowledge
- Do NOT invent candidates
- Do NOT assume missing information
- Return ALL relevant candidates from the context
- If no relevant candidates found say:
  "No matching candidates found in the provided resumes."

For each matching candidate include:
- Name
- Years of experience
- Relevant technologies
- Short reason they match the query

Context:
{context}

Recruiter Query:
{question}

Answer:
""")


# =========================
# Format chunks into context
# Chunks are raw dicts from Qdrant payload.
# Print removed — LangSmith traces give better visibility.
# =========================

def format_chunks(candidate_chunks: dict) -> str:

    formatted = []

    for candidate_name, payloads in candidate_chunks.items():

        all_content = "\n".join([
            p.get("page_content", "")
            for p in payloads
        ])

        first  = payloads[0] if payloads else {}
        skills = first.get("skills", [])
        if isinstance(skills, list):
            skills = ", ".join(skills)

        block = f"""
--- Candidate: {candidate_name} ---
Years of Experience : {first.get('years_experience', 'Unknown')}
Email               : {first.get('email', 'Unknown')}
Skills              : {skills}
Resume Content      :
{all_content}
"""
        formatted.append(block)

    return "\n\n".join(formatted)


# =========================
# Main Ask Function
# =========================

def ask(query, min_experience=None, top_n=5):

    candidate_chunks = retriever_system.retrieve(
        query=query,
        min_experience=min_experience,
        top_n=top_n
    )

    if not candidate_chunks:
        return "No matching candidates found in the provided resumes."

    context = format_chunks(candidate_chunks)

    filled_prompt = prompt.format(
        context=context,
        question=query
    )

    print("\nAI RESPONSE:\n")
    response = ""
    for chunk in llm.stream(filled_prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content

    print()
    return response