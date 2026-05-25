from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
# Chunks are now raw dicts
# from Qdrant payload
# =========================

def format_chunks(candidate_chunks):

    formatted = []

    for candidate_name, payloads in candidate_chunks.items():

        # Merge all chunk text for this candidate
        all_content = "\n".join([
            p.get("page_content", "")
            for p in payloads
        ])

        # Grab metadata from first payload
        first = payloads[0] if payloads else {}

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
        print(f"  Context built for: {candidate_name} ({len(payloads)} chunks)")

    return "\n\n".join(formatted)


# =========================
# Main Ask Function
# =========================

def ask(query, min_experience=None, top_n=5):

    # Retrieve best candidates and their chunks
    candidate_chunks = retriever_system.retrieve(
        query=query,
        min_experience=min_experience,
        top_n=top_n
    )

    if not candidate_chunks:
        return "No matching candidates found in the provided resumes."

    print("\nBuilding context:")
    context = format_chunks(candidate_chunks)

    filled_prompt = prompt.format(
        context=context,
        question=query
    )

    # Stream response word by word
    print("\nAI RESPONSE:\n")
    response = ""
    for chunk in llm.stream(filled_prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content

    print()
    return response