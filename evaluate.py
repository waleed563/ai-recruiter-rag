"""
RAGAS Evaluation for AI Recruiter RAG System
=============================================
No datasets library needed — pure Python + OpenAI.

Run with:
    python evaluate.py
"""

import json
import time
from openai import OpenAI
from src.retrieval.retriever import ResumeRetriever
from src.chains.rag_chain import format_chunks
from src.config.settings import OPENAI_API_KEY


client = OpenAI(api_key=OPENAI_API_KEY)
retriever = ResumeRetriever()


# =========================
# 30 Test Questions
# =========================

TEST_QUESTIONS = [
    "Find Java backend engineers with Spring Boot experience",
    "Who has experience with Java microservices and Docker?",
    "Find candidates with Hibernate and JPA experience",
    "Which candidates have worked with Spring MVC and REST APIs?",
    "Find Java developers with 5 or more years of experience",
    "Who has AWS experience with EC2 and S3?",
    "Find candidates with Docker and Kubernetes experience",
    "Which developers have worked with Jenkins for CI/CD?",
    "Find engineers with cloud deployment experience on AWS",
    "Who has experience with Maven and Gradle build tools?",
    "Find candidates with Oracle database experience",
    "Who has worked with MongoDB and NoSQL databases?",
    "Find developers with SQL and PL/SQL experience",
    "Which candidates have Cassandra or Redis experience?",
    "Find engineers with both SQL and NoSQL database skills",
    "Find candidates with React or Angular frontend skills",
    "Who has experience with JavaScript and jQuery?",
    "Find full stack developers with Java backend and React frontend",
    "Which candidates have HTML CSS and Bootstrap experience?",
    "Find developers with AngularJS experience",
    "Find senior Java developers with 8 or more years experience",
    "Who has worked as a full stack Java developer?",
    "Find candidates with both frontend and backend experience",
    "Which candidates have worked in banking or finance projects?",
    "Find developers with messaging systems like Kafka or RabbitMQ",
    "Find Java developers with AWS Docker and microservices",
    "Who has Spring Boot and REST API and PostgreSQL experience?",
    "Find candidates with Python and Django experience",
    "Which developers have worked with Elasticsearch?",
    "Find engineers with Agile or Scrum methodology experience",
]


# =========================
# Score one result using GPT
# as the judge — no RAGAS needed
# =========================

def score_with_gpt(question, answer, context):

    judge_prompt = f"""
You are evaluating a RAG system for recruiters.
Score the following response on 4 metrics.
Return ONLY a JSON object, nothing else.

Question: {question}

Context retrieved:
{context[:2000]}

Answer given:
{answer[:1000]}

Score each metric from 0.0 to 1.0:

{{
  "faithfulness": <did the answer use only the context? 1.0=yes, 0.0=made things up>,
  "answer_relevancy": <did the answer address the question? 1.0=fully, 0.0=off topic>,
  "context_precision": <were the retrieved candidates relevant? 1.0=all relevant, 0.0=noise>,
  "context_recall": <did context contain enough info to answer? 1.0=complete, 0.0=missing info>
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    try:
        # Strip markdown fences if present
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0
        }


# =========================
# Run one question through
# the full pipeline
# =========================

def run_question(question, index, total):

    print(f"\n[{index}/{total}] {question}")

    try:
        # Retrieve candidates
        candidate_chunks = retriever.retrieve(query=question, top_n=5)

        if not candidate_chunks:
            print("  No candidates found")
            return None

        # Build context
        context = format_chunks(candidate_chunks)

        # Get answer from LLM
        answer_prompt = f"""
You are an AI recruitment assistant.
Answer ONLY using the provided resume context.
For each matching candidate include name, years of experience,
relevant technologies, and short reason they match.

Context:
{context}

Question: {question}

Answer:
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": answer_prompt}],
            temperature=0
        )
        answer = response.choices[0].message.content

        # Score with GPT judge
        scores = score_with_gpt(question, answer, context)

        print(f"  faithfulness={scores['faithfulness']:.2f} "
              f"relevancy={scores['answer_relevancy']:.2f} "
              f"precision={scores['context_precision']:.2f} "
              f"recall={scores['context_recall']:.2f}")

        return {
            "question":          question,
            "answer":            answer,
            "faithfulness":      scores["faithfulness"],
            "answer_relevancy":  scores["answer_relevancy"],
            "context_precision": scores["context_precision"],
            "context_recall":    scores["context_recall"],
        }

    except Exception as e:
        print(f"  FAILED: {e}")
        return None


# =========================
# Run Full Evaluation
# =========================

def run_evaluation():

    print("=" * 55)
    print("AI RECRUITER — RAGAS-style Evaluation")
    print("=" * 55)
    print(f"Questions : {len(TEST_QUESTIONS)}")
    print(f"Model     : gpt-4o-mini\n")

    all_results = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        result = run_question(question, i, len(TEST_QUESTIONS))
        if result:
            all_results.append(result)
        time.sleep(0.5)   # avoid rate limits

    if not all_results:
        print("No results collected.")
        return

    # =========================
    # Calculate averages
    # =========================

    n = len(all_results)
    metrics = {
        "faithfulness":      sum(r["faithfulness"]      for r in all_results) / n,
        "answer_relevancy":  sum(r["answer_relevancy"]  for r in all_results) / n,
        "context_precision": sum(r["context_precision"] for r in all_results) / n,
        "context_recall":    sum(r["context_recall"]    for r in all_results) / n,
    }

    # =========================
    # Print Report
    # =========================

    print("\n" + "=" * 55)
    print("EVALUATION REPORT")
    print("=" * 55)
    print(f"  Questions evaluated : {n}/{len(TEST_QUESTIONS)}\n")

    for metric, value in metrics.items():
        bar    = "█" * int(value * 20)
        status = "✓ GOOD" if value >= 0.7 else "✗ NEEDS WORK"
        print(f"  {metric:<22} {value:.3f}  {bar:<20} {status}")

    overall = sum(metrics.values()) / len(metrics)
    print(f"\n  {'Overall Score':<22} {overall:.3f}")
    print("=" * 55)

    # Save full results
    with open("evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nFull results saved to: evaluation_results.json")

    return metrics


if __name__ == "__main__":
    run_evaluation()