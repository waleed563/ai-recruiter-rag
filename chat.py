from src.chains.rag_chain import ask


# =========================
# Run a query
# Change the query here to test
# different recruiter searches
# =========================

if __name__ == "__main__":

    print("\n" + "=" * 50)
    print("AI RECRUITER — Powered by RAG")
    print("=" * 50)

    # Example 1: search by skills
    query = (
        "Find Java backend engineers with "
        "AWS, Docker, and microservices experience"
    )

    # Example 2: search with experience filter
    # result = ask(query, min_experience=5)

    # Example 3: get more candidates
    # result = ask(query, top_n=10)

    print(f"\nQuery: {query}\n")

    result = ask(query)

    print("\n" + "=" * 50)