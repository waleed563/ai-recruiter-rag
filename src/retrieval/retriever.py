import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
from langchain_openai import OpenAIEmbeddings

from src.config.settings import (
    OPENAI_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION
)


# =========================
# MMR — Maximal Marginal Relevance
# Picks k diverse chunks from a larger pool
# balancing relevance + diversity
#
# lambda_mult=1.0 → pure similarity (no diversity)
# lambda_mult=0.7 → 70% similarity, 30% diversity (recommended)
# lambda_mult=0.0 → pure diversity (ignores similarity)
# =========================

def apply_mmr(vectors, payloads, query_vector, k=10, lambda_mult=0.7):

    if not vectors or not payloads:
        return payloads[:k]

    # Cap k to available chunks
    k = min(k, len(payloads))

    query = np.array(query_vector)
    vecs  = np.array(vectors)

    def cosine(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    selected_idx  = []
    remaining_idx = list(range(len(vectors)))

    while len(selected_idx) < k and remaining_idx:

        mmr_scores = []

        for i in remaining_idx:

            # How relevant is this chunk to the query?
            relevance = cosine(vecs[i], query)

            # How similar is it to already selected chunks?
            if selected_idx:
                redundancy = max(
                    cosine(vecs[i], vecs[j])
                    for j in selected_idx
                )
            else:
                redundancy = 0.0

            # MMR formula: reward relevance, penalise redundancy
            mmr = lambda_mult * relevance - (1 - lambda_mult) * redundancy
            mmr_scores.append((i, mmr))

        # Pick the chunk with highest MMR score
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)

    return [payloads[i] for i in selected_idx]


class ResumeRetriever:

    def __init__(self):

        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )


    def embed_query(self, query):
        return self.embeddings.embed_query(query)


    # =========================
    # Stage 1: Candidate Discovery
    # Cosine similarity on summary_index
    # Fast — only 231 summary chunks
    # Returns top 5 candidate names
    # =========================

    def find_top_candidates(self, query, min_experience=None):

        query_vector = self.embed_query(query)

        must_conditions = [
            FieldCondition(
                key="section",
                match=MatchValue(value="summary_index")
            )
        ]

        if min_experience:
            must_conditions.append(
                FieldCondition(
                    key="years_experience",
                    range=Range(gte=min_experience)
                )
            )

        results = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            query_filter=Filter(must=must_conditions),
            limit=20,              # cast wide net
            with_payload=True,
            with_vectors=False     # don't need vectors here
        )

        candidates = {}
        for point in results.points:
            name = point.payload.get("candidate_name", "Unknown")
            if name and name != "Unknown" and name not in candidates:
                candidates[name] = point.score

        return candidates


    # =========================
    # Stage 2: Deep Dive with MMR
    # Fetch 30 chunks per candidate
    # MMR picks 10 most diverse ones
    # So LLM gets varied context
    # not 10 repetitive chunks
    # =========================

    def get_candidate_chunks(self, query, candidate_names):

        query_vector = self.embed_query(query)
        all_chunks   = {}

        for name in candidate_names:

            # Fetch 30 chunks — MMR will pick best 10 from these
            results = self.client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="candidate_name",
                            match=MatchValue(value=name)
                        )
                    ]
                ),
                limit=30,              # fetch more than needed
                with_payload=True,
                with_vectors=True      # need vectors for MMR math
            )

            payloads = [p.payload      for p in results.points]
            vectors  = [p.vector       for p in results.points]

            # Apply MMR to pick 10 diverse chunks from 30
            diverse_payloads = apply_mmr(
                vectors=vectors,
                payloads=payloads,
                query_vector=query_vector,
                k=10,
                lambda_mult=0.7        # 70% relevant, 30% diverse
            )

            all_chunks[name] = diverse_payloads
            print(f"  MMR selected {len(diverse_payloads)}/30 "
                  f"chunks for: {name}")

        return all_chunks


    # =========================
    # Main Retrieve Method
    # =========================

    def retrieve(self, query, min_experience=None, top_n=5):

        print(f"\nSearching: {query}")

        # Stage 1: cosine similarity → top 5 candidates
        candidates = self.find_top_candidates(query, min_experience)

        if not candidates:
            print("No candidates found.")
            return {}

        top_candidates = dict(
            sorted(
                candidates.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
        )

        print(f"\nTop {top_n} candidates:")
        for name, score in top_candidates.items():
            print(f"  {name} — score: {score:.4f}")

        # Stage 2: MMR → 10 diverse chunks per candidate
        print("\nFetching diverse chunks (MMR):")
        chunks = self.get_candidate_chunks(
            query,
            list(top_candidates.keys())
        )

        return chunks