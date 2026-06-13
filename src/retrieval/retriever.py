import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable

from src.config.settings import (
    OPENAI_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION
)


# =========================
# Config flags
# Flip USE_MMR to compare quality vs speed.
#   USE_MMR = False  → fast path, single fetch, no vector download
#   USE_MMR = True   → diverse chunks, but fetches vectors (slower)
# =========================

USE_MMR        = False   # start fast; turn on only if answers get repetitive
MMR_POOL       = 15      # how many chunks to fetch before MMR trims (was 30)
CHUNKS_PER_CAND = 10     # final chunks handed to the LLM per candidate


# =========================
# MMR — Maximal Marginal Relevance
# Picks k diverse chunks from a larger pool
# balancing relevance + diversity
#
# lambda_mult=1.0 → pure similarity (no diversity)
# lambda_mult=0.7 → 70% similarity, 30% diversity (recommended)
# lambda_mult=0.0 → pure diversity (ignores similarity)
# =========================

@traceable(name="apply-mmr")
def apply_mmr(vectors, payloads, query_vector, k=10, lambda_mult=0.7):

    if not vectors or not payloads:
        return payloads[:k]

    k = min(k, len(payloads))

    query = np.array(query_vector)
    vecs  = np.array(vectors)

    # Pre-normalise once so cosine is a single dot product.
    # Avoids recomputing norms inside the nested loop.
    def normalize(matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    vecs_n  = normalize(vecs)
    query_n = query / (np.linalg.norm(query) or 1.0)

    # Relevance of every chunk to the query — vectorised, computed once
    relevance = vecs_n @ query_n          # shape: (n,)

    selected_idx  = []
    remaining_idx = list(range(len(vectors)))

    while len(selected_idx) < k and remaining_idx:

        best_i     = None
        best_score = -np.inf

        for i in remaining_idx:
            if selected_idx:
                # similarity to already-picked chunks
                redundancy = float(np.max(vecs_n[selected_idx] @ vecs_n[i]))
            else:
                redundancy = 0.0

            mmr = lambda_mult * relevance[i] - (1 - lambda_mult) * redundancy
            if mmr > best_score:
                best_score = mmr
                best_i     = i

        selected_idx.append(best_i)
        remaining_idx.remove(best_i)

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


    @traceable(name="embed-query")
    def embed_query(self, query):
        return self.embeddings.embed_query(query)


    # =========================
    # Stage 1: Candidate Discovery
    # Cosine similarity on summary_index
    # Fast — only 231 summary chunks
    # NOTE: now takes a pre-computed query_vector
    #       (no embedding call inside)
    # =========================

    @traceable(name="find-top-candidates")
    def find_top_candidates(self, query_vector, min_experience=None):

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
            limit=20,
            with_payload=True,
            with_vectors=False
        )

        candidates = {}
        for point in results.points:
            name = point.payload.get("candidate_name", "Unknown")
            if name and name != "Unknown" and name not in candidates:
                candidates[name] = point.score

        return candidates


    # =========================
    # Stage 2: Per-candidate chunks
    # Still one Qdrant call per candidate, but:
    #   - reuses the query_vector (no re-embed)
    #   - only downloads vectors when MMR is on
    #   - fetches a smaller pool
    # =========================

    @traceable(name="get-candidate-chunks")
    def get_candidate_chunks(self, query_vector, candidate_names):

        all_chunks = {}

        for name in candidate_names:

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
                # fetch a small pool if MMR is on, else just the final count
                limit=MMR_POOL if USE_MMR else CHUNKS_PER_CAND,
                with_payload=True,
                with_vectors=USE_MMR        # only pay for vectors when needed
            )

            payloads = [p.payload for p in results.points]

            if USE_MMR:
                vectors = [p.vector for p in results.points]
                chosen  = apply_mmr(
                    vectors=vectors,
                    payloads=payloads,
                    query_vector=query_vector,
                    k=CHUNKS_PER_CAND,
                    lambda_mult=0.7
                )
            else:
                # Qdrant already returned these in similarity order
                chosen = payloads[:CHUNKS_PER_CAND]

            all_chunks[name] = chosen
            print(f"  selected {len(chosen)} chunks for: {name}")

        return all_chunks


    # =========================
    # Main Retrieve Method
    # Embeds the query ONCE, then reuses the vector
    # for both stages.
    # =========================

    @traceable(name="retrieve")
    def retrieve(self, query, min_experience=None, top_n=5):

        print(f"\nSearching: {query}")

        # Embed ONCE — both stages reuse this vector
        query_vector = self.embed_query(query)

        # Stage 1: cosine similarity → candidate shortlist
        candidates = self.find_top_candidates(query_vector, min_experience)

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

        # Stage 2: per-candidate chunks (MMR optional)
        mode = "MMR" if USE_MMR else "fast"
        print(f"\nFetching chunks ({mode}):")
        chunks = self.get_candidate_chunks(
            query_vector,
            list(top_candidates.keys())
        )

        return chunks