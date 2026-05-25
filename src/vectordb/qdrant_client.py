from qdrant_client import QdrantClient

from qdrant_client.models import (
    VectorParams,
    Distance,
    PayloadSchemaType,
    PointStruct
)

import uuid

from langchain_openai import OpenAIEmbeddings

from src.config.settings import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION
)


# =========================
# Initialize Qdrant Client
# =========================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=300
)


# =========================
# Create Collection
# =========================

def create_collection():

    collections = client.get_collections().collections

    collection_names = [
        collection.name
        for collection in collections
    ]

    if QDRANT_COLLECTION not in collection_names:

        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=1536,        # text-embedding-3-small output size
                distance=Distance.COSINE
            )
        )

        # Index years_experience so recruiters can filter by it
        # e.g. "show me candidates with 5+ years"
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="years_experience",
            field_schema=PayloadSchemaType.INTEGER
        )

        # Index candidate_name for name-based filtering
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="candidate_name",
            field_schema=PayloadSchemaType.KEYWORD
        )

        # Index section so we can search only skills or only experience
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="section",
            field_schema=PayloadSchemaType.KEYWORD
        )

        print(f"Created collection: {QDRANT_COLLECTION}")

    else:
        print(f"Collection already exists: {QDRANT_COLLECTION}")


# =========================
# Upload Documents
# FIX: payload is now flat — all metadata at top level
# so Qdrant can filter by candidate_name, section, skills directly
# =========================

def upload_documents(documents):

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    batch_size = 10
    total_docs = len(documents)

    for i in range(0, total_docs, batch_size):

        batch = documents[i:i + batch_size]

        print(f"\nUploading batch {i // batch_size + 1}")

        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]

        # Translate text into numbers (embeddings)
        vectors = embeddings.embed_documents(texts)

        points = []

        for text, metadata, vector in zip(texts, metadatas, vectors):

            # FIX: flatten payload — no nested metadata box
            # Every key sits at the top level for easy filtering
            payload = {
                "page_content": text,
                **metadata          # spreads all keys directly here
            }

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload
                )
            )

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
            wait=True           # wait until Qdrant confirms before moving on
        )

        print(f"Uploaded {len(points)} chunks")

    print("\nAll chunks uploaded successfully.")