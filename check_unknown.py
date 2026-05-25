"""
Show the general section of Unknown candidates
so we can see what format their names are in
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.config.settings import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Get general section chunks for Unknown candidates
result, _ = client.scroll(
    collection_name=QDRANT_COLLECTION,
    limit=1000,
    with_payload=True,
    with_vectors=False,
    scroll_filter=Filter(
        must=[
            FieldCondition(
                key="candidate_name",
                match=MatchValue(value="Unknown")
            ),
            FieldCondition(
                key="section",
                match=MatchValue(value="general")
            )
        ]
    )
)

print(f"Unknown candidates general sections: {len(result)}\n")

# Show first 10 so we can see the pattern
for i, point in enumerate(result[:10], 1):
    print(f"--- Unknown #{i} ---")
    print(f"file     : {point.payload.get('file_name', 'N/A')}")
    print(f"content  : {point.payload.get('page_content', '')[:300]}")
    print()