"""
Check all candidate names currently in Qdrant
to find badly extracted ones before fixing anything
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.config.settings import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Get all summary chunks — one per candidate
result, _ = client.scroll(
    collection_name=QDRANT_COLLECTION,
    limit=1000,
    with_payload=True,
    with_vectors=False,
    scroll_filter=Filter(
        must=[FieldCondition(
            key="section",
            match=MatchValue(value="summary_index")
        )]
    )
)

names = sorted([p.payload.get("candidate_name", "Unknown") for p in result])

print(f"Total candidates: {len(names)}\n")

# Flag suspicious names
good    = []
bad     = []

for name in names:
    words = name.split()
    is_bad = (
        name == "Unknown"                    or
        len(words) > 4                       or  # too long = job title crept in
        any(w.lower() in [
            "developer", "engineer", "manager",
            "analyst", "consultant", "architect",
            "senior", "junior", "lead", "java",
            "python", "resume", "cv"
        ] for w in words)                    or  # job title words
        any(char.isdigit() for char in name) or  # contains numbers
        len(name) < 2                            # too short
    )
    if is_bad:
        bad.append(name)
    else:
        good.append(name)

print(f"✓ Good names  : {len(good)}")
print(f"✗ Bad names   : {len(bad)}\n")

if bad:
    print("BAD NAMES FOUND:")
    for name in bad:
        print(f"  → {name}")