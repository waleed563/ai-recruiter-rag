import os

# dotenv may not be installed in some environments (linters/CI). Provide a
# safe fallback so import errors won't break static analysis or runtime usage.
try:
	from dotenv import load_dotenv  # type: ignore
except Exception:
	def load_dotenv():
		return None

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")