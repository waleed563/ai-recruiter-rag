import time

from src.ingestion.loader import ResumeLoader
from src.ingestion.parser import ResumeParser
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.chunker import ResumeChunker

from src.vectordb.qdrant_client import (
    client,
    create_collection,
    upload_documents
)

from src.config.settings import QDRANT_COLLECTION


loader             = ResumeLoader("data/raw_resumes")
parser             = ResumeParser()
metadata_extractor = MetadataExtractor()
chunker            = ResumeChunker()


def reset_collection():
    collections    = client.get_collections().collections
    existing_names = [col.name for col in collections]
    if QDRANT_COLLECTION in existing_names:
        print(f"Deleting old collection: {QDRANT_COLLECTION}")
        client.delete_collection(collection_name=QDRANT_COLLECTION)
        print("Old collection deleted.")
    create_collection()
    print("Fresh collection created.\n")


def upload_with_retry(documents, retries=3, delay=5):
    for attempt in range(1, retries + 1):
        try:
            upload_documents(documents)
            return True
        except Exception as e:
            print(f"  Upload attempt {attempt} failed: {e}")
            if attempt < retries:
                print(f"  Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("  All retries exhausted. Skipping.")
                return False


def process_resume(resume, index, total):
    print(f"\nProcessing {index}/{total}: {resume['file_name']}")
    try:
        parsed_sections = parser.parse_sections(resume["raw_text"])

        # FIX: pass file_name so name extractor can use it as fallback
        metadata = metadata_extractor.extract_metadata(
            parsed_sections,
            file_name=resume["file_name"]
        )

        documents = chunker.chunk_sections(parsed_sections, metadata)

        print(f"  candidate : {metadata.get('candidate_name', 'Unknown')}")
        print(f"  chunks    : {len(documents)}")
        print(f"  skills    : {len(metadata.get('skills', []))} found")

        return documents

    except Exception as e:
        print(f"  FAILED: {resume['file_name']} — {e}")
        return []


def run_pipeline():
    reset_collection()

    resumes = loader.load_resumes()
    total   = len(resumes)
    print(f"Loaded {total} resumes\n")

    succeeded    = 0
    failed       = 0
    total_chunks = 0

    for index, resume in enumerate(resumes, start=1):
        documents = process_resume(resume, index, total)
        if not documents:
            failed += 1
            continue
        success = upload_with_retry(documents)
        if success:
            succeeded    += 1
            total_chunks += len(documents)
        else:
            failed += 1

    print("\n" + "=" * 40)
    print("PIPELINE COMPLETE")
    print("=" * 40)
    print(f"  Total resumes  : {total}")
    print(f"  Succeeded      : {succeeded}")
    print(f"  Failed         : {failed}")
    print(f"  Total chunks   : {total_chunks}")
    print("=" * 40)


if __name__ == "__main__":
    run_pipeline()