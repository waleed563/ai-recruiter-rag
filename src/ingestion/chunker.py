from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ResumeChunker:

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,       # max characters per chunk
            chunk_overlap=50,     # overlap so context is not lost at edges
            separators=[
                "\n\n",   # try to split on double newline first (job breaks)
                "\n",     # then single newline
                ".",      # then sentence endings
                " "       # last resort: split on a space
            ]
        )

    def chunk_sections(self, parsed_sections, metadata):

        chunks = []

        # =========================
        # Regular Section Chunks
        # =========================

        for section_name, content in parsed_sections.items():

            if not content.strip():
                continue

            splits = self.splitter.split_text(content)

            for i, split in enumerate(splits):

                chunk = Document(
                    page_content=split,
                    metadata={
                        **metadata,
                        "section": section_name,
                        "chunk_index": i
                    }
                )

                chunks.append(chunk)

        # =========================
        # Summary Index Chunk
        # One special card per resume
        # Used for fast first-stage search
        # so retriever finds the right candidate
        # without reading all 50 chunks
        # =========================

        summary_text = f"""
Candidate: {metadata.get('candidate_name', 'Unknown')}
Years of experience: {metadata.get('years_experience', 'Unknown')}
Skills: {', '.join(metadata.get('skills', []))}
Email: {metadata.get('email', '')}
        """.strip()

        summary_chunk = Document(
            page_content=summary_text,
            metadata={
                **metadata,
                "section": "summary_index"
            }
        )

        chunks.append(summary_chunk)

        return chunks