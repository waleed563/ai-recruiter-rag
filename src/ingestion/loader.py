from pathlib import Path

try:
    from docx import Document
except ImportError as e:
    raise ImportError("python-docx is not installed. Install it with: pip install python-docx") from e


class ResumeLoader:

    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)

    def extract_text_from_docx(self, file_path):
        doc = Document(file_path)

        paragraphs = []

        for para in doc.paragraphs:
            text = para.text.strip()

            if text:
                paragraphs.append(text)

        return "\n".join(paragraphs)

    def load_resumes(self):

        resumes = []

        for file in self.folder_path.glob("*.docx"):

            try:
                text = self.extract_text_from_docx(file)

                resumes.append({
                    "file_name": file.name,
                    "raw_text": text
                })

                print(f"Loaded: {file.name}")

            except Exception as e:
                print(f"Error loading {file.name}: {e}")

        return resumes