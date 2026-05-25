SECTION_HEADERS = {
    "summary": [
        "summary",
        "professional summary",
        "profile",
        "objective"
    ],

    "skills": [
        "skills",
        "technical skills",
        "core competencies",
        "expertise"
    ],

    "experience": [
        "experience",
        "professional experience",
        "work experience",
        "employment history"
    ],

    "education": [
        "education",
        "academic background"
    ],

    "certifications": [
        "certifications",
        "licenses"
    ],

    "projects": [
        "projects"
    ]
}



import re


def normalize_text(text):

    text = text.strip().lower()

    text = re.sub(r"[:\-]+$", "", text)

    text = re.sub(r"\s+", " ", text)

    return text

class ResumeParser:

    def __init__(self):
        self.section_headers = SECTION_HEADERS

    def detect_section(self, line):

        normalized = normalize_text(line)

        for section, keywords in self.section_headers.items():

            for keyword in keywords:

                if normalized.startswith(keyword):
                    return section

        return None

    def parse_sections(self, raw_text):

        lines = raw_text.split("\n")

        parsed_sections = {}

        current_section = "general"

        parsed_sections[current_section] = []


        for line in lines:

            line = line.strip()

            if not line:
                continue

            detected_section = self.detect_section(line)

            if detected_section:

                current_section = detected_section

                if current_section not in parsed_sections:
                    parsed_sections[current_section] = []
                print(f"Detected section: {detected_section}")

            else:
                parsed_sections[current_section].append(line)

        return {
            section: "\n".join(content)
            for section, content in parsed_sections.items()
        }