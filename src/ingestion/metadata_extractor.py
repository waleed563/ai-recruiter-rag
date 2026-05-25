import re


# =========================
# Skills Master List
# =========================

SKILL_KEYWORDS = [
    # Languages
    "python", "java", "javascript", "typescript", "sql", "pl/sql",
    "c", "c++", "groovy", "kotlin", "scala", "shell", "bash",
    # Frontend
    "react", "angular", "angularjs", "html", "css", "jquery",
    "bootstrap", "ajax", "json", "node", "nodejs",
    # Backend / Frameworks
    "spring", "spring boot", "spring mvc", "spring security",
    "spring batch", "spring cloud", "spring aop", "spring ioc",
    "hibernate", "struts", "jsf", "jsp", "servlets", "jpa",
    "ejb", "jstl", "jms", "jndi", "jdbc",
    "django", "fastapi", "flask",
    # Cloud
    "aws", "ec2", "s3", "rds", "lambda", "azure", "gcp",
    "cloud foundry", "docker", "kubernetes",
    # Databases
    "oracle", "mysql", "postgresql", "mongodb", "db2",
    "cassandra", "redis", "elasticsearch", "dynamodb", "sql server",
    # Web Services
    "rest", "soap", "graphql", "microservices", "wsdl", "jax-rs",
    "jax-ws", "apache camel", "rabbitmq", "kafka", "ibm mq",
    # DevOps / Tools
    "jenkins", "maven", "gradle", "ant", "git", "svn",
    "jira", "docker", "linux", "unix",
    # Testing
    "junit", "mockito", "selenium",
    # Messaging
    "activemq", "jms",
]


# =========================
# Invalid name words
# Words that are NOT part of a person's name
# =========================

INVALID_NAME_WORDS = set([
    "resume", "cv", "profile", "summary", "objective",
    "address", "phone", "email", "page", "http", "www",
    "developer", "engineer", "manager", "analyst", "consultant",
    "architect", "senior", "junior", "lead", "director", "head",
    "specialist", "coordinator", "executive", "officer", "president",
    "professional", "scrum", "master", "project", "business",
    "systems", "technical", "technology", "certified", "expert",
    "possess", "proficient", "experienced", "ability", "have",
    "strong", "it", "sr", "mr", "ms", "dr", "skype", "mob",
    "java", "python", "sql", "aws", "data", "software", "web",
    "psm", "csm", "scjp", "pmp", "lean", "six", "sigma", "belt",
    "sun", "microsoft", "oracle", "ibm", "information",
])


class MetadataExtractor:

    def __init__(self):
        pass


    # =========================
    # Helper: is this word part of a name?
    # =========================

    def _is_name_word(self, word):
        w = word.lower().strip(".,/():;-")
        return (
            len(w) >= 2 and
            w not in INVALID_NAME_WORDS and
            not any(c.isdigit() for c in w) and
            len(word) > 0 and
            word[0].isupper()
        )


    # =========================
    # Helper: extract consecutive
    # name-like words from a string
    # =========================

    def _extract_name_words(self, text):
        words = text.split()
        name_words = []
        for word in words:
            if self._is_name_word(word):
                name_words.append(word)
            else:
                if name_words:
                    break   # stop at first non-name word
        if 1 <= len(name_words) <= 4:
            return " ".join(name_words)
        return None


    # =========================
    # Helper: clean a line
    # removes email, phone, URLs,
    # certifications, labels
    # =========================

    def _clean_line(self, line):
        line = re.sub(r"\S+@\S+", "", line)
        line = re.sub(r"(\+?[\d][\d\-\(\)\.\s]{7,}[\d])", "", line)
        line = re.sub(r"http\S+", "", line)
        line = re.sub(r"\(.*?\)", "", line)
        line = re.sub(
            r"(?i)(email|phone|mob|mobile|tel|skype|linkedin)\s*[:/]?\s*",
            "", line
        )
        return " ".join(line.split()).strip()


    # =========================
    # Helper: extract name from filename
    # Last resort fallback
    # 'Nithin_Katapally_Java.docx' → 'Nithin Katapally'
    # =========================

    def _name_from_filename(self, filename):
        name = filename.replace(".docx", "").replace("_", " ").replace("-", " ")
        return self._extract_name_words(name) or "Unknown"


    # =========================
    # Extract Name
    # Tries 4 strategies in order
    # =========================

    def extract_name(self, general_section, file_name=""):

        lines = general_section.split("\n")

        for line in lines[:15]:
            line = line.strip()
            if not line:
                continue

            # Strategy 1: explicit Name: prefix
            if re.match(r"(?i)^name\s*:", line):
                name = re.sub(r"(?i)^name\s*:", "", line).strip()
                name = self._clean_line(name)
                result = self._extract_name_words(name)
                if result:
                    return result

            # Strategy 2: slash-separated line
            # handles: "Lorenzo Zackery / email@gmail.com / 215.980.1011"
            if "/" in line:
                parts = [p.strip() for p in line.split("/")]
                for part in parts:
                    part = self._clean_line(part)
                    result = self._extract_name_words(part)
                    if result:
                        return result

            # Strategy 3: clean the line and extract name words
            # handles: "Ajay Kumar (CSM)  Email/Skype: ajaydt@gmail.com"
            #      and: "Harshitha         Email:harshithac511@gmail.com"
            cleaned = self._clean_line(line)
            result = self._extract_name_words(cleaned)
            if result:
                return result

        # Strategy 4: fallback to filename
        # handles: resumes where name is buried deep
        if file_name:
            return self._name_from_filename(file_name)

        return "Unknown"


    # =========================
    # Extract Email
    # =========================

    def extract_email(self, text):
        match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        return match.group(0) if match else None


    # =========================
    # Extract Phone
    # =========================

    def extract_phone(self, text):
        match = re.search(r"(\+?\d[\d\-\(\) ]{8,}\d)", text)
        return match.group(0) if match else None


    # =========================
    # Extract Years Experience
    # =========================

    def extract_years_experience(self, summary, full_text):
        patterns = [
            r"(\d+)\+?\s*(?:years?|yrs?)\s*of\s*(?:strong\s*)?(?:software|professional|work|industry)",
            r"around\s+(\d+)\s+years",
            r"over\s+(\d+)\s+years",
            r"(\d+)\+?\s*(?:years?|yrs?)",
        ]
        for source in [summary, full_text]:
            for pattern in patterns:
                match = re.search(pattern, source.lower())
                if match:
                    return int(match.group(1))
        return None


    # =========================
    # Extract Skills
    # =========================

    def extract_skills(self, skills_section, full_text):
        source = skills_section if skills_section.strip() else full_text
        source_lower = source.lower()
        found = []
        for skill in SKILL_KEYWORDS:
            if skill in source_lower:
                found.append(skill)
        return list(set(found))


    # =========================
    # Main: Extract All Metadata
    # FIX: now passes file_name to extract_name
    # =========================

    def extract_metadata(self, parsed_sections, file_name=""):

        general   = parsed_sections.get("general", "")
        summary   = (
            parsed_sections.get("summary", "") or
            parsed_sections.get("objective", "")
        )
        skills    = parsed_sections.get("skills", "")
        full_text = "\n".join(parsed_sections.values())

        return {
            "candidate_name":   self.extract_name(general, file_name),
            "email":            self.extract_email(full_text),
            "phone":            self.extract_phone(full_text),
            "years_experience": self.extract_years_experience(summary, full_text),
            "skills":           self.extract_skills(skills, full_text),
            "file_name":        file_name,
        }