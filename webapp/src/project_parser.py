import re

KNOWN_SKILLS = [
    "Python",
    "Java",
    "JavaScript",
    "React",
    "Angular",
    "Node.js",
    "Flask",
    "Django",
    "Spring Boot",
    "SQL",
    "MySQL",
    "PostgreSQL",
    "MongoDB",
    "DevOps",
    "Docker",
    "Kubernetes",
    "AWS",
    "Azure",
    "Git",
    "Linux",
    "Cybersecurity",
    "Testing",
    "Automation Testing",
    "Machine Learning",
    "Deep Learning",
    "Data Analysis",
    "Data Science",
    "UI/UX Design",
    "PHP",
    "Laravel",
    "Go",
    "Rust",
    "Scala",
    "Android",
    "iOS",
]


def extract_skills_from_text(project_description: str) -> list[str]:
    text = project_description.lower()
    found = []

    for skill in KNOWN_SKILLS:
        if re.search(re.escape(skill.lower()), text):
            found.append(skill)

    return sorted(set(found))