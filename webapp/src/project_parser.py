import re

# All supported skills used for extracting project requirements
# from uploaded PDF project descriptions.
#
# IMPORTANT:
# - Programming skills
# - Management skills
# both are supported now.
#
# These skills are matched against PDF text using regex search.

KNOWN_SKILLS = [
    # =========================
    # Programming / Technical Skills
    # =========================
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "React",
    "Angular",
    "Vue",
    "Node.js",
    "Express.js",
    "Next.js",
    "Flask",
    "Django",
    "Spring Boot",
    "SQL",
    "MySQL",
    "PostgreSQL",
    "MongoDB",
    "Redis",
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
    "TensorFlow",
    "PyTorch",
    "REST API",
    "GraphQL",
    "Microservices",
    "CI/CD",
    "Firebase",
    "HTML",
    "CSS",
    "Tailwind CSS",
    "Bootstrap",
    "C++",
    "C#",
    "Kotlin",
    "Swift",

    # =========================
    # Management / Soft Skills
    # =========================
    "Project Management",
    "Agile",
    "Scrum",
    "Leadership",
    "Team Management",
    "Communication",
    "Planning",
    "Coordination",
    "Risk Management",
    "Time Management",
    "Stakeholder Management",
    "Resource Management",
    "Budgeting",
    "Documentation",
    "Presentation",
    "Decision Making",
    "Problem Solving",
    "Requirement Analysis",
    "Business Analysis",
    "Quality Management",
    "Client Communication",
    "Project Planning",
    "Sprint Planning",
    "Task Management",
    "Team Leadership",
    "Conflict Resolution",
    "Strategic Thinking",
    "Negotiation",
    "Process Improvement",
]


def extract_skills_from_text(project_description: str) -> list[str]:
    """
    Extract matching skills from uploaded PDF project description text.

    Returns:
        list[str]: detected project skills
    """

    text = project_description.lower()
    found = []

    for skill in KNOWN_SKILLS:
        # safer regex matching with word boundary support
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"

        if re.search(pattern, text):
            found.append(skill)

    # remove duplicates and sort result
    return sorted(set(found))