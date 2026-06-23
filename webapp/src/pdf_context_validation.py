import re
 
REQUIRED_SECTIONS = {
    "project_overview": [
        r"\bproject overview\b",
        r"\boverview\b",
    ],
    "technology_requirements": [
        r"\btechnology requirements\b",
        r"\btechnical requirements\b",
        r"\btech stack\b",
        r"\bfrontend\b",
        r"\bbackend\b",
        r"\bdatabase\b",
    ],
    "required_skills": [
        r"\brequired skills\b",
        r"\bskills required\b",
    ],
    "expected_outcome": [
        r"\bexpected outcome\b",
        r"\bdeliverables\b",
        r"\bobjectives\b",
    ],
}

INVALID_HINTS = [
    r"\bthesis\b",
    r"\bdeclaration\b",
    r"\bsignature\b",
    r"\bmatriculation\b",
    r"\bregulation\b",
    r"\bpolicy\b",
]

def validate_project_pdf_or_raise(project_text: str) -> None:
    text = project_text.lower()

    found_sections = {}
    for section_name, patterns in REQUIRED_SECTIONS.items():
        found_sections[section_name] = any(re.search(pattern, text) for pattern in patterns)

    missing_sections = [name for name, ok in found_sections.items() if not ok]
    has_invalid_hints = any(re.search(pattern, text) for pattern in INVALID_HINTS)

    # Minimum acceptance rule:
    # 1) must have at least 3 required sections
    # 2) must contain "required skills" section
    # 3) if clearly non-project hints exist and required sections are weak, reject
    if (
        len([x for x in found_sections.values() if x]) < 3
        or not found_sections["required_skills"]
        or (has_invalid_hints and len(missing_sections) >= 2)
    ):
        raise ValueError("Uploaded pdf file are not able to recognaize HAF-EPA training model")


