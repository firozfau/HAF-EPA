from typing import Dict


def load_datasets() -> Dict[str, str]:
    """
    Week 1 starter dataset loader.
    Actual dataset loading will be implemented in Week 2.
    """
    return {
        "status": "ready",
        "message": "Dataset loading module initialized. Full CSV loading will be added in Week 2.",
        "planned_datasets": "employees, projects, tasks, employee_skills, project_skills, skills",
    }
