"""Shared prompt text formatting helpers."""


def format_user_turn_text(question: str, guide_text: str = "") -> str:
    """Return user-turn text with optional guide and a terminal newline."""
    question_text = str(question).rstrip()
    guide = str(guide_text).strip()
    if guide:
        return f"{question_text}\n{guide}\n"
    return f"{question_text}\n"
