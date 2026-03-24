"""Skill — LLM domain instructions loaded from markdown files."""

from __future__ import annotations

from pathlib import Path

__all__ = ["Skill"]

_BUILT_IN_SKILLS_DIR = Path(__file__).parent / "skills"


class Skill:
    """Domain-specific instructions for LLM classification.

    Instructions are loaded from a markdown file by default. Built-in skills
    live in errorsense/skills/. Custom skills can point to any file path.

    For programmatic use (e.g. trailing review), inline instructions=
    is also supported.

    Args:
        name: Skill name. If no path or instructions given, looks for {name}.md
              in the built-in skills directory.
        path: Explicit path to a .md file. Overrides built-in lookup.
        instructions: Inline instructions string. Overrides file loading.
        prompt_format: Override the default LLM prompt format.
        temperature: LLM temperature (default: 0.0 for determinism).
    """

    def __init__(
        self,
        name: str,
        path: str | Path | None = None,
        instructions: str | None = None,
        prompt_format: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        if not name:
            raise ValueError("Skill requires a non-empty 'name'")

        self.name = name
        self.prompt_format = prompt_format
        self.temperature = temperature

        if instructions:
            self.instructions = instructions
            return

        # Load from file
        if path is not None:
            skill_path = Path(path)
        else:
            skill_path = _BUILT_IN_SKILLS_DIR / f"{name}.md"

        if not skill_path.exists():
            raise FileNotFoundError(
                f"Skill {name!r}: file not found at {skill_path}. "
                f"Create {skill_path} or pass path= to point to your skill file."
            )

        self.instructions = skill_path.read_text().strip()
        if not self.instructions:
            raise ValueError(f"Skill {name!r}: file {skill_path} is empty")
