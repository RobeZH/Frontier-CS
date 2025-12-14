"""
Pair expansion and management for batch evaluation.

A Pair represents a (solution, problem) combination to evaluate.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..models import get_model_prefix, sanitize_problem_name


@dataclass
class Pair:
    """Represents a solution-problem pair for evaluation."""

    solution: str  # Solution identifier (e.g., "gpt5_flash_attn")
    problem: str   # Problem identifier (e.g., "flash_attn")

    @property
    def id(self) -> str:
        """Unique identifier for this pair."""
        return f"{self.solution}:{self.problem}"

    @property
    def safe_name(self) -> str:
        """Filesystem-safe name for this pair."""
        base = f"{self.solution}-{self.problem}"
        digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
        sanitized = _sanitize_name(base)
        suffix = f"-{digest}"
        max_len = 63
        available = max_len - len(suffix)
        trimmed = sanitized[:available].rstrip("-")
        return _sanitize_name(f"{trimmed}{suffix}")

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Pair):
            return self.id == other.id
        return False


def _sanitize_name(name: str) -> str:
    """Sanitize a name to be a valid cluster/file name."""
    cleaned = []
    valid = "abcdefghijklmnopqrstuvwxyz0123456789-"
    last_dash = False
    for ch in name.lower():
        if ch in valid:
            cleaned.append(ch)
            last_dash = ch == "-"
        else:
            if not last_dash:
                cleaned.append("-")
                last_dash = True
    sanitized = "".join(cleaned).strip("-")
    return sanitized or "job"


# get_model_prefix and sanitize_problem_name imported from ..models


def expand_pairs(
    problems: List[str],
    models: List[str],
    variants: Optional[List[int]] = None,
    *,
    solutions_dir: Optional[Path] = None,
    validate_paths: bool = True,
) -> List[Pair]:
    """
    Expand problems × models × variants into pairs.

    Args:
        problems: List of problem IDs (e.g., ["flash_attn", "cross_entropy"])
        models: List of model names (e.g., ["gpt-5", "claude-sonnet-4-5"])
        variants: List of variant indices (default: [0] for no suffix)
        solutions_dir: Directory containing solutions (for validation)
        validate_paths: Whether to validate solution paths exist

    Returns:
        List of Pair objects
    """
    if variants is None:
        variants = [0]

    pairs: List[Pair] = []

    for problem in problems:
        problem_name = sanitize_problem_name(problem)

        for model in models:
            model_prefix = get_model_prefix(model)

            for variant_idx in variants:
                suffix = "" if variant_idx == 0 else f"_{variant_idx}"
                solution_name = f"{model_prefix}_{problem_name}{suffix}"

                if validate_paths and solutions_dir:
                    solution_path = solutions_dir / solution_name
                    if not solution_path.exists():
                        continue

                pairs.append(Pair(solution=solution_name, problem=problem))

    return pairs


def read_pairs_file(path: Path) -> List[Pair]:
    """
    Read pairs from a pairs file.

    Format: one pair per line as "solution:problem"
    Lines starting with # are comments.
    """
    pairs: List[Pair] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                raise ValueError(f"Invalid pair line (expected solution:problem): {stripped}")
            solution, problem = stripped.split(":", 1)
            pairs.append(Pair(solution=solution.strip(), problem=problem.strip()))

    return pairs


def read_problems_file(path: Path) -> List[str]:
    """Read problems from a problems file (one per line)."""
    problems: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Normalize: remove 'research/' prefix if present
            if stripped.startswith("research/"):
                stripped = stripped[len("research/"):]
            problems.append(stripped)

    return problems


def read_models_file(path: Path) -> List[str]:
    """Read models from a models file (one per line)."""
    models: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            models.append(stripped)

    return models


def read_variants_file(path: Path) -> List[int]:
    """Read variant indices from a file (one per line)."""
    variants: List[int] = []

    if not path.exists():
        return [0]  # Default: just index 0 (no suffix)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                variants.append(int(stripped))
            except ValueError:
                pass

    return variants if variants else [0]
