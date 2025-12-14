"""
Solution matrix checker for Frontier-CS.

Validates solution directory coverage against models/problems configuration.
Generates pairs.txt for batch evaluation.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    get_model_prefix,
    sanitize_problem_name,
    normalize_solution_name,
    MODEL_PREFIX_ALIASES,
)


def read_list_file(path: Path) -> List[str]:
    """Read a list file (one item per line, # comments, blank lines ignored)."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    items: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(line)
    if not items:
        raise ValueError(f"No valid entries found in {path}")
    return items


def read_problem_list(path: Path) -> List[str]:
    """Read problems from problems.txt, normalizing 'research/' prefix."""
    problems = []
    for entry in read_list_file(path):
        # Remove 'research/' prefix if present
        normalized = entry.split("research/", 1)[-1]
        problems.append(normalized)
    return problems


def read_models_list(path: Path) -> List[str]:
    """Read unique models from models.txt."""
    models: List[str] = []
    seen: Set[str] = set()
    for entry in read_list_file(path):
        if entry not in seen:
            models.append(entry)
            seen.add(entry)
    return models


def read_variant_indices(path: Path) -> List[int]:
    """
    Read variant indices from num_solutions.txt.

    Supports two formats:
    - Single integer N: expands to [0, 1, ..., N-1]
    - Multiple integers: uses those as the indices
    """
    values = read_list_file(path)

    # Single integer -> expand to range
    if len(values) == 1:
        try:
            n = int(values[0])
            if n <= 0:
                return [0]
            return list(range(n))
        except ValueError:
            pass

    # Multiple integers
    indices: List[int] = []
    seen: Set[int] = set()
    for v in values:
        try:
            idx = int(v)
        except ValueError as exc:
            raise ValueError(f"Invalid variant index in {path}: '{v}'") from exc
        if idx < 0:
            raise ValueError(f"Variant indices must be >= 0, got {idx}")
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    if not indices:
        return [0]
    return indices


def expected_solution_names(
    problems: List[str],
    model_prefixes: List[str],
    variant_indices: List[int],
) -> Tuple[Dict[Tuple[str, str], List[str]], Dict[str, str]]:
    """
    Generate expected solution directory names.

    Returns:
        - mapping: {(model_prefix, problem): [solution_names]}
        - slug_to_problem: {problem_slug: problem_path}
    """
    mapping: Dict[Tuple[str, str], List[str]] = {}
    slug_to_problem: Dict[str, str] = {}

    for problem in problems:
        slug = sanitize_problem_name(problem)
        slug_to_problem[slug] = problem

        for prefix in model_prefixes:
            names: List[str] = []
            base = f"{prefix}_{slug}"
            for idx in variant_indices:
                if idx == 0:
                    names.append(base)
                else:
                    names.append(f"{base}_{idx}")
            mapping[(prefix, problem)] = names

    return mapping, slug_to_problem


def collect_solution_dirs(root: Path) -> List[str]:
    """List all solution directories."""
    if not root.is_dir():
        return []
    return sorted(entry.name for entry in root.iterdir() if entry.is_dir())


def collect_problem_dirs(root: Path) -> List[str]:
    """
    Find all problem directories (directories with readme files).

    Returns paths relative to root, up to 2 levels deep.
    """
    candidates: Set[str] = set()
    if not root.is_dir():
        return []

    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if len(relative.parts) > 2:
            continue
        entries = {child.name.lower() for child in path.iterdir() if child.is_file()}
        if any(name in entries for name in {"readme.md", "readme"}):
            candidates.add(str(relative))

    return sorted(candidates)


class MatrixCheckResult:
    """Result of a solution matrix check."""

    def __init__(self):
        self.problems: List[str] = []
        self.models: List[str] = []
        self.model_prefixes: List[str] = []
        self.variant_indices: List[int] = []

        self.expected_count: int = 0
        self.actual_count: int = 0

        self.missing_by_key: Dict[Tuple[str, str], List[str]] = {}
        self.extra_dirs: List[str] = []

        self.missing_problem_dirs: List[str] = []
        self.extra_problem_dirs: List[str] = []

        self.slug_to_problem: Dict[str, str] = {}

    @property
    def total_missing(self) -> int:
        return sum(len(names) for names in self.missing_by_key.values())

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = []
        lines.append("=== Solution Matrix Check ===")
        lines.append(f"Problems listed: {len(self.problems)}")
        lines.append(f"Models listed:   {len(self.models)} ({', '.join(self.model_prefixes)})")
        lines.append(
            f"Variants per model/problem: {len(self.variant_indices)} "
            f"(indices: {', '.join(map(str, self.variant_indices))})"
        )
        lines.append(f"Expected solution directories: {self.expected_count}")
        lines.append(f"Actual solution directories:   {self.actual_count}")
        lines.append("")

        lines.append(f"Missing solution directories: {self.total_missing}")
        if self.total_missing:
            for (model_prefix, problem), names in sorted(self.missing_by_key.items()):
                lines.append(f"  - {model_prefix} -> {problem}: missing {len(names)}")
                for entry in names[:5]:
                    lines.append(f"      · {entry}")
                if len(names) > 5:
                    lines.append(f"      · ... ({len(names) - 5} more)")
        lines.append("")

        lines.append(f"Extra solution directories: {len(self.extra_dirs)}")
        if self.extra_dirs:
            for name in self.extra_dirs[:20]:
                model_hint = name.split("_", 1)[0] if "_" in name else name
                slug = name.split("_", 1)[1] if "_" in name else ""
                hint = ""
                if slug:
                    problem_hint = self.slug_to_problem.get(slug)
                    if problem_hint:
                        hint = f" (possible problem: {problem_hint})"
                lines.append(f"  - {name}{hint}")
            if len(self.extra_dirs) > 20:
                lines.append(f"  - ... ({len(self.extra_dirs) - 20} more)")
        lines.append("")

        lines.append("=== Problems Directory Check ===")
        lines.append(f"Missing problem directories: {len(self.missing_problem_dirs)}")
        for path in self.missing_problem_dirs[:10]:
            lines.append(f"  - {path}")
        if len(self.missing_problem_dirs) > 10:
            lines.append(f"  - ... ({len(self.missing_problem_dirs) - 10} more)")

        lines.append(f"Extra problem directories: {len(self.extra_problem_dirs)}")
        for path in self.extra_problem_dirs[:10]:
            lines.append(f"  - {path}")
        if len(self.extra_problem_dirs) > 10:
            lines.append(f"  - ... ({len(self.extra_problem_dirs) - 10} more)")

        return "\n".join(lines)


def check_solution_matrix(
    problems_file: Path,
    models_file: Path,
    variants_file: Path,
    solutions_dir: Path,
    problems_dir: Path,
) -> MatrixCheckResult:
    """
    Check solution directory coverage against configuration.

    Args:
        problems_file: Path to problems.txt
        models_file: Path to models.txt
        variants_file: Path to num_solutions.txt
        solutions_dir: Path to solutions directory
        problems_dir: Path to problems directory (e.g., research/)

    Returns:
        MatrixCheckResult with coverage analysis
    """
    result = MatrixCheckResult()

    # Read configuration
    result.problems = read_problem_list(problems_file)
    result.models = read_models_list(models_file)
    result.variant_indices = read_variant_indices(variants_file)

    if not result.problems:
        raise ValueError("No problems configured")
    if not result.models:
        raise ValueError("No models configured")

    # Convert model names to prefixes
    result.model_prefixes = [get_model_prefix(model) for model in result.models]

    # Generate expected solution names
    expected_map, result.slug_to_problem = expected_solution_names(
        result.problems,
        result.model_prefixes,
        result.variant_indices,
    )
    expected_dirs = {name for names in expected_map.values() for name in names}
    result.expected_count = len(expected_dirs)

    # Collect actual solution directories
    actual_dirs = set(collect_solution_dirs(solutions_dir))

    # Normalize names (apply aliases)
    normalized_dirs = set()
    for name in actual_dirs:
        normalized_dirs.add(normalize_solution_name(name))
    actual_dirs = normalized_dirs
    result.actual_count = len(actual_dirs)

    # Find missing solutions
    for key, names in expected_map.items():
        missing = [n for n in names if n not in actual_dirs]
        if missing:
            result.missing_by_key[key] = missing

    # Find extra solutions
    result.extra_dirs = sorted(actual_dirs - expected_dirs)

    # Check problem directories
    result.missing_problem_dirs = [
        p for p in result.problems if not (problems_dir / p).exists()
    ]

    actual_problem_dirs = collect_problem_dirs(problems_dir)
    result.extra_problem_dirs = sorted(set(actual_problem_dirs) - set(result.problems))

    return result


def generate_pairs_file(
    result: MatrixCheckResult,
    output_path: Path,
    include_missing: bool = True,
) -> int:
    """
    Generate a pairs file from the check result.

    Args:
        result: MatrixCheckResult from check_solution_matrix
        output_path: Path to write pairs file
        include_missing: If True, include missing solutions (for generation)
                        If False, only include existing solutions (for evaluation)

    Returns:
        Number of pairs written
    """
    lines = ["# auto-generated pairs from frontier-eval check"]

    # Generate expected solution names again
    expected_map, _ = expected_solution_names(
        result.problems,
        result.model_prefixes,
        result.variant_indices,
    )

    count = 0
    for (prefix, problem), solutions in sorted(expected_map.items(), key=lambda x: (x[0][1], x[0][0])):
        for solution in sorted(solutions):
            if include_missing or solution not in result.missing_by_key.get((prefix, problem), []):
                lines.append(f"{solution}:{problem}")
                count += 1

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return count
