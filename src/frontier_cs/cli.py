#!/usr/bin/env python3
"""
CLI interface for Frontier-CS evaluation.

Usage:
    # Single problem evaluation
    frontier-eval flash_attn solution.py
    frontier-eval --algorithmic 1 solution.cpp

    # With SkyPilot
    frontier-eval flash_attn solution.py --skypilot

    # All problems for a solution
    frontier-eval --all-problems solution.py

    # Specific problems
    frontier-eval --problems flash_attn,cross_entropy solution.py

    # List problems
    frontier-eval --list
    frontier-eval --list --algorithmic

    # Batch evaluation
    frontier-eval batch --model gpt-5 --problems flash_attn,cross_entropy
    frontier-eval batch --problems-file problems.txt --models-file models.txt
    frontier-eval batch --pairs-file pairs.txt
    frontier-eval batch --resume --results-dir results/batch1
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .evaluator import FrontierCSEvaluator
from .runner import EvaluationResult

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="frontier-eval",
        description="Evaluate solutions for Frontier-CS problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a research problem
  frontier-eval flash_attn solution.py

  # Evaluate an algorithmic problem
  frontier-eval --algorithmic 1 solution.cpp

  # Evaluate with SkyPilot (cloud)
  frontier-eval flash_attn solution.py --skypilot

  # Evaluate multiple problems
  frontier-eval --problems flash_attn,cross_entropy solution.py

  # Evaluate all research problems
  frontier-eval --all-problems solution.py

  # List available problems
  frontier-eval --list
        """,
    )

    # Problem and solution arguments (as options to avoid conflict with subcommands)
    parser.add_argument(
        "problem_id",
        nargs="?",
        default=None,
        help="Problem ID (e.g., flash_attn, gemm_optimization/squares)",
    )
    parser.add_argument(
        "solution",
        nargs="?",
        default=None,
        help="Path to solution file",
    )

    # Problem selection
    problem_group = parser.add_argument_group("Problem Selection")
    problem_group.add_argument(
        "--algorithmic",
        action="store_true",
        help="Evaluate algorithmic problem (expects numeric ID)",
    )
    problem_group.add_argument(
        "--problems",
        type=str,
        help="Comma-separated list of problem IDs to evaluate",
    )
    problem_group.add_argument(
        "--all-problems",
        action="store_true",
        help="Evaluate all problems in the track",
    )
    problem_group.add_argument(
        "--problems-file",
        type=Path,
        help="File containing problem IDs (one per line)",
    )

    # Backend options
    backend_group = parser.add_argument_group("Backend Options")
    backend_group.add_argument(
        "--skypilot",
        action="store_true",
        help="Use SkyPilot for cloud evaluation",
    )
    backend_group.add_argument(
        "--cloud",
        type=str,
        default="gcp",
        help="Cloud provider for SkyPilot (default: gcp)",
    )
    backend_group.add_argument(
        "--region",
        type=str,
        help="Cloud region for SkyPilot",
    )
    backend_group.add_argument(
        "--judge-url",
        type=str,
        default="http://localhost:8081",
        help="Judge server URL for algorithmic problems",
    )

    # Evaluation options
    eval_group = parser.add_argument_group("Evaluation Options")
    eval_group.add_argument(
        "--timeout",
        type=int,
        help="Timeout in seconds per problem",
    )
    eval_group.add_argument(
        "--code",
        type=str,
        help="Solution code as string (alternative to file)",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only output scores",
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output including logs",
    )

    # Info commands
    info_group = parser.add_argument_group("Info Commands")
    info_group.add_argument(
        "--list",
        action="store_true",
        help="List available problems",
    )
    info_group.add_argument(
        "--show",
        action="store_true",
        help="Show problem statement",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Batch subcommand
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch evaluation with incremental progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all problems for a model
  frontier-eval batch --model gpt-5 --problems flash_attn,cross_entropy

  # Evaluate from problem and model files
  frontier-eval batch --problems-file problems.txt --models-file models.txt

  # Evaluate from pairs file
  frontier-eval batch --pairs-file pairs.txt

  # Resume interrupted evaluation
  frontier-eval batch --resume --results-dir results/batch1

  # Check evaluation status
  frontier-eval batch --status --results-dir results/batch1
        """,
    )

    # Pairs input (mutually exclusive)
    pairs_group = batch_parser.add_mutually_exclusive_group()
    pairs_group.add_argument(
        "--pairs",
        type=str,
        help="Comma-separated pairs (solution:problem,solution:problem)",
    )
    pairs_group.add_argument(
        "--pairs-file",
        type=Path,
        help="Pairs file (solution:problem per line)",
    )

    # Problems input (mutually exclusive)
    problems_group = batch_parser.add_mutually_exclusive_group()
    problems_group.add_argument(
        "--problems",
        type=str,
        help="Comma-separated problem IDs",
    )
    problems_group.add_argument(
        "--problems-file",
        type=Path,
        help="Problems file (one per line)",
    )

    # Models input (mutually exclusive)
    models_group = batch_parser.add_mutually_exclusive_group()
    models_group.add_argument(
        "--models",
        type=str,
        help="Comma-separated model names (e.g., gpt-5,claude-sonnet-4-5)",
    )
    models_group.add_argument(
        "--models-file",
        type=Path,
        help="Models file (one per line)",
    )

    # Variants input (mutually exclusive)
    variants_group = batch_parser.add_mutually_exclusive_group()
    variants_group.add_argument(
        "--variants",
        type=str,
        help="Comma-separated variant indices (e.g., 0,1,2)",
    )
    variants_group.add_argument(
        "--variants-file",
        type=Path,
        help="Variants file (indices, one per line)",
    )

    batch_output = batch_parser.add_argument_group("Output Options")
    batch_output.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/batch"),
        help="Directory for results and state (default: results/batch)",
    )

    batch_backend = batch_parser.add_argument_group("Backend Options")
    batch_backend.add_argument(
        "--skypilot",
        action="store_true",
        help="Use SkyPilot for cloud evaluation",
    )
    batch_backend.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum concurrent evaluations (default: 1)",
    )
    batch_backend.add_argument(
        "--timeout",
        type=int,
        help="Timeout per evaluation in seconds",
    )
    batch_backend.add_argument(
        "--bucket-url",
        type=str,
        help="Bucket URL for result storage (s3://... or gs://...). "
             "Results are written directly to the bucket by each worker and "
             "synced incrementally. Enables reliable resume across runs.",
    )

    batch_control = batch_parser.add_argument_group("Control Options")
    batch_control.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted evaluation",
    )
    batch_control.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring previous state",
    )
    batch_control.add_argument(
        "--status",
        action="store_true",
        help="Show evaluation status and exit",
    )
    batch_control.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry all failed pairs",
    )
    batch_control.add_argument(
        "--complete",
        action="store_true",
        help="Evaluate only missing pairs (requires --problems-file and --models-file)",
    )
    batch_control.add_argument(
        "--report",
        action="store_true",
        help="Show aggregated report and exit",
    )
    batch_control.add_argument(
        "--export-failed",
        type=Path,
        help="Export failed pairs to file",
    )
    batch_control.add_argument(
        "--sync-bucket",
        action="store_true",
        help="Sync results from bucket to local state and export reports",
    )

    # Check subcommand
    check_parser = subparsers.add_parser(
        "check",
        help="Check solution matrix coverage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check coverage with default files (problems.txt, models.txt, num_solutions.txt)
  frontier-eval check

  # Check with custom files
  frontier-eval check --problems-file my_problems.txt --models-file my_models.txt

  # Generate pairs file for batch evaluation
  frontier-eval check --output-pairs pairs.txt

  # Only include existing solutions in pairs file
  frontier-eval check --output-pairs pairs.txt --existing-only
        """,
    )

    check_parser.add_argument(
        "--problems-file",
        type=Path,
        default=Path("research/problems.txt"),
        help="Problems file (default: research/problems.txt)",
    )
    check_parser.add_argument(
        "--models-file",
        type=Path,
        default=Path("research/models.txt"),
        help="Models file (default: research/models.txt)",
    )
    check_parser.add_argument(
        "--variants-file",
        type=Path,
        default=Path("research/num_solutions.txt"),
        help="Variants file (default: research/num_solutions.txt)",
    )
    check_parser.add_argument(
        "--solutions-dir",
        type=Path,
        default=Path("solutions"),
        help="Solutions directory (default: solutions)",
    )
    check_parser.add_argument(
        "--problems-dir",
        type=Path,
        default=Path("research"),
        help="Problems directory (default: research)",
    )
    check_parser.add_argument(
        "--output-pairs",
        type=Path,
        help="Write pairs file to this path",
    )
    check_parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Only include existing solutions in pairs file (for evaluation)",
    )

    return parser


def print_result(result: EvaluationResult, quiet: bool = False, verbose: bool = False) -> None:
    """Print evaluation result."""
    if quiet:
        if result.success:
            print(f"{result.problem_id}: {result.score}")
        else:
            print(f"{result.problem_id}: ERROR")
        return

    print(f"\n{'='*60}")
    print(f"Problem: {result.problem_id}")
    print(f"Status: {result.status.value}")

    if result.success:
        print(f"Score: {result.score}")
    else:
        print(f"Message: {result.message}")

    if result.duration_seconds:
        print(f"Duration: {result.duration_seconds:.1f}s")

    if verbose and result.logs:
        print(f"\n--- Logs ---\n{result.logs}")

    print("=" * 60)


def print_results_json(results: List[EvaluationResult]) -> None:
    """Print results as JSON."""
    import json

    data = []
    for r in results:
        data.append({
            "problem_id": r.problem_id,
            "score": r.score,
            "status": r.status.value,
            "message": r.message,
            "duration_seconds": r.duration_seconds,
        })
    print(json.dumps(data, indent=2))


def get_problem_ids(
    args: argparse.Namespace,
    evaluator: FrontierCSEvaluator,
    track: str,
) -> List[str]:
    """Get list of problem IDs to evaluate."""
    if args.all_problems:
        return evaluator.list_problems(track)

    if args.problems:
        return [p.strip() for p in args.problems.split(",")]

    if args.problems_file:
        if not args.problems_file.exists():
            print(f"Error: Problems file not found: {args.problems_file}", file=sys.stderr)
            sys.exit(1)
        problems = []
        for line in args.problems_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                problems.append(line)
        return problems

    if args.problem_id:
        return [args.problem_id]

    return []


def run_batch(args: argparse.Namespace) -> int:
    """Run batch evaluation command."""
    from .batch import BatchEvaluator
    from .batch.pair import Pair, read_problems_file, read_models_file

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    # Create batch evaluator
    backend = "skypilot" if args.skypilot else "docker"
    bucket_url = getattr(args, "bucket_url", None)
    batch = BatchEvaluator(
        results_dir=args.results_dir,
        backend=backend,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
        bucket_url=bucket_url,
    )

    # Handle status command
    if args.status:
        status = batch.get_status()
        print("\nBatch Evaluation Status")
        print("=" * 40)
        print(f"Total pairs: {status['total_pairs']}")
        print(f"Completed: {status['completed']}")
        print(f"Successful: {status['successful']}")
        print(f"Errors: {status['errors']}")
        print(f"Pending: {status['pending']}")
        print(f"Started: {status['started_at'] or 'N/A'}")
        print(f"Updated: {status['updated_at'] or 'N/A'}")
        return 0

    # Handle sync-bucket command
    if getattr(args, "sync_bucket", False):
        if not bucket_url:
            print("Error: --sync-bucket requires --bucket-url", file=sys.stderr)
            return 1
        print(f"\nSyncing results from {bucket_url}...")
        count = batch.sync_from_bucket()
        print(f"Merged {count} results from bucket")

        # Export reports
        batch._export_all_results()
        status = batch.get_status()
        print(f"\nStatus: {status['completed']}/{status['total_pairs']} completed")
        print(f"Results exported to {args.results_dir}")
        return 0

    # Handle report command
    if args.report:
        print("\nAggregated Results by Model")
        print("=" * 60)
        by_model = batch.state.aggregate_by_model()
        for model, stats in sorted(by_model.items()):
            avg = f"{stats['avg_score']:.4f}" if stats['avg_score'] is not None else "N/A"
            print(f"  {model}: {stats['successful']}/{stats['total']} successful, avg={avg}")

        print("\nAggregated Results by Problem")
        print("=" * 60)
        by_problem = batch.state.aggregate_by_problem()
        for problem, stats in sorted(by_problem.items()):
            avg = f"{stats['avg_score']:.4f}" if stats['avg_score'] is not None else "N/A"
            print(f"  {problem}: {stats['successful']}/{stats['total']} successful, avg={avg}")
        return 0

    # Handle export-failed command
    if args.export_failed:
        count = batch.state.export_failed(args.export_failed)
        print(f"Exported {count} failed pairs to {args.export_failed}")
        return 0

    # Handle retry-failed command
    if args.retry_failed:
        print(f"\nRetrying failed pairs from {args.results_dir}")
        state = batch.retry_failed()
        print(f"\nComplete: {state.success_count}/{state.total_pairs} successful")
        return 0 if state.error_count == 0 else 1

    # Handle resume command
    if args.resume:
        print(f"\nResuming batch evaluation from {args.results_dir}")
        state = batch.resume()
        print(f"\nComplete: {state.success_count}/{state.total_pairs} successful")
        return 0 if state.error_count == 0 else 1

    # Helper: get problems list from --problems or --problems-file
    def get_problems():
        if args.problems:
            return [p.strip() for p in args.problems.split(",")]
        elif args.problems_file:
            if not args.problems_file.exists():
                print(f"Error: Problems file not found: {args.problems_file}", file=sys.stderr)
                return None
            return read_problems_file(args.problems_file)
        return None

    # Helper: get models list from --models or --models-file
    def get_models():
        if args.models:
            return [m.strip() for m in args.models.split(",")]
        elif args.models_file:
            if not args.models_file.exists():
                print(f"Error: Models file not found: {args.models_file}", file=sys.stderr)
                return None
            return read_models_file(args.models_file)
        return None

    # Helper: get variants list from --variants or --variants-file
    def get_variants():
        if args.variants:
            return [int(v.strip()) for v in args.variants.split(",")]
        elif args.variants_file:
            if not args.variants_file.exists():
                print(f"Error: Variants file not found: {args.variants_file}", file=sys.stderr)
                return None
            from .batch.pair import read_variants_file
            return read_variants_file(args.variants_file)
        return None

    # Handle complete command (evaluate missing pairs)
    if args.complete:
        problems = get_problems()
        models = get_models()
        if not problems or not models:
            print("Error: --complete requires --problems/--problems-file and --models/--models-file", file=sys.stderr)
            return 1

        variants = get_variants()
        print(f"\nEvaluating missing pairs ({len(problems)} problems × {len(models)} models)")
        state = batch.evaluate_missing(problems, models, variants=variants)
        print(f"\nComplete: {state.success_count}/{state.total_pairs} successful")
        return 0 if state.error_count == 0 else 1

    # Determine input mode
    resume = not args.no_resume
    state = None

    has_pairs = args.pairs or args.pairs_file
    has_expansion = (args.problems or args.problems_file) and (args.models or args.models_file)

    if has_pairs and has_expansion:
        print("Error: Cannot use --pairs/--pairs-file together with --problems + --models", file=sys.stderr)
        return 1

    if args.pairs:
        # Mode: pairs from command line
        from .batch.pair import Pair
        pairs = []
        for p in args.pairs.split(","):
            p = p.strip()
            if ":" not in p:
                print(f"Error: Invalid pair format (expected solution:problem): {p}", file=sys.stderr)
                return 1
            solution, problem = p.split(":", 1)
            pairs.append(Pair(solution=solution.strip(), problem=problem.strip()))

        print(f"\nBatch evaluation: {len(pairs)} pairs")
        state = batch.evaluate_pairs(pairs, resume=resume)

    elif args.pairs_file:
        # Mode: pairs file
        if not args.pairs_file.exists():
            print(f"Error: Pairs file not found: {args.pairs_file}", file=sys.stderr)
            return 1

        print(f"\nBatch evaluation from pairs file: {args.pairs_file}")
        state = batch.evaluate_pairs_file(args.pairs_file, resume=resume)

    elif (args.problems or args.problems_file) and (args.models or args.models_file):
        # Mode: problems × models expansion
        problems = get_problems()
        models = get_models()
        variants = get_variants()
        if not problems or not models:
            return 1

        from .batch.pair import expand_pairs
        pairs = expand_pairs(
            problems,
            models,
            variants,
            solutions_dir=batch.base_dir / "solutions",
            validate_paths=True,
        )

        print(f"\nBatch evaluation: {len(problems)} problems × {len(models)} models = {len(pairs)} pairs")
        state = batch.evaluate_pairs(pairs, resume=resume)

    else:
        print("Error: Specify input with --pairs, --pairs-file, or --problems + --models", file=sys.stderr)
        return 1

    # Print summary
    print(f"\n{'='*40}")
    print("Batch Evaluation Summary")
    print("=" * 40)
    print(f"Total: {state.total_pairs}")
    print(f"Successful: {state.success_count}")
    print(f"Errors: {state.error_count}")
    print(f"Results saved to: {args.results_dir}")
    print(f"\nOutput files:")
    print(f"  - results.csv: All results")
    print(f"  - by_model.csv: Aggregated by model")
    print(f"  - by_problem.csv: Aggregated by problem")
    if state.error_count > 0:
        print(f"  - failed.txt: {state.error_count} failed pairs")

    return 0 if state.error_count == 0 else 1


def run_check(args: argparse.Namespace) -> int:
    """Run check command to verify solution matrix coverage."""
    from .check import check_solution_matrix, generate_pairs_file

    try:
        result = check_solution_matrix(
            problems_file=args.problems_file,
            models_file=args.models_file,
            variants_file=args.variants_file,
            solutions_dir=args.solutions_dir,
            problems_dir=args.problems_dir,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Print summary
    print(result.summary())

    # Generate pairs file if requested
    if args.output_pairs:
        include_missing = not args.existing_only
        count = generate_pairs_file(result, args.output_pairs, include_missing=include_missing)
        mode = "all expected" if include_missing else "existing only"
        print(f"\nWrote {count} pairs ({mode}) to {args.output_pairs}")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    if argv is None:
        argv = sys.argv[1:]

    # Handle subcommands first by checking if first arg is a subcommand
    subcommands = {"batch", "check"}
    if argv and argv[0] in subcommands:
        parser = create_parser()
        args = parser.parse_args(argv)
        if args.command == "batch":
            return run_batch(args)
        if args.command == "check":
            return run_check(args)

    # For single evaluations, create a parser without subcommands
    # to avoid argparse confusion with positional args
    parser = create_parser()
    # Inject a dummy command to satisfy the subparser
    args = parser.parse_args(argv + ["batch", "--status"]) if not any(a in argv for a in subcommands) else parser.parse_args(argv)
    # Reset command since we used a dummy
    args.command = None
    args.status = False

    # Determine track
    track = "algorithmic" if args.algorithmic else "research"

    # Create evaluator
    backend = "skypilot" if args.skypilot else "docker"
    evaluator = FrontierCSEvaluator(
        backend=backend,
        judge_url=args.judge_url,
        cloud=args.cloud,
        region=args.region,
    )

    # Handle info commands
    if args.list:
        problems = evaluator.list_problems(track)
        print(f"\n{track.title()} Problems ({len(problems)} total):\n")
        for p in problems:
            print(f"  {p}")
        return 0

    if args.show:
        if not args.problem_id:
            print("Error: --show requires a problem_id", file=sys.stderr)
            return 1
        statement = evaluator.get_problem_statement(track, args.problem_id)
        if statement:
            print(statement)
        else:
            print(f"Problem not found: {args.problem_id}", file=sys.stderr)
            return 1
        return 0

    # Get problem IDs
    problem_ids = get_problem_ids(args, evaluator, track)

    if not problem_ids:
        print("Error: No problems specified. Use --help for usage.", file=sys.stderr)
        return 1

    # Get solution code
    if args.code:
        code = args.code
    elif args.solution:
        solution_path = Path(args.solution)
        if not solution_path.exists():
            print(f"Error: Solution file not found: {solution_path}", file=sys.stderr)
            return 1
        code = solution_path.read_text(encoding="utf-8")
    else:
        print("Error: No solution provided. Use --code or provide a file path.", file=sys.stderr)
        return 1

    # Run evaluations
    results = []
    for pid in problem_ids:
        if not args.quiet:
            print(f"Evaluating {pid}...", end=" ", flush=True)

        result = evaluator.evaluate(track, pid, code, timeout=args.timeout)
        results.append(result)

        if not args.quiet:
            if result.success:
                print(f"Score: {result.score}")
            else:
                print(f"ERROR: {result.message}")

    # Output results
    if args.json:
        print_results_json(results)
    elif not args.quiet:
        print(f"\n{'='*60}")
        print("Summary")
        print("=" * 60)

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print(f"Total: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")

        if successful:
            avg_score = sum(r.score for r in successful) / len(successful)
            print(f"Average Score: {avg_score:.2f}")

        if failed and args.verbose:
            print("\nFailed problems:")
            for r in failed:
                print(f"  {r.problem_id}: {r.message}")

    # Return non-zero if any failures
    return 0 if all(r.success for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
