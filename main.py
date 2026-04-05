"""CLI entry point for the SEBI Circular Reference Extractor."""

import argparse
import json
import sys

from dotenv import load_dotenv

load_dotenv()

from extract import extract_references
from evaluate import run_evaluation


def cmd_extract(args):
    """Extract references from a PDF."""
    result = extract_references(args.pdf)
    refs = result["references"]
    stats = result["stats"]

    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total references found: {stats['total']}")
    print(f"  - Regex only:  {stats['regex_only']}")
    print(f"  - LLM only:    {stats['llm_only']}")
    print(f"  - Both:        {stats['both']}")
    print(f"{'='*60}\n")

    # Print references grouped by type
    by_type = {}
    for ref in refs:
        by_type.setdefault(ref["type"], []).append(ref)

    for ref_type, type_refs in sorted(by_type.items()):
        print(f"[{ref_type.upper()}] ({len(type_refs)} found)")
        for ref in type_refs:
            pages = ", ".join(str(p) for p in ref["page_numbers"])
            source_tag = f" [{ref['source']}]" if ref["source"] != "both" else ""
            print(f"  - {ref['title']}")
            print(f"    Pages: {pages}{source_tag}")
        print()

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")


def cmd_evaluate(args):
    """Run evaluation against ground truth."""
    run_evaluation(args.pdf, args.ground_truth)


def cmd_optimize(args):
    """Run prompt optimization with judge LLM."""
    from optimize import optimize_prompt
    optimize_prompt(args.pdf, args.ground_truth, iterations=args.iterations)


def main():
    parser = argparse.ArgumentParser(
        description="SEBI Circular Reference Extractor - Extract and evaluate document references"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract references from a PDF")
    extract_parser.add_argument("pdf", help="Path to the SEBI circular PDF")
    extract_parser.add_argument("--output", "-o", help="Output JSON file path")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate extraction against ground truth")
    eval_parser.add_argument("pdf", help="Path to the SEBI circular PDF")
    eval_parser.add_argument("ground_truth", help="Path to ground truth JSON file")

    # Optimize command
    opt_parser = subparsers.add_parser("optimize", help="Run prompt optimization with judge LLM")
    opt_parser.add_argument("pdf", help="Path to the SEBI circular PDF")
    opt_parser.add_argument("ground_truth", help="Path to ground truth JSON file")
    opt_parser.add_argument("--iterations", "-n", type=int, default=2, help="Number of optimization iterations")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
