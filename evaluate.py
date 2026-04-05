"""Evaluation framework: compare regex-only vs hybrid extraction against ground truth."""

import json
from difflib import SequenceMatcher

from extract import extract_text_by_page, extract_references_regex, extract_references


def load_ground_truth(json_path: str) -> list[dict]:
    """Load ground truth references from a JSON file."""
    with open(json_path) as f:
        return json.load(f)


def _match_title(pred_title: str, gt_title: str, threshold: float = 0.7) -> bool:
    """Fuzzy match two reference titles."""
    a = pred_title.lower().strip()
    b = gt_title.lower().strip()
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


def compute_metrics(predicted: list[dict], ground_truth: list[dict]) -> dict:
    """Compute precision, recall, and F1 against ground truth."""
    if not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}

    gt_matched = [False] * len(ground_truth)
    tp = 0
    fp = 0

    for pred in predicted:
        matched = False
        for i, gt in enumerate(ground_truth):
            if not gt_matched[i] and _match_title(pred["title"], gt["title"]):
                gt_matched[i] = True
                tp += 1
                matched = True
                break
        if not matched:
            fp += 1

    fn = sum(1 for m in gt_matched if not m)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def run_evaluation(pdf_path: str, ground_truth_path: str):
    """Run full evaluation comparing regex-only vs hybrid extraction."""
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    print(f"PDF: {pdf_path}")
    print(f"Ground truth: {ground_truth_path}\n")

    ground_truth = load_ground_truth(ground_truth_path)
    print(f"Ground truth contains {len(ground_truth)} references\n")

    # --- Regex-only ---
    print("Running regex-only extraction...")
    pages = extract_text_by_page(pdf_path)
    regex_refs = extract_references_regex(pages)
    regex_metrics = compute_metrics(regex_refs, ground_truth)
    print(f"  Found {len(regex_refs)} references")

    # --- Hybrid (regex + LLM) ---
    print("\nRunning hybrid extraction (regex + LLM)...")
    hybrid_result = extract_references(pdf_path)
    hybrid_refs = hybrid_result["references"]
    hybrid_metrics = compute_metrics(hybrid_refs, ground_truth)

    # --- Print comparison ---
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Found':>8}")
    print(f"{'-'*60}")
    print(f"{'Regex-only':<20} {regex_metrics['precision']:>10.3f} {regex_metrics['recall']:>10.3f} {regex_metrics['f1']:>10.3f} {len(regex_refs):>8}")
    print(f"{'Hybrid (regex+LLM)':<20} {hybrid_metrics['precision']:>10.3f} {hybrid_metrics['recall']:>10.3f} {hybrid_metrics['f1']:>10.3f} {len(hybrid_refs):>8}")
    print(f"{'-'*60}")

    # Improvement
    p_diff = hybrid_metrics["precision"] - regex_metrics["precision"]
    r_diff = hybrid_metrics["recall"] - regex_metrics["recall"]
    f1_diff = hybrid_metrics["f1"] - regex_metrics["f1"]
    print(f"{'Improvement':<20} {p_diff:>+10.3f} {r_diff:>+10.3f} {f1_diff:>+10.3f}")
    print()

    # Show what LLM added
    llm_additions = [r for r in hybrid_refs if r["source"] == "llm"]
    if llm_additions:
        print("References found ONLY by LLM:")
        for ref in llm_additions:
            print(f"  - [{ref['type']}] {ref['title']} (pages: {ref['page_numbers']})")

    # Show missed references
    if hybrid_metrics["fn"] > 0:
        print(f"\nMissed references ({hybrid_metrics['fn']}):")
        hybrid_titles = [r["title"] for r in hybrid_refs]
        for gt in ground_truth:
            matched = any(_match_title(ht, gt["title"]) for ht in hybrid_titles)
            if not matched:
                print(f"  - [{gt.get('type', '?')}] {gt['title']}")

    return {
        "regex_only": regex_metrics,
        "hybrid": hybrid_metrics,
    }
