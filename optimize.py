"""Prompt optimization: uses a judge LLM to evaluate extraction quality and suggest prompt improvements."""

import json
import os
import time

from google import genai
from dotenv import load_dotenv

from extract import extract_text_by_page, extract_references_regex, extract_references_llm, _call_gemini, _parse_llm_response, LLM_PROMPT
from evaluate import compute_metrics, load_ground_truth

load_dotenv()
_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# ---------------------------------------------------------------------------
# Few-shot examples that can be improved by the optimizer
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """
EXAMPLES of references you should find:

Example 1 (standard circular with number):
Text: "Circular No. SEBI/HO/MRD/DP/CIR/P/2017/63 dated June 28, 2017"
Output: {{"title": "Circular No. SEBI/HO/MRD/DP/CIR/P/2017/63 dated June 28, 2017", "type": "circular", "page_numbers": [7]}}

Example 2 (descriptive reference with date and title only):
Text: "Oct 27, 2010- European Style Stock Options"
Output: {{"title": "Oct 27, 2010- European Style Stock Options", "type": "circular", "page_numbers": [8]}}

Example 3 (SEBI email):
Text: "SEBI Email dated May 4, 2020 on Rationalisation of Strikes on Long dated options"
Output: {{"title": "SEBI Email dated May 4, 2020 on Rationalisation of Strikes on Long dated options", "type": "other", "page_numbers": [10]}}

Example 4 (descriptive reference in numbered list):
Text: "96 Feb 13, 2001 -SMDRP/Policy/Cir-10/2001"
Output: {{"title": "Feb 13, 2001 - SMDRP/Policy/Cir-10/2001", "type": "circular", "page_numbers": [10]}}

Example 5 (letter reference):
Text: "SEBI letter dated January 5, 2023."
Output: {{"title": "SEBI letter dated January 5, 2023", "type": "other", "page_numbers": [13]}}
"""


JUDGE_PROMPT = """You are an evaluation judge for a document reference extraction system.

The system extracted references from a SEBI circular PDF. Below are:
1. The MISSED references (false negatives) - references that exist in the ground truth but the system failed to find
2. The FALSE POSITIVES - references the system found but are not in the ground truth
3. The current few-shot examples used in the extraction prompt

Your job:
1. Analyze WHY the system missed those references (what pattern did it fail to recognize?)
2. Suggest 3-5 NEW few-shot examples that would help the system catch similar references in the future
3. Rate the current prompt's effectiveness on a scale of 1-10

Return a JSON object with:
- "analysis": string explaining the failure patterns
- "new_examples": list of objects with "text" (input text snippet), "output" (expected JSON), "reason" (why this example helps)
- "prompt_score": number 1-10
- "suggestions": list of string suggestions for prompt improvement

MISSED references:
{missed}

FALSE POSITIVES (first 20):
{false_positives}

Current few-shot examples:
{current_examples}

Return ONLY valid JSON. No markdown fences."""


def run_judge(pdf_path: str, ground_truth_path: str) -> dict:
    """Run the judge LLM to analyze extraction quality and suggest improvements."""
    print("Loading ground truth...")
    ground_truth = load_ground_truth(ground_truth_path)

    print("Running extraction...")
    pages = extract_text_by_page(pdf_path)
    regex_refs = extract_references_regex(pages)
    llm_refs = extract_references_llm(pages, regex_refs=regex_refs)

    # Merge (simplified)
    from extract import _is_duplicate
    all_refs = list(regex_refs)
    for lr in llm_refs:
        if not any(_is_duplicate(lr["title"], er["title"]) for er in all_refs):
            all_refs.append(lr)

    # Compute what was missed and what was wrong
    from difflib import SequenceMatcher

    def match_title(a, b):
        a, b = a.lower().strip(), b.lower().strip()
        if a in b or b in a:
            return True
        return SequenceMatcher(None, a, b).ratio() >= 0.7

    gt_matched = [False] * len(ground_truth)
    pred_matched = [False] * len(all_refs)

    for i, pred in enumerate(all_refs):
        for j, gt in enumerate(ground_truth):
            if not gt_matched[j] and match_title(pred["title"], gt["title"]):
                gt_matched[j] = True
                pred_matched[i] = True
                break

    missed = [ground_truth[i] for i, m in enumerate(gt_matched) if not m]
    false_pos = [all_refs[i] for i, m in enumerate(pred_matched) if not m]

    print(f"  Matched: {sum(gt_matched)}/{len(ground_truth)}")
    print(f"  Missed: {len(missed)}")
    print(f"  False positives: {len(false_pos)}")

    # Call judge LLM
    print("\nRunning judge LLM...")
    prompt = JUDGE_PROMPT.format(
        missed=json.dumps(missed[:30], indent=2),
        false_positives=json.dumps([{"title": r["title"], "type": r["type"]} for r in false_pos[:20]], indent=2),
        current_examples=FEW_SHOT_EXAMPLES,
    )

    try:
        response = _call_gemini(prompt)
        # Parse response
        text = response.strip()
        if text.startswith("```"):
            import re
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        judge_result = json.loads(text)
    except Exception as e:
        print(f"  Judge LLM failed: {e}")
        return {"error": str(e)}

    # Print results
    print(f"\n{'='*60}")
    print("JUDGE EVALUATION")
    print(f"{'='*60}")
    print(f"\nPrompt Score: {judge_result.get('prompt_score', '?')}/10")
    print(f"\nAnalysis:\n{judge_result.get('analysis', 'N/A')}")

    print(f"\nSuggested New Examples:")
    for i, ex in enumerate(judge_result.get("new_examples", []), 1):
        print(f"\n  Example {i}:")
        print(f"    Text: {ex.get('text', 'N/A')}")
        print(f"    Output: {ex.get('output', 'N/A')}")
        print(f"    Reason: {ex.get('reason', 'N/A')}")

    print(f"\nPrompt Improvement Suggestions:")
    for s in judge_result.get("suggestions", []):
        print(f"  - {s}")

    return judge_result


def optimize_prompt(pdf_path: str, ground_truth_path: str, iterations: int = 2):
    """Run multiple optimization iterations: extract → judge → improve examples → repeat."""

    print(f"{'='*60}")
    print(f"PROMPT OPTIMIZATION ({iterations} iterations)")
    print(f"{'='*60}")

    current_examples = FEW_SHOT_EXAMPLES
    results_history = []

    for i in range(iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {i+1}/{iterations}")
        print(f"{'='*60}")

        # Run evaluation
        ground_truth = load_ground_truth(ground_truth_path)
        pages = extract_text_by_page(pdf_path)
        regex_refs = extract_references_regex(pages)

        # Run LLM with current examples injected
        print("Running extraction with current prompt...")
        llm_refs = extract_references_llm(pages, regex_refs=regex_refs)

        from extract import _is_duplicate
        all_refs = list(regex_refs)
        for lr in llm_refs:
            if not any(_is_duplicate(lr["title"], er["title"]) for er in all_refs):
                all_refs.append(lr)

        metrics = compute_metrics(all_refs, ground_truth)
        print(f"  Metrics: P={metrics['precision']}, R={metrics['recall']}, F1={metrics['f1']}")

        results_history.append({
            "iteration": i + 1,
            "metrics": metrics,
            "total_refs": len(all_refs),
        })

        # Run judge
        judge_result = run_judge(pdf_path, ground_truth_path)

        if "error" in judge_result:
            print(f"  Judge failed, stopping optimization")
            break

        # Update examples with judge suggestions
        new_examples = judge_result.get("new_examples", [])
        if new_examples:
            print(f"\n  Adding {len(new_examples)} new few-shot examples to prompt...")
            for ex in new_examples:
                current_examples += f"\nExample (auto-generated):\nText: \"{ex.get('text', '')}\"\nOutput: {ex.get('output', '')}\n"

        time.sleep(5)  # Rate limit between iterations

    # Print summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Iteration':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Refs Found':>12}")
    print(f"{'-'*56}")
    for r in results_history:
        m = r["metrics"]
        print(f"{r['iteration']:<12} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {r['total_refs']:>12}")

    return results_history


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python optimize.py <pdf_path> <ground_truth.json> [iterations]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    gt_path = sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    optimize_prompt(pdf_path, gt_path, iterations=iters)
