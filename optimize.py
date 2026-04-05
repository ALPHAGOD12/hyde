"""Prompt optimization: uses a judge LLM to rewrite the extraction prompt based on eval results."""

import json
import os
import re
import time
from difflib import SequenceMatcher

from google import genai
from dotenv import load_dotenv

from extract import (
    extract_text_by_page,
    extract_references_regex,
    _is_duplicate,
    _parse_llm_response,
    LLM_PROMPT,
)
from evaluate import compute_metrics, load_ground_truth

load_dotenv()
_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def _call_gemini(prompt: str) -> str:
    """Call Gemini with retry."""
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=60), retry=retry_if_exception_type(Exception))
    def _call(p):
        response = _client.models.generate_content(model="gemini-2.5-flash", contents=p)
        return response.text

    return _call(prompt)


def _match_title(a: str, b: str) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.7


def _run_extraction_with_prompt(pages, regex_refs, prompt_template):
    """Run LLM extraction using a custom prompt template."""
    all_refs = {}
    batch_size = 3

    existing_list = "\n".join(f"- {r['title']}" for r in regex_refs[:200])

    for i in range(0, len(pages), batch_size):
        batch = pages[i:i + batch_size]
        batch_text = ""
        for page in batch:
            batch_text += f"\n--- Page {page['page_num']} ---\n{page['text']}"

        prompt = prompt_template.format(text=batch_text, existing_refs=existing_list)

        try:
            response_text = _call_gemini(prompt)
            refs = _parse_llm_response(response_text)
        except Exception as e:
            print(f"    Warning: LLM failed for pages {batch[0]['page_num']}-{batch[-1]['page_num']}: {e}")
            continue

        for ref in refs:
            if not isinstance(ref, dict) or "title" not in ref:
                continue
            title = ref.get("title", "").strip()
            if not title:
                continue
            key = title.lower()
            if key in all_refs:
                for pn in ref.get("page_numbers", []):
                    if pn not in all_refs[key]["page_numbers"]:
                        all_refs[key]["page_numbers"].append(pn)
            else:
                all_refs[key] = {
                    "title": title,
                    "type": ref.get("type", "other"),
                    "page_numbers": ref.get("page_numbers", []),
                    "source": "llm",
                }

        if i + batch_size < len(pages):
            time.sleep(2)

    return list(all_refs.values())


def _compute_errors(all_refs, ground_truth):
    """Compute missed refs and false positives."""
    gt_matched = [False] * len(ground_truth)
    pred_matched = [False] * len(all_refs)

    for i, pred in enumerate(all_refs):
        for j, gt in enumerate(ground_truth):
            if not gt_matched[j] and _match_title(pred["title"], gt["title"]):
                gt_matched[j] = True
                pred_matched[i] = True
                break

    missed = [ground_truth[i] for i, m in enumerate(gt_matched) if not m]
    false_pos = [all_refs[i] for i, m in enumerate(pred_matched) if not m]
    return missed, false_pos


# ---------------------------------------------------------------------------
# Judge prompt — asks the LLM to rewrite the ENTIRE extraction prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are an expert prompt engineer optimizing a document reference extraction system.

The system extracts references from SEBI (Securities and Exchange Board of India) circular PDFs.
It uses two passes: regex (already done) and then an LLM (your prompt to optimize).

Here is the CURRENT extraction prompt being used:
--- START CURRENT PROMPT ---
{current_prompt}
--- END CURRENT PROMPT ---

Here are the EVALUATION RESULTS from running this prompt:

Precision: {precision} (of extracted refs, what fraction are real)
Recall: {recall} (of real refs, what fraction did we find)
F1: {f1}

MISSED references (the system failed to find these — FALSE NEGATIVES):
{missed}

FALSE POSITIVES (the system found these but they are not real references):
{false_positives}

SAMPLE of correctly found references (for context):
{correct_samples}

YOUR TASK:
1. Analyze the error patterns — what types of references are being missed? What's being hallucinated?
2. Rewrite the ENTIRE extraction prompt to fix these issues. The new prompt must:
   - Keep the {{existing_refs}} and {{text}} template variables (they get filled in at runtime)
   - Include improved instructions based on error analysis
   - Include 5-7 few-shot examples chosen to address the specific failure patterns
   - Be clear about output format (JSON array)
   - Address any hallucination patterns you see in the false positives

Return a JSON object with:
- "analysis": string explaining what's going wrong and why
- "optimized_prompt": the complete new prompt as a string (must contain {{existing_refs}} and {{text}} placeholders)
- "changes_made": list of strings describing each change and why
- "expected_improvement": string predicting how metrics should change

Return ONLY valid JSON. No markdown fences."""


def optimize_prompt(pdf_path: str, ground_truth_path: str, iterations: int = 2):
    """Run iterative prompt optimization: extract → evaluate → judge rewrites prompt → repeat."""

    print(f"{'='*60}")
    print(f"PROMPT OPTIMIZATION ({iterations} iterations)")
    print(f"{'='*60}")

    ground_truth = load_ground_truth(ground_truth_path)
    pages = extract_text_by_page(pdf_path)
    regex_refs = extract_references_regex(pages)

    current_prompt = LLM_PROMPT
    results_history = []

    for iteration in range(iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{iterations}")
        print(f"{'='*60}")

        # Step 1: Run extraction with current prompt
        print("\n[Step 1] Running extraction with current prompt...")
        llm_refs = _run_extraction_with_prompt(pages, regex_refs, current_prompt)

        # Merge with regex
        all_refs = list(regex_refs)
        for lr in llm_refs:
            if not any(_is_duplicate(lr["title"], er["title"]) for er in all_refs):
                all_refs.append(lr)

        # Step 2: Evaluate
        print("\n[Step 2] Evaluating...")
        metrics = compute_metrics(all_refs, ground_truth)
        missed, false_pos = _compute_errors(all_refs, ground_truth)

        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall:    {metrics['recall']}")
        print(f"  F1:        {metrics['f1']}")
        print(f"  Missed:    {len(missed)}")
        print(f"  False pos: {len(false_pos)}")

        results_history.append({
            "iteration": iteration + 1,
            "metrics": metrics,
            "total_refs": len(all_refs),
            "missed": len(missed),
            "false_pos": len(false_pos),
        })

        # Step 3: Judge rewrites the prompt
        if iteration < iterations - 1:  # Don't judge on last iteration
            print("\n[Step 3] Judge LLM analyzing errors and rewriting prompt...")

            # Get some correct samples for context
            correct = []
            for ref in all_refs:
                for gt in ground_truth:
                    if _match_title(ref["title"], gt["title"]):
                        correct.append(ref)
                        break
                if len(correct) >= 10:
                    break

            judge_prompt = JUDGE_PROMPT.format(
                current_prompt=current_prompt,
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                missed=json.dumps(missed[:20], indent=2),
                false_positives=json.dumps(
                    [{"title": r["title"], "type": r["type"]} for r in false_pos[:15]],
                    indent=2,
                ),
                correct_samples=json.dumps(
                    [{"title": r["title"], "type": r["type"]} for r in correct[:10]],
                    indent=2,
                ),
            )

            try:
                response = _call_gemini(judge_prompt)
                text = response.strip()
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
                    text = re.sub(r"\n?```\s*$", "", text)
                judge_result = json.loads(text)
            except Exception as e:
                print(f"  Judge failed: {e}")
                break

            # Print judge analysis
            print(f"\n  Analysis: {judge_result.get('analysis', 'N/A')[:500]}")
            print(f"\n  Changes made:")
            for change in judge_result.get("changes_made", []):
                print(f"    - {change}")
            print(f"\n  Expected improvement: {judge_result.get('expected_improvement', 'N/A')}")

            # Update prompt
            new_prompt = judge_result.get("optimized_prompt", "")
            if new_prompt and "{existing_refs}" in new_prompt and "{text}" in new_prompt:
                current_prompt = new_prompt
                print(f"\n  Prompt updated successfully ({len(new_prompt)} chars)")
            else:
                print(f"\n  Warning: Judge returned invalid prompt (missing placeholders), keeping current")

            time.sleep(5)  # Rate limit

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Iter':<6} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Found':>8} {'Missed':>8} {'FP':>8}")
    print(f"{'-'*62}")
    for r in results_history:
        m = r["metrics"]
        print(f"{r['iteration']:<6} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {r['total_refs']:>8} {r['missed']:>8} {r['false_pos']:>8}")

    # Show improvement
    if len(results_history) >= 2:
        first = results_history[0]["metrics"]
        last = results_history[-1]["metrics"]
        print(f"\n  F1 improvement: {first['f1']:.3f} → {last['f1']:.3f} ({last['f1'] - first['f1']:+.3f})")
        print(f"  Recall improvement: {first['recall']:.3f} → {last['recall']:.3f} ({last['recall'] - first['recall']:+.3f})")

    # Save the optimized prompt
    if current_prompt != LLM_PROMPT:
        with open("optimized_prompt.txt", "w") as f:
            f.write(current_prompt)
        print(f"\n  Optimized prompt saved to optimized_prompt.txt")

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
