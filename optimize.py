"""Prompt optimization: uses a judge LLM to rewrite the extraction prompt based on eval results."""

import json
import os
import re
import time
from difflib import SequenceMatcher

from openai import OpenAI
from dotenv import load_dotenv

from extract import (
    extract_text_by_page,
    extract_references_regex,
    _is_duplicate,
    _parse_llm_response,
    LLM_PROMPT,
    MATCH_THRESHOLD,
    _LLM_MODEL,
)
from evaluate import compute_metrics, load_ground_truth

load_dotenv()
_client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")


def _call_llm(prompt: str) -> str:
    """Call local LLM via Ollama's OpenAI-compatible API with retry."""
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=120), retry=retry_if_exception_type(Exception))
    def _call(p):
        response = _client.chat.completions.create(
            messages=[{"role": "user", "content": p}],
            model=_LLM_MODEL,
            max_tokens=4096,
            temperature=0.2,
        )
        return response.choices[0].message.content

    return _call(prompt)


def _match_title(a: str, b: str) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= MATCH_THRESHOLD


def _sanitize_prompt_braces(prompt: str) -> str:
    """Escape all {/} in the prompt except the {text} and {existing_refs} placeholders."""
    prompt = prompt.replace("{", "{{").replace("}", "}}")
    prompt = prompt.replace("{{text}}", "{text}")
    prompt = prompt.replace("{{existing_refs}}", "{existing_refs}")
    return prompt


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
            response_text = _call_llm(prompt)
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
            time.sleep(1)

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
__CURRENT_PROMPT__
--- END CURRENT PROMPT ---

Here are the EVALUATION RESULTS from running this prompt:

Precision: __PRECISION__ (of extracted refs, what fraction are real)
Recall: __RECALL__ (of real refs, what fraction did we find)
F1: __F1__

MISSED references (the system failed to find these — FALSE NEGATIVES):
__MISSED__

FALSE POSITIVES (the system found these but they are not real references):
__FALSE_POSITIVES__

SAMPLE of correctly found references (for context):
__CORRECT_SAMPLES__

YOUR TASK:
1. Analyze the error patterns — what types of references are being missed? What's being hallucinated?
2. Rewrite the ENTIRE extraction prompt to fix these issues. The new prompt must:
   - Keep the {existing_refs} and {text} template variables exactly as shown (single curly braces — these are Python .format() placeholders)
   - For any literal JSON braces in few-shot examples, use DOUBLE curly braces {{ and }} so Python's .format() doesn't break
   - Include improved instructions based on error analysis
   - Include 5-7 few-shot examples chosen to address the specific failure patterns
   - Be clear about output format (JSON array)
   - Address any hallucination patterns you see in the false positives

IMPORTANT: Return your response as TWO parts separated by the marker "===PROMPT_SEPARATOR===":
- Part 1: Your analysis, changes made, and expected improvement as plain text
- Part 2: The complete optimized prompt (raw text, ready to use with Python .format())

Do NOT return JSON. Use the separator format described above."""


def optimize_prompt(pdf_path: str, ground_truth_path: str, iterations: int = 2):
    """Run iterative prompt optimization: extract → evaluate → judge rewrites prompt → repeat."""

    print(f"{'='*60}")
    print(f"PROMPT OPTIMIZATION ({iterations} iterations)")
    print(f"{'='*60}")

    ground_truth = load_ground_truth(ground_truth_path)
    pages = extract_text_by_page(pdf_path)
    regex_refs = extract_references_regex(pages)

    current_prompt = LLM_PROMPT
    best_prompt = current_prompt
    best_f1 = 0.0
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

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_prompt = current_prompt
            print(f"  ✓ New best F1: {best_f1:.3f}")
        elif iteration > 0:
            print(f"  ✗ F1 regressed ({metrics['f1']:.3f} < {best_f1:.3f}), will rollback to best prompt")
            current_prompt = best_prompt

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

            judge_prompt = JUDGE_PROMPT
            judge_prompt = judge_prompt.replace("__CURRENT_PROMPT__", current_prompt)
            judge_prompt = judge_prompt.replace("__PRECISION__", str(metrics["precision"]))
            judge_prompt = judge_prompt.replace("__RECALL__", str(metrics["recall"]))
            judge_prompt = judge_prompt.replace("__F1__", str(metrics["f1"]))
            judge_prompt = judge_prompt.replace("__MISSED__", json.dumps(missed[:20], indent=2))
            judge_prompt = judge_prompt.replace("__FALSE_POSITIVES__", json.dumps(
                [{"title": r["title"], "type": r["type"]} for r in false_pos[:15]], indent=2
            ))
            judge_prompt = judge_prompt.replace("__CORRECT_SAMPLES__", json.dumps(
                [{"title": r["title"], "type": r["type"]} for r in correct[:10]], indent=2
            ))

            try:
                response = _call_llm(judge_prompt)
                text = response.strip()

                if "===PROMPT_SEPARATOR===" in text:
                    parts = text.split("===PROMPT_SEPARATOR===", 1)
                    analysis_text = parts[0].strip()
                    new_prompt = parts[1].strip()
                    # Strip markdown fences if present around the prompt
                    if new_prompt.startswith("```"):
                        new_prompt = re.sub(r"^```(?:\w*)?\s*\n?", "", new_prompt)
                        new_prompt = re.sub(r"\n?```\s*$", "", new_prompt)
                else:
                    # Fallback: try to find the prompt after common headers
                    analysis_text = text[:500]
                    new_prompt = ""

                print(f"\n  Analysis:\n{analysis_text[:600]}")

            except Exception as e:
                print(f"  Judge failed: {e}")
                break

            # Update prompt if valid
            if new_prompt and "{existing_refs}" in new_prompt and "{text}" in new_prompt:
                current_prompt = _sanitize_prompt_braces(new_prompt)
                print(f"\n  Prompt updated successfully ({len(current_prompt)} chars)")
            elif new_prompt:
                # Try fixing common placeholder issues
                fixed = new_prompt
                for variant in ["{{existing_refs}}", "<<existing_refs>>", "[existing_refs]", "{existing_refs}"]:
                    if variant in fixed and variant != "{existing_refs}":
                        fixed = fixed.replace(variant, "{existing_refs}")
                for variant in ["{{text}}", "<<text>>", "[text]", "{text}"]:
                    if variant in fixed and variant != "{text}":
                        fixed = fixed.replace(variant, "{text}")
                if "{existing_refs}" in fixed and "{text}" in fixed:
                    current_prompt = _sanitize_prompt_braces(fixed)
                    print(f"\n  Prompt updated (after fixing placeholders, {len(current_prompt)} chars)")
                else:
                    print(f"\n  Warning: Judge returned prompt without valid placeholders, keeping current")
                    print(f"  (prompt preview: {new_prompt[:300]}...)")
            else:
                print(f"\n  Warning: No prompt returned by judge, keeping current")

            time.sleep(2)  # Brief pause between iterations

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

    # Save the best prompt found across all iterations
    if best_prompt != LLM_PROMPT:
        with open("optimized_prompt.txt", "w") as f:
            f.write(best_prompt)
        print(f"\n  Best prompt (F1={best_f1:.3f}) saved to optimized_prompt.txt")

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
