"""GEPA (Genetic-Evolutionary Prompt Architecture) optimizer for reference extraction.

Implements population-based prompt optimization with:
- Multiple candidate prompts evolved in parallel
- Pareto-optimal selection across precision/recall objectives
- Minibatch training with full validation
- Rich execution-trace reflection for targeted mutations
"""

import copy
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    retry=retry_if_exception_type(Exception),
)
def _call_llm(prompt: str, max_tokens: int = 4096) -> str:
    response = _client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=_LLM_MODEL,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content


def _match_title(a: str, b: str) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= MATCH_THRESHOLD


def _sanitize_prompt_braces(prompt: str) -> str:
    """Escape all {/} except the {text} and {existing_refs} placeholders."""
    prompt = prompt.replace("{", "{{").replace("}", "}}")
    prompt = prompt.replace("{{text}}", "{text}")
    prompt = prompt.replace("{{existing_refs}}", "{existing_refs}")
    return prompt


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalTrace:
    """Rich execution trace for a single candidate evaluation."""
    missed_refs: list[dict] = field(default_factory=list)
    false_positives: list[dict] = field(default_factory=list)
    correct_refs: list[dict] = field(default_factory=list)
    llm_unique: list[dict] = field(default_factory=list)


@dataclass
class Candidate:
    """A single prompt candidate in the population."""
    prompt: str
    generation: int = 0
    parent_id: int = -1
    scores: dict = field(default_factory=dict)
    val_scores: dict = field(default_factory=dict)
    trace: EvalTrace = field(default_factory=EvalTrace)
    id: int = 0

    def dominates(self, other: "Candidate") -> bool:
        """True if self Pareto-dominates other (better on all, strictly better on at least one)."""
        if not self.scores or not other.scores:
            return False
        objectives = ["precision", "recall"]
        at_least_one_better = False
        for obj in objectives:
            if self.scores.get(obj, 0) < other.scores.get(obj, 0):
                return False
            if self.scores.get(obj, 0) > other.scores.get(obj, 0):
                at_least_one_better = True
        return at_least_one_better


# ---------------------------------------------------------------------------
# Core GEPA engine
# ---------------------------------------------------------------------------

class GEPAOptimizer:
    def __init__(
        self,
        pdf_path: str,
        ground_truth_path: str,
        population_size: int = 4,
        iterations: int = 5,
        minibatch_pages: int = 4,
        validation_fraction: float = 0.3,
    ):
        self.pdf_path = pdf_path
        self.ground_truth_path = ground_truth_path
        self.population_size = population_size
        self.iterations = iterations
        self.minibatch_pages = minibatch_pages
        self.validation_fraction = validation_fraction

        self.all_pages = extract_text_by_page(pdf_path)
        self.regex_refs = extract_references_regex(self.all_pages)
        self.ground_truth = load_ground_truth(ground_truth_path)

        self._split_train_val()
        self._next_id = 0

    def _split_train_val(self):
        """Split ground truth into training and validation sets by page."""
        all_page_nums = sorted(set(p["page_num"] for p in self.all_pages))
        random.seed(42)
        val_count = max(2, int(len(all_page_nums) * self.validation_fraction))
        val_pages = set(random.sample(all_page_nums, val_count))

        self.train_pages = [p for p in self.all_pages if p["page_num"] not in val_pages]
        self.val_pages = [p for p in self.all_pages if p["page_num"] in val_pages]

        self.train_gt = [g for g in self.ground_truth if g.get("page", 0) not in val_pages]
        self.val_gt = [g for g in self.ground_truth if g.get("page", 0) in val_pages]

        if not self.val_gt:
            self.val_gt = self.ground_truth
            self.val_pages = self.all_pages
            self.train_gt = self.ground_truth
            self.train_pages = self.all_pages

        print(f"  Train: {len(self.train_pages)} pages, {len(self.train_gt)} ground truth refs")
        print(f"  Val:   {len(self.val_pages)} pages, {len(self.val_gt)} ground truth refs")

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    # ----- Extraction -----

    def _extract_with_prompt(self, pages: list[dict], prompt_template: str) -> list[dict]:
        """Run LLM extraction on given pages with a prompt, merge with regex."""
        all_refs = {}
        batch_size = 3
        existing_list = "\n".join(f"- {r['title']}" for r in self.regex_refs[:200])

        for i in range(0, len(pages), batch_size):
            batch = pages[i:i + batch_size]
            batch_text = ""
            for page in batch:
                batch_text += f"\n--- Page {page['page_num']} ---\n{page['text']}"

            try:
                prompt = prompt_template.format(text=batch_text, existing_refs=existing_list)
            except (KeyError, IndexError):
                continue

            try:
                response_text = _call_llm(prompt)
                refs = _parse_llm_response(response_text)
            except Exception as e:
                print(f"    [extract] pages {batch[0]['page_num']}-{batch[-1]['page_num']} failed: {e}")
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

    def _merge_with_regex(self, llm_refs: list[dict]) -> list[dict]:
        """Merge LLM refs with regex refs, deduplicating."""
        merged = list(self.regex_refs)
        for lr in llm_refs:
            if not any(_is_duplicate(lr["title"], er["title"]) for er in merged):
                merged.append(lr)
        return merged

    # ----- Evaluation -----

    def _evaluate(self, candidate: Candidate, pages: list[dict], gt: list[dict]) -> dict:
        """Evaluate a candidate on given pages and ground truth. Returns scores and fills trace."""
        llm_refs = self._extract_with_prompt(pages, candidate.prompt)
        all_refs = self._merge_with_regex(llm_refs)

        metrics = compute_metrics(all_refs, gt)

        gt_matched = [False] * len(gt)
        pred_matched = [False] * len(all_refs)
        for i, pred in enumerate(all_refs):
            for j, g in enumerate(gt):
                if not gt_matched[j] and _match_title(pred["title"], g["title"]):
                    gt_matched[j] = True
                    pred_matched[i] = True
                    break

        trace = EvalTrace(
            missed_refs=[gt[i] for i, m in enumerate(gt_matched) if not m],
            false_positives=[all_refs[i] for i, m in enumerate(pred_matched) if not m],
            correct_refs=[all_refs[i] for i, m in enumerate(pred_matched) if m],
            llm_unique=[r for r in all_refs if r.get("source") == "llm"],
        )
        candidate.trace = trace

        return {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "total": len(all_refs),
            "missed": len(trace.missed_refs),
            "fp": len(trace.false_positives),
            "llm_unique": len(trace.llm_unique),
        }

    # ----- Pareto selection -----

    def _pareto_front(self, candidates: list[Candidate]) -> list[Candidate]:
        """Return the Pareto-optimal candidates (non-dominated set)."""
        front = []
        for c in candidates:
            dominated = False
            for other in candidates:
                if other is not c and other.dominates(c):
                    dominated = True
                    break
            if not dominated:
                front.append(c)
        return front if front else candidates[:1]

    def _select_parent(self, population: list[Candidate]) -> Candidate:
        """Select a parent from the Pareto front, preferring diversity."""
        front = self._pareto_front(population)
        if len(front) == 1:
            return front[0]
        return random.choice(front)

    # ----- Reflection & mutation -----

    def _reflect_and_mutate(self, parent: Candidate, generation: int) -> Candidate | None:
        """Use LLM reflection on execution traces to produce an improved candidate."""
        trace = parent.trace
        scores = parent.scores

        missed_sample = trace.missed_refs[:15]
        fp_sample = trace.false_positives[:10]
        correct_sample = trace.correct_refs[:8]
        llm_additions = trace.llm_unique[:8]

        missed_json = json.dumps(missed_sample, indent=2, default=str)
        fp_json = json.dumps(
            [{"title": r["title"], "type": r.get("type", "?")} for r in fp_sample], indent=2
        )
        llm_json = json.dumps(
            [{"title": r["title"], "type": r.get("type", "?")} for r in llm_additions], indent=2
        )
        correct_json = json.dumps(
            [{"title": r["title"], "type": r.get("type", "?")} for r in correct_sample], indent=2
        )

        reflection_prompt = (
            "You are an expert prompt engineer. Your task: rewrite a reference extraction prompt to fix the problems below.\n\n"
            "CURRENT PROMPT:\n--- START ---\n"
            f"{parent.prompt}\n"
            "--- END ---\n\n"
            "SCORES: "
            f"Precision={scores.get('precision', 0)}, Recall={scores.get('recall', 0)}, F1={scores.get('f1', 0)}, "
            f"Missed={scores.get('missed', 0)}, False Positives={scores.get('fp', 0)}\n\n"
            f"MISSED (system failed to find these references):\n{missed_json}\n\n"
            f"FALSE POSITIVES (system hallucinated these):\n{fp_json}\n\n"
            f"CORRECTLY FOUND examples:\n{correct_json}\n\n"
            "INSTRUCTIONS:\n"
            "1. Analyze what types of references are missed and what is hallucinated\n"
            "2. Rewrite the ENTIRE prompt to fix these issues\n"
            "3. The rewritten prompt MUST contain {existing_refs} and {text} as placeholders\n"
            "4. Use double curly braces {{ and }} for literal JSON braces in examples\n"
            "5. Include 5 few-shot examples targeting the missed patterns\n\n"
            "Return the rewritten prompt between START_PROMPT and END_PROMPT tags.\n\n"
            "START_PROMPT\n[your rewritten prompt here]\nEND_PROMPT"
        )

        try:
            response = _call_llm(reflection_prompt, max_tokens=4096)
            text = response.strip()
        except Exception as e:
            print(f"    [reflect] failed: {e}")
            return None

        # Parse between START_PROMPT and END_PROMPT tags
        start_tag = "START_PROMPT"
        end_tag = "END_PROMPT"
        if start_tag in text and end_tag in text:
            new_prompt = text.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
        elif "===PROMPT_SEPARATOR===" in text:
            new_prompt = text.split("===PROMPT_SEPARATOR===", 1)[1].strip()
        else:
            print("    [reflect] could not parse prompt from response")
            return None

        if new_prompt.startswith("```"):
            new_prompt = re.sub(r"^```(?:\w*)?\s*\n?", "", new_prompt)
            new_prompt = re.sub(r"\n?```\s*$", "", new_prompt)

        if "{existing_refs}" not in new_prompt or "{text}" not in new_prompt:
            fixed = new_prompt
            for variant in ["{{existing_refs}}", "<<existing_refs>>", "[existing_refs]"]:
                if variant in fixed:
                    fixed = fixed.replace(variant, "{existing_refs}")
            for variant in ["{{text}}", "<<text>>", "[text]"]:
                if variant in fixed:
                    fixed = fixed.replace(variant, "{text}")
            if "{existing_refs}" in fixed and "{text}" in fixed:
                new_prompt = fixed
            else:
                print("    [reflect] invalid placeholders in generated prompt")
                return None

        new_prompt = _sanitize_prompt_braces(new_prompt)

        child = Candidate(
            prompt=new_prompt,
            generation=generation,
            parent_id=parent.id,
            id=self._new_id(),
        )

        print(f"    [reflect] generated prompt ({len(new_prompt)} chars)")
        return child

    # ----- Main optimization loop -----

    def run(self) -> list[dict]:
        print(f"\n{'='*70}")
        print(f"GEPA OPTIMIZATION")
        print(f"  Population: {self.population_size} | Iterations: {self.iterations}")
        print(f"  Minibatch: {self.minibatch_pages} pages | Model: {_LLM_MODEL}")
        print(f"{'='*70}")

        # Step 1: Seed population with the original prompt + variants
        print(f"\n[Phase 1] Seeding population...")
        seed = Candidate(prompt=LLM_PROMPT, generation=0, id=self._new_id())
        population = [seed]

        # Evaluate seed on training set
        print(f"  Evaluating seed candidate...")
        seed.scores = self._evaluate(seed, self.train_pages, self.train_gt)
        self._print_candidate(seed, "SEED")

        # Generate initial population through mutations of seed
        for i in range(self.population_size - 1):
            print(f"  Generating initial variant {i+1}...")
            child = self._reflect_and_mutate(seed, generation=0)
            if child:
                child.scores = self._evaluate(child, self.train_pages, self.train_gt)
                population.append(child)
                self._print_candidate(child, f"VARIANT-{i+1}")

        history = []

        # Step 2: Evolutionary loop
        for iteration in range(self.iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{self.iterations}")
            print(f"{'='*70}")

            # Sample minibatch pages for this iteration
            minibatch_pages = self._sample_minibatch()
            minibatch_gt = [g for g in self.train_gt
                           if g.get("page", 0) in {p["page_num"] for p in minibatch_pages}]
            if not minibatch_gt:
                minibatch_gt = self.train_gt
                minibatch_pages = self.train_pages

            print(f"  Minibatch: {len(minibatch_pages)} pages, {len(minibatch_gt)} ground truth refs")

            # Select parent from Pareto front
            parent = self._select_parent(population)
            print(f"  Selected parent: C{parent.id} (gen={parent.generation}, F1={parent.scores.get('f1', 0):.3f})")

            # Reflect and mutate
            print(f"  Reflecting on traces and mutating...")
            child = self._reflect_and_mutate(parent, generation=iteration + 1)

            if child is None:
                print(f"  Mutation failed, trying another parent...")
                other_parents = [c for c in population if c is not parent]
                if other_parents:
                    parent2 = random.choice(other_parents)
                    child = self._reflect_and_mutate(parent2, generation=iteration + 1)

            if child is None:
                print(f"  All mutations failed this iteration, skipping...")
                continue

            # Evaluate child on minibatch
            print(f"  Evaluating child C{child.id} on minibatch...")
            child.scores = self._evaluate(child, minibatch_pages, minibatch_gt)
            self._print_candidate(child, f"CHILD-{iteration+1}")

            # Re-evaluate parent on same minibatch for fair comparison
            parent_mb_scores = self._evaluate(parent, minibatch_pages, minibatch_gt)

            # Accept if child doesn't regress badly
            if child.scores.get("f1", 0) >= parent_mb_scores.get("f1", 0) * 0.95:
                population.append(child)
                print(f"  ACCEPTED C{child.id} into population (size={len(population)})")

                # Prune population to maintain size
                if len(population) > self.population_size * 2:
                    population = self._prune_population(population)
                    print(f"  Pruned to {len(population)} candidates")
            else:
                print(f"  REJECTED C{child.id} (F1={child.scores.get('f1',0):.3f} "
                      f"vs parent={parent_mb_scores.get('f1',0):.3f})")

            # Record iteration state
            front = self._pareto_front(population)
            best = max(population, key=lambda c: c.scores.get("f1", 0))
            history.append({
                "iteration": iteration + 1,
                "population_size": len(population),
                "pareto_front_size": len(front),
                "best_f1": best.scores.get("f1", 0),
                "best_precision": best.scores.get("precision", 0),
                "best_recall": best.scores.get("recall", 0),
            })

            self._print_population_summary(population)

        # Step 3: Full validation
        print(f"\n{'='*70}")
        print(f"VALIDATION PHASE")
        print(f"{'='*70}")

        front = self._pareto_front(population)
        print(f"  Validating {len(front)} Pareto-optimal candidates on full validation set...")

        for c in front:
            c.val_scores = self._evaluate(c, self.val_pages, self.val_gt)
            print(f"  C{c.id}: P={c.val_scores['precision']:.3f} "
                  f"R={c.val_scores['recall']:.3f} F1={c.val_scores['f1']:.3f}")

        best_val = max(front, key=lambda c: c.val_scores.get("f1", 0))

        # Also validate seed for comparison
        seed_val = self._evaluate(seed, self.val_pages, self.val_gt)

        # Step 4: Summary
        print(f"\n{'='*70}")
        print(f"GEPA OPTIMIZATION SUMMARY")
        print(f"{'='*70}")

        print(f"\nEvolution history:")
        print(f"{'Iter':<6} {'Pop':>5} {'Front':>6} {'Best F1':>9} {'Precision':>10} {'Recall':>8}")
        print(f"{'-'*46}")
        for h in history:
            print(f"{h['iteration']:<6} {h['population_size']:>5} {h['pareto_front_size']:>6} "
                  f"{h['best_f1']:>9.3f} {h['best_precision']:>10.3f} {h['best_recall']:>8.3f}")

        print(f"\nValidation results:")
        print(f"  Seed (original):  P={seed_val['precision']:.3f} R={seed_val['recall']:.3f} "
              f"F1={seed_val['f1']:.3f}")
        print(f"  Best (GEPA):      P={best_val.val_scores['precision']:.3f} "
              f"R={best_val.val_scores['recall']:.3f} F1={best_val.val_scores['f1']:.3f}")

        f1_diff = best_val.val_scores["f1"] - seed_val["f1"]
        print(f"  Improvement:      F1 {f1_diff:+.3f}")

        # Save best prompt
        if best_val.val_scores.get("f1", 0) > seed_val.get("f1", 0):
            with open("gepa_optimized_prompt.txt", "w") as f:
                f.write(best_val.prompt)
            print(f"\n  Best prompt (C{best_val.id}, gen={best_val.generation}) "
                  f"saved to gepa_optimized_prompt.txt")
        else:
            print(f"\n  No improvement over seed — original prompt is already optimal")

        # Full run on all pages with best prompt for final numbers
        print(f"\n  Running final extraction on ALL pages with best prompt...")
        final_candidate = Candidate(prompt=best_val.prompt, id=-1)
        final_scores = self._evaluate(final_candidate, self.all_pages, self.ground_truth)
        print(f"  FINAL (all pages): P={final_scores['precision']:.3f} "
              f"R={final_scores['recall']:.3f} F1={final_scores['f1']:.3f} "
              f"Found={final_scores['total']}")

        return history

    # ----- Helpers -----

    def _sample_minibatch(self) -> list[dict]:
        """Sample a random subset of training pages."""
        if len(self.train_pages) <= self.minibatch_pages:
            return self.train_pages
        return random.sample(self.train_pages, self.minibatch_pages)

    def _prune_population(self, population: list[Candidate]) -> list[Candidate]:
        """Prune population keeping Pareto front + best by F1."""
        front = self._pareto_front(population)
        remaining = [c for c in population if c not in front]
        remaining.sort(key=lambda c: c.scores.get("f1", 0), reverse=True)
        keep = self.population_size - len(front)
        return front + remaining[:max(0, keep)]

    def _print_candidate(self, c: Candidate, label: str):
        s = c.scores
        print(f"    {label} C{c.id}: P={s.get('precision',0):.3f} R={s.get('recall',0):.3f} "
              f"F1={s.get('f1',0):.3f} | found={s.get('total',0)} missed={s.get('missed',0)} "
              f"fp={s.get('fp',0)} llm_unique={s.get('llm_unique',0)}")

    def _print_population_summary(self, population: list[Candidate]):
        front = self._pareto_front(population)
        print(f"\n  Population ({len(population)} candidates, {len(front)} on Pareto front):")
        for c in sorted(population, key=lambda x: x.scores.get("f1", 0), reverse=True):
            marker = " *" if c in front else "  "
            print(f"  {marker} C{c.id} gen={c.generation}: "
                  f"P={c.scores.get('precision',0):.3f} R={c.scores.get('recall',0):.3f} "
                  f"F1={c.scores.get('f1',0):.3f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python gepa_optimize.py <pdf_path> <ground_truth.json> [iterations] [population]")
        sys.exit(1)

    pdf = sys.argv[1]
    gt = sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    pop = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    optimizer = GEPAOptimizer(pdf, gt, population_size=pop, iterations=iters)
    optimizer.run()
