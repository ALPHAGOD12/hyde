"""Core extraction engine: PDF parsing, regex extraction, LLM extraction, and hybrid merge."""

import json
import os
import re
from difflib import SequenceMatcher

import fitz  # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()
_client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
_LLM_MODEL = "qwen2.5:7b"

MATCH_THRESHOLD = 0.75

# ---------------------------------------------------------------------------
# Regex patterns: (compiled_regex, reference_type)
# ---------------------------------------------------------------------------
REFERENCE_PATTERNS = [
    # SEBI circular numbers – new format
    (re.compile(
        r"SEBI/HO/[A-Z0-9/\-]+(?:/P)?/CIR/\d{4}/\d+",
        re.IGNORECASE,
    ), "circular"),

    # SEBI circular numbers – older CIR/ format
    (re.compile(
        r"CIR/[A-Z]+/\d+/\d{4}",
        re.IGNORECASE,
    ), "circular"),

    # SEBI circular – IMD/CIR, MRD/DoP etc. format
    (re.compile(
        r"(?:IMD|MRD|MIRSD|OIAE|CFD|CDMRD)[/-][A-Z/]+/\d+/\d{4}",
        re.IGNORECASE,
    ), "circular"),

    # Master Circular references
    (re.compile(
        r"Master\s+Circular\s+(?:No\.?\s*)?[A-Z0-9/\-]+(?:\s+dated\s+[\w\s,]+\d{4})?",
        re.IGNORECASE,
    ), "circular"),

    # SEBI Regulations (full name with year)
    (re.compile(
        r"SEBI\s*\([\w\s,&\-]+\)\s*Regulations?,?\s*\d{4}",
        re.IGNORECASE,
    ), "regulation"),

    # Securities and Exchange Board of India Act
    (re.compile(
        r"Securities\s+and\s+Exchange\s+Board\s+of\s+India\s+Act,?\s*\d{4}",
        re.IGNORECASE,
    ), "act"),

    # Securities Contracts (Regulation) Act
    (re.compile(
        r"Securities\s+Contracts?\s*\(Regulation\)\s*Act,?\s*\d{4}",
        re.IGNORECASE,
    ), "act"),

    # Depositories Act
    (re.compile(
        r"Depositories\s+Act,?\s*\d{4}",
        re.IGNORECASE,
    ), "act"),

    # Companies Act
    (re.compile(
        r"Companies\s+Act,?\s*\d{4}",
        re.IGNORECASE,
    ), "act"),

    # Other common Indian acts
    (re.compile(
        r"(?:Prevention\s+of\s+Money\s+Laundering\s+Act|"
        r"Income\s+Tax\s+Act|"
        r"Indian\s+Contract\s+Act|"
        r"Foreign\s+Exchange\s+Management\s+Act|"
        r"Insolvency\s+and\s+Bankruptcy\s+Code)"
        r",?\s*\d{4}",
        re.IGNORECASE,
    ), "act"),

    # Section/Regulation X of <document>
    (re.compile(
        r"(?:Section|Regulation|Rule|Clause|Chapter)\s+\d+[A-Za-z]?"
        r"(?:\s*\(\d+\))*\s+of\s+(?:the\s+)?(.{10,120}?)(?=\.|,|\n|$)",
        re.IGNORECASE,
    ), "section_reference"),

    # Gazette of India notifications
    (re.compile(
        r"(?:Official\s+)?Gazette\s+of\s+India.*?(?:Part\s+[IVX]+[A-Za-z\s]*)?",
        re.IGNORECASE,
    ), "gazette"),

    # Generic circular number with date pattern
    (re.compile(
        r"(?:circular|letter)\s+(?:no\.?\s*)?[A-Z0-9/\-]+\s+dated\s+[\w\s,]+\d{4}",
        re.IGNORECASE,
    ), "circular"),
]


# ---------------------------------------------------------------------------
# Step 1: PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_by_page(pdf_path: str) -> list[dict]:
    """Extract text from each page of a PDF, preserving page numbers."""
    pages = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({"page_num": i + 1, "text": text})
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Step 2a: Regex-based extraction
# ---------------------------------------------------------------------------

def extract_references_regex(pages: list[dict]) -> list[dict]:
    """Extract references using regex patterns. Returns deduplicated list."""
    raw_refs: dict[str, dict] = {}  # normalized_title -> reference dict

    for page in pages:
        text = page["text"]
        for pattern, ref_type in REFERENCE_PATTERNS:
            for match in pattern.finditer(text):
                title = match.group(0).strip()
                title = re.sub(r"\s+", " ", title)  # normalize whitespace
                key = title.lower()

                if key in raw_refs:
                    if page["page_num"] not in raw_refs[key]["page_numbers"]:
                        raw_refs[key]["page_numbers"].append(page["page_num"])
                else:
                    raw_refs[key] = {
                        "title": title,
                        "type": ref_type,
                        "page_numbers": [page["page_num"]],
                        "source": "regex",
                    }

    return list(raw_refs.values())


# ---------------------------------------------------------------------------
# Step 2b: LLM-based extraction via Hugging Face Inference API
# ---------------------------------------------------------------------------

LLM_PROMPT = """You are a legal document analyst specializing in Indian securities regulation (SEBI).

The following references have ALREADY been found by pattern matching:
{existing_refs}

Your job is to find ADDITIONAL cross-references to other documents that the pattern matcher missed.

PRIORITY FORMATS TO FIND (most commonly missed):

1. DATE-TITLE ENTRIES in numbered lists — these are the most important to catch.
   They appear as a serial number followed by a date and a descriptive subject, with NO formal circular number.
   Examples from real SEBI documents:
   - "39 Oct 27, 2010- European Style Stock Options"
   - "72 Jul 08, 2004 - Clarification for circular no. DNPD/Cir-25/04 dated June 10, 2004"
   - "97 Dec 15, 2000 - Use of Digital Signature on Contract Notes"
   - "55 Jun 16, 1998 - Derivatives trading in India"
   - "67 December 05, 2013 - Exchange Traded Cash Settled Interest Rate Futures (IRF) on 10-Year Government of India Security"
   - "40 Jul 28, 1999 - Risk Containment Measures for the Index Futures Market"

2. REF. NO. FORMAT — references starting with "Ref. No." or "Ref.No." or "Ref. SMD/..." :
   - "Ref. No. DNPD/Cir-23/04 dated April 27, 2004"
   - "Ref.No. DNPD/Cir-25/04 dated June 10, 2004"
   - "Ref. SMD/6059 dated October 17, 1994"

3. LETTERS AND EMAILS — SEBI letters, emails, and informal references:
   - "SEBI letter dated January 5, 2023"
   - "SEBI email dated February 06, 2020"
   - "Letter dated November 6, 2008"
   - "Letter dated May 29, 2019"
   - "SEBI Email dated May 4, 2020 on Rationalisation of Strikes on Long dated options"

4. PRESS RELEASES:
   - "Press Release No. 49/2018 dated December 03, 2018"

5. CIRCULAR NUMBERS with non-standard prefixes (SMD, SMDRP, DNPD, etc.):
   - "Circular No. SMD/536/95 dated March 28, 1995"
   - "Circular No. Ref. SMD-II/52 dated January 10, 1996"
   - "Circular No. SMD/POLICY/CIR (DBA-II)-37/98 dated December 04, 1998"

Do NOT repeat references already listed above. Only return NEW ones.

EXAMPLES:

Example 1 (date-title entry from numbered list):
Text: "39 Oct 27, 2010- European Style Stock Options"
Output: {{"title": "Oct 27, 2010- European Style Stock Options", "type": "circular", "page_numbers": [8]}}

Example 2 (date-title entry with full date):
Text: "67 December 05, 2013 - Exchange Traded Cash Settled Interest Rate Futures (IRF) on 10-Year Government of India Security"
Output: {{"title": "December 05, 2013 - Exchange Traded Cash Settled Interest Rate Futures (IRF) on 10-Year Government of India Security", "type": "circular", "page_numbers": [9]}}

Example 3 (Ref. No. format):
Text: "Ref. No. DNPD/Cir-23/04 dated April 27, 2004"
Output: {{"title": "Ref. No. DNPD/Cir-23/04 dated April 27, 2004", "type": "other", "page_numbers": [7]}}

Example 4 (SEBI email):
Text: "108 SEBI Email dated May 4, 2020 on Rationalisation of Strikes on Long dated options"
Output: {{"title": "SEBI Email dated May 4, 2020 on Rationalisation of Strikes on Long dated options", "type": "other", "page_numbers": [10]}}

Example 5 (letter reference):
Text: "Letter dated November 6, 2008"
Output: {{"title": "Letter dated November 6, 2008", "type": "other", "page_numbers": [12]}}

Example 6 (non-standard circular prefix):
Text: "Circular No. SMD/536/95 dated March 28, 1995"
Output: {{"title": "Circular No. SMD/536/95 dated March 28, 1995", "type": "circular", "page_numbers": [1]}}

Example 7 (date-title entry with risk topic):
Text: "40 Jul 28, 1999 - Risk Containment Measures for the Index Futures Market"
Output: {{"title": "Jul 28, 1999 - Risk Containment Measures for the Index Futures Market", "type": "circular", "page_numbers": [9]}}

For each new reference return:
- "title": the document title or identifier as it appears in the text
- "type": one of "circular", "regulation", "act", "gazette", "section_reference", "other"
- "page_numbers": list of page numbers where it appears

Return ONLY a valid JSON array. No markdown fences, no explanation, no extra text.
If there are no additional references, return an empty array: []

Text:
{text}"""


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    retry=retry_if_exception_type(Exception),
)
def _call_llm(prompt: str) -> str:
    """Call local LLM via Ollama's OpenAI-compatible API with retry logic."""
    response = _client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=_LLM_MODEL,
        max_tokens=4096,
        temperature=0.2,
    )
    return response.choices[0].message.content


def _parse_llm_response(text: str) -> list[dict]:
    """Parse JSON from LLM response, handling markdown fences."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        refs = json.loads(text)
        if isinstance(refs, list):
            return refs
    except json.JSONDecodeError:
        pass
    return []


def extract_references_llm(pages: list[dict], regex_refs: list[dict] = None) -> list[dict]:
    """Extract references using LLM via HF Inference API, processing in batches of 3 pages.

    If regex_refs is provided, the LLM is told what was already found
    and asked to only find additional references.
    """
    import time

    # Format existing regex refs as a simple list for the prompt
    if regex_refs:
        existing_list = "\n".join(f"- {r['title']}" for r in regex_refs[:50])
        if len(regex_refs) > 50:
            existing_list += f"\n... and {len(regex_refs) - 50} more regex-found references"
    else:
        existing_list = "(none)"

    all_refs: dict[str, dict] = {}  # normalized_title -> reference dict
    batch_size = 3

    for i in range(0, len(pages), batch_size):
        batch = pages[i:i + batch_size]
        batch_text = ""
        for page in batch:
            batch_text += f"\n--- Page {page['page_num']} ---\n{page['text']}"

        prompt = LLM_PROMPT.format(text=batch_text, existing_refs=existing_list)

        try:
            response_text = _call_llm(prompt)
            refs = _parse_llm_response(response_text)
        except Exception as e:
            print(f"    Warning: LLM extraction failed for pages {batch[0]['page_num']}-{batch[-1]['page_num']}: {e}")
            continue

        for ref in refs:
            if not isinstance(ref, dict):
                continue

            if "title" in ref:
                title = ref.get("title", "").strip()
            elif "Date" in ref and "Details" in ref:
                title = f"{ref['Date']} - {ref['Details']}".strip()
            elif "date" in ref and "details" in ref:
                title = f"{ref['date']} - {ref['details']}".strip()
            else:
                continue

            if not title:
                continue

            ref_type = ref.get("type", "circular")
            page_nums = ref.get("page_numbers", [pn for p in batch for pn in [p["page_num"]]])

            key = title.lower()
            if key in all_refs:
                for pn in (page_nums if isinstance(page_nums, list) else [page_nums]):
                    if pn not in all_refs[key]["page_numbers"]:
                        all_refs[key]["page_numbers"].append(pn)
            else:
                all_refs[key] = {
                    "title": title,
                    "type": ref_type,
                    "page_numbers": page_nums if isinstance(page_nums, list) else [page_nums],
                    "source": "llm",
                }

        if i + batch_size < len(pages):
            time.sleep(1)

    return list(all_refs.values())


# ---------------------------------------------------------------------------
# Step 3: Hybrid merge
# ---------------------------------------------------------------------------

def _is_duplicate(title_a: str, title_b: str, threshold: float = MATCH_THRESHOLD) -> bool:
    """Check if two reference titles are duplicates using fuzzy matching."""
    a = title_a.lower().strip()
    b = title_b.lower().strip()
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


_DEDUP_PROMPT = """You are given a cluster of reference titles that may refer to the same document.
Pick the SINGLE most complete canonical form — the one with the fullest circular number, date, and title.

Cluster:
{cluster_json}

Return ONLY a single JSON object with:
- "canonical_title": the best/most complete title from the cluster
- "merged_page_numbers": combined list of all page numbers from the cluster

No explanation, no markdown fences. Just the JSON object."""


def _llm_deduplicate(refs: list[dict]) -> list[dict]:
    """Cluster near-duplicate refs and use LLM to pick canonical forms.

    Only clusters refs where one title is a strict substring of another
    (a clear fragment/full-form pair), NOT merely similar strings.
    """
    import time

    n = len(refs)
    assigned = [False] * n
    clusters: list[list[int]] = []

    for i in range(n):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        ti = refs[i]["title"].lower().strip()
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            tj = refs[j]["title"].lower().strip()
            if ti in tj or tj in ti:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    deduplicated = []
    for cluster in clusters:
        if len(cluster) == 1:
            deduplicated.append(refs[cluster[0]])
            continue

        items = [refs[idx] for idx in cluster]
        all_pages = []
        all_sources = set()
        for item in items:
            for pn in item.get("page_numbers", []):
                if pn not in all_pages:
                    all_pages.append(pn)
            all_sources.add(item.get("source", "regex"))

        best = max(items, key=lambda r: len(r["title"]))

        if len(items) <= 4:
            cluster_data = [{"title": it["title"], "page_numbers": it["page_numbers"]} for it in items]
            prompt = _DEDUP_PROMPT.format(cluster_json=json.dumps(cluster_data, indent=2))
            try:
                raw = _call_llm(prompt)
                parsed = _parse_llm_response_obj(raw)
                if parsed and "canonical_title" in parsed:
                    best_title = parsed["canonical_title"].strip()
                    if best_title:
                        best = {"title": best_title, "type": best["type"]}
                    if "merged_page_numbers" in parsed and isinstance(parsed["merged_page_numbers"], list):
                        all_pages = sorted(set(all_pages + parsed["merged_page_numbers"]))
            except Exception:
                pass
            time.sleep(0.5)

        source = "both" if len(all_sources) > 1 else all_sources.pop()
        deduplicated.append({
            "title": best["title"],
            "type": best.get("type", "circular"),
            "page_numbers": sorted(all_pages),
            "source": source,
        })

    return deduplicated


_VERIFY_PROMPT = """You are a legal document analyst. Below is a list of extracted cross-references from a SEBI circular.
Your job is to verify each one: is it a REAL cross-reference to another regulatory document (circular, regulation, act, letter, email, press release, gazette notification)?

Remove entries that are NOT actual cross-references, such as:
- Internal section headings or table of contents entries from the SAME document
- Partial fragments that are just department codes (e.g. bare "MRD/DP/14/2010" without "Circular No." prefix)
- Page numbers, serial numbers, or index entries
- Generic descriptions that are not references to specific documents

Keep entries that ARE real cross-references, even if informal (e.g. "SEBI letter dated January 5, 2023").

References to verify:
{refs_json}

Return ONLY a JSON array of the titles that are VALID cross-references. No markdown fences, no explanation.
Example: ["Circular No. CIR/MRD/DP/21/2010 dated July 15, 2010", "SEBI letter dated January 5, 2023"]"""


def _llm_verify(refs: list[dict], pages: list[dict]) -> list[dict]:
    """LLM verification pass to filter out false positives."""
    import time

    if not refs:
        return refs

    batch_size = 30
    verified_titles: set[str] = set()
    total_input = 0
    total_kept = 0

    for i in range(0, len(refs), batch_size):
        batch = refs[i:i + batch_size]
        titles = [r["title"] for r in batch]
        total_input += len(titles)
        prompt = _VERIFY_PROMPT.format(refs_json=json.dumps(titles, indent=2))

        try:
            raw = _call_llm(prompt)
            kept = _parse_llm_response(raw)
            if isinstance(kept, list) and len(kept) > 0:
                batch_kept = 0
                for title in kept:
                    if isinstance(title, str):
                        verified_titles.add(title.strip().lower())
                        batch_kept += 1
                    elif isinstance(title, dict) and "title" in title:
                        verified_titles.add(title["title"].strip().lower())
                        batch_kept += 1
                total_kept += batch_kept
                if batch_kept < len(titles) * 0.3:
                    for r in batch:
                        verified_titles.add(r["title"].lower())
            else:
                for r in batch:
                    verified_titles.add(r["title"].lower())
        except Exception:
            for r in batch:
                verified_titles.add(r["title"].lower())

        if i + batch_size < len(refs):
            time.sleep(0.5)

    if total_kept < total_input * 0.5:
        print(f"    Warning: LLM verification was too aggressive ({total_kept}/{total_input} kept), skipping filter")
        return refs

    return [r for r in refs if r["title"].strip().lower() in verified_titles]


def _parse_llm_response_obj(text: str) -> dict | None:
    """Parse a single JSON object from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return None


def extract_references(pdf_path: str) -> dict:
    """
    Main extraction function. Returns dict with:
    - references: merged list of all references
    - stats: counts by source
    """
    print(f"Parsing PDF: {pdf_path}")
    pages = extract_text_by_page(pdf_path)
    print(f"  Extracted text from {len(pages)} pages")

    # Regex pass
    print("  Running regex extraction...")
    regex_refs = extract_references_regex(pages)
    print(f"  Found {len(regex_refs)} references via regex")

    # LLM pass (informed by regex results)
    print("  Running LLM extraction (informed by regex results)...")
    llm_refs = extract_references_llm(pages, regex_refs=regex_refs)
    print(f"  Found {len(llm_refs)} references via LLM")

    # Merge: start with regex results, add non-duplicate LLM results
    merged = list(regex_refs)
    llm_only_count = 0

    for llm_ref in llm_refs:
        is_dup = False
        for existing in merged:
            if _is_duplicate(llm_ref["title"], existing["title"]):
                # Mark as found by both
                if existing["source"] == "regex":
                    existing["source"] = "both"
                # Merge page numbers
                for pn in llm_ref.get("page_numbers", []):
                    if pn not in existing["page_numbers"]:
                        existing["page_numbers"].append(pn)
                is_dup = True
                break
        if not is_dup and llm_ref["title"]:
            merged.append(llm_ref)
            llm_only_count += 1

    # Sort by first page number
    merged.sort(key=lambda r: r["page_numbers"][0] if r["page_numbers"] else 999)

    # LLM-powered deduplication: merge near-duplicate fragments into canonical forms
    print("  Running LLM-powered deduplication...")
    merged = _llm_deduplicate(merged)
    print(f"  {len(merged)} references after deduplication")

    # LLM verification pass: filter out false positives
    print("  Running LLM verification pass...")
    merged = _llm_verify(merged, pages)
    print(f"  {len(merged)} references after verification")

    # Stats
    stats = {
        "total": len(merged),
        "regex_only": sum(1 for r in merged if r["source"] == "regex"),
        "llm_only": sum(1 for r in merged if r["source"] == "llm"),
        "both": sum(1 for r in merged if r["source"] == "both"),
    }

    return {"references": merged, "stats": stats}
