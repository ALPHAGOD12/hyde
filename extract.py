"""Core extraction engine: PDF parsing, regex extraction, LLM extraction, and hybrid merge."""

import json
import os
import re
from difflib import SequenceMatcher

import fitz  # PyMuPDF
from google import genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()
_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

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
# Step 2b: LLM-based extraction via Gemini
# ---------------------------------------------------------------------------

LLM_PROMPT = """You are a legal document analyst specializing in Indian securities regulation (SEBI).

Given the following text from a SEBI circular (with page markers like "--- Page X ---"), extract ALL references to external documents including:
- Other SEBI circulars (any format of circular number)
- SEBI regulations (e.g. SEBI (Listing Obligations) Regulations, 2015)
- Indian acts and laws (e.g. Companies Act, 2013)
- Gazette notifications
- Any other regulatory or legal documents referenced

For each reference return:
- "title": the document title or identifier as it appears in the text
- "type": one of "circular", "regulation", "act", "gazette", "section_reference", "other"
- "page_numbers": list of page numbers where it appears

Return ONLY a valid JSON array. No markdown fences, no explanation, no extra text.

Text:
{text}"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(Exception),
)
def _call_gemini(prompt: str) -> str:
    """Call Gemini API with retry logic."""
    response = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


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


def extract_references_llm(pages: list[dict]) -> list[dict]:
    """Extract references using Gemini LLM, processing in batches of 3 pages."""
    import time

    all_refs: dict[str, dict] = {}  # normalized_title -> reference dict
    batch_size = 3

    for i in range(0, len(pages), batch_size):
        batch = pages[i:i + batch_size]
        batch_text = ""
        for page in batch:
            batch_text += f"\n--- Page {page['page_num']} ---\n{page['text']}"

        prompt = LLM_PROMPT.format(text=batch_text)

        try:
            response_text = _call_gemini(prompt)
            refs = _parse_llm_response(response_text)
        except Exception as e:
            print(f"    Warning: LLM extraction failed for pages {batch[0]['page_num']}-{batch[-1]['page_num']}: {e}")
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

        # Rate limit: small delay between batches
        if i + batch_size < len(pages):
            time.sleep(2)

    return list(all_refs.values())


# ---------------------------------------------------------------------------
# Step 3: Hybrid merge
# ---------------------------------------------------------------------------

def _is_duplicate(title_a: str, title_b: str, threshold: float = 0.75) -> bool:
    """Check if two reference titles are duplicates using fuzzy matching."""
    a = title_a.lower().strip()
    b = title_b.lower().strip()
    # Exact substring match
    if a in b or b in a:
        return True
    # Fuzzy ratio
    return SequenceMatcher(None, a, b).ratio() >= threshold


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

    # LLM pass
    print("  Running LLM extraction...")
    llm_refs = extract_references_llm(pages)
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

    # Stats
    stats = {
        "total": len(merged),
        "regex_only": sum(1 for r in merged if r["source"] == "regex"),
        "llm_only": sum(1 for r in merged if r["source"] == "llm"),
        "both": sum(1 for r in merged if r["source"] == "both"),
    }

    return {"references": merged, "stats": stats}
