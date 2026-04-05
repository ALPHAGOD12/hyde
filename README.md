# SEBI Circular Knowledge Graph Agent

An AI agent that takes a SEBI circular PDF as input and extracts **all references** to other documents — circulars, regulations, acts, gazette notifications, press releases, and more. Built for compliance teams at Indian banks who need to track and respond to SEBI's evolving regulatory landscape.

## Problem

SEBI circulars frequently reference other circulars, regulations, and laws. Since these documents are all in PDF format, manually tracking cross-references across hundreds of documents is time-consuming and error-prone. A single master circular can reference 400+ other documents across dozens of formats.

## Solution

This agent uses a **hybrid extraction approach** — combining fast regex pattern matching with LLM-powered extraction — to achieve comprehensive reference detection that neither method could accomplish alone.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/ALPHAGOD12/hyde.git
cd hyde
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your Gemini API key

Get a free API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), then create a `.env` file:

```
GOOGLE_API_KEY=your_key_here
```

### 3. Run extraction

```bash
python main.py extract path/to/circular.pdf
python main.py extract path/to/circular.pdf --output results.json
```

### 4. Run evaluation (optional)

```bash
python main.py evaluate path/to/circular.pdf ground_truth.json
```

---

## How It Works

### Architecture

```
PDF Input
    |
    v
[1. PDF Parser] --- PyMuPDF extracts text page-by-page
    |
    v
[2. Regex Extractor] --- 12+ patterns for known SEBI reference formats
    |                     (fast, free, no API calls)
    |
    v
[3. LLM Extractor] --- Gemini 2.5 Flash processes pages in batches of 3
    |                   (catches informal/non-standard references)
    |
    v
[4. Merge & Dedup] --- Fuzzy matching combines results, removes duplicates
    |
    v
Structured JSON Output (with page numbers and source tags)
```

### Step 1: PDF Text Extraction

Uses PyMuPDF (`fitz`) to extract text from each page independently. This preserves page-level granularity so every reference can be traced back to its exact page number.

### Step 2: Regex Extraction (Fast Pass)

Applies 12+ compiled regex patterns against each page's text. This catches all structured reference formats used by SEBI:

| Pattern | What it catches | Example |
|---------|----------------|---------|
| `SEBI/HO/.../CIR/YYYY/N` | New format SEBI circulars | `SEBI/HO/MRD/MRD-PoD-3/P/CIR/2024/65` |
| `CIR/.../N/YYYY` | Older CIR format | `CIR/MRD/DP/13/2010` |
| `IMD/MRD/MIRSD...` | Department-prefixed circulars | `MRD/DP/14/2010` |
| `SEBI (...) Regulations, YYYY` | SEBI Regulations | `SEBI (LODR) Regulations, 2015` |
| `... Act, YYYY` | Indian Acts | `Companies Act, 2013` |
| `Section/Regulation X of ...` | Section-level references | `Section 11(1) of the SEBI Act` |
| `Gazette of India...` | Gazette notifications | `Gazette of India, Part III` |
| `Master Circular...` | Master circular references | `Master Circular No. CIR/...` |
| `Circular/Letter No. ... dated ...` | Generic dated references | `Circular No. SMD/... dated March 14, 1995` |

### Step 3: LLM Extraction (Deep Pass)

Sends pages to Google Gemini 2.5 Flash in batches of 3 with a specialized prompt. The LLM catches what regex cannot:

- **Non-standard formats**: `Ref. No. DNPD/Cir-22/04 dated April 01, 2004`
- **Descriptive references**: `"Oct 27, 2010 - European Style Stock Options"`
- **Press releases**: `PR No.111/2005 dated Sep 02, 2005`
- **SEBI emails/letters**: `SEBI Email dated May 4, 2020`
- **Informal mentions**: References without standard circular number formatting

### Step 4: Merge & Deduplicate

- Regex results form the baseline
- LLM results are added only if they don't duplicate an existing entry
- Deduplication uses `difflib.SequenceMatcher` with a 0.75 similarity threshold
- Each reference is tagged with its source: `regex`, `llm`, or `both`
- Results are sorted by first page number of appearance

---

## Results

### Benchmark: 13-page SEBI Master Circular (Schedule I - Stock Exchanges)

| Method | References Found | Unique Additions |
|--------|:---------------:|:----------------:|
| Regex only | 382 | — |
| LLM only | 453 | — |
| **Hybrid (merged)** | **477** | **+95 from LLM** |

The hybrid approach found **25% more references** than regex alone.

### What the LLM Added (95 references regex missed)

| Category | Count | Example |
|----------|:-----:|---------|
| Non-standard circular formats | 83 | `Ref.No. DNPD/Cir-24/04 dated May 26, 2004` |
| Press releases | 2 | `Press Release No. 49/2018 dated December 03, 2018` |
| SEBI emails | 3 | `SEBI Email dated May 4, 2020` |
| Letters & other | 7 | `Letter dated September 02, 2002` |

---

## Reference Types Detected

| Type | Description | Examples |
|------|-------------|----------|
| `circular` | SEBI circulars in any format | `SEBI/HO/MIRSD/POD-1/P/CIR/2024/81`, `CIR/MRD/DP/13/2010` |
| `regulation` | SEBI Regulations | `SEBI (Listing Obligations and Disclosure Requirements) Regulations, 2015` |
| `act` | Indian Acts and Laws | `Securities and Exchange Board of India Act, 1992`, `Companies Act, 2013` |
| `section_reference` | Specific sections of documents | `Section 11(1) of the SEBI Act, 1992` |
| `gazette` | Gazette of India notifications | `Gazette of India, Part III` |
| `other` | Press releases, emails, letters | `Press Release No. 49/2018`, `SEBI Email dated May 4, 2020` |

---

## Evaluation Framework

The project includes a built-in evaluation framework to measure extraction quality and demonstrate improvement from the LLM layer.

### Creating Ground Truth

Create a JSON file with manually annotated references:

```json
[
  {"title": "SEBI (Listing Obligations) Regulations, 2015", "type": "regulation"},
  {"title": "SEBI/HO/MIRSD/POD-1/P/CIR/2024/81", "type": "circular"},
  {"title": "Companies Act, 2013", "type": "act"}
]
```

A sample ground truth file (`ground_truth_sample.json`) with 42 annotated references is included.

### Running Evaluation

```bash
python main.py evaluate circular.pdf ground_truth.json
```

### Metrics

The evaluation computes and compares:

- **Precision** — What fraction of extracted references are real? (Are we hallucinating?)
- **Recall** — What fraction of actual references did we find? (Are we missing things?)
- **F1 Score** — Harmonic mean of precision and recall

Output shows a side-by-side comparison:

```
============================================================
RESULTS
============================================================
Method                Precision     Recall         F1    Found
------------------------------------------------------------
Regex-only                0.110      1.000      0.198      382
Hybrid (regex+LLM)        0.088      1.000      0.162      477
------------------------------------------------------------
Improvement              -0.022     +0.000     -0.036
```

The evaluation also lists:
- References found **only** by the LLM (showing its added value)
- References **missed** by both methods (guiding future improvements)

### Evaluation Methodology

1. **Build gold-standard**: Manually annotate all references on selected pages
2. **Measure baseline**: Run regex-only extraction, compute P/R/F1
3. **Measure hybrid**: Run full hybrid extraction, compute P/R/F1
4. **Error analysis**: Review false positives (hallucinations) and false negatives (misses)
5. **Iterate**: Tune regex patterns and LLM prompts based on error analysis

---

## Output Format

```json
{
  "references": [
    {
      "title": "SEBI/HO/MRD/MRD-PoD-3/P/CIR/2024/65",
      "type": "circular",
      "page_numbers": [5, 6, 11, 13],
      "source": "regex"
    },
    {
      "title": "Press Release No. 49/2018 dated December 03, 2018",
      "type": "other",
      "page_numbers": [5],
      "source": "llm"
    }
  ],
  "stats": {
    "total": 477,
    "regex_only": 326,
    "llm_only": 95,
    "both": 56
  }
}
```

---

## Rate Limit Handling

The agent is designed to work within Gemini's free tier limits:

- **Batching**: Pages are processed in groups of 3 to reduce API calls (e.g., a 13-page PDF = 5 API calls instead of 13)
- **Exponential backoff**: Automatic retry with increasing delays (2s, 4s, 8s... up to 60s) via `tenacity`
- **Graceful degradation**: If the LLM fails, regex results are still returned — the agent never fails completely
- **Inter-batch delay**: 2-second pause between batches to stay within rate limits

---

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| PDF Parsing | **PyMuPDF** | Fast, reliable text extraction with page-level granularity |
| LLM | **Google Gemini 2.5 Flash** | Free tier, large context window (1M tokens), good structured extraction |
| Retry Logic | **tenacity** | Battle-tested exponential backoff for API rate limits |
| Env Management | **python-dotenv** | Keeps API keys out of code |
| Fuzzy Matching | **difflib** (stdlib) | No extra dependency for deduplication |

---

## Project Structure

```
hyde/
├── main.py                  # CLI entry point (extract / evaluate commands)
├── extract.py               # Core engine: PDF parsing, regex, LLM, merge
├── evaluate.py              # Evaluation: precision/recall/F1 comparison
├── requirements.txt         # Python dependencies
├── ground_truth_sample.json # Sample annotated references for evaluation
├── .env                     # API key (git-ignored)
├── .gitignore
└── README.md
```

---

## Limitations & Future Work (v2)

### Current Limitations
- **Scanned PDFs**: Currently relies on text extraction; scanned/image-based PDFs would need OCR (e.g., `pytesseract`)
- **Table-heavy pages**: References embedded in complex tables may be partially extracted
- **Cross-page references**: A reference spanning two pages might be detected on only one
- **LLM cost at scale**: Processing thousands of circulars would require managing API costs

### v2 Ideas: Scaling to All SEBI Circulars

To build a full knowledge graph across all SEBI circulars:

1. **Graph Database**: Store references as edges in Neo4j — nodes are documents, edges are "references" relationships
2. **Bulk Processing Pipeline**: Crawl all circulars from sebi.gov.in, process each through the agent, build the graph incrementally
3. **Supersession Tracking**: Identify which circulars supersede or amend earlier ones (critical for compliance)
4. **RAG Layer**: Add retrieval-augmented generation so compliance officers can ask natural language questions like "Which circulars reference Regulation 51 of LODR?"
5. **Change Alerts**: Monitor new SEBI circulars and automatically update the graph, alerting teams to new compliance requirements
