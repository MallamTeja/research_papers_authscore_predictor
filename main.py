"""
main.py — FastAPI inference server for paper quality scoring.

Endpoints:
  GET  /          →  redirect to /docs  (Swagger UI)
  POST /predict   →  score a paper from its text fields

Deployment: Render (render.yml points here)
Run locally:
    uvicorn main:app --reload --port 8000
"""

import io
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("lightgbmv1.joblib")

# These MUST match the column order used in lbg1.py exactly
SCALAR_FEATURES = [
    "author_count",
    "summary_len",
    "text_len",
    "years_since_published",
]
CAT_FEATURE = "primary_category"
EMB_DIM     = 384   # all-MiniLM-L6-v2 output dimension

# Lazy-loaded globals
_model: Optional[lgb.Booster]         = None
_embedder                              = None   # SentenceTransformer


# ── Model / embedder loading ────────────────────────────────────────────────────
def get_model() -> lgb.Booster:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model file '{MODEL_PATH}' not found. "
                "Run lbg1.py to train and save the model first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ── Paper parsing helpers ───────────────────────────────────────────────────────
_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")


def extract_year_from_text(text: str) -> Optional[int]:
    """
    Try to find a 4-digit publication year in the text.
    Looks for patterns like 'Published: 2023', 'arXiv:2312.', '© 2022', etc.
    Returns the most likely publication year or None.
    """
    # Prioritise lines that look like metadata
    priority_patterns = [
        re.compile(r"(?:published|submitted|received|accepted|date)[^\n]*?(\b20[0-2]\d\b|\b19[5-9]\d\b)", re.I),
        re.compile(r"arxiv[:\s]*(\d{4})\.\d+", re.I),
        re.compile(r"©\s*(20[0-2]\d|19[5-9]\d)"),
        re.compile(r"\b(20[0-2]\d|19[5-9]\d)\b"),
    ]
    for pat in priority_patterns:
        m = pat.search(text[:5000])   # only scan first 5 000 chars — metadata lives there
        if m:
            return int(m.group(1))
    return None


def years_since_published(year: Optional[int]) -> float:
    if year is None:
        return 0.0
    now_year = datetime.now(timezone.utc).year
    delta = now_year - year
    return max(float(delta), 0.0)


def extract_authors_from_text(text: str) -> int:
    """
    Rough heuristic: look for an author line near the top of the paper.
    Returns author count; defaults to 1 if nothing found.
    """
    # Common PDF-to-text patterns: "Author1, Author2 and Author3"
    header = text[:3000]
    # Look for comma/and separated name-like tokens before the abstract
    abstract_pos = header.lower().find("abstract")
    if abstract_pos == -1:
        abstract_pos = 1500
    header = header[:abstract_pos]

    # Count 'and' occurrences as separators between author names
    and_count = len(re.findall(r"\band\b", header, re.I))
    comma_count = header.count(",")
    # Heuristic: each 'and' ≈ separator between last two; commas separate others
    if and_count > 0 or comma_count > 0:
        return max(1, and_count + 1 + max(0, comma_count - and_count * 2))
    return 1


def infer_primary_category(text: str, summary: str) -> str:
    """
    Lightweight keyword-based category inference for papers without metadata.
    Returns an arXiv-style category string.
    """
    combined = (summary + " " + text[:2000]).lower()

    rules = [
        ("cs.CV",   ["image", "vision", "object detection", "segmentation", "convolutional", "pixel"]),
        ("cs.CL",   ["language model", "nlp", "text generation", "translation", "bert", "gpt", "transformer", "token"]),
        ("cs.LG",   ["machine learning", "deep learning", "neural network", "training", "gradient", "loss function"]),
        ("stat.ML", ["bayesian", "gaussian process", "variational inference", "latent", "probabilistic"]),
        ("cs.AI",   ["reinforcement learning", "reward", "agent", "policy", "planning"]),
        ("cs.RO",   ["robot", "locomotion", "manipulation", "control"]),
    ]
    scores = {}
    for cat, keywords in rules:
        scores[cat] = sum(1 for kw in keywords if kw in combined)

    best_cat = max(scores, key=scores.get)
    if scores[best_cat] == 0:
        return "cs.LG"   # default fallback
    return best_cat


def extract_summary(text: str, provided_summary: Optional[str]) -> str:
    """
    Use provided summary if given; otherwise extract abstract from text.
    """
    if provided_summary and provided_summary.strip():
        return provided_summary.strip()

    # Try to extract abstract from raw text
    lower = text.lower()
    abs_start = lower.find("abstract")
    if abs_start != -1:
        snippet = text[abs_start + 8 : abs_start + 2000]
        # Stop at introduction or first double newline
        for sentinel in ["\n\n", "\r\n\r\n", "1 introduction", "1. introduction"]:
            pos = snippet.lower().find(sentinel)
            if pos != -1:
                snippet = snippet[:pos]
        
        if len(snippet) > 500:
            snippet = snippet[:500]
        return snippet.strip()

    # Fallback: first 500 chars
    return text[:500].strip()


def read_uploaded_file(upload: UploadFile) -> str:
    """
    Read text from an uploaded file. Supports:
    - .txt, .md  → decode as UTF-8
    - .pdf       → extract with PyMuPDF (fitz) or pdfplumber
    """
    filename = (upload.filename or "").lower()
    raw_bytes = upload.file.read()

    if filename and filename.endswith(".pdf"):
        return _extract_pdf_text(raw_bytes)

    # Assume plain text / markdown
    for enc in ("utf-8", "latin-1", "utf-16"):
        try:
            return raw_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    raise HTTPException(status_code=400, detail="Could not decode uploaded file as text.")


def _extract_pdf_text(raw_bytes: bytes) -> str:
    """Try PyMuPDF first, fall back to pdfplumber."""
    try:
        import fitz  # PyMuPDF
        doc  = fitz.open(stream=raw_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        if text.strip():
            return text
    except Exception:
        pass

    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
        if text.strip():
            return text
    except Exception:
        pass

    try:
        from arxiv2text import arxiv_to_text  # type: ignore
        # arxiv2text needs a URL; not usable here directly
        pass
    except Exception:
        pass

    raise HTTPException(
        status_code=422,
        detail=(
            "Could not extract text from PDF. "
            "Install PyMuPDF (`pip install PyMuPDF`) or pdfplumber (`pip install pdfplumber`)."
        ),
    )


# ── Feature builder (mirrors lbg1.py) ─────────────────────────────────────────
def build_inference_features(
    text: str,
    summary: str,
    author_count: int,
    primary_category: str,
    publish_year: Optional[int],
) -> pd.DataFrame:
    """
    Build a single-row DataFrame with the exact same columns as lbg1.py training.
    """
    ysp    = years_since_published(publish_year)
    # ── Request timeout/memory guard: truncate input to 5000 chars for embedding ──
    trunc_text = text[:5000]
    emb    = get_embedder().encode(trunc_text).astype(np.float32)

    if len(emb) != EMB_DIM:
        raise RuntimeError(
            f"Embedding dimension mismatch: expected {EMB_DIM}, got {len(emb)}. "
            "Make sure the same SentenceTransformer model is used as in emb.py."
        )

    row = {
        "author_count":        float(author_count),
        "summary_len":         float(len(summary)),
        "text_len":            float(len(text)),
        "years_since_published": ysp,
        CAT_FEATURE:           pd.Categorical([primary_category]),
    }

    scalar_df = pd.DataFrame([{k: v for k, v in row.items() if k != CAT_FEATURE}])
    cat_df    = pd.DataFrame({CAT_FEATURE: pd.Categorical([primary_category])})
    emb_df    = pd.DataFrame([emb], columns=[f"emb_{i}" for i in range(EMB_DIM)])

    X = pd.concat(
        [scalar_df.reset_index(drop=True),
         cat_df.reset_index(drop=True),
         emb_df.reset_index(drop=True)],
        axis=1,
    )
    return X


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Research Papers Authenticity Score Predictor",
    description=(
        "Upload an AI/ML research paper (PDF or plain text) and get a predicted "
        "quality score (0–100) based on authencity, novelty, rigor, and impact — powered by "
        "a LightGBM model trained on hundreds of papers.\n\n"
        "**Workflow**: `POST /predict` → JSON with `score`, `confidence_band`, and extracted metadata."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)





@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to Swagger UI."""
    return RedirectResponse(url="/docs")


# ── Response schema ─────────────────────────────────────────────────────────────
class PredictResponse(BaseModel):
    score: float
    score_rounded: int
    confidence_band: str
    extracted: dict


# ── /predict endpoint ──────────────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Score a research paper",
    description=(
        "Upload a research paper file (PDF, .txt, or .md) with optional metadata fields. "
        "The API extracts features automatically from the text, runs the LightGBM model, "
        "and returns a predicted quality score (0–100).\n\n"
        "**Optional fields** (if not provided, extracted from text heuristically):\n"
        "- `summary`: abstract / summary text\n"
        "- `author_count`: number of authors\n"
        "- `publish_year`: 4-digit year (e.g. 2023)\n"
        "- `primary_category`: arXiv category (e.g. cs.LG, cs.CV)\n"
    ),
)
async def predict(
    file: UploadFile = File(
        ...,
        description="Paper file: PDF, plain text (.txt), or Markdown (.md)",
    ),
    summary: Optional[str] = Form(
        default=None,
        description="Abstract / summary (optional — extracted from text if absent)",
    ),
    author_count: Optional[int] = Form(
        default=None,
        description="Number of authors (optional — heuristically inferred if absent)",
    ),
    publish_year: Optional[int] = Form(
        default=None,
        description="Publication year e.g. 2023 (optional — scanned from text if absent)",
    ),
    primary_category: Optional[str] = Form(
        default=None,
        description="arXiv category e.g. cs.LG (optional — inferred from content if absent)",
    ),
):
    # 1. Read file → full text
    text = read_uploaded_file(file)
    if not text.strip():
        raise HTTPException(status_code=422, detail="Uploaded file produced no readable text.")

    # 2. Fill in missing metadata from text
    resolved_summary  = extract_summary(text, summary)
    resolved_authors  = author_count if author_count is not None else extract_authors_from_text(text)
    resolved_year     = publish_year if publish_year is not None else extract_year_from_text(text)
    resolved_category = primary_category if primary_category else infer_primary_category(text, resolved_summary)

    # 3. Build feature matrix
    try:
        X = build_inference_features(
            text              = text,
            summary           = resolved_summary,
            author_count      = resolved_authors,
            primary_category  = resolved_category,
            publish_year      = resolved_year,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")

    # 4. Predict
    model = get_model()
    try:
        raw_score = float(model.predict(X, num_iteration=model.best_iteration)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # Clamp to valid range
    score = float(np.clip(raw_score, 0.0, 100.0))

    # 5. Confidence band
    if score >= 85:
        band = "exceptional (top tier)"
    elif score >= 70:
        band = "strong"
    elif score >= 50:
        band = "moderate"
    elif score >= 30:
        band = "below average"
    else:
        band = "weak"

    return PredictResponse(
        score         = round(score, 2),
        score_rounded = int(round(score)),
        confidence_band = band,
        extracted = {
            "author_count":          resolved_authors,
            "summary_len":           len(resolved_summary),
            "text_len":              len(text),
            "years_since_published": years_since_published(resolved_year),
            "publish_year":          resolved_year,
            "primary_category":      resolved_category,
            "summary_preview":       resolved_summary[:300] + ("..." if len(resolved_summary) > 300 else ""),
        },
    )


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health", include_in_schema=False)
async def health():
    try:
        # Harden check: ensure model can actually be loaded/exists
        model = get_model()
        model_ok = True
    except Exception:
        model_ok = False
    return JSONResponse({"status": "ok" if model_ok else "degraded", "model_loaded": model_ok})