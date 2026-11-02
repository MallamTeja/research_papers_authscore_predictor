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
import time
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

# ONNX runtime replaces sentence-transformers: torch alone blew Render's 512 MB free-tier RAM
try:
    # onnxruntime runs the same MiniLM weights without importing torch (~10x smaller in RAM)
    import onnxruntime as ort
    # tokenizers is the Rust tokenizer sentence-transformers used internally, so token ids match
    from tokenizers import Tokenizer
    # hf_hub_download fetches single files, so we skip the repo's PyTorch/TF weight copies
    from huggingface_hub import hf_hub_download
except ImportError:
    # Sentinel lets get_embedder() raise a clear error instead of a NameError mid-request
    ort = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

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
# Same HF repo emb.py trained on, so inference embeddings live in the training vector space
EMBED_REPO  = "sentence-transformers/all-MiniLM-L6-v2"
# Repo-relative dir: Render only carries /opt/render/project/src from build to runtime, not ~/.cache
EMBED_DIR   = Path("model_cache/all-MiniLM-L6-v2")
# sentence-transformers' max_seq_length for this model; tokenizer.json alone says 128, which would drift
MAX_SEQ_LEN = 256

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
        print(f"Loading model from {MODEL_PATH}...")
        _model = joblib.load(MODEL_PATH)
    return _model


class OnnxMiniLM:
    """Drop-in for SentenceTransformer('all-MiniLM-L6-v2').encode(), minus torch."""

    def __init__(self):
        # Download (or reuse) the fp32 ONNX export; fp32 keeps parity with the torch training embeddings
        model_file = hf_hub_download(EMBED_REPO, "onnx/model.onnx", local_dir=EMBED_DIR)
        # The tokenizer file from the same repo guarantees identical vocab and lowercasing
        tok_file = hf_hub_download(EMBED_REPO, "tokenizer.json", local_dir=EMBED_DIR)
        # Load the fast Rust tokenizer from the downloaded file
        self.tok = Tokenizer.from_file(tok_file)
        # Override the file's 128-token truncation with the 256 sentence-transformers actually used
        self.tok.enable_truncation(max_length=MAX_SEQ_LEN)
        # One text per call, so padding would only add masked-out work
        self.tok.no_padding()
        # Session options let us cap threads for Render's 0.1-CPU container
        opts = ort.SessionOptions()
        # One thread: the container sees host CPUs it can't use, and extra threads cost RAM and contention
        opts.intra_op_num_threads = 1
        # CPU provider only: Render free tier has no GPU, and naming it avoids provider probing
        self.sess = ort.InferenceSession(model_file, sess_options=opts, providers=["CPUExecutionProvider"])
        # Some ONNX exports drop token_type_ids; feed only the inputs this graph declares
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def encode(self, text: str) -> np.ndarray:
        # Tokenize exactly as sentence-transformers would: [CLS] ... [SEP], truncated to 256
        enc = self.tok.encode(text)
        # ONNX BERT graphs expect int64 tensors shaped (batch=1, seq_len)
        feeds = {
            # Token ids select rows of the embedding table
            "input_ids": np.array([enc.ids], dtype=np.int64),
            # Mask marks real tokens; all ones here since there is no padding
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            # Segment ids are all zero for single-sentence input
            "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
        }
        # Drop any input the graph doesn't declare, or onnxruntime rejects the call
        feeds = {k: v for k, v in feeds.items() if k in self.input_names}
        # last_hidden_state is (1, seq_len, 384); take batch row 0 to get per-token vectors
        tokens = self.sess.run(["last_hidden_state"], feeds)[0][0]
        # Column-shaped mask so it broadcasts across the 384 dims
        mask = feeds["attention_mask"][0][:, None].astype(np.float32)
        # Masked mean pooling, the same Pooling layer sentence-transformers applies for this model
        pooled = (tokens * mask).sum(axis=0) / max(float(mask.sum()), 1e-9)
        # L2-normalize, matching the model's Normalize layer, so vectors are unit length like training
        return (pooled / max(float(np.linalg.norm(pooled)), 1e-12)).astype(np.float32)


def get_embedder():
    global _embedder
    if _embedder is None:
        # Fail with an actionable message if req.txt wasn't installed, not a NameError
        if ort is None:
            raise RuntimeError("onnxruntime / tokenizers / huggingface_hub not installed (see req.txt).")
        # Log the load so Render logs show whether first-request latency came from this step
        print(f"Loading ONNX embedder '{EMBED_REPO}'...")
        # Cache the instance: the ONNX session is the expensive part, built once per process
        _embedder = OnnxMiniLM()
    return _embedder


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly load only the fast, local model on startup."""
    print("--- Startup: Loading LightGBM model ---")
    try:
        get_model()  # Fast, local file
        print("--- Startup: LightGBM model loaded successfully ---")
    except Exception as e:
        print(f"--- Startup ERROR: Could not load LightGBM model: {e} ---")
        # In production, you might want to raise here to prevent serving traffic without a model:
        # raise
    yield
    print("--- Shutdown: Cleaning up ---")


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

    # Remove potential noise like emails, affiliations, or institutions
    # (Simplified for optimal performance without complex NLP)
    header = re.sub(r'[\w\.-]+@[\w\.-]+', '', header)  # remove emails
    
    # Look for name-like structures: "First Last" or "Last, F."
    # Count separators: comma (if not part of initials) and "and"
    and_count = len(re.findall(r"\band\b", header, re.I))
    
    # To avoid counting commas in "Smith, A.", we only count commas that are followed
    # by a space and then a word starting with uppercase, or a comma that doesn't have a period nearby.
    comma_count = len(re.findall(r",\s+(?=[A-Z])", header))
    if comma_count == 0 and "," in header:
        # Fallback if no uppercase follows: just count commas but be conservative
        comma_count = header.count(",")
    
    # Optimal count for lists like "A, B and C"
    return max(1, and_count + comma_count + 1)


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
        except (UnicodeDecodeError, LookupError):
            continue
    raise HTTPException(status_code=400, detail="Could not decode uploaded file as text.")


def _extract_pdf_text(raw_bytes: bytes) -> str:
    """Try PyMuPDF first, fall back to pdfplumber."""
    if fitz:
        try:
            doc  = fitz.open(stream=raw_bytes, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            if text.strip():
                return text
        except Exception:
            pass

    if pdfplumber:
        try:
            import io
            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                text = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
            if text.strip():
                return text
        except Exception:
            pass

    raise HTTPException(
        status_code=422,
        detail=(
            "Could not extract text from PDF. "
            "Ensure PyMuPDF or pdfplumber is installed."
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
    lifespan=lifespan,
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
):
    # 1. Basic File Validation
    t0 = time.perf_counter()
    filename = (file.filename or "").lower()
    allowed_extensions = {".pdf", ".txt", ".md"}
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported extensions: {', '.join(allowed_extensions)}"
        )

    # 2. Read file → full text
    text = read_uploaded_file(file)
    clean_text = text.strip()
    t_file_read = time.perf_counter()
    
    # 3. Content Validation (Edge Case: empty or near-empty files)
    if len(clean_text) < 200:
        raise HTTPException(
            status_code=422, 
            detail="File content is too short to be a valid research paper (minimum 200 characters)."
        )

    # 4. Auto-Extraction Strategy (Optimal Solution)
    # We ignore any manual inputs and derive everything from the content itself.
    resolved_summary  = extract_summary(clean_text, None)
    resolved_authors  = extract_authors_from_text(clean_text)
    resolved_year     = extract_year_from_text(clean_text)
    resolved_category = infer_primary_category(clean_text, resolved_summary)
    t_extraction = time.perf_counter()

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
    t_features = time.perf_counter()

    # 4. Predict
    model = get_model()
    try:
        raw_score = float(model.predict(X, num_iteration=model.best_iteration)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")
    t_predict = time.perf_counter()

    # Log timing
    print(f"\n--- Inference Timing for {filename} ---")
    print(f"File Read/PDF Extraction: {(t_file_read - t0):.4f}s")
    print(f"Metadata Extraction:      {(t_extraction - t_file_read):.4f}s")
    print(f"Feature Building (Emb):   {(t_features - t_extraction):.4f}s")
    print(f"Model Prediction:         {(t_predict - t_features):.4f}s")
    print(f"Total Inference Time:     {(t_predict - t0):.4f}s")
    print("------------------------------------------\n")

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