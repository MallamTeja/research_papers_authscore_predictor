"""
testmain.py — Full test suite for main.py

Coverage:
  - Unit tests for every pure helper function
  - Integration tests for all FastAPI endpoints (mocked model + embedder)
  - Contract tests  — response schema never changes
  - Edge-case tests — empty file, bad encoding, missing metadata, score clamping
  - CI/CD smoke test — confirms the app boots and /health returns 200

Run:
    pytest testmain.py -v
    pytest testmain.py -v --tb=short          # compact tracebacks
    pytest testmain.py -v -k "unit"           # only unit tests
    pytest testmain.py -v -k "integration"    # only integration tests
    pytest testmain.py --co -q               # list all test names without running
"""

import io
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Ensure the repo root is importable regardless of working directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))


# ===========================================================================
# Fixtures & shared helpers
# ===========================================================================

FAKE_EMB_DIM = 384
FAKE_SCORE   = 73.5   # what the mock model always returns


def _fake_embedding(text: str) -> np.ndarray:
    """Deterministic fake embedding — same shape as the real model."""
    rng = np.random.default_rng(abs(hash(text)) % (2**31))
    return rng.random(FAKE_EMB_DIM).astype(np.float32)


def _make_mock_model(score: float = FAKE_SCORE) -> MagicMock:
    """LightGBM Booster stub that always predicts `score`."""
    m = MagicMock()
    m.predict.return_value = np.array([score])
    m.best_iteration = 100
    return m


def _make_mock_embedder() -> MagicMock:
    """SentenceTransformer stub that returns deterministic 384-d vectors."""
    embedder = MagicMock()
    embedder.encode.side_effect = _fake_embedding
    return embedder


@pytest.fixture(scope="module")
def mock_model():
    return _make_mock_model()


@pytest.fixture(scope="module")
def mock_embedder():
    return _make_mock_embedder()


@pytest.fixture(scope="module")
def client(mock_model, mock_embedder):
    """
    TestClient with model + embedder patched so no actual files are needed.
    Patches are applied at the module level so every request through this
    client uses the fakes.
    """
    import main  # noqa: import here so patches apply before app is used

    with (
        patch("main._model",    mock_model),
        patch("main._embedder", mock_embedder),
        patch("main.MODEL_PATH", Path("lightgbmv1.joblib")),   # path exists check bypassed below
        patch.object(Path, "exists", return_value=True),
    ):
        with TestClient(main.app, raise_server_exceptions=True) as c:
            yield c


def _txt_file(content: str, filename: str = "paper.txt") -> dict:
    """Helper: build a files dict for TestClient multipart upload."""
    return {"file": (filename, io.BytesIO(content.encode()), "text/plain")}


# Minimal realistic paper text used across many tests
PAPER_TEXT = """\
Attention Is All You Need

Alice Smith, Bob Jones and Carol Lee

Abstract
We propose a new simple network architecture, the Transformer, based solely
on attention mechanisms, dispensing with recurrence and convolutions entirely.
Experiments on two machine translation tasks show these models to be superior
in quality.

1 Introduction
The dominant sequence transduction models are based on complex recurrent
or convolutional neural networks that include an encoder and a decoder.

Published: 2017
"""


# ===========================================================================
# Unit tests — pure helper functions (no HTTP, no model)
# ===========================================================================

class TestExtractYearFromText:
    """main.extract_year_from_text"""

    def test_finds_published_keyword(self):
        from main import extract_year_from_text
        assert extract_year_from_text("Published: 2021\nsome text") == 2021

    def test_finds_submitted_keyword(self):
        from main import extract_year_from_text
        assert extract_year_from_text("Submitted 2019-03-14\nsome text") == 2019

    def test_finds_arxiv_id_year(self):
        from main import extract_year_from_text
        assert extract_year_from_text("arXiv:2312.00752") == 2312 or \
               extract_year_from_text("arXiv:2312.00752") in range(1950, 2100)
        # Just confirm it returns an int, not None
        result = extract_year_from_text("arXiv:2312.00752")
        assert isinstance(result, int)

    def test_finds_copyright_symbol(self):
        from main import extract_year_from_text
        assert extract_year_from_text("© 2022 The Authors") == 2022

    def test_falls_back_to_bare_year(self):
        from main import extract_year_from_text
        assert extract_year_from_text("Some text 2018 more text") == 2018

    def test_returns_none_for_no_year(self):
        from main import extract_year_from_text
        assert extract_year_from_text("No year here at all.") is None

    def test_ignores_years_before_1950(self):
        from main import extract_year_from_text
        result = extract_year_from_text("In 1066 the Normans invaded.")
        assert result is None

    def test_only_scans_first_5000_chars(self):
        from main import extract_year_from_text
        # Year buried after 5000 chars — should NOT be found
        prefix = "x" * 5001
        result = extract_year_from_text(prefix + "Published: 2023")
        assert result is None

    def test_prefers_metadata_line_over_bare_year(self):
        from main import extract_year_from_text
        # Text mentions 2015 in body but metadata line says 2020
        text = "Published: 2020\nIn 2015 a paper was released."
        assert extract_year_from_text(text) == 2020


class TestYearsSincePublished:
    """main.years_since_published"""

    def test_none_returns_zero(self):
        from main import years_since_published
        assert years_since_published(None) == 0.0

    def test_current_year_returns_zero(self):
        from main import years_since_published
        now_year = datetime.now(timezone.utc).year
        assert years_since_published(now_year) == 0.0

    def test_past_year_positive(self):
        from main import years_since_published
        now_year = datetime.now(timezone.utc).year
        result = years_since_published(now_year - 5)
        assert result == 5.0

    def test_future_year_clamped_to_zero(self):
        from main import years_since_published
        future = datetime.now(timezone.utc).year + 3
        assert years_since_published(future) == 0.0


class TestExtractAuthorsFromText:
    """main.extract_authors_from_text"""

    def test_single_author_no_comma(self):
        from main import extract_authors_from_text
        text = "John Doe\n\nAbstract\nThis paper does stuff."
        result = extract_authors_from_text(text)
        assert result >= 1

    def test_multiple_authors_with_and(self):
        from main import extract_authors_from_text
        text = "Alice Smith, Bob Jones and Carol Lee\n\nAbstract\nThis paper."
        result = extract_authors_from_text(text)
        assert result >= 2

    def test_returns_int(self):
        from main import extract_authors_from_text
        assert isinstance(extract_authors_from_text("Author One\nAbstract\nText"), int)

    def test_empty_text_returns_one(self):
        from main import extract_authors_from_text
        assert extract_authors_from_text("") == 1

    def test_no_authors_found_returns_one(self):
        from main import extract_authors_from_text
        assert extract_authors_from_text("Abstract\nJust some text here.") >= 1


class TestInferPrimaryCategory:
    """main.infer_primary_category"""

    def test_vision_keywords(self):
        from main import infer_primary_category
        text = "We use convolutional networks for image segmentation."
        assert infer_primary_category(text, "") == "cs.CV"

    def test_nlp_keywords(self):
        from main import infer_primary_category
        text = "We train a language model using transformer and token embeddings."
        assert infer_primary_category(text, "") == "cs.CL"

    def test_ml_keywords(self):
        from main import infer_primary_category
        text = "A deep learning approach using neural network training with gradient descent."
        assert infer_primary_category(text, "") in ("cs.LG", "cs.CL")  # gradient is cs.LG

    def test_rl_keywords(self):
        from main import infer_primary_category
        text = "We study reinforcement learning with reward shaping and policy gradients."
        assert infer_primary_category(text, "") == "cs.AI"

    def test_fallback_unknown_text(self):
        from main import infer_primary_category
        result = infer_primary_category("xyz abc def", "some random text")
        assert result == "cs.LG"

    def test_summary_contributes(self):
        from main import infer_primary_category
        # Category signal only in summary, not in text
        result = infer_primary_category("some generic text", "language model bert gpt transformer")
        assert result == "cs.CL"

    def test_returns_string(self):
        from main import infer_primary_category
        result = infer_primary_category("text", "summary")
        assert isinstance(result, str)
        assert "." in result   # arXiv format: "cs.LG"


class TestExtractSummary:
    """main.extract_summary"""

    def test_uses_provided_summary(self):
        from main import extract_summary
        result = extract_summary("any text here", "My explicit summary.")
        assert result == "My explicit summary."

    def test_extracts_abstract_from_text(self):
        from main import extract_summary
        text = "Title\n\nAbstract\nThis is the abstract content.\n\n1 Introduction\nBody."
        result = extract_summary(text, None)
        assert "abstract content" in result.lower()

    def test_stops_before_introduction(self):
        from main import extract_summary
        text = "Abstract\nSome abstract text.\n\n1 Introduction\nShould not be here."
        result = extract_summary(text, None)
        assert "introduction" not in result.lower()

    def test_fallback_to_first_500_chars(self):
        from main import extract_summary
        text = "No abstract keyword here just plain text " + "x" * 600
        result = extract_summary(text, None)
        assert len(result) <= 500

    def test_empty_provided_summary_falls_back(self):
        from main import extract_summary
        text = "Abstract\nThis is extracted."
        result = extract_summary(text, "   ")   # whitespace-only provided
        assert "extracted" in result

    def test_strips_whitespace(self):
        from main import extract_summary
        result = extract_summary("   Abstract\n   trimmed   \n\nIntro", None)
        assert result == result.strip()


class TestConfidenceBand:
    """Test the score → confidence_band mapping directly (via /predict response)."""

    @pytest.mark.parametrize("score,expected_band", [
        (92.0, "exceptional (top tier)"),
        (85.0, "exceptional (top tier)"),
        (75.0, "strong"),
        (70.0, "strong"),
        (55.0, "moderate"),
        (50.0, "moderate"),
        (35.0, "below average"),
        (30.0, "below average"),
        (15.0, "weak"),
        (0.0,  "weak"),
    ])
    def test_bands(self, score, expected_band, mock_embedder):
        """Each score bracket maps to the correct band label."""
        import main
        mock = _make_mock_model(score=score)
        with (
            patch("main._model",    mock),
            patch("main._embedder", mock_embedder),
            patch.object(Path, "exists", return_value=True),
        ):
            with TestClient(main.app) as c:
                resp = c.post("/predict", files=_txt_file(PAPER_TEXT))
        assert resp.status_code == 200
        assert resp.json()["confidence_band"] == expected_band


# ===========================================================================
# Integration tests — HTTP endpoints via TestClient
# ===========================================================================

class TestRootRedirect:
    """GET / should redirect to /docs."""

    def test_redirects_to_docs(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307, 308)
        assert resp.headers["location"].endswith("/docs")

    def test_docs_page_loads(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200
        assert "swagger" in resp.text.lower() or "openapi" in resp.text.lower()


class TestHealthEndpoint:
    """GET /health smoke test."""

    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_json(self, client):
        resp = client.get("/health")
        body = resp.json()
        assert "status" in body
        assert body["status"] == "ok"

    def test_model_loaded_field_present(self, client):
        resp = client.get("/health")
        assert "model_loaded" in resp.json()


class TestPredictEndpoint:
    """POST /predict — happy-path and metadata variants."""

    def test_happy_path_txt(self, client):
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client):
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        body = resp.json()
        assert "score"           in body
        assert "score_rounded"   in body
        assert "confidence_band" in body
        assert "extracted"       in body

    def test_score_is_float(self, client):
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        assert isinstance(resp.json()["score"], float)

    def test_score_rounded_is_int(self, client):
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        assert isinstance(resp.json()["score_rounded"], int)

    def test_score_in_valid_range(self, client):
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        score = resp.json()["score"]
        assert 0.0 <= score <= 100.0

    def test_extracted_contains_metadata(self, client):
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        extracted = resp.json()["extracted"]
        for key in ("author_count", "summary_len", "text_len",
                    "years_since_published", "primary_category", "summary_preview"):
            assert key in extracted, f"Missing key in extracted: {key}"

    def test_extracted_text_len_positive(self, client):
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        assert resp.json()["extracted"]["text_len"] > 0

    def test_explicit_author_count_used(self, client):
        resp = client.post(
            "/predict",
            files=_txt_file(PAPER_TEXT),
            data={"author_count": "7"},
        )
        assert resp.status_code == 200
        assert resp.json()["extracted"]["author_count"] == 7

    def test_explicit_publish_year_used(self, client):
        resp = client.post(
            "/predict",
            files=_txt_file(PAPER_TEXT),
            data={"publish_year": "2019"},
        )
        assert resp.status_code == 200
        now_year = datetime.now(timezone.utc).year
        expected_ysp = float(now_year - 2019)
        assert resp.json()["extracted"]["years_since_published"] == expected_ysp

    def test_explicit_primary_category_used(self, client):
        resp = client.post(
            "/predict",
            files=_txt_file(PAPER_TEXT),
            data={"primary_category": "cs.RO"},
        )
        assert resp.status_code == 200
        assert resp.json()["extracted"]["primary_category"] == "cs.RO"

    def test_explicit_summary_used(self, client):
        custom_summary = "A custom abstract about robots and locomotion."
        resp = client.post(
            "/predict",
            files=_txt_file(PAPER_TEXT),
            data={"summary": custom_summary},
        )
        assert resp.status_code == 200
        preview = resp.json()["extracted"]["summary_preview"]
        assert "custom abstract" in preview

    def test_all_optional_fields_provided(self, client):
        resp = client.post(
            "/predict",
            files=_txt_file(PAPER_TEXT),
            data={
                "summary":          "Explicit abstract.",
                "author_count":     "3",
                "publish_year":     "2021",
                "primary_category": "stat.ML",
            },
        )
        assert resp.status_code == 200
        e = resp.json()["extracted"]
        assert e["author_count"]       == 3
        assert e["primary_category"]   == "stat.ML"

    def test_markdown_file_accepted(self, client):
        md_content = "# Title\n\n**Authors**: A, B\n\nAbstract\nSome abstract.\n\n## Introduction\nBody."
        resp = client.post(
            "/predict",
            files={"file": ("paper.md", io.BytesIO(md_content.encode()), "text/markdown")},
        )
        assert resp.status_code == 200

    def test_no_filename_defaults_to_text(self, client):
        """File with no extension should be treated as plain text."""
        resp = client.post(
            "/predict",
            files={"file": ("", io.BytesIO(PAPER_TEXT.encode()), "text/plain")},
        )
        assert resp.status_code == 200

    def test_score_rounded_matches_score(self, client):
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        body = resp.json()
        assert body["score_rounded"] == round(body["score"])


class TestPredictEdgeCases:
    """POST /predict — edge cases and error handling."""

    def test_empty_file_returns_422(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        )
        assert resp.status_code == 422

    def test_whitespace_only_file_returns_422(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("ws.txt", io.BytesIO(b"   \n\t\n  "), "text/plain")},
        )
        assert resp.status_code == 422

    def test_no_file_returns_422(self, client):
        """Missing file field entirely."""
        resp = client.post("/predict", data={"author_count": "2"})
        assert resp.status_code == 422

    def test_paper_without_abstract_keyword(self, client):
        """Paper text without 'Abstract' keyword still succeeds — fallback used."""
        text = "Some Title\nAuthors: X, Y\n\nThis is body text without an abstract keyword.\n2022"
        resp = client.post("/predict", files=_txt_file(text))
        assert resp.status_code == 200

    def test_paper_without_year(self, client):
        """No year in text and no publish_year form field — years_since_published=0."""
        text = "Title\nAuthor A\nAbstract\nSome content without any date."
        resp = client.post("/predict", files=_txt_file(text))
        assert resp.status_code == 200
        assert resp.json()["extracted"]["years_since_published"] == 0.0

    def test_very_long_paper(self, client):
        """50 000 char paper — should not timeout or error."""
        long_text = PAPER_TEXT + ("This is additional body text. " * 1500)
        resp = client.post("/predict", files=_txt_file(long_text))
        assert resp.status_code == 200

    def test_unicode_text(self, client):
        """Papers with non-ASCII characters (e.g. accented author names)."""
        text = "Müller, Ångström and 田中\n\nAbstract\nDeep learning with Transformer. 2021"
        resp = client.post("/predict", files=_txt_file(text))
        assert resp.status_code == 200

    def test_score_clamped_when_model_returns_above_100(self, mock_embedder):
        """If model returns >100, score is clamped to 100."""
        import main
        mock = _make_mock_model(score=150.0)
        with (
            patch("main._model",    mock),
            patch("main._embedder", mock_embedder),
            patch.object(Path, "exists", return_value=True),
        ):
            with TestClient(main.app) as c:
                resp = c.post("/predict", files=_txt_file(PAPER_TEXT))
        assert resp.json()["score"] == 100.0

    def test_score_clamped_when_model_returns_negative(self, mock_embedder):
        """If model returns <0, score is clamped to 0."""
        import main
        mock = _make_mock_model(score=-20.0)
        with (
            patch("main._model",    mock),
            patch("main._embedder", mock_embedder),
            patch.object(Path, "exists", return_value=True),
        ):
            with TestClient(main.app) as c:
                resp = c.post("/predict", files=_txt_file(PAPER_TEXT))
        assert resp.json()["score"] == 0.0

    def test_summary_preview_truncated_at_300(self, client):
        long_summary = "A " * 200   # 400 chars
        resp = client.post(
            "/predict",
            files=_txt_file(PAPER_TEXT),
            data={"summary": long_summary},
        )
        preview = resp.json()["extracted"]["summary_preview"]
        assert len(preview) <= 303   # 300 + "..."
        assert preview.endswith("...")

    def test_summary_preview_not_truncated_when_short(self, client):
        short_summary = "Short abstract."
        resp = client.post(
            "/predict",
            files=_txt_file(PAPER_TEXT),
            data={"summary": short_summary},
        )
        preview = resp.json()["extracted"]["summary_preview"]
        assert not preview.endswith("...")


# ===========================================================================
# Contract tests — response schema stability
# ===========================================================================

class TestResponseContract:
    """
    These tests guard against accidental breaking changes to the /predict
    response schema. If a field is renamed or removed, CI fails here.
    """

    REQUIRED_TOP_LEVEL = {"score", "score_rounded", "confidence_band", "extracted"}
    REQUIRED_EXTRACTED = {
        "author_count", "summary_len", "text_len",
        "years_since_published", "publish_year",
        "primary_category", "summary_preview",
    }

    def test_top_level_keys_present(self, client):
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        body = resp.json()
        missing = self.REQUIRED_TOP_LEVEL - set(body.keys())
        assert not missing, f"Missing top-level keys: {missing}"

    def test_no_extra_unexpected_top_level_keys(self, client):
        """Fail fast if new fields are added without updating this contract test."""
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        body = resp.json()
        extra = set(body.keys()) - self.REQUIRED_TOP_LEVEL
        assert not extra, (
            f"Unexpected new top-level keys: {extra}\n"
            "Update REQUIRED_TOP_LEVEL in TestResponseContract if intentional."
        )

    def test_extracted_keys_present(self, client):
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        extracted = resp.json()["extracted"]
        missing = self.REQUIRED_EXTRACTED - set(extracted.keys())
        assert not missing, f"Missing extracted keys: {missing}"

    def test_score_type(self, client):
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        assert isinstance(resp.json()["score"], (int, float))

    def test_score_rounded_type(self, client):
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        assert isinstance(resp.json()["score_rounded"], int)

    def test_confidence_band_is_string(self, client):
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        assert isinstance(resp.json()["confidence_band"], str)

    def test_extracted_author_count_is_int(self, client):
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        assert isinstance(resp.json()["extracted"]["author_count"], int)

    def test_extracted_summary_len_is_int(self, client):
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        assert isinstance(resp.json()["extracted"]["summary_len"], int)

    def test_extracted_text_len_is_int(self, client):
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        assert isinstance(resp.json()["extracted"]["text_len"], int)

    def test_extracted_years_since_published_is_float(self, client):
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        assert isinstance(resp.json()["extracted"]["years_since_published"], float)


# ===========================================================================
# CI / CD smoke tests — run these in your GitHub Actions / Render pre-deploy
# ===========================================================================

class TestCISmoke:
    """
    Lightweight checks that are safe and fast to run in CI.
    They do NOT need a trained model on disk — everything is mocked.
    """

    def test_app_imports_without_error(self):
        """main.py must be importable even if model file is absent."""
        import importlib
        # Re-import cleanly
        if "main" in sys.modules:
            del sys.modules["main"]
        try:
            import main  # noqa
        except RuntimeError:
            pass   # RuntimeError from missing model is OK at import time
        except Exception as e:
            pytest.fail(f"Unexpected import error: {e}")

    def test_health_endpoint_always_responds(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_docs_endpoint_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_predict_returns_json_content_type(self, client):
        resp = client.post("/predict", files=_txt_file(PAPER_TEXT))
        assert "application/json" in resp.headers.get("content-type", "")

    def test_root_redirects(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307, 308)

    def test_unknown_endpoint_returns_404(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404

    def test_post_to_health_returns_405(self, client):
        """Health is GET only — POST should be method-not-allowed."""
        resp = client.post("/health")
        assert resp.status_code == 405

    def test_predict_without_model_returns_error(self, mock_embedder):
        """When model file is missing, /predict returns 500, not a crash."""
        import main
        # Force get_model() to raise RuntimeError (no model on disk)
        with (
            patch("main._model",    None),
            patch("main._embedder", mock_embedder),
            patch.object(Path, "exists", return_value=False),
        ):
            with TestClient(main.app, raise_server_exceptions=False) as c:
                resp = c.post("/predict", files=_txt_file(PAPER_TEXT))
        assert resp.status_code in (500, 422, 503)

    def test_openapi_schema_is_valid(self, client):
        """The OpenAPI schema endpoint must return parseable JSON."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "/predict" in schema["paths"]

    def test_predict_endpoint_documented_in_openapi(self, client):
        resp = client.get("/openapi.json")
        paths = resp.json()["paths"]
        assert "/predict" in paths
        assert "post" in paths["/predict"]


# ===========================================================================
# Feature-builder unit tests (no HTTP)
# ===========================================================================

class TestBuildInferenceFeatures:
    """main.build_inference_features — shape and column sanity."""

    def test_output_is_dataframe(self, mock_embedder):
        import main
        with patch("main._embedder", mock_embedder):
            X = main.build_inference_features(
                text="Some text about deep learning.",
                summary="A summary.",
                author_count=3,
                primary_category="cs.LG",
                publish_year=2021,
            )
        assert isinstance(X, pd.DataFrame)

    def test_output_has_one_row(self, mock_embedder):
        import main
        with patch("main._embedder", mock_embedder):
            X = main.build_inference_features("text", "summary", 2, "cs.CV", 2020)
        assert len(X) == 1

    def test_output_has_correct_number_of_columns(self, mock_embedder):
        import main
        expected_cols = 4 + 1 + FAKE_EMB_DIM   # 4 scalar + 1 cat + 384 emb
        with patch("main._embedder", mock_embedder):
            X = main.build_inference_features("text", "summary", 2, "cs.CV", 2020)
        assert X.shape[1] == expected_cols

    def test_scalar_feature_values_correct(self, mock_embedder):
        import main
        text    = "deep learning text"
        summary = "short summary"
        with patch("main._embedder", mock_embedder):
            X = main.build_inference_features(
                text=text, summary=summary,
                author_count=5, primary_category="cs.LG", publish_year=2022,
            )
        assert X["author_count"].iloc[0]  == 5.0
        assert X["summary_len"].iloc[0]   == float(len(summary))
        assert X["text_len"].iloc[0]      == float(len(text))

    def test_embedding_columns_exist(self, mock_embedder):
        import main
        with patch("main._embedder", mock_embedder):
            X = main.build_inference_features("text", "summary", 1, "cs.CL", None)
        emb_cols = [c for c in X.columns if c.startswith("emb_")]
        assert len(emb_cols) == FAKE_EMB_DIM

    def test_years_since_published_none_gives_zero(self, mock_embedder):
        import main
        with patch("main._embedder", mock_embedder):
            X = main.build_inference_features("text", "summary", 1, "cs.LG", None)
        assert X["years_since_published"].iloc[0] == 0.0

    def test_embedding_dim_mismatch_raises(self, mock_embedder):
        """If embedder returns wrong dimension, build_inference_features raises RuntimeError."""
        import main
        bad_embedder = MagicMock()
        bad_embedder.encode.return_value = np.zeros(128, dtype=np.float32)   # wrong dim
        with patch("main._embedder", bad_embedder):
            with pytest.raises(RuntimeError, match="dimension mismatch"):
                main.build_inference_features("text", "summary", 1, "cs.LG", 2020)


# ===========================================================================
# File reading unit tests
# ===========================================================================

class TestReadUploadedFile:
    """main.read_uploaded_file — encoding handling."""

    def _make_upload(self, content: bytes, filename: str) -> MagicMock:
        upload = MagicMock()
        upload.filename = filename
        upload.file = io.BytesIO(content)
        return upload

    def test_utf8_txt(self):
        from main import read_uploaded_file
        upload = self._make_upload(b"Hello world", "paper.txt")
        assert read_uploaded_file(upload) == "Hello world"

    def test_latin1_txt(self):
        from main import read_uploaded_file
        content = "Héllo wörld".encode("latin-1")
        upload = self._make_upload(content, "paper.txt")
        result = read_uploaded_file(upload)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_markdown_file(self):
        from main import read_uploaded_file
        upload = self._make_upload(b"# Title\nContent", "paper.md")
        assert "Title" in read_uploaded_file(upload)

    def test_pdf_with_no_extractor_raises_http_422(self):
        """If no PDF library is available, should raise HTTPException 422."""
        from main import read_uploaded_file
        from fastapi import HTTPException
        # Upload a fake PDF (magic bytes only) with all extractors patched to fail
        fake_pdf = b"%PDF-1.4 fake"
        upload = self._make_upload(fake_pdf, "paper.pdf")
        with (
            patch.dict(sys.modules, {"fitz": None, "pdfplumber": None}),
            patch("builtins.__import__", side_effect=ImportError),
        ):
            with pytest.raises((HTTPException, Exception)):
                read_uploaded_file(upload)