"""Integration tests -- require OPENAI_API_KEY to be set.

Run with: pytest tests/test_feedback_integration.py -v

These tests make real API calls. Skip them in CI or when no key is available.
"""

import os
import pytest
from app.feedback import get_feedback, _cache
from app.models import FeedbackRequest, VALID_ERROR_TYPES, VALID_DIFFICULTIES
 
pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping integration tests",
)
 
 
@pytest.fixture(autouse=True)
def clear_cache():
    _cache.clear()
    yield
    _cache.clear()
 
 
def _assert_valid_response(result, expect_correct: bool | None = None):
    """Shared assertions for every response."""
    assert result.difficulty in VALID_DIFFICULTIES
    for err in result.errors:
        assert err.error_type in VALID_ERROR_TYPES
        assert len(err.explanation) > 0
        assert len(err.original) > 0
 
    if expect_correct is True:
        assert result.is_correct is True
        assert result.errors == []
    elif expect_correct is False:
        assert result.is_correct is False
        assert len(result.errors) >= 1
 
 
# ---------- 1. Spanish conjugation error ----------
 
@pytest.mark.asyncio
async def test_spanish_conjugation_error():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    # Should correct to something with "fui"
    assert "fui" in result.corrected_sentence.lower()
 
 
# ---------- 2. Correct German sentence ----------
 
@pytest.mark.asyncio
async def test_correct_german_sentence():
    original = "Ich habe gestern einen interessanten Film gesehen."
    result = await get_feedback(
        FeedbackRequest(
            sentence=original,
            target_language="German",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=True)
    assert result.corrected_sentence == original
 
 
# ---------- 3. French gender agreement (multiple errors) ----------
 
@pytest.mark.asyncio
async def test_french_gender_agreement():
    result = await get_feedback(
        FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    assert len(result.errors) >= 2
    assert "Le chat" in result.corrected_sentence or "le chat" in result.corrected_sentence
 
 
# ---------- 4. Japanese particle error (non-Latin script) ----------
 
@pytest.mark.asyncio
async def test_japanese_particle_error():
    result = await get_feedback(
        FeedbackRequest(
            sentence="私は東京を住んでいます。",
            target_language="Japanese",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    assert "に" in result.corrected_sentence
 
 
# ---------- 5. Korean spelling error (non-Latin script) ----------
 
@pytest.mark.asyncio
async def test_korean_spelling_error():
    result = await get_feedback(
        FeedbackRequest(
            sentence="저는 한국어를 배우고 싶습니데.",
            target_language="Korean",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    # "싶습니데" should be corrected to "싶습니다"
    assert "싶습니다" in result.corrected_sentence
 
 
# ---------- 6. Russian case error (non-Latin script) ----------
 
@pytest.mark.asyncio
async def test_russian_case_error():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Я читаю интересная книга.",
            target_language="Russian",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    # "интересная" should become "интересную" (accusative)
    assert "интересную" in result.corrected_sentence
 
 
# ---------- 7. Chinese word choice error ----------
 
@pytest.mark.asyncio
async def test_chinese_word_choice():
    result = await get_feedback(
        FeedbackRequest(
            sentence="我很高兴看你。",
            target_language="Chinese",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    # "看" should be "见" — 很高兴见到你
    assert "见" in result.corrected_sentence
 
 
# ---------- 8. Portuguese spelling + grammar ----------
 
@pytest.mark.asyncio
async def test_portuguese_multiple_errors():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Eu quero comprar um prezente para minha irmã, mas não sei o que ela gosta.",
            target_language="Portuguese",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    assert "presente" in result.corrected_sentence
 
 
# ---------- 9. Italian correct sentence ----------
 
@pytest.mark.asyncio
async def test_correct_italian_sentence():
    original = "Mi piace molto la cucina italiana."
    result = await get_feedback(
        FeedbackRequest(
            sentence=original,
            target_language="Italian",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=True)
    assert result.corrected_sentence == original
 
 
# ---------- 10. Explanation in native language ----------
 
@pytest.mark.asyncio
async def test_explanation_in_native_language_spanish():
    """When native language is Spanish, explanations should be in Spanish."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="Spanish",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    # Check that at least one explanation contains common Spanish words
    explanations = " ".join(e.explanation for e in result.errors).lower()
    spanish_indicators = ["es", "el", "la", "en", "un", "una", "masculino", "femenino", "porque"]
    assert any(word in explanations for word in spanish_indicators), (
        f"Explanations don't appear to be in Spanish: {explanations[:200]}"
    )
 