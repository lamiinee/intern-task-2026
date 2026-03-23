"""Unit tests -- run without an API key using mocked LLM responses."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.feedback import get_feedback, _validate_and_fix, _cache
from app.models import FeedbackRequest

def _mock_response(response_data: dict) -> MagicMock:
    """Build a mock Anthropic message response."""
    text_block = MagicMock()
    text_block.text = json.dumps(response_data)
    message = MagicMock()
    message.content = [text_block]
    return message

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the response cache before each test."""
    _cache.clear()
    yield
    _cache.clear()
 
class TestValidateAndFix:
    def test_fixes_invalid_difficulty(self):
        data = {
            "corrected_sentence": "Hola",
            "is_correct": True,
            "errors": [],
            "difficulty": "Z9",
        }
        fixed = _validate_and_fix(data, "Hola")
        assert fixed["difficulty"] == "B1"
 
    def test_fixes_invalid_error_type(self):
        data = {
            "corrected_sentence": "Le chat",
            "is_correct": False,
            "errors": [
                {
                    "original": "La chat",
                    "correction": "Le chat",
                    "error_type": "made_up_type",
                    "explanation": "Test",
                }
            ],
            "difficulty": "A1",
        }
        fixed = _validate_and_fix(data, "La chat")
        assert fixed["errors"][0]["error_type"] == "other"
 
    def test_reconciles_empty_errors_to_correct(self):
        data = {
            "corrected_sentence": "Changed something",
            "is_correct": False,
            "errors": [],
            "difficulty": "A2",
        }
        fixed = _validate_and_fix(data, "Original sentence")
        assert fixed["is_correct"] is True
        assert fixed["corrected_sentence"] == "Original sentence"
 
    def test_reconciles_errors_present_to_incorrect(self):
        data = {
            "corrected_sentence": "Fixed",
            "is_correct": True,  # wrong — there are errors
            "errors": [
                {
                    "original": "x",
                    "correction": "y",
                    "error_type": "grammar",
                    "explanation": "test",
                }
            ],
            "difficulty": "A1",
        }
        fixed = _validate_and_fix(data, "Original")
        assert fixed["is_correct"] is False
 
    def test_strips_malformed_errors(self):
        data = {
            "corrected_sentence": "Ok",
            "is_correct": False,
            "errors": [
                {"original": "a"},  # missing required keys
                "not a dict",
                {
                    "original": "b",
                    "correction": "c",
                    "error_type": "grammar",
                    "explanation": "good one",
                },
            ],
            "difficulty": "B1",
        }
        fixed = _validate_and_fix(data, "Ok")
        assert len(fixed["errors"]) == 1
 
 
# Mocked API call tests 
 
@pytest.mark.asyncio
async def test_spanish_conjugation_error():
    mock_data = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "You mixed two verb forms. Use 'fui' (I went) for past tense.",
            }
        ],
        "difficulty": "A2",
    }
 
    with patch("app.feedback._get_client") as mock_get:
        client = MagicMock()
        client.messages.create = AsyncMock(return_value=_mock_response(mock_data))
        mock_get.return_value = client
 
        result = await get_feedback(
            FeedbackRequest(
                sentence="Yo soy fue al mercado ayer.",
                target_language="Spanish",
                native_language="English",
            )
        )
 
    assert result.is_correct is False
    assert result.corrected_sentence == "Yo fui al mercado ayer."
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "conjugation"
    assert result.difficulty == "A2"
 
 
@pytest.mark.asyncio
async def test_correct_sentence_returns_original():
    original = "Ich habe gestern einen interessanten Film gesehen."
    mock_data = {
        "corrected_sentence": original,
        "is_correct": True,
        "errors": [],
        "difficulty": "B1",
    }
 
    with patch("app.feedback._get_client") as mock_get:
        client = MagicMock()
        client.messages.create = AsyncMock(return_value=_mock_response(mock_data))
        mock_get.return_value = client
 
        result = await get_feedback(
            FeedbackRequest(
                sentence=original,
                target_language="German",
                native_language="English",
            )
        )
 
    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == original
 
 
@pytest.mark.asyncio
async def test_french_multiple_gender_errors():
    mock_data = {
        "corrected_sentence": "Le chat noir est sur la table.",
        "is_correct": False,
        "errors": [
            {
                "original": "La chat",
                "correction": "Le chat",
                "error_type": "gender_agreement",
                "explanation": "'Chat' is masculine in French.",
            },
            {
                "original": "le table",
                "correction": "la table",
                "error_type": "gender_agreement",
                "explanation": "'Table' is feminine in French.",
            },
        ],
        "difficulty": "A1",
    }
 
    with patch("app.feedback._get_client") as mock_get:
        client = MagicMock()
        client.messages.create = AsyncMock(return_value=_mock_response(mock_data))
        mock_get.return_value = client
 
        result = await get_feedback(
            FeedbackRequest(
                sentence="La chat noir est sur le table.",
                target_language="French",
                native_language="English",
            )
        )
 
    assert result.is_correct is False
    assert len(result.errors) == 2
    assert all(e.error_type == "gender_agreement" for e in result.errors)
 
 
@pytest.mark.asyncio
async def test_cache_returns_same_result():
    mock_data = {
        "corrected_sentence": "Hola mundo.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }
 
    with patch("app.feedback._get_client") as mock_get:
        client = MagicMock()
        client.messages.create = AsyncMock(return_value=_mock_response(mock_data))
        mock_get.return_value = client
 
        req = FeedbackRequest(
            sentence="Hola mundo.",
            target_language="Spanish",
            native_language="English",
        )
 
        result1 = await get_feedback(req)
        result2 = await get_feedback(req)
 
    # API should only be called once, second call hits cache
    assert client.messages.create.await_count == 1
    assert result1 == result2
 
 
@pytest.mark.asyncio
async def test_markdown_fence_stripping():
    """If the LLM wraps its JSON in ```json ... ```, we strip it."""
    raw_json = {
        "corrected_sentence": "Test.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }
    fenced = f"```json\n{json.dumps(raw_json)}\n```"
 
    text_block = MagicMock()
    text_block.text = fenced
    message = MagicMock()
    message.content = [text_block]
 
    with patch("app.feedback._get_client") as mock_get:
        client = MagicMock()
        client.messages.create = AsyncMock(return_value=message)
        mock_get.return_value = client
 
        result = await get_feedback(
            FeedbackRequest(
                sentence="Test.",
                target_language="Spanish",
                native_language="English",
            )
        )
 
    assert result.is_correct is True