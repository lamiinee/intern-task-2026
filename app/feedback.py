"""System prompt and LLM interaction for language feedback."""

import json
import hashlib
import logging

import anthropic
from cachetools import TTLCache

from app.models import(
  FeedbackRequest,
  FeedbackResponse,
  VALID_ERROR_TYPES,
  VALID_DIFFICULTIES,
)

logger = logging.getLogger(__name__)

# Cache: up to 256 entries, expire after 1 hour.
# Saves money when the same sentence is submitted more than once.
_cache: TTLCache = TTLCache(maxsize=256, ttl=3600)

SYSTEM_PROMPT = """\
## Role
You are a warm, encouraging language tutor who genuinely wants your students to \
succeed. You have expertise in over 50 languages and understand how learners at \
different levels make mistakes. You know that a good correction teaches a rule the \
learner can apply next time, not just a fix they copy blindly. You analyze sentences \
in any language and return structured JSON feedback.

## Task
Analyze a learner written sentence in their target language. Identify every error, \
provide minimal corrections, explain what went wrong and give \
them a memorable way to remember the correct form. Write all explanations in their \
native language so they can understand regardless of their target language level. Rate the sentence's CEFR difficulty level.

## Context
- Learners range from absolute beginners to advanced speakers.
- The app supports every language — Latin scripts, CJK, Cyrillic, Arabic, Devanagari, etc.
- Some sentences will be perfectly correct. Celebrate that! Return is_correct=true and \
an empty errors array. Do not invent errors in correct sentences.
- Some sentences will have multiple errors. Catch all of them, but don't nitpick style \
preferences as errors. Only flag things that are genuinely wrong.
- Register: casual speech that fits the context is not an error. Only flag \
tone_register if the sentence mixes formal and informal in a way that sounds unnatural.
- Punctuation norms vary by language (e.g., Spanish ¿¡, French spaces before :;!?) — \
judge by the TARGET language's conventions, not English.

## Reasoning
The learner sees your feedback immediately after writing. If you mark a correct \
sentence as wrong, they lose trust. If you miss a real error, they learn the wrong \
thing. If your explanation is in the wrong language or too technical, they can't learn \
from it. Each piece of feedback is a tiny teaching moment so make it count.
 
The corrected_sentence should feel like "what I was trying to say, but correct" — not \
a rewrite. Preserve their word choices, their sentence structure, their voice. Only \
change what's actually broken.

## Stop Conditions
- Every "original" field must be a substring that appears verbatim in the input sentence
- error_type must be exactly one of: grammar, spelling, word_choice, punctuation, \
word_order, missing_word, extra_word, conjugation, gender_agreement, number_agreement, \
tone_register, other
- difficulty must be one of: A1, A2, B1, B2, C1, C2, based on the sentence's \
vocabulary and grammar complexity, not on whether it has errors
- is_correct must be true when errors is [] and false when errors is non-empty
- Explanations are in the native language, 1-2 sentences, friendly and educational, be creative
- Output is pure JSON — no markdown, no preamble, no trailing text

## Output
Respond with ONLY this JSON:
{
  "corrected_sentence": "string",
  "is_correct": boolean,
  "errors": [
    {
      "original": "string (exact text from the input)",
      "correction": "string (the fix)",
      "error_type": "string (from the allowed list)",
      "explanation": "string (in native language, warm and helpful)"
    }
  ],
  "difficulty": "A1|A2|B1|B2|C1|C2"
}
"""

# Anthropic client is created once
_client: anthropic.AsyncAnthropic | None = None

def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic()
    return _client

def _cache_key(request: FeedbackRequest) -> str:
    # deterministic hash for a request so we can cache responses
    raw = f"{request.sentence}|{request.target_language}|{request.native_language}"
    return hashlib.sha256(raw.encode()).hexdigest()

def _validate_and_fix(data: dict, original_sentence: str) -> dict:
    """ Clean the LLM output so it always conforms to the schema,
    Fixes common LLM mistakes like invalid error_type values or mismatched
    is_correct.
    """
    # Fix difficulty
    if data.get("difficulty") not in VALID_DIFFICULTIES:
        data["difficulty"] = "B1"  # safe default (Can be determine if we have more data from the users)
 
    # Fix error types
    errors = data.get("errors", [])
    cleaned_errors = []
    for err in errors:
        if not isinstance(err, dict):
            continue
        if err.get("error_type") not in VALID_ERROR_TYPES:
            err["error_type"] = "other"
        # Make sure all required keys exist
        if all(k in err for k in ("original", "correction", "error_type", "explanation")):
            cleaned_errors.append(err)
    data["errors"] = cleaned_errors
 
    # Reconcile is_correct with errors list
    if len(cleaned_errors) == 0:
        data["is_correct"] = True
        data["corrected_sentence"] = original_sentence
    else:
        data["is_correct"] = False
 
    return data

async def get_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Call Claude to analyze the learner's sentence and return feedback"""
 
    # Check cache first
    key = _cache_key(request)
    if key in _cache:
        logger.info("Cache hit for sentence: %s", request.sentence[:40])
        return _cache[key]
 
    client = _get_client()
 
    user_message = (
        f"Target language: {request.target_language}\n"
        f"Native language: {request.native_language}\n"
        f"Sentence to analyze: {request.sentence}"
    )
 
    # Retry up to 3 times on transient failures (temporary self-correcting error)
    last_error = None
    for attempt in range(3):
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                temperature=0.1,
            )
 
            content = response.content[0].text
            # Strip markdown fences if the model wraps its response
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[: content.rfind("```")]
 
            data = json.loads(content)
            data = _validate_and_fix(data, request.sentence)
            result = FeedbackResponse(**data)
 
            # Cache the successful result
            _cache[key] = result
            return result
 
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning("Attempt %d: LLM returned invalid JSON, retrying...", attempt + 1)
        except anthropic.APIStatusError as e:
            last_error = e
            if e.status_code in (429, 500, 502, 503, 529):
                logger.warning("Attempt %d: API error %d, retrying...", attempt + 1, e.status_code)
            else:
                raise
        except anthropic.APIConnectionError as e:
            last_error = e
            logger.warning("Attempt %d: Connection error, retrying...", attempt + 1)
 
    raise RuntimeError(f"Failed after 3 attempts. Last error: {last_error}")