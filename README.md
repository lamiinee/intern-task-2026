# Pangea Chat: Gen AI Intern Task (Summer 2026)

## Language Feedback API

An LLM-powered API that analyzes learner-written sentences and returns structured correction feedback. Built with **Python + FastAPI + Anthropic Claude**.

### Run tests

```bash
# Unit tests (no API key needed)
docker compose exec feedback-api pytest tests/test_feedback_unit.py tests/test_schema.py -v

# Integration tests (requires API key in .env)
docker compose exec feedback-api pytest tests/test_feedback_integration.py -v
```

## Design Decisions

### Model choice: Claude Sonnet 4

I chose `claude-sonnet-4-20250514` for the best balance of accuracy, speed, and cost:

- Accuracy : Sonnet 4 handles multilingual grammar analysis well across Latin and non-Latin scripts. It reliably follows structured output instructions.
- Speed : Responses typically come back in 2-5 seconds, well under the 30-second timeout.
- Cost : Significantly cheaper per token than Opus while being more accurate than Haiku for nuanced linguistic tasks. For a production language-learning app, this is the sweet spot.

### Prompt strategy

The system prompt follows the Claude prompt anatomy (Role → Task → Context → Reasoning → Stop Conditions → Output) to give the model maximum clarity on what's expected:

1. **Role**: Establishes Claude as a warm, encouraging tutor with expertise across 50+ languages. This framing produces explanations that teach rather than just correct, the model aims to give learners a memorable rule they can reuse, not just a fix to copy.
2. **Task**: Clearly scoped, find errors, correct minimally, explain in the native language, rate CEFR difficulty. No ambiguity about what the model should return.
3. **Context**: Handles edge cases upfront, correct sentences (don't invent errors), multiple errors (catch all of them), register sensitivity (casual speech isn't wrong), and language-specific punctuation norms (judge by the target language, not English). Also explicitly states support for non-Latin scripts (CJK, Cyrillic, Arabic, Devanagari).
4. **Reasoning**: Explains why each rule matters from the learner's perspective. Marking a correct sentence wrong loses trust. Missing a real error teaches the wrong thing. Writing explanations in the wrong language makes them useless. This section gives the model the intent behind the rules, which helps it handle situations the rules don't explicitly cover.
5. **Stop Conditions**: Hard constraints that map directly to the JSON schema, allowed error types, allowed CEFR levels, the `is_correct ↔ errors` consistency rule, and the requirement for pure JSON output with no markdown.
6. **Output**: The exact JSON shape the model must return, with field descriptions inline so there's no guessing.

This structure means the model knows _what_ to do (Task), _who_ it's doing it for (Context), _why_ it matters (Reasoning), _when it's done_ (Stop Conditions), and _how to format it_ (Output). The result is more reliable structured output with fewer schema violations.

### Validation layer

LLMs don't always follow instructions perfectly. The `_validate_and_fix` function acts as a safety net:

- If the model returns an invalid `error_type`, it gets mapped to `"other"` instead of crashing.
- If the model returns an invalid `difficulty`, it defaults to `"B1"`.
- If `is_correct` contradicts the `errors` array (e.g., says `true` but has errors), the code reconciles them.
- Malformed error objects (missing required fields) are silently dropped.

This means the API never returns schema-invalid JSON, even if the model has a bad day.

### Caching

An in-memory TTL cache (256 entries, 1-hour expiry) prevents duplicate API calls. If the same sentence + language pair is submitted twice, the second response is instant and free. In production this could be replaced with Redis, but for this use case I judge that a simple in-process cache is sufficient.

### Retry logic

Transient failures (rate limits, server errors, malformed JSON) are retried up to 3 times before returning an error. This makes the API more resilient without adding significant latency in the happy path.

### Error handling

The `/feedback` endpoint catches errors from the LLM pipeline and returns appropriate HTTP status codes (502 for upstream failures, 500 for unexpected errors) instead of crashing.

## How I verified accuracy for languages I don't speak

I speak French and English, so I could directly verify the French test cases myself. For everything else, here's my approach:

1. **Sample inputs/outputs** served as ground truth for Spanish, Japanese, German, and Portuguese.
2. **AI-generated test cases**: I used Claude to generate test sentences with known errors for Korean, Russian, Chinese, and Italian.
3. **The integration tests** serve as regression tests if a future prompt change breaks Korean particle correction or Russian case agreement, the test catches it.

## Test coverage

**Unit tests** (no API key needed):

- Validation/sanitization logic: invalid difficulty, invalid error types, malformed errors, is_correct reconciliation
- Mocked API responses: Spanish errors, correct sentences, multiple errors, cache behavior, markdown fence stripping

**Integration tests** (real API calls):

- 10 tests covering: Spanish, German, French, Japanese, Korean, Russian, Chinese, Portuguese, Italian
- Edge cases: correct sentences, multiple errors, non-Latin scripts, explanation language verification

## Use of AI

I used AI (Claude) throughout this project. Here's exactly where and how:

- **Test cases**: The integration tests for Korean, Russian, Chinese, and Italian were AI-generated. I verified them against translation tools and grammar references, but the initial sentences and expected corrections came from Claude.
- **Boilerplate code**: The retry logic, caching setup, and validation function were written with AI assistance. I reviewed and adjusted the code, but didn't write it from scratch.
