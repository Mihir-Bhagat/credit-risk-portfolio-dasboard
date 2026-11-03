# Which LLM for "Ask the Portfolio"? Research and decision

Researched 2026-11-03. Figures come from the third-party pages linked below, which read the providers' own pages between
2026-10-16 and 2026-10-23. **Limits change often; check the provider dashboard before an interview.** Nothing here was
measured with a live key (the build sandbox has no keys and no route to these providers); see `MANUAL_CHECKS.md` for the
script that does.

## What the job actually is

The LLM is used for one narrow thing: turn a question into a small **JSON query plan** over a closed list of 15 metrics,
17 dimensions and numeric filters. It never sees data rows, never computes a number and never writes the answer text
(pandas executes the plan, the answer is rendered locally and every number is checked against the executed result).
So the model needs: reliable JSON output, good instruction following (decline what the catalog cannot answer, ignore
injected instructions), low latency, a free tier, and an OpenAI-compatible endpoint. It does **not** need frontier-level
reasoning, so small or "flash" models are enough. Each request is about 1,400 input tokens (the catalog) plus at most 500
output tokens.

## Shortlist

| Provider (free tier) | Candidate model | Free limits found | Data terms | Fit |
|---|---|---|---|---|
| **Groq** | `openai/gpt-oss-120b` (also `qwen/qwen3.8-27b`) | 30 requests/min, 1,000 requests/day, 8,000 tokens/min, 200,000 tokens/day per organisation | Not stated on the page used: read Groq's terms before sending anything beyond the catalog (this app sends only the question and catalog) | **Recommended default.** Fast inference, JSON mode, OpenAI-compatible. The 8k tokens/min cap means roughly 4 uncached questions a minute; repeated questions are cached and the app falls back to the offline interpreter with a notice if the cap is hit |
| **Google Gemini** | `gemini-3.5-flash-lite` (also `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.1-flash-lite`) | Google no longer publishes per-model numbers; they are shown per project in AI Studio. RPD resets at midnight Pacific | Free-tier content may be used by Google to improve products and may be read by human reviewers | Good quality, generous in practice, but the data-use term matters for a bank. Fine here only because no row data is sent |
| **OpenRouter** | `openai/gpt-oss-120b:free` | 20 requests/min; 50 requests/day (1,000/day after buying 10 credits) | Not stated on the page used | Useful fallback for variety of models; 50/day is thin for repeated interview demos |
| Offline interpreter (built in) | none | unlimited | nothing leaves the machine | Always on as the fallback; covers the example questions and declines what it cannot ground |

## Decision

1. Default: **Groq `openai/gpt-oss-120b`** (set `GROQ_API_KEY`). It has the highest free request allowance and
   fastest responses of the three, which matters for a live demo.
2. Alternative: Gemini `gemini-3.5-flash-lite` (set `GEMINI_API_KEY`) if you already have a Google key.
3. Production at a real bank: an approved gateway or on-premise model. The adapter only needs `LLM_BASE_URL`, `LLM_MODEL`
   and `LLM_API_KEY`, so it can point at one without code changes.

The model ids above come from the vendor pages linked below and **were not called live**; if a provider renamed a model, set
`LLM_MODEL` to the current id. Quality should be confirmed with `python scripts/live_llm_check.py` (same question suite as
the automated tests, with the real provider; roughly 40 calls).

## What was and was not tested

| Claim | How tested | Status |
|---|---|---|
| Adapter sends only the question and catalog; key only in the `Authorization` header | Real HTTP requests to a local OpenAI-compatible fake server, request bodies inspected (`tests/test_f_ask.py`) | Tested |
| 429, 5xx (with retry), 401, timeout, malformed and non-JSON replies, invalid metrics, injected fields, JSON-mode rejection, fenced JSON | Same fake server | Tested |
| Fallback to offline interpreter with visible notice | Same, plus Streamlit AppTest | Tested |
| A real provider returns a valid plan for the suite | Needs a key | **MANUAL** (`scripts/live_llm_check.py`) |
| Real latency and cost per question | Needs a key | **MANUAL** |
| Comparative quality of Groq vs Gemini vs OpenRouter | Needs keys for each | **MANUAL / not done** |

Real LLM API calls made during development and verification: **0**.

## Sources

- [Groq free-tier limits (BenchLM, read 2026-10-16)](https://benchlm.ai/md/free-tier/groq.md)
- [Gemini API free tier: models and data use (Apidog, 2026-10-23)](https://apidog.com/blog/gemini-api-free-tier/)
- [OpenRouter free-model limits (BenchLM, read 2026-10-16)](https://benchlm.ai/md/free-tier/openrouter.md)
