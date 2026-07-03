# ⚖️ LLM Arbitration System

**Multi-agent evaluation for AI outputs — three independent critics judge accuracy, logic, and completeness, and an adjudicator resolves their disagreements into one auditable verdict.**

## The problem

LLMs increasingly grade their own homework — a single model self-critiques its own output, or one "judge" model rubber-stamps everything. That's a weak evaluation design: it inherits whatever blind spots the underlying model already has.

This project takes a different approach: **three independent critic agents**, each judging one dimension only (factual accuracy, logical consistency, completeness), running in parallel. Where they disagree, a fourth agent — the **Adjudicator** — doesn't just average their scores. It reasons about *which* critic's concern is legitimate, dismisses overreach, and produces one final, explainable verdict.

## How it works

```
                    ┌──> Accuracy Critic  ──┐
User Input ─────────┼──> Logic Critic ──────┼──> Disagreement Detector ──> Adjudicator ──> Verdict
                    └──> Completeness Critic┘
```

1. **3 critics run in parallel** (LangGraph fan-out) — each sees the same output but scores it against one dimension only
2. **Disagreement detection** compares critic scores; a ≥2-point spread flags a real conflict for review
3. **The Adjudicator** reviews all 3 critiques + any disagreements, decides which flagged issues are confirmed vs. dismissed as noise, and produces one overall score + plain-language summary

## Example: where critics genuinely split

| Critic | Score | Why |
|---|---|---|
| Accuracy | 4/5 | Individual claims stated aren't factually false |
| Logic | 4/5 | Flags the unsupported leap from "higher test accuracy" to "deploy immediately" |
| Completeness | 2/5 | Catches that production-readiness factors (monitoring, bias, rollback) are entirely missing |

This is the system's core value: an output can pass a shallow fact-check and still be badly reasoned or incomplete — catching that requires more than one lens.

## Tech stack

- **Groq** (LLaMA 3.3 70B / 3.1 8B) — LLM inference for all 4 agents
- **LangGraph** — multi-agent orchestration, parallel execution, shared state
- **FastAPI** — REST API (`/v1/arbitrate`), auto-documented via OpenAPI/Swagger
- **Pydantic v2** — strict schema validation at every LLM call boundary
- **Streamlit** — interactive UI, calls the API over HTTP (decoupled from the backend)
- **Docker / docker-compose** — containerized API + UI services

## Architecture decisions worth noting

- **Structured output everywhere** — every LLM response is forced into JSON (`response_format`) and validated against a Pydantic model before it enters the pipeline. Malformed responses fail loudly instead of corrupting downstream state.
- **UI talks to the API over HTTP, not direct function calls** — proves the backend is a real service, not just glue code. The same API could serve a Slack bot, a mobile app, or a CLI without changes.
- **Different models per critic role** — critics don't all run on the same model instance, which keeps "disagreement" meaningful rather than just noise from one model talking to itself.

## Running locally

```bash
git clone <repo-url>
cd llm-arbitration-system
python -m venv venv
venv\Scripts\Activate.ps1      # Windows
pip install -r requirements.txt

# Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Terminal 1
uvicorn app.api:app --reload

# Terminal 2
streamlit run streamlit_app.py
```

Or with Docker:
```bash
docker-compose up
```
API on `:8000`, UI on `:8501`.

## API

```bash
POST /v1/arbitrate
{
  "original_question": "optional context",
  "output_to_evaluate": "the AI output being judged"
}
```
Returns all 3 critiques, detected disagreements, and the final adjudicated verdict. Interactive docs at `/docs`.

## What I'd build next

- Persistent audit trail (store every arbitration run for trend analysis over time)
- Configurable critic weighting per use case (e.g. weight accuracy higher for factual QA, completeness higher for task instructions)
- Batch evaluation mode for regression-testing prompt changes against a fixed test set

---
Built by [Your Name] — [LinkedIn] · [GitHub]
