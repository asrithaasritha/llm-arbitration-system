# LLM Arbitration System

**Multi-agent evaluation for AI outputs.** Three independent critics judge accuracy, logic, and completeness in parallel; an adjudicator resolves their disagreements into one auditable verdict.

## The Problem

LLMs increasingly grade their own homework — a single model self-critiques its own output, or one "judge" model rubber-stamps everything. That inherits whatever blind spots the underlying model already has.

This system uses three independent critic agents, each scoring one dimension only, running in parallel. Where they disagree, a fourth agent — the **Adjudicator** — doesn't just average their scores. It reasons about which critic's concern is legitimate, dismisses overreach, and produces one final, explainable verdict.

## Architecture

```
                    ┌──> Accuracy Critic  ──┐
User Input ─────────┼──> Logic Critic ──────┼──> Disagreement Detector ──> Adjudicator ──> Verdict
                    └──> Completeness Critic┘
```

1. **3 critics run in parallel** (LangGraph fan-out) — each scores the output against one dimension only
2. **Disagreement detection** flags a real conflict when critic scores diverge by ≥2 points
3. **The Adjudicator** reviews all critiques and disagreements, confirms or dismisses each flagged issue, and produces one overall score with a plain-language summary

## Example: Critics Genuinely Split

| Critic | Score | Reasoning |
|---|---|---|
| Accuracy | 4/5 | Individual claims stated aren't factually false |
| Logic | 4/5 | Flags the unsupported leap from "higher test accuracy" to "deploy immediately" |
| Completeness | 2/5 | Production-readiness factors — monitoring, bias, rollback — are entirely missing |

An output can pass a shallow fact-check and still be badly reasoned or incomplete. Catching that requires more than one lens.

## Tech Stack

| Layer | Tool |
|---|---|
| LLM inference | Groq (LLaMA 3.3 70B / 3.1 8B) |
| Orchestration | LangGraph — parallel execution, shared state |
| API | FastAPI — `/v1/arbitrate`, auto-documented via OpenAPI |
| Validation | Pydantic v2 — enforced at every LLM call boundary |
| UI | Streamlit — calls the API over HTTP, decoupled from backend |
| Packaging | Docker / docker-compose |

## Design Decisions

- **Structured output everywhere.** Every LLM response is forced into JSON and validated against a Pydantic model before entering the pipeline — malformed responses fail loudly instead of corrupting downstream state.
- **UI talks to the API over HTTP**, not direct function calls — proves the backend is a real, independent service. The same API could serve a Slack bot or CLI without changes.
- **Different models per critic role** — keeps disagreement meaningful rather than noise from one model talking to itself.

## Running Locally

```bash
git clone https://github.com/asrithaasritha/llm-arbitration-system
cd llm-arbitration-system
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
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

```
POST /v1/arbitrate
{
  "original_question": "optional context",
  "output_to_evaluate": "the AI output being judged"
}
```
Returns all 3 critiques, detected disagreements, and the final verdict. Interactive docs at `/docs`.

## What's Next

- Persistent audit trail for trend analysis across runs
- Configurable critic weighting per use case
- Batch evaluation mode for regression-testing prompt changes

---

**Asritha** · [LinkedIn](https://www.linkedin.com/in/asritha-p-7a312a382/) · [GitHub](https://github.com/asrithaasritha/llm-arbitration-system)
