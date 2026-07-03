"""
FastAPI wrapper -- exposes the LangGraph arbitration pipeline as
a real HTTP API that any client can call.
"""

from fastapi import FastAPI, HTTPException
from app.models import ArbitrationRequest, ArbitrationResponse
from app.orchestrator import build_graph

app = FastAPI(
    title="LLM Arbitration System",
    description="Multi-agent evaluation of LLM outputs -- 3 critics + an adjudicator.",
    version="0.1.0",
)

# Build the graph once at startup, not on every request -- this is
# more efficient since building the graph structure has some overhead.
_graph = build_graph()


@app.get("/health")
def health():
    """Simple liveness check -- useful for Docker/deployment monitoring."""
    return {"status": "ok"}


@app.post("/v1/arbitrate", response_model=ArbitrationResponse)
def arbitrate(request: ArbitrationRequest):
    """
    Runs an AI output through the full arbitration pipeline:
    3 parallel critics -> disagreement detection -> adjudicator verdict.
    """
    try:
        result = _graph.invoke({
            "original_question": request.original_question or "",
            "output_to_evaluate": request.output_to_evaluate,
            "accuracy_critique": None,
            "logic_critique": None,
            "completeness_critique": None,
            "disagreements": [],
            "verdict": None,
        })
    except Exception as e:
        # If anything in the pipeline fails (bad LLM response, network
        # issue, etc.), surface it as a proper HTTP error instead of
        # crashing the server.
        raise HTTPException(status_code=500, detail=str(e))

    return ArbitrationResponse(
        critiques=[
            result["accuracy_critique"],
            result["logic_critique"],
            result["completeness_critique"],
        ],
        disagreements=result["disagreements"],
        verdict=result["verdict"],
    )
