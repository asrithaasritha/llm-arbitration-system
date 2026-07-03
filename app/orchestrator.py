"""
LangGraph orchestration: runs the 3 critics in parallel, detects
disagreements between them, then sends everything to the adjudicator
for a final verdict.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from app.models import Critique, Disagreement, Verdict
from app.critics import run_accuracy_critic, run_logic_critic, run_completeness_critic
from app.adjudicator import run_adjudicator


class ArbitrationState(TypedDict):
    """The shared 'clipboard' passed between every node in the graph."""
    original_question: str
    output_to_evaluate: str
    accuracy_critique: Critique | None
    logic_critique: Critique | None
    completeness_critique: Critique | None
    disagreements: list[Disagreement]
    verdict: Verdict | None


# ---- Node functions ----

def accuracy_node(state: ArbitrationState) -> dict:
    return {"accuracy_critique": run_accuracy_critic(state["output_to_evaluate"])}


def logic_node(state: ArbitrationState) -> dict:
    return {"logic_critique": run_logic_critic(state["output_to_evaluate"])}


def completeness_node(state: ArbitrationState) -> dict:
    critique = run_completeness_critic(
        state["original_question"], state["output_to_evaluate"]
    )
    return {"completeness_critique": critique}


def detect_disagreements_node(state: ArbitrationState) -> dict:
    """Flags a disagreement if the highest/lowest critic scores differ by >= 2."""
    critiques = {
        "accuracy": state["accuracy_critique"],
        "logic": state["logic_critique"],
        "completeness": state["completeness_critique"],
    }
    scores = {dim: c.score for dim, c in critiques.items()}
    highest_dim = max(scores, key=scores.get)
    lowest_dim = min(scores, key=scores.get)
    spread = scores[highest_dim] - scores[lowest_dim]

    disagreements = []
    if spread >= 2:
        disagreements.append(
            Disagreement(
                description=(
                    f"{highest_dim} critic scored {scores[highest_dim]}/5 while "
                    f"{lowest_dim} critic scored {scores[lowest_dim]}/5 -- "
                    f"a {spread}-point gap suggests they're evaluating this "
                    f"output very differently."
                ),
                dimensions_involved=[highest_dim, lowest_dim],
                severity_gap=spread,
            )
        )
    return {"disagreements": disagreements}


def adjudicator_node(state: ArbitrationState) -> dict:
    verdict = run_adjudicator(
        output_to_evaluate=state["output_to_evaluate"],
        accuracy=state["accuracy_critique"],
        logic=state["logic_critique"],
        completeness=state["completeness_critique"],
        disagreements=state["disagreements"],
    )
    return {"verdict": verdict}


# ---- Build the graph ----

def build_graph():
    graph = StateGraph(ArbitrationState)

    graph.add_node("accuracy", accuracy_node)
    graph.add_node("logic", logic_node)
    graph.add_node("completeness", completeness_node)
    graph.add_node("detect_disagreements", detect_disagreements_node)
    graph.add_node("adjudicator", adjudicator_node)

    # Parallel fan-out: all 3 critics start simultaneously from START
    graph.add_edge(START, "accuracy")
    graph.add_edge(START, "logic")
    graph.add_edge(START, "completeness")

    # Fan-in: wait for all 3 critics before detecting disagreements
    graph.add_edge("accuracy", "detect_disagreements")
    graph.add_edge("logic", "detect_disagreements")
    graph.add_edge("completeness", "detect_disagreements")

    # Sequential: adjudicator runs last, after disagreements are known
    graph.add_edge("detect_disagreements", "adjudicator")
    graph.add_edge("adjudicator", END)

    return graph.compile()


if __name__ == "__main__":
    import time

    app = build_graph()

    test_question = "Why did the company's stock price rise after the earnings call?"
    test_output = (
        "The company's stock rose 12% after the earnings call. Apple was "
        "founded in 1976 by Steve Jobs. Therefore, the earnings call caused "
        "investors to buy more Apple products, which is why the stock rose."
    )

    t0 = time.time()
    result = app.invoke({
        "original_question": test_question,
        "output_to_evaluate": test_output,
        "accuracy_critique": None,
        "logic_critique": None,
        "completeness_critique": None,
        "disagreements": [],
        "verdict": None,
    })
    elapsed = time.time() - t0

    print(f"(full pipeline took {elapsed:.1f}s)\n")
    print("=== DISAGREEMENTS ===")
    if result["disagreements"]:
        for d in result["disagreements"]:
            print(d.model_dump_json(indent=2))
    else:
        print("None detected this run.")

    print("\n=== FINAL VERDICT ===")
    print(result["verdict"].model_dump_json(indent=2))
