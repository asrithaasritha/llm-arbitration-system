"""
LangGraph orchestration: runs the 3 critics in parallel, detects
disagreements between them, and hands everything to the adjudicator.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from app.models import Critique, Disagreement
from app.critics import run_accuracy_critic, run_logic_critic, run_completeness_critic


class ArbitrationState(TypedDict):
    """
    The shared 'clipboard' passed between every node in the graph.
    Each node reads what it needs and writes its result back in.
    """
    original_question: str
    output_to_evaluate: str
    accuracy_critique: Critique | None
    logic_critique: Critique | None
    completeness_critique: Critique | None
    disagreements: list[Disagreement]


# ---- Node functions ----
# Each node takes the current state, does its job, and returns
# a dict of updates to merge back into the state.

def accuracy_node(state: ArbitrationState) -> dict:
    critique = run_accuracy_critic(state["output_to_evaluate"])
    return {"accuracy_critique": critique}


def logic_node(state: ArbitrationState) -> dict:
    critique = run_logic_critic(state["output_to_evaluate"])
    return {"logic_critique": critique}


def completeness_node(state: ArbitrationState) -> dict:
    critique = run_completeness_critic(
        state["original_question"], state["output_to_evaluate"]
    )
    return {"completeness_critique": critique}


def detect_disagreements_node(state: ArbitrationState) -> dict:
    """
    Compares the 3 critics' scores. If the spread between the
    highest and lowest score is >= 2, that's a real disagreement
    worth flagging for the adjudicator to resolve.
    """
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


# ---- Build the graph ----

def build_graph():
    graph = StateGraph(ArbitrationState)

    graph.add_node("accuracy", accuracy_node)
    graph.add_node("logic", logic_node)
    graph.add_node("completeness", completeness_node)
    graph.add_node("detect_disagreements", detect_disagreements_node)

    # Parallel fan-out: START connects to all 3 critics at once.
    # LangGraph runs all nodes reachable from START with no
    # dependency between them concurrently.
    graph.add_edge(START, "accuracy")
    graph.add_edge(START, "logic")
    graph.add_edge(START, "completeness")

    # Fan-in: detect_disagreements waits until ALL THREE critics
    # have finished before it runs.
    graph.add_edge("accuracy", "detect_disagreements")
    graph.add_edge("logic", "detect_disagreements")
    graph.add_edge("completeness", "detect_disagreements")

    graph.add_edge("detect_disagreements", END)

    return graph.compile()


if __name__ == "__main__":
    import time

    app = build_graph()

    # A case DESIGNED to make critics disagree:
    # factually correct, but the reasoning connecting cause -> effect is broken.
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
    })
    elapsed = time.time() - t0

    print(f"(pipeline took {elapsed:.1f}s -- proves critics ran in parallel)\n")

    print("=== ACCURACY ===")
    print(result["accuracy_critique"].model_dump_json(indent=2))
    print("\n=== LOGIC ===")
    print(result["logic_critique"].model_dump_json(indent=2))
    print("\n=== COMPLETENESS ===")
    print(result["completeness_critique"].model_dump_json(indent=2))
    print("\n=== DISAGREEMENTS ===")
    if result["disagreements"]:
        for d in result["disagreements"]:
            print(d.model_dump_json(indent=2))
    else:
        print("None detected this run.")
