"""
Critic agents -- each one independently evaluates an AI output
along one dimension (accuracy, logic, completeness).
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv
from app.models import Critique

load_dotenv(override=True)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

ACCURACY_SYSTEM_PROMPT = """You are a Factual Accuracy Critic. You evaluate whether claims in an AI-generated output are verifiable, internally consistent, and not contradicted by the provided context (if any).

For the given output, identify every factual claim. For each questionable claim, note: the specific claim, why it's questionable, and severity (1=minor, 5=major).

Respond with ONLY valid JSON in this exact shape, no other text:
{
  "dimension": "accuracy",
  "score": <integer 1-5>,
  "issues": [{"claim": "...", "problem": "...", "severity": <integer 1-5>}],
  "confidence": <integer 1-5>
}"""


def run_accuracy_critic(output_to_evaluate: str) -> Critique:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": ACCURACY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Output to evaluate:\n\n{output_to_evaluate}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    parsed = json.loads(response.choices[0].message.content)
    return Critique(**parsed)


LOGIC_SYSTEM_PROMPT = """You are a Logical Consistency Critic. You evaluate whether the reasoning in an output follows logically and whether conclusions are actually supported by the premises given.

Do not assess factual accuracy -- assume claims are true and check only whether the argument structure holds. Flag: non-sequiturs, unsupported leaps, circular reasoning, contradictions between statements within the output.

Respond with ONLY valid JSON in this exact shape, no other text:
{
  "dimension": "logic",
  "score": <integer 1-5>,
  "issues": [{"claim": "...", "problem": "...", "severity": <integer 1-5>}],
  "confidence": <integer 1-5>
}"""


def run_logic_critic(output_to_evaluate: str) -> Critique:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": LOGIC_SYSTEM_PROMPT},
            {"role": "user", "content": f"Output to evaluate:\n\n{output_to_evaluate}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    parsed = json.loads(response.choices[0].message.content)
    return Critique(**parsed)


COMPLETENESS_SYSTEM_PROMPT = """You are a Completeness Critic. You are given the ORIGINAL QUESTION and the OUTPUT that was produced in response. Evaluate whether the output addresses every part of what was asked, and flag any gaps, ignored constraints, or partial answers dressed up as complete ones.

Respond with ONLY valid JSON in this exact shape, no other text:
{
  "dimension": "completeness",
  "score": <integer 1-5>,
  "issues": [{"claim": "...", "problem": "...", "severity": <integer 1-5>}],
  "confidence": <integer 1-5>
}"""


def run_completeness_critic(original_question: str, output_to_evaluate: str) -> Critique:
    user_content = f"ORIGINAL QUESTION:\n{original_question}\n\nOUTPUT:\n{output_to_evaluate}"
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": COMPLETENESS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    parsed = json.loads(response.choices[0].message.content)
    return Critique(**parsed)


if __name__ == "__main__":
    test_question = "What is the Eiffel Tower and when/where was it built?"
    test_output = "The Eiffel Tower was built in 1822 and is located in Berlin, Germany."

    print("=== ACCURACY ===")
    print(run_accuracy_critic(test_output).model_dump_json(indent=2))

    print("\n=== LOGIC ===")
    print(run_logic_critic(test_output).model_dump_json(indent=2))

    print("\n=== COMPLETENESS ===")
    print(run_completeness_critic(test_question, test_output).model_dump_json(indent=2))
