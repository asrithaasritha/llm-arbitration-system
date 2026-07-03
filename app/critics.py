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
    """Calls Groq with the accuracy persona, validates and returns a Critique."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": ACCURACY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Output to evaluate:\n\n{output_to_evaluate}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    raw_json = response.choices[0].message.content
    parsed = json.loads(raw_json)          # turn the text into a Python dict
    return Critique(**parsed)               # validate it against our schema


if __name__ == "__main__":
    # Quick manual test -- run this file directly to see it work
    test_output = "The Eiffel Tower was built in 1822 and is located in Berlin, Germany."
    result = run_accuracy_critic(test_output)
    print(result.model_dump_json(indent=2))