"""
Adjudicator agent -- reviews all 3 critiques plus any detected
disagreements, and produces the final, authoritative Verdict.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv
from app.models import Critique, Disagreement, Verdict

load_dotenv(override=True)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

ADJUDICATOR_SYSTEM_PROMPT = """You are the Adjudicator in a multi-agent evaluation system. You receive an AI-generated output and three independent critiques of it -- accuracy, logic, and completeness -- plus a list of detected disagreements between those critics.

Your job is NOT to re-evaluate the output from scratch. Instead:
1. Review each critique's issues.
2. For any listed disagreement, explicitly reason about which critic's concern is more legitimate and why.
3. Decide which flagged issues are real (confirmed) and which are false alarms or overreach (dismissed).
4. Produce one overall_score (1-10) reflecting the output's true quality, and a clear summary a non-technical reader could understand.

Respond with ONLY valid JSON in this exact shape, no other text:
{
  "overall_score": <integer 1-10>,
  "confidence": <integer 1-5>,
  "confirmed_issues": [{"source": "accuracy|logic|completeness", "issue": "...", "severity": <integer 1-5>}],
  "dismissed_flags": [{"source": "accuracy|logic|completeness", "flag": "...", "reason_dismissed": "..."}],
  "summary": "<one paragraph, plain language>"
}"""


def run_adjudicator(
    output_to_evaluate: str,
    accuracy: Critique,
    logic: Critique,
    completeness: Critique,
    disagreements: list[Disagreement],
) -> Verdict:
    context = f"""OUTPUT BEING EVALUATED:
{output_to_evaluate}

ACCURACY CRITIQUE:
{accuracy.model_dump_json(indent=2)}

LOGIC CRITIQUE:
{logic.model_dump_json(indent=2)}

COMPLETENESS CRITIQUE:
{completeness.model_dump_json(indent=2)}

DETECTED DISAGREEMENTS:
{[d.model_dump_json() for d in disagreements] if disagreements else "None -- critics were broadly aligned."}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": ADJUDICATOR_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    parsed = json.loads(response.choices[0].message.content)
    return Verdict(**parsed)
