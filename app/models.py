
from pydantic import BaseModel, Field
from typing import Literal


class Issue(BaseModel):
    claim: str = Field(..., description="The specific text/claim being flagged")
    problem: str = Field(..., description="Why this is an issue")
    severity: int = Field(..., ge=1, le=5, description="1=minor, 5=major")


class Critique(BaseModel):
    dimension: Literal["accuracy", "logic", "completeness"]
    score: int = Field(..., ge=1, le=5, description="Overall score for this dimension")
    issues: list[Issue] = Field(default_factory=list)
    confidence: int = Field(..., ge=1, le=5, description="Critic's confidence in its own assessment")


class Disagreement(BaseModel):
    description: str
    dimensions_involved: list[str]
    severity_gap: int = Field(..., description="How far apart the critics' severity ratings were")


class ConfirmedIssue(BaseModel):
    source: Literal["accuracy", "logic", "completeness"]
    issue: str
    severity: int = Field(..., ge=1, le=5)


class DismissedFlag(BaseModel):
    source: Literal["accuracy", "logic", "completeness"]
    flag: str
    reason_dismissed: str


class Verdict(BaseModel):
    overall_score: int = Field(..., ge=1, le=10)
    confidence: int = Field(..., ge=1, le=5)
    confirmed_issues: list[ConfirmedIssue] = Field(default_factory=list)
    dismissed_flags: list[DismissedFlag] = Field(default_factory=list)
    summary: str


class ArbitrationRequest(BaseModel):
    original_question: str | None = Field(None, description="Optional: the prompt that produced the output")
    output_to_evaluate: str = Field(..., description="The LLM output being judged")


class ArbitrationResponse(BaseModel):
    critiques: list[Critique]
    disagreements: list[Disagreement]
    verdict: Verdict
