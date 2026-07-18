"""LLM review-assist layer: turn model evidence into a structured review brief.

Design boundaries (these are the points worth defending in an interview):

1. The LLM never makes or executes a decision. Auto-approve / manual-review /
   audit routing is owned by the calibrated scoring model + threshold. This
   layer only drafts a brief for the human reviewing a flagged transaction.
2. The LLM only receives a de-identified evidence package (see src/explain) —
   no PII, no payment details, no raw identifiers.
3. The LLM may only recommend actions whose rule fired (src/rules). Facts it
   cites must come from the evidence. Anything else is a hallucination and is
   measurable in evals/.
4. The LLM is not on the critical path. If the API key is absent, the model
   errors, or the output fails schema validation, we fall back to a
   deterministic template built from the same evidence + rules. Serving never
   depends on the LLM being available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from src.explain import rule_namespace
from src.rules import allowed_actions, evaluate_rules

# Load .env so OPENAI_API_KEY / LLM_MODEL are picked up without exporting them.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass  # python-dotenv optional; env vars can be set manually

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


class RiskFactor(BaseModel):
    factor: str = Field(..., description="Short human-readable risk factor.")
    evidence: str = Field(..., description="The concrete value/percentile supporting it.")
    model_contribution: float = Field(
        ..., description="Signed contribution of this feature to the risk margin."
    )


class RecommendedCheck(BaseModel):
    action: str = Field(..., description="Action string; must be an allowed rule action.")
    rule_id: str = Field(..., description="Rule that triggered this action.")
    rationale: str = Field(..., description="Why this check applies to this transaction.")


class ReviewBrief(BaseModel):
    summary: str
    risk_factors: list[RiskFactor]
    recommended_checks: list[RecommendedCheck]
    recommended_disposition: Literal["hold_for_review", "escalate", "likely_ok"]
    generated_by: Literal["llm", "fallback_template"] = "llm"


SYSTEM_PROMPT = """You are a risk-review assistant for a ticket marketplace.
You help a human reviewer by summarizing why a transaction was flagged. You do
NOT approve, decline, or take any action yourself.

Strict rules:
- Use ONLY the facts in the provided evidence JSON. Never invent values,
  history, or context that is not present.
- You may ONLY recommend actions from the provided fired_rules list. Include a
  recommended_check for EVERY rule in fired_rules — do not omit or merge any —
  and cite each check's rule_id. Do not suggest any action not in the list.
- Every risk_factor must reference a real field/value from the evidence.
- Be concise and factual. No speculation about identity or intent beyond what
  the evidence supports.

Return ONLY a JSON object matching this schema:
{
  "summary": str,
  "risk_factors": [{"factor": str, "evidence": str, "model_contribution": float}],
  "recommended_checks": [{"action": str, "rule_id": str, "rationale": str}],
  "recommended_disposition": "hold_for_review" | "escalate" | "likely_ok"
}
"""


def _build_user_prompt(evidence: dict[str, Any], fired_rules: list[dict[str, Any]]) -> str:
    return (
        "evidence:\n"
        + json.dumps(evidence, indent=2)
        + "\n\nfired_rules (the only actions you may recommend):\n"
        + json.dumps(fired_rules, indent=2)
    )


def _call_openai(system: str, user: str, model: str) -> str | None:
    """Call OpenAI if the SDK and an API key are available; else return None."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


def _fallback_brief(evidence: dict[str, Any], fired_rules: list[dict[str, Any]]) -> ReviewBrief:
    """Deterministic brief from evidence + rules — no LLM, guaranteed schema.

    This is what serving returns whenever the LLM is unavailable or invalid. It
    is intentionally faithful (only evidence-backed facts, only fired actions),
    so the system degrades gracefully instead of failing.
    """
    contributors = [
        c for c in evidence.get("top_contributors", []) if c["direction"] == "increases_risk"
    ]
    risk_factors = [
        RiskFactor(
            factor=c["label"],
            evidence=(
                f"value={c['value']}"
                + (f", {c['percentile']}th percentile" if c.get("percentile") is not None else "")
            ),
            model_contribution=c["contribution"],
        )
        for c in contributors[:4]
    ]
    checks = [
        RecommendedCheck(action=r["action"], rule_id=r["rule_id"], rationale=r["description"])
        for r in fired_rules
    ]

    score = evidence.get("risk_score", 0.0)
    if any(r["rule_id"] == "elevated_score" for r in fired_rules):
        disposition = "escalate"
    elif fired_rules:
        disposition = "hold_for_review"
    else:
        disposition = "likely_ok"

    n = len(fired_rules)
    summary = (
        f"Transaction flagged with risk score {score}. "
        f"{len(risk_factors)} risk-increasing factor(s) identified; "
        f"{n} review action(s) recommended by policy."
    )
    return ReviewBrief(
        summary=summary,
        risk_factors=risk_factors,
        recommended_checks=checks,
        recommended_disposition=disposition,
        generated_by="fallback_template",
    )


def _sanitize(brief: ReviewBrief, fired_rules: list[dict[str, Any]]) -> ReviewBrief:
    """Drop any recommended check whose action/rule was not actually fired.

    Even a well-behaved LLM can drift; this guarantees the served brief never
    recommends an action outside policy.
    """
    allowed = allowed_actions()
    fired_ids = {r["rule_id"] for r in fired_rules}
    brief.recommended_checks = [
        c for c in brief.recommended_checks if c.action in allowed and c.rule_id in fired_ids
    ]
    return brief


def generate_review_brief(
    evidence: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    rules_path: str | None = None,
) -> ReviewBrief:
    """Produce a structured review brief for a flagged transaction.

    Tries the LLM first (if configured); validates against the schema and
    sanitizes against fired rules; falls back to a deterministic template on
    any failure.
    """
    namespace = rule_namespace(evidence)
    fired_rules = evaluate_rules(namespace, rules_path)

    raw = _call_openai(SYSTEM_PROMPT, _build_user_prompt(evidence, fired_rules), model)
    if raw:
        try:
            data = json.loads(raw)
            data.setdefault("generated_by", "llm")
            brief = ReviewBrief.model_validate(data)
            return _sanitize(brief, fired_rules)
        except (json.JSONDecodeError, ValidationError):
            pass  # fall through to deterministic template

    return _fallback_brief(evidence, fired_rules)
