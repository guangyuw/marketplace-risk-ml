"""Load and evaluate the manual-review action rules.

Rules live in ``review_rules.yaml`` as declarative ``condition -> action``
entries. They are evaluated against the flattened evidence namespace produced
by :func:`src.explain.rule_namespace`. Conditions are simple boolean
expressions evaluated with no builtins available, so a rule file can express
thresholds/membership without being able to execute arbitrary code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

RULES_PATH = Path(__file__).resolve().parent / "review_rules.yaml"


@lru_cache(maxsize=1)
def load_rules(path: str | None = None) -> list[dict[str, Any]]:
    rules_file = Path(path) if path else RULES_PATH
    data = yaml.safe_load(rules_file.read_text())
    return list(data.get("rules", []))


def _safe_eval(condition: str, namespace: dict[str, Any]) -> bool:
    # No builtins: conditions can only reference the evidence variables and use
    # comparison/boolean/membership operators.
    return bool(eval(condition, {"__builtins__": {}}, namespace))  # noqa: S307


def evaluate_rules(namespace: dict[str, Any], path: str | None = None) -> list[dict[str, Any]]:
    """Return the rules whose condition fires for this transaction.

    A rule that fails to evaluate (e.g. references a missing variable) is
    skipped rather than crashing the review request.
    """
    fired: list[dict[str, Any]] = []
    for rule in load_rules(path):
        try:
            if _safe_eval(rule["condition"], namespace):
                fired.append(
                    {
                        "rule_id": rule["id"],
                        "action": rule["action"],
                        "description": " ".join(rule.get("description", "").split()),
                    }
                )
        except Exception:
            continue
    return fired


def allowed_actions(path: str | None = None) -> set[str]:
    """Set of action strings the assist layer is permitted to recommend."""
    return {r["action"] for r in load_rules(path)}


def allowed_rule_ids(path: str | None = None) -> set[str]:
    return {r["id"] for r in load_rules(path)}
