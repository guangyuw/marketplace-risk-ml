"""Business thresholds, monitoring, and routing logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AUDIT_FRACTION, C_FALSE_CLEAR, C_MANUAL_REVIEW, RANDOM_STATE


def automation_vs_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Trade automation rate vs false-clear (auto-approve high-risk) rate."""
    y_true = np.asarray(y_true)
    rows = []
    for t in thresholds:
        auto = y_prob < t
        n_auto = int(auto.sum())
        fcr = float(y_true[auto].mean()) if n_auto else 0.0
        rows.append(
            {
                "threshold": round(float(t), 4),
                "automation_rate": n_auto / len(y_true),
                "false_clear_rate": fcr,
            }
        )
    return pd.DataFrame(rows)


def choose_threshold_by_tolerance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    tolerance: float,
    grid: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    grid = grid if grid is not None else np.linspace(0.01, 0.5, 50)
    tbl = automation_vs_error(y_true, y_prob, grid)
    ok = tbl[tbl["false_clear_rate"] <= tolerance]
    if len(ok):
        chosen = ok.sort_values("automation_rate").iloc[-1]
        return float(chosen.threshold), tbl
    fallback = tbl.iloc[0]
    return float(fallback.threshold), tbl


def expected_cost(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    y_true = np.asarray(y_true)
    auto = y_prob < threshold
    false_clear = int((auto & (y_true == 1)).sum())
    manual = int((~auto).sum())
    return false_clear * C_FALSE_CLEAR + manual * C_MANUAL_REVIEW


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index for score / feature drift monitoring."""
    q = np.quantile(expected, np.linspace(0, 1, bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    e = np.histogram(expected, q)[0] / len(expected)
    a = np.histogram(actual, q)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def route_transactions(
    y_prob: np.ndarray,
    threshold: float,
    audit_fraction: float = AUDIT_FRACTION,
    seed: int | None = None,
) -> np.ndarray:
    """Production routing with random audit to mitigate selective-label bias.

    seed should be None in production so audit sampling is truly random.
    Pass an integer seed only in tests for reproducibility.
    """
    rng = np.random.default_rng(seed)
    auto = y_prob < threshold
    audit = auto & (rng.random(len(y_prob)) < audit_fraction)
    return np.where(audit, "audit", np.where(auto, "auto_approve", "manual_review"))
