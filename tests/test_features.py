"""Basic tests for feature pipeline."""

import numpy as np

from src.data import generate_synthetic_transactions, temporal_train_test_split
from src.features import build_features, fit_freq_maps, prepare_datasets
from src.monitoring import choose_threshold_by_tolerance, psi, route_transactions


def test_temporal_split_no_overlap():
    df = generate_synthetic_transactions(n=1000)
    train, test = temporal_train_test_split(df)
    assert train["event_date"].max() <= test["event_date"].min()
    assert len(train) + len(test) == len(df)


def test_features_same_columns_train_test():
    df = generate_synthetic_transactions(n=2000)
    train, test = temporal_train_test_split(df)
    x_train, _, x_test, _, _ = prepare_datasets(train, test, "is_high_risk")
    assert list(x_train.columns) == list(x_test.columns)
    assert x_train.shape[1] == 12


def test_unseen_category_maps_to_zero():
    df = generate_synthetic_transactions(n=500)
    train, test = temporal_train_test_split(df)
    freq_maps = fit_freq_maps(train)
    test = test.copy()
    test.loc[0, "section_code"] = "UNSEEN_SECTION"
    feats = build_features(test.iloc[[0]], freq_maps)
    assert feats["section_code_freq"].iloc[0] == 0.0


def test_threshold_selection_respects_tolerance():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.9])
    t, tbl = choose_threshold_by_tolerance(y, p, tolerance=0.5)
    row = tbl.loc[tbl["threshold"] == round(t, 4)].iloc[0]
    assert row["false_clear_rate"] <= 0.5


def test_psi_stable_on_same_distribution():
    x = np.random.default_rng(0).normal(size=1000)
    assert psi(x, x) < 0.01


def test_routing_returns_valid_labels():
    routes = route_transactions(np.array([0.01, 0.5, 0.9]), threshold=0.2, seed=0)
    assert set(routes).issubset({"auto_approve", "manual_review", "audit"})
