"""Tests for leakage-free ML evaluation pipeline."""

import numpy as np
import pandas as pd

from utils.ml_classifier import FoldInternalDESelector, run_ml_classification


def test_fold_selector_uses_training_labels_only():
    rng = np.random.default_rng(42)
    n_genes = 200
    n_per_class = 10
    X0 = rng.normal(0, 1, (n_per_class, n_genes))
    X1 = rng.normal(0, 1, (n_per_class, n_genes))
    X1[:, :5] += 4.0
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    sel = FoldInternalDESelector(k=5, pval_threshold=0.05, fc_threshold=1.5)
    sel.fit(X, y)
    assert len(sel.selected_indices_) == 5
    assert all(idx < 5 for idx in sel.selected_indices_)


def test_cv_pipeline_runs_on_synthetic_matrix():
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(50)]
    cols_ctrl = [f"C{i}" for i in range(6)]
    cols_trt = [f"T{i}" for i in range(6)]
    data = rng.poisson(30, size=(50, 12)).astype(float)
    df = pd.DataFrame(data, index=genes, columns=cols_ctrl + cols_trt)
    results = pd.DataFrame({"Gene": genes, "significance": ["Not significant"] * 50})
    out = run_ml_classification(df, results, cols_ctrl, cols_trt, top_n_features=10, n_splits=3)
    assert out["n_features"] == 10
    assert not np.isnan(out["accuracy"])


def test_global_de_table_not_used_for_features():
    rng = np.random.default_rng(1)
    genes = [f"G{i}" for i in range(80)]
    cols_ctrl = [f"C{i}" for i in range(8)]
    cols_trt = [f"T{i}" for i in range(8)]
    data = rng.poisson(20, size=(80, 16)).astype(float)
    data[:3, 8:] += 80
    df = pd.DataFrame(data, index=genes, columns=cols_ctrl + cols_trt)
    misleading = pd.DataFrame({
        "Gene": genes[40:70],
        "significance": ["Upregulated"] * 30,
    })
    out = run_ml_classification(
        df, misleading, cols_ctrl, cols_trt, top_n_features=5, n_splits=3
    )
    top = out["feature_importances"]["Gene"].head(3).tolist()
    assert any(g in {"G0", "G1", "G2"} for g in top)
