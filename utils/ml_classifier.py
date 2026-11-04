import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline

from utils.de_analysis import _bh_correction


COLORS = {
    "upregulated": "#E05C5C",
    "downregulated": "#5C8FE0",
    "accent": "#7C6AF7",
    "background": "#0F1117",
    "surface": "#1A1D27",
    "text": "#E8EAF0",
}


class FoldInternalDESelector(BaseEstimator, TransformerMixin):
    """
    Select top-k genes using Welch t-test + BH FDR on the training fold only
    (same rules as utils.de_analysis significance labels).
    """

    def __init__(
        self,
        k: int = 30,
        pval_threshold: float = 0.05,
        fc_threshold: float = 1.5,
    ):
        self.k = k
        self.pval_threshold = pval_threshold
        self.fc_threshold = fc_threshold

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        control = X[y == 0]
        treated = X[y == 1]
        if control.shape[0] < 2 or treated.shape[0] < 2:
            n_features = X.shape[1]
            k = min(self.k, n_features)
            self.selected_indices_ = np.arange(k)
            return self

        _, p_values = stats.ttest_ind(treated, control, axis=0, equal_var=False)
        p_values = np.nan_to_num(p_values, nan=1.0)
        adj_p = _bh_correction(p_values)
        log2_fc = treated.mean(axis=0) - control.mean(axis=0)
        log2_fc_thresh = np.log2(self.fc_threshold)

        sig_mask = (adj_p < self.pval_threshold) & (np.abs(log2_fc) > log2_fc_thresh)
        order = np.argsort(adj_p)

        selected = []
        for idx in order:
            if sig_mask[idx]:
                selected.append(idx)
            if len(selected) >= self.k:
                break
        if len(selected) < self.k:
            for idx in order:
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= self.k:
                    break

        self.selected_indices_ = np.array(selected[: self.k], dtype=int)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.selected_indices_]


def _build_expression_matrix(
    df: pd.DataFrame,
    control_cols: list,
    treated_cols: list,
    already_log2: bool,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Samples × genes matrix (log2), labels, gene names."""
    all_cols = control_cols + treated_cols
    values = df.loc[:, all_cols].values.astype(float)
    if already_log2:
        log2_values = values
    else:
        log2_values = np.log2(values + 1.0)
    X = log2_values.T
    y = np.array([0] * len(control_cols) + [1] * len(treated_cols))
    gene_names = list(df.index)
    return X, y, gene_names


def _make_ml_pipeline(
    top_n_features: int,
    pval_threshold: float,
    fc_threshold: float,
    random_state: int,
) -> Pipeline:
    return Pipeline([
        (
            "selector",
            FoldInternalDESelector(
                k=top_n_features,
                pval_threshold=pval_threshold,
                fc_threshold=fc_threshold,
            ),
        ),
        ("scaler", StandardScaler()),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                random_state=random_state,
                class_weight="balanced",
            ),
        ),
    ])


def run_ml_classification(
    df: pd.DataFrame,
    results: pd.DataFrame,
    control_cols: list,
    treated_cols: list,
    top_n_features: int = 30,
    n_splits: int = 5,
    random_state: int = 42,
    pval_threshold: float = 0.05,
    fc_threshold: float = 1.5,
    already_log2: bool = False,
) -> dict:
    """
    Train a Random Forest classifier to distinguish Control vs Treated.

    Cross-validated AUC uses in-fold Welch DE feature selection (no label leakage
    into gene picking). ROC curve and feature importances come from a final model
    refit on all samples (exploratory visualization only).

    The ``results`` DataFrame is kept for API compatibility with the Streamlit app
    but is not used to choose ML features.

    Returns:
        dict with: accuracy, std, feature_importances, roc_fig, cv_scores
    """
    del results  # global DE table must not drive CV features

    if df.index.duplicated().any():
        df = df.groupby(df.index).mean()

    X, y, gene_names = _build_expression_matrix(
        df, control_cols, treated_cols, already_log2
    )
    gene_names_arr = np.array(gene_names)

    pipeline = _make_ml_pipeline(
        top_n_features, pval_threshold, fc_threshold, random_state
    )

    n_cls_min = min(len(control_cols), len(treated_cols))
    if n_cls_min >= 2:
        cv = StratifiedKFold(
            n_splits=min(n_splits, n_cls_min),
            shuffle=True,
            random_state=random_state,
        )
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    else:
        cv_scores = np.array([float("nan")])

    pipeline.fit(X, y)
    selector = pipeline.named_steps["selector"]
    rf_model = pipeline.named_steps["rf"]
    selected_genes = gene_names_arr[selector.selected_indices_]

    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Gene": selected_genes,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    X_selected = selector.transform(X)
    X_scaled = pipeline.named_steps["scaler"].transform(X_selected)
    probs = rf_model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)

    roc_fig = _plot_roc(fpr, tpr, roc_auc, float(np.nanmean(cv_scores)))
    importance_fig = _plot_feature_importance(feature_importance_df.head(20))

    return {
        "accuracy": float(np.nanmean(cv_scores)),
        "std": float(np.nanstd(cv_scores)),
        "cv_scores": cv_scores,
        "roc_auc": roc_auc,
        "feature_importances": feature_importance_df,
        "roc_fig": roc_fig,
        "importance_fig": importance_fig,
        "n_features": len(selected_genes),
    }


def _plot_roc(fpr, tpr, roc_auc: float, cv_auc: float) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"Full-data fit (AUC = {roc_auc:.3f})",
        line=dict(color=COLORS["accent"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(124,106,247,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random (AUC = 0.5)",
        line=dict(color=COLORS["text"], width=1, dash="dash"),
        opacity=0.4,
    ))
    fig.update_layout(
        plot_bgcolor=COLORS["surface"],
        paper_bgcolor=COLORS["background"],
        font=dict(family="Courier New, monospace", color=COLORS["text"]),
        title=dict(
            text=f"ROC (full-data fit)  |  5-fold CV AUC: {cv_auc:.3f}",
            font=dict(size=15), x=0.5
        ),
        xaxis=dict(title="False Positive Rate", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="True Positive Rate", gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(bgcolor="rgba(255,255,255,0.05)"),
        height=420,
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def _plot_feature_importance(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Gene"],
        orientation="h",
        marker=dict(
            color=df["Importance"],
            colorscale=[[0, COLORS["downregulated"]], [0.5, COLORS["accent"]], [1, COLORS["upregulated"]]],
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor=COLORS["surface"],
        paper_bgcolor=COLORS["background"],
        font=dict(family="Courier New, monospace", color=COLORS["text"]),
        title=dict(
            text="Feature Importance — Full-Data Model (exploratory)",
            font=dict(size=15),
            x=0.5,
        ),
        xaxis=dict(title="Mean Decrease in Impurity", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(autorange="reversed"),
        height=480,
        margin=dict(l=120, r=40, t=60, b=60),
    )
    return fig
