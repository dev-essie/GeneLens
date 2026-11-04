"""Reproduce GSE152418 ML cross-validation metrics (leakage-free pipeline)."""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _Stub:
    def cache_data(self, **kw):
        return lambda fn: fn


sys.modules.setdefault("streamlit", _Stub())
import streamlit as st  # noqa: E402

st.cache_data = lambda **kw: (lambda fn: fn)

from utils.gene_mapper import map_to_gene_symbol  # noqa: E402
from utils.de_analysis import run_differential_expression  # noqa: E402
from utils.ml_classifier import run_ml_classification  # noqa: E402

COUNTS_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE152nnn/GSE152418/suppl/"
    "GSE152418_p20047_Study1_RawCounts.txt.gz"
)
CACHE = Path("/tmp/GSE152418_p20047_Study1_RawCounts.txt.gz")


def load_counts() -> pd.DataFrame:
    if not CACHE.exists():
        import requests

        resp = requests.get(COUNTS_URL, timeout=120)
        resp.raise_for_status()
        CACHE.write_bytes(resp.content)
    with gzip.open(CACHE, "rt") as f:
        return pd.read_csv(f, sep="\t", index_col=0)


def app_default_split(cols: list[str]) -> tuple[list[str], list[str]]:
    """Same 50/50 fallback as app.py when column names lack group keywords."""
    _ctrl_kw = {"normal", "control", "ctrl", "wt", "healthy", "untreated", "vehicle", "mock", "neg"}
    _treat_kw = {
        "treated", "treatment", "disease", "ko", "knockdown", "cancer", "tumor",
        "diabetes", "t2d", "mutant", "infected", "patient", "case", "high", "low",
    }
    control = [c for c in cols if any(k in c.lower() for k in _ctrl_kw)]
    treated = [c for c in cols if any(k in c.lower() for k in _treat_kw)]
    if not control and not treated:
        control = cols[: len(cols) // 2]
        treated = cols[len(cols) // 2 :]
    return control, treated


def main() -> None:
    df = load_counts()
    control_cols, treated_cols = app_default_split(list(df.columns))
    mapped, _, _ = map_to_gene_symbol(df)
    results = run_differential_expression(
        mapped, control_cols, treated_cols, already_log2=False
    )
    ml = run_ml_classification(
        mapped,
        results,
        control_cols,
        treated_cols,
        already_log2=False,
    )
    print(f"Samples: {len(control_cols)} control + {len(treated_cols)} treated")
    print(f"Genes after mapping: {mapped.shape[0]}")
    print(f"5-fold CV AUC: {ml['accuracy']:.3f} ± {ml['std']:.3f}")
    print("Per-fold:", [round(float(x), 4) for x in ml["cv_scores"]])
    print("Top feature-importance genes (full-data model):")
    for _, row in ml["feature_importances"].head(8).iterrows():
        print(f"  {row['Gene']}: {row['Importance']:.4f}")


if __name__ == "__main__":
    main()
