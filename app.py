import streamlit as st
import pandas as pd
import numpy as np
import io
import re


def parse_count_matrix(file, already_log2: bool) -> pd.DataFrame:
    """Parse a gene expression matrix from CSV, TSV, or TXT.

    Detects separator from extension, then falls back to the other common
    delimiter if the first attempt produces only one column. Skips lines
    starting with '#' (common in GEO supplementary text files).
    """
    name = file.name.lower()
    raw = file.read()

    primary_sep = "\t" if (name.endswith(".tsv") or name.endswith(".txt")) else ","
    fallback_sep = "," if primary_sep == "\t" else "\t"

    df = None
    for sep in (primary_sep, fallback_sep):
        try:
            candidate = pd.read_csv(
                io.BytesIO(raw), sep=sep, index_col=0, comment="#"
            )
            if candidate.shape[1] >= 1:
                df = candidate
                break
        except Exception:
            continue

    if df is None:
        raise ValueError(
            "Could not parse the file as CSV or TSV. "
            "Ensure the first column contains gene names and remaining columns are samples."
        )

    df.index.name = "Gene"
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    if not already_log2:
        df = df.astype(int)
    return df

from utils.geo_loader import EXAMPLE_DATASETS, fetch_geo_dataset
from utils.de_analysis import run_differential_expression, get_summary_stats, get_top_degs
from utils.visualisations import plot_volcano, plot_heatmap, plot_pca, plot_deg_bar
from utils.ml_classifier import run_ml_classification
from utils.ai_interpretation import generate_ai_interpretation
from utils.pathway_enrichment import run_pathway_enrichment, plot_go_bar, plot_kegg_bar
from utils.gene_mapper import map_to_gene_symbol
from utils.report_generator import generate_pdf_report, ensure_kaleido_chrome, render_chart_png


def chart_download_button(fig, file_name: str, key: str, width: int = 1000, height: int = 600):
    """Small PNG download button for a single chart, so it can be reused outside the report."""
    ensure_kaleido_chrome()
    try:
        png = render_chart_png(fig, width=width, height=height, scale=2)
    except Exception:
        return
    st.download_button("⬇️ PNG", data=png, file_name=file_name, mime="image/png", key=key)


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GeneLens",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a0c14 0%, #0f1117 50%, #0d1020 100%);
}

/* Header */
.main-header {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7C6AF7, #5C8FE0, #E05C5C);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    letter-spacing: -1px;
}

.sub-header {
    font-family: 'Space Mono', monospace;
    color: #6B7280;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
}

.metric-label {
    color: #6B7280;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.2rem;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #7C6AF7;
    letter-spacing: 1px;
    border-bottom: 1px solid rgba(124,106,247,0.3);
    padding-bottom: 0.5rem;
    margin: 2rem 0 1rem 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(10,12,20,0.9) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Buttons */
.stButton > button {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 1px;
    background: linear-gradient(135deg, #7C6AF7, #5C8FE0);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.4rem;
    transition: all 0.2s ease;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(124,106,247,0.4);
}

/* AI box */
.ai-box {
    background: linear-gradient(135deg, rgba(124,106,247,0.08), rgba(92,143,224,0.08));
    border: 1px solid rgba(124,106,247,0.25);
    border-radius: 12px;
    padding: 1.5rem;
    font-size: 0.92rem;
    line-height: 1.8;
    color: #D1D5DB;
}

/* Table */
.dataframe {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.06);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
if "geo_trigger" not in st.session_state:
    st.session_state.geo_trigger = ""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧬 GeneLens")
    st.markdown("<small style='color:#6B7280'>Gene Expression Analysis Tool</small>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**📂 Data Input**")
    data_source = st.radio(
        "Source",
        ["🌐 GEO Database", "📁 Upload File"],
        label_visibility="collapsed",
    )

    uploaded_file = None

    if data_source == "🌐 GEO Database":
        st.markdown("<small style='color:#9CA3AF'>Enter a GEO accession (e.g. GSE96870)</small>", unsafe_allow_html=True)
        geo_accession = st.text_input(
            "GEO Accession",
            value=st.session_state.geo_trigger,
            placeholder="GSExxxxxx",
            label_visibility="collapsed",
        )
        load_geo_btn = st.button("Load from GEO")

        st.markdown("<small style='color:#6B7280'>— or pick a real dataset —</small>", unsafe_allow_html=True)
        example_choice = st.selectbox(
            "Example datasets",
            [""] + [f"{d['accession']}  —  {d['title']}" for d in EXAMPLE_DATASETS],
            format_func=lambda x: "Select an example…" if x == "" else x,
            label_visibility="collapsed",
        )
        if example_choice:
            st.session_state.geo_trigger = example_choice.split("  —  ")[0].strip()
            st.rerun()

    else:
        uploaded_file = st.file_uploader(
            "Upload count matrix (CSV, TSV, or TXT)",
            type=["csv", "tsv", "txt"],
            help="Rows = genes, columns = samples. First column must be gene names. Tab or comma separated.",
            label_visibility="collapsed",
        )
        st.markdown("<small style='color:#9CA3AF'>Format: Gene | Sample1 | Sample2 … (.csv / .tsv / .txt)</small>", unsafe_allow_html=True)

    already_log2 = st.checkbox(
        "Pre-normalized data (microarray / TPM / FPKM)",
        value=False,
        help=(
            "Check for microarray intensities or normalized RNA-seq (TPM, FPKM, log2). "
            "Leave unchecked for raw RNA-seq read counts."
        ),
    )

    st.markdown("---")
    st.markdown("**⚙️ Analysis Parameters**")
    pval_threshold = st.slider("Adj. p-value threshold", 0.01, 0.10, 0.05, 0.01)
    fc_threshold = st.slider("Fold change threshold", 1.2, 4.0, 1.5, 0.1)
    top_n_heatmap = st.slider("Heatmap top genes", 10, 60, 40, 5)
    label_top_n = st.slider("Volcano gene labels", 5, 30, 15, 5)

    st.markdown("---")
    st.markdown("**🤖 AI Interpretation**")
    api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
    run_ai = st.button("✨ Generate AI Interpretation")

    st.markdown("---")
    st.markdown(
        "<small style='color:#4B5563'>Built by Alabi Esther · GitHub: ESTIE-CREATOR</small>",
        unsafe_allow_html=True
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">GeneLens</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Differential Expression · ML Classification · AI Interpretation</div>', unsafe_allow_html=True)


# ── Cached computation helpers ────────────────────────────────────────────────
@st.cache_data(show_spinner="Downloading from NCBI GEO — this may take a minute…")
def cached_fetch_geo(accession: str):
    return fetch_geo_dataset(accession)

@st.cache_data(show_spinner=False)
def cached_de_analysis(df, control_t, treated_t, pval_threshold, fc_threshold, already_log2):
    return run_differential_expression(
        df, list(control_t), list(treated_t), pval_threshold, fc_threshold, already_log2
    )

@st.cache_data(show_spinner=False)
def cached_ml(df, results, control_t, treated_t, pval_threshold, fc_threshold, already_log2):
    return run_ml_classification(
        df,
        results,
        list(control_t),
        list(treated_t),
        pval_threshold=pval_threshold,
        fc_threshold=fc_threshold,
        already_log2=already_log2,
    )

@st.cache_data(show_spinner=False)
def cached_pathway_enrichment(results):
    return run_pathway_enrichment(results)


df = None
control_cols = []
treated_cols = []
dataset_label = ""
_gene_id_type = None   # tracks detected ID format for GO/KEGG gating

if data_source == "🌐 GEO Database":
    active_accession = st.session_state.geo_trigger or (geo_accession if load_geo_btn else "")
    if active_accession:
        try:
            df, sample_info, is_counts = cached_fetch_geo(active_accession.strip().upper())
            if not already_log2 and is_counts:
                pass  # raw counts — DE analysis will log2-transform
            df_mapped, _gene_id_type, _map_stats = map_to_gene_symbol(df)
            if _gene_id_type is not None:
                if _map_stats["n_mapped"] > 0:
                    st.info(
                        f"Detected **{_gene_id_type.replace('_', ' ')}** IDs — "
                        f"mapped {_map_stats['n_mapped']:,} genes to HGNC symbols "
                        f"({_map_stats['n_dropped']:,} dropped, no match in local table)."
                    )
                    df = df_mapped
                else:
                    st.warning(
                        f"Detected **{_gene_id_type.replace('_', ' ')}** IDs but none matched "
                        "the local gene symbol table. GO/KEGG enrichment will be unavailable. "
                        "DE analysis will proceed with the original IDs."
                    )
            dataset_label = f"**{active_accession}** — {len(df):,} genes × {len(df.columns)} samples"
        except Exception as exc:
            st.error(f"Could not load {active_accession}: {exc}")
            st.stop()
    else:
        st.info(
            "**Getting started with real data**\n\n"
            "Pick one of the example datasets from the sidebar, or enter any "
            "[NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/) accession number (GSExxxxxx).\n\n"
            "For RNA-seq datasets whose counts are in supplementary files, download the "
            "count matrix CSV from GEO and use **Upload File** instead."
        )
        st.stop()

elif data_source == "📁 Upload File":
    if uploaded_file is not None:
        try:
            df = parse_count_matrix(uploaded_file, already_log2)
            df_mapped, _gene_id_type, _map_stats = map_to_gene_symbol(df)
            if _gene_id_type is not None:
                if _map_stats["n_mapped"] > 0:
                    st.info(
                        f"Detected **{_gene_id_type.replace('_', ' ')}** IDs — "
                        f"mapped {_map_stats['n_mapped']:,} genes to HGNC symbols "
                        f"({_map_stats['n_dropped']:,} dropped, no match in local table)."
                    )
                    df = df_mapped
                else:
                    st.warning(
                        f"Detected **{_gene_id_type.replace('_', ' ')}** IDs but none matched "
                        "the local gene symbol table. GO/KEGG enrichment will be unavailable. "
                        "DE analysis will proceed with the original IDs."
                    )
            dataset_label = f"**{uploaded_file.name}** — {len(df):,} genes × {len(df.columns)} samples"
        except Exception as exc:
            st.error(f"Error loading file: {exc}")
            st.stop()
    else:
        st.info(
            "**Upload a count matrix**\n\n"
            "Download a count matrix CSV from [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/) "
            "or [ArrayExpress](https://www.ebi.ac.uk/arrayexpress/) and upload it here.\n\n"
            "**Format:** rows = genes, columns = samples. First column must be gene names. "
            "Accepts `.csv`, `.tsv`, and `.txt` (tab or comma separated)."
        )
        st.stop()

# ── Sample Assignment ─────────────────────────────────────────────────────────
st.success(f"Loaded {dataset_label}")

all_cols = list(df.columns)

# Auto-detect groups from column name keywords
_ctrl_kw  = {"normal", "control", "ctrl", "wt", "healthy", "untreated", "vehicle", "mock", "neg"}
_treat_kw = {"treated", "treatment", "disease", "ko", "knockdown", "cancer", "tumor",
             "diabetes", "t2d", "mutant", "infected", "patient", "case", "high", "low"}
auto_control = [c for c in all_cols if any(k in c.lower() for k in _ctrl_kw)]
auto_treated  = [c for c in all_cols if any(k in c.lower() for k in _treat_kw)]
# Fall back to 50/50 split only if keyword detection found nothing
if not auto_control and not auto_treated:
    auto_control = all_cols[: len(all_cols) // 2]
    auto_treated  = all_cols[len(all_cols) // 2 :]

col1, col2 = st.columns(2)
with col1:
    control_cols = st.multiselect(
        "Control samples",
        options=all_cols,
        default=auto_control,
        key="control_samples",
        help="Select the columns that represent your control / untreated condition.",
    )
with col2:
    remaining_for_treated = [s for s in all_cols if s not in control_cols]
    treated_cols = st.multiselect(
        "Treated / disease samples",
        options=remaining_for_treated,
        default=[s for s in auto_treated if s in remaining_for_treated],
        key="treated_samples",
        help="Select the columns that represent the treated / disease condition.",
    )

if set(control_cols) & set(treated_cols):
    st.error(
        "⚠️ The same sample cannot be in both groups. "
        "Please check your sample assignments."
    )
    st.stop()

if len(control_cols) < 2 or len(treated_cols) < 2:
    st.warning(
        "⚠️ Each group needs at least 2 samples for statistical analysis."
    )
    st.stop()


# ── Run DE Analysis ───────────────────────────────────────────────────────────
try:
    with st.spinner("Running differential expression analysis..."):
        results = cached_de_analysis(
            df, tuple(control_cols), tuple(treated_cols),
            pval_threshold, fc_threshold, already_log2,
        )
        summary = get_summary_stats(results)
except Exception as _de_err:
    st.error(f"Differential expression analysis failed: {_de_err}")
    st.stop()

# ── Summary Cards ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">SUMMARY</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
cards = [
    (summary["total_genes"], "Total Genes", "#E8EAF0"),
    (summary["upregulated"], "Upregulated", "#E05C5C"),
    (summary["downregulated"], "Downregulated", "#5C8FE0"),
    (summary["not_significant"], "Not Significant", "#6B7280"),
]
for col, (val, label, color) in zip([c1, c2, c3, c4], cards):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color}">{val}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Plots: Volcano + Bar ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">DIFFERENTIAL EXPRESSION</div>', unsafe_allow_html=True)

col_v, col_b = st.columns([2, 1])
with col_v:
    fig_volcano = plot_volcano(results, pval_threshold, fc_threshold, label_top_n)
    st.plotly_chart(fig_volcano, width='stretch')
    chart_download_button(fig_volcano, "genelens_volcano.png", key="dl_volcano")
with col_b:
    fig_bar = plot_deg_bar(results)
    st.plotly_chart(fig_bar, width='stretch')
    chart_download_button(fig_bar, "genelens_deg_bar.png", key="dl_bar", width=500, height=450)


# ── Heatmap ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">HEATMAP</div>', unsafe_allow_html=True)
fig_heatmap = plot_heatmap(df, results, control_cols, treated_cols, top_n=top_n_heatmap)
st.plotly_chart(fig_heatmap, width='stretch')
chart_download_button(fig_heatmap, "genelens_heatmap.png", key="dl_heatmap", width=1000, height=700)


# ── PCA ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">PCA — SAMPLE CLUSTERING</div>', unsafe_allow_html=True)
fig_pca = plot_pca(df, control_cols, treated_cols)
if fig_pca is not None:
    st.plotly_chart(fig_pca, width='stretch')
    chart_download_button(fig_pca, "genelens_pca.png", key="dl_pca", width=800, height=550)
else:
    st.info("PCA requires at least 2 samples in each group.")


# ── ML Classification ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">MACHINE LEARNING CLASSIFICATION</div>', unsafe_allow_html=True)

try:
    with st.spinner("Training Random Forest classifier..."):
        ml_results = cached_ml(
            df,
            results,
            tuple(control_cols),
            tuple(treated_cols),
            pval_threshold,
            fc_threshold,
            already_log2,
        )
    ml1, ml2 = st.columns(2)
    with ml1:
        st.plotly_chart(ml_results["roc_fig"], width='stretch')
        chart_download_button(ml_results["roc_fig"], "genelens_roc.png", key="dl_roc", width=600, height=500)
    with ml2:
        st.plotly_chart(ml_results["importance_fig"], width='stretch')
        chart_download_button(ml_results["importance_fig"], "genelens_feature_importance.png", key="dl_importance", width=650, height=520)
except Exception as _ml_err:
    st.warning(f"ML classification could not run: {_ml_err}")
    ml_results = {"accuracy": 0.0, "std": 0.0, "n_features": 0, "roc_auc": 0.0,
                  "feature_importances": pd.DataFrame({"Gene": [], "Importance": []}),
                  "error": str(_ml_err)}

st.markdown(f"""
<div style='text-align:center; padding:0.8rem; background:rgba(124,106,247,0.08); border-radius:8px; font-family:Space Mono,monospace; color:#D1D5DB; font-size:0.85rem;'>
Cross-Validation AUC: <b style='color:#7C6AF7'>{ml_results['accuracy']:.3f} ± {ml_results['std']:.3f}</b>
&nbsp;|&nbsp; Features used: <b>{ml_results['n_features']}</b> genes
</div>
""", unsafe_allow_html=True)


# ── DE Results Table ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">TOP DE GENES TABLE</div>', unsafe_allow_html=True)

top_degs = get_top_degs(results, n=30)
display_cols = ["Gene", "log2FoldChange", "pvalue", "padj", "significance"]
st.dataframe(
    top_degs[display_cols].style.format({
        "log2FoldChange": "{:.3f}",
        "pvalue": "{:.2e}",
        "padj": "{:.2e}",
    }),
    use_container_width=True,
    height=380
)

# Download button
csv_buffer = io.StringIO()
results[display_cols].to_csv(csv_buffer, index=False)
st.download_button(
    "⬇️ Download Full DE Results (CSV)",
    data=csv_buffer.getvalue(),
    file_name="geneinsight_de_results.csv",
    mime="text/csv"
)


# ── Pathway Enrichment ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">PATHWAY ENRICHMENT — GO & KEGG</div>', unsafe_allow_html=True)

# Determine if gene index is HGNC symbols (required for Enrichr).
# True when: GEO data (already processed), or upload with no ID conversion needed,
# or upload where conversion succeeded (genes no longer look like numeric/ENSG IDs).
_sample_ids = [str(g) for g in df.index[:30]]
_looks_like_symbols = (
    sum(1 for g in _sample_ids if re.match(r'^[A-Za-z][A-Z0-9a-z\-\.]+$', g) and not g.isdigit()) / max(len(_sample_ids), 1) >= 0.70
)

if _looks_like_symbols:
    run_enrichment = st.button(
        "🔬 Run GO & KEGG Enrichment",
        help="Queries the Enrichr database — requires internet connection",
    )
    if run_enrichment:
        with st.spinner("Running GO and KEGG enrichment via Enrichr..."):
            enr_results = cached_pathway_enrichment(results)
        st.session_state["enr_results"] = enr_results

        if enr_results["success"]:
            st.success(
                f"✅ Enrichment complete — {enr_results['n_de_genes']} DE genes analysed "
                f"({enr_results['n_up']} up, {enr_results['n_down']} down)",
                icon="🧬"
            )
            tab_go, tab_kegg = st.tabs(["GO Biological Process", "KEGG Pathways"])
            with tab_go:
                if not enr_results["go_results"].empty:
                    fig_go = plot_go_bar(enr_results["go_results"])
                    if fig_go:
                        st.plotly_chart(fig_go, width='stretch')
                        chart_download_button(fig_go, "genelens_go_enrichment.png", key="dl_go", width=900, height=550)
                    go_display = enr_results["go_results"][["Term", "Overlap", "Adjusted P-value", "Genes"]].copy()
                    go_display["Adjusted P-value"] = go_display["Adjusted P-value"].map("{:.2e}".format)
                    st.dataframe(go_display, use_container_width=True, height=320)
                else:
                    st.info("No significant GO terms found at padj < 0.05 for this gene set.")
            with tab_kegg:
                if not enr_results["kegg_results"].empty:
                    fig_kegg = plot_kegg_bar(enr_results["kegg_results"])
                    if fig_kegg:
                        st.plotly_chart(fig_kegg, width='stretch')
                        chart_download_button(fig_kegg, "genelens_kegg_enrichment.png", key="dl_kegg", width=900, height=550)
                    kegg_display = enr_results["kegg_results"][["Term", "Overlap", "Adjusted P-value", "Genes"]].copy()
                    kegg_display["Adjusted P-value"] = kegg_display["Adjusted P-value"].map("{:.2e}".format)
                    st.dataframe(kegg_display, use_container_width=True, height=320)
                else:
                    st.info("No significant KEGG pathways found at padj < 0.05 for this gene set.")
        else:
            st.info(
                f"Pathway enrichment requires an internet connection to query the Enrichr database. "
                f"Error: {enr_results.get('error', 'Unknown error')}. "
                "The rest of the analysis above is fully available.",
                icon="🌐"
            )
    else:
        st.info("Click **Run GO & KEGG Enrichment** above to query the Enrichr database for pathway hits.")
else:
    st.info(
        "GO & KEGG enrichment is not available for this dataset — gene IDs could not be "
        "converted to HGNC symbols (detected format: "
        f"**{_gene_id_type or 'unknown'}**). "
        "All other analysis features above are fully available.",
        icon="ℹ️"
    )


# ── AI Interpretation ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">AI BIOLOGICAL INTERPRETATION</div>', unsafe_allow_html=True)

interpretation = ""  # populated below if AI runs

if run_ai:
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
    else:
        top_up = results[results["significance"] == "Upregulated"]["Gene"].head(10).tolist()
        top_down = results[results["significance"] == "Downregulated"]["Gene"].head(10).tolist()
        top_features = ml_results["feature_importances"]["Gene"].head(8).tolist()

        with st.spinner("Claude is analysing your results..."):
            try:
                interpretation = generate_ai_interpretation(
                    api_key=api_key,
                    summary_stats=summary,
                    top_up_genes=top_up,
                    top_down_genes=top_down,
                    ml_auc=ml_results["roc_auc"],
                    top_features=top_features,
                )
                st.session_state["interpretation"] = interpretation
                st.markdown(f'<div class="ai-box">{interpretation}</div>', unsafe_allow_html=True)

                report = f"""GeneLens — AI Biological Interpretation
Generated by Claude (Anthropic)
{'='*60}

DATASET
{dataset_label}

ANALYSIS SUMMARY
Total genes: {summary['total_genes']}
Upregulated: {summary['upregulated']}
Downregulated: {summary['downregulated']}
ML Classifier AUC: {ml_results['roc_auc']:.3f}

INTERPRETATION
{interpretation}

TOP UPREGULATED GENES
{', '.join(top_up)}

TOP DOWNREGULATED GENES
{', '.join(top_down)}
"""
                st.download_button(
                    "⬇️ Download Interpretation Report (TXT)",
                    data=report,
                    file_name="geneinsight_interpretation.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"API error: {e}")
else:
    st.markdown("""
    <div class="ai-box" style="color:#6B7280; text-align:center; padding:2rem;">
        🤖 Enter your Anthropic API key in the sidebar and click<br>
        <b style="color:#7C6AF7">✨ Generate AI Interpretation</b> to get a biological analysis of your results.
    </div>
    """, unsafe_allow_html=True)

# ── Export Report ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">EXPORT REPORT</div>', unsafe_allow_html=True)

col_exp1, col_exp2 = st.columns([1, 3])
with col_exp1:
    if st.button("📄 Generate Full Report", help="Creates a PDF report with all charts, tables, and analysis"):
        with st.spinner("Building report..."):
            saved_interp = st.session_state.get("interpretation", "")
            pdf_bytes = generate_pdf_report(
                dataset_label=dataset_label,
                summary=summary,
                results=results,
                ml_results=ml_results,
                top_degs=top_degs,
                fig_volcano=fig_volcano,
                fig_heatmap=fig_heatmap,
                fig_pca=fig_pca,
                fig_bar=fig_bar,
                interpretation=saved_interp,
                enr_results=st.session_state.get("enr_results"),
            )
        st.download_button(
            "⬇️ Download Report (PDF)",
            data=pdf_bytes,
            file_name="genelens_report.pdf",
            mime="application/pdf",
        )
with col_exp2:
    st.markdown(
        "<small style='color:#6B7280'>Print-ready PDF · includes charts as high-res images and full tables · "
        "each chart can also be downloaded individually as PNG above</small>",
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    "<center><small style='color:#374151; font-family:Space Mono,monospace'>GeneLens · Built by Alabi Esther Oluwatimilehin · github.com/ESTIE-CREATOR</small></center>",
    unsafe_allow_html=True
)
