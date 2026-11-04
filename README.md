# GeneLens 🧬
### Differential Expression · GO Enrichment · KEGG Pathways · ML Classification · AI Interpretation

A complete bioinformatics pipeline — from raw RNA-seq count matrices to pathway-level biological interpretation.

<img width="1919" height="949" alt="Image" src="https://github.com/user-attachments/assets/e5665890-5e5b-4216-9320-1a332603526f" />

---

## 🔬 Live App
👉 **[genelens.streamlit.app](https://genelens.streamlit.app)**

---

## What GeneLens Does

| Feature | Method |
|---------|--------|
| Differential Expression | Welch's t-test + Benjamini-Hochberg FDR correction |
| Volcano Plot | Interactive — labelled gene names, threshold lines |
| Heatmap | Z-scored expression across all samples |
| PCA | Sample clustering to verify group separation |
| GO Enrichment | GO_Biological_Process_2021 via Enrichr API |
| KEGG Pathways | KEGG_2021_Human via Enrichr API |
| ML Classification | Random Forest (200 trees, 5-fold CV, AUC + feature importance) |
| AI Interpretation | Anthropic Claude API — automated biological interpretation |

> GO and KEGG enrichment works for datasets with real gene names (e.g. SOD2, TP53) —
> or IDs GeneLens can map to gene symbols automatically (see below).

---

## Gene ID Mapping (New in v1.0)

Uploaded or GEO-fetched datasets aren't always keyed by HGNC gene symbol — RNA-seq
pipelines commonly output Ensembl gene IDs (`ENSG...`), and some platforms use Entrez,
RefSeq, or UniProt IDs instead. GeneLens now detects the ID format automatically and
maps it to HGNC symbols using two local, static reference tables (no network calls at
analysis time):

- **`hgnc_mapping.tsv`** — NCBI `gene_info`, covering **193,877 genes** with Ensembl,
  Entrez, RefSeq, and UniProt cross-references
- **`ensembl_to_symbol.tsv`** — a BioMart supplement (**49,132 entries**) filling in
  lncRNA and other non-coding genes NCBI's table misses

This is what lets a raw Ensembl-ID RNA-seq matrix (see the COVID-19 dataset below,
60,683 raw IDs) get GO/KEGG enrichment at all — the mapping step happens automatically
before differential expression, and the app reports how many genes mapped vs. were
dropped.

---

## Validated on Real Published Data

### Dataset 1 — COVID-19 PBMC RNA-seq (GSE152418, *Science* 2020)

**Scale:** 44,242 genes × 34 samples | 17 COVID-19 patients vs. 17 healthy controls

> GEO's raw matrix ships 60,683 Ensembl gene IDs. GeneLens maps Ensembl → HGNC symbols
> automatically (see [Gene ID Mapping](#gene-id-mapping-new-in-v10) above); after
> dropping unmapped and duplicate IDs, 44,242 genes remain for analysis.

<img width="959" height="474" alt="Volcano plot — GSE152418 differential expression" src="https://github.com/user-attachments/assets/c3f17865-3161-43f2-b139-e6a2688458e9" />

*3,200 upregulated · 2,216 downregulated*

<img width="959" height="476" alt="PCA — GSE152418 sample clustering" src="https://github.com/user-attachments/assets/fd47d0de-f499-4f62-9faf-d3c58d4ca24a" />

*Clean separation of COVID-19 patients from healthy controls on PC1*

**Key findings:**
- `CDC20`, `FOXM1`, `RRM2` downregulated → immune cell cycle arrest
- `CYP1B1` upregulated → oxidative stress response
- `TGFBR2`, `SETDB1` upregulated → immune evasion and epigenetic reprogramming

**ML classification (Random Forest, GeneLens defaults):** top 30 differentially expressed genes as features; 200 trees, max depth 5, balanced class weights; 5-fold stratified cross-validation (ROC AUC). Reproduced locally on the published GSE152418 raw count matrix (`GSE152418_p20047_Study1_RawCounts.txt.gz`) after Ensembl→symbol mapping, with adj. p-value < 0.05 and |FC| > 1.5 matching the DE panel above.

- **5-fold CV AUC:** 1.000 ± 0.000
- **Top feature-importance genes:** `TGFBR2`, `RNF38`, `ESPL1`, `RNF111`, `ZCCHC3`, `ASPM`, `TK1`, `CSRNP2`

---

### Dataset 2 — Type 2 Diabetes Liver — Workflow Demonstration

**Purpose:** Demonstration of the full GeneLens analysis workflow and biological interpretation capabilities, connecting RNA-seq analysis to oxidative stress pathways studied in my undergraduate thesis.

**Structure:** 226 curated metabolic and oxidative stress genes × 13 samples | 5 Normal vs. 8 T2DM conditions

**Note:** This dataset was constructed as a biologically informed demonstration using gene expression directions and magnitudes derived from published T2DM transcriptomics literature (including Pihlajamäki et al. 2009, GSE15653). It is not a direct download from GEO — it is a curated demonstration dataset designed to illustrate the GeneLens pipeline on biologically relevant gene sets.

<img width="1918" height="877" alt="Heatmap — T2DM demonstration dataset" src="https://github.com/user-attachments/assets/d2528e19-c859-4b4f-a5ef-e8ab85319d89" />

*GPX1, GCLM, GSR downregulated — antioxidant enzymes from thesis research*

<img width="1919" height="943" alt="GO enrichment — T2DM demonstration" src="https://github.com/user-attachments/assets/1f415041-4918-4d2a-932d-954a9172ff04" />

*Glutathione metabolic process — enriched among downregulated genes*

<img width="1919" height="948" alt="KEGG pathways — T2DM demonstration" src="https://github.com/user-attachments/assets/e79afb9c-c05f-4051-a77d-dbd09504edc8" />


**Key findings from demonstration:**
- `GPX1`, `GCLM`, `GSR` downregulated — antioxidant suppression
- `SOD2`, `CAT` strong downward trends — consistent with thesis biochemical findings
- `CYP2E1` upregulated → ROS generation
- **Top GO term:** Glucose homeostasis (GO:0042593)
- **Top KEGG pathway:** Insulin signalling (p=1.86e-08)

> 📁 Full case study and methods: [genelens-t2dm-study](https://github.com/dev-essie/genelens-t2dm-study)

---

## Supported File Formats

| Format | Notes |
|--------|-------|
| `.csv` | Standard comma-separated count matrix |
| `.tsv` | Tab-separated count matrix |
| `.txt` | GEO Series Matrix files — auto-parsed |

**Expected structure:** rows = genes, columns = samples, first column = gene names.

---

## How to Run Locally

```bash
git clone https://github.com/dev-essie/GeneLens.git
cd GeneLens
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Project Structure

```
GeneLens/
├── app.py                      ← Main Streamlit app (incl. count-matrix file parsing)
├── requirements.txt
├── packages.txt                ← apt deps (Chromium, for chart/PDF export on Streamlit Cloud)
├── runtime.txt / .python-version
├── LICENSE · CITATION.cff
├── .streamlit/config.toml      ← Dark theme
├── data/
│   ├── hgnc_mapping.tsv        ← NCBI gene_info: Ensembl/Entrez/RefSeq/UniProt → HGNC symbol
│   └── ensembl_to_symbol.tsv   ← BioMart supplement (lncRNA / non-coding genes)
└── utils/
    ├── de_analysis.py          ← Welch t-test + BH FDR
    ├── gene_mapper.py          ← Detects & maps gene IDs to HGNC symbols
    ├── geo_loader.py           ← NCBI GEO fetch (GSM tables + supplementary-matrix fallback)
    ├── visualisations.py       ← Volcano, heatmap, PCA, bar chart
    ├── ml_classifier.py        ← Random Forest + ROC curve
    ├── pathway_enrichment.py   ← GO and KEGG via Enrichr API
    ├── ai_interpretation.py    ← Claude API interpretation
    ├── report_generator.py     ← PDF report export (charts, tables, AI interpretation)
    └── data_generator.py       ← Demo dataset generator
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Frontend | Streamlit, Plotly |
| Analysis | Python, SciPy, pandas, NumPy |
| Pathway Enrichment | gseapy (Enrichr API) |
| Machine Learning | scikit-learn |
| AI Interpretation | Anthropic Claude API |
| Deployment | Streamlit Cloud |

---

## Research Interests & Next Steps

GeneLens is an active research-oriented project, not a finished product. Planned and exploratory directions include:

- Applying ML on bulk RNA-seq and other transcriptomic data across additional disease contexts, with emphasis on reproducible benchmarks on public GEO series
- Validating GeneLens outputs on more independent datasets and comparing results to published differential-expression tables where available
- Exploring statistically richer DE models (e.g. negative-binomial / DESeq2- or limma-voom-style workflows) alongside the current Welch t-test implementation
- Comparing classifiers beyond Random Forest (e.g. regularized logistic regression, gradient boosting) and reporting calibration as well as AUC
- Extending toward single-cell or pseudobulk workflows where sample metadata and matrix formats allow
- Connecting findings to metabolic and infectious disease relevance in Nigeria and broader African health contexts, in collaboration with domain experts

---

## Related Research

📁 **T2DM Workflow Case Study:** [github.com/dev-essie/genelens-t2dm-study](https://github.com/dev-essie/genelens-t2dm-study)

---

## How to Cite

If you use GeneLens in a presentation, thesis, or publication, please cite the software record in [`CITATION.cff`](CITATION.cff). Example:

> Alabi, E. O. (2026). *GeneLens: A web application for RNA-seq differential expression analysis, pathway enrichment, and ML classification* (Version 1.0.0) [Computer software]. https://github.com/dev-essie/GeneLens

---

## Built By

**Alabi Esther Oluwatimilehin**  
BSc Biochemistry — First Class Honours, University of Medical Sciences, Nigeria  
[github.com/dev-essie](https://github.com/dev-essie) · [devessie.xyz](https://devessie.xyz/) · [linkedin.com/in/alabi-esther-essie](https://linkedin.com/in/alabi-esther-essie)

---
*MIT License*
