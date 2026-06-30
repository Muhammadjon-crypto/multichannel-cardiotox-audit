# Measurement Scarcity Limits Multi-Channel Cardiac Ion-Channel Liability Prediction: A Public-Data Audit

**Authors:** Muhammadjon Tursunbadalov, Mustafojon Tursunbadalov
**Affiliation:** School of Science and Technology, Champions College Prep, Houston, TX, USA

A reproducible audit of whether the public data needed for multi-channel cardiac-safety (cardiotoxicity) machine learning actually exists. We study three CiPA ion channels — **hERG** (KCNH2), **Nav1.5** (SCN5A), and **Cav1.2** (CACNA1C) — and show that the bottleneck for multi-channel modeling is **measurement scarcity**, not channel difficulty.

> This repository accompanies a paper submitted to IEEE BIBM 2026. It is a **data and benchmarking analysis**; it does not introduce a new modeling method.

## Key findings

1. **Scarcity gradient.** hERG: 12,101 compounds (50.1% active); Nav1.5: 2,830 (8.2%); Cav1.2: 463 (6.9%, only 32 actives total).
2. **Overlap ceiling.** Only **90 compounds** are measured against all three channels — a true per-compound multi-channel dataset barely exists.
3. **Volume bounds performance.** A controlled experiment that starves the hERG model of data drops ROC-AUC from 0.928 (full) to 0.782 (at Cav1.2-size) to 0.698 (at 120 compounds).
4. **Low-active channels are unmeasurable.** Cav1.2 posts ROC-AUC ≈ 0.92 but with a 95% CI of [0.70, 1.00]; scaffold vs. random splitting changes estimates by ≤ 0.007, so the limit is the number of labeled actives, not train–test similarity.

## Figures

| File | Description |
|---|---|
| `figures/fig5_pipeline.png` | End-to-end workflow |
| `figures/fig2_scarcity.png` | Per-channel compound counts and class balance |
| `figures/fig3_overlap_venn.png` | Cross-channel overlap (90 shared across all three) |
| `figures/fig1_learning_curve.png` | hERG data-starvation learning curve |
| `figures/fig4_ci_forest.png` | Bootstrap 95% CIs for Nav1.5 / Cav1.2 |

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── reproduce_pipeline.py   # full analysis, end to end
│   └── make_figures.py         # regenerate all figures from results
├── data/
│   ├── scarcity_table.csv
│   ├── overlap_counts.csv
│   ├── learning_curve.csv
│   └── split_comparison.csv
├── figures/                    # publication-quality PNGs (600 dpi)
└── paper/
    ├── BIBM2026_SUBMISSION_anonymous.docx   # double-blind submission version
    └── BIBM2026_NAMED_for_repo.docx         # named version
```

## Reproduce

```bash
git clone https://github.com/Muhammadjon-crypto/multichannel-cardiotox-audit.git
cd multichannel-cardiotox-audit
pip install -r requirements.txt
python src/reproduce_pipeline.py     # runs the full pipeline (uses public TDC + ChEMBL data)
python src/make_figures.py           # regenerate the figures
```

Environment: Python 3.12, scikit-learn 1.2.2, NumPy 1.26.4, RDKit, XGBoost, PyTDC, chembl_webresource_client. Fixed random seed = 42 throughout. The ChEMBL pull requires internet access; it was performed in June 2026.

## Data sources

- **hERG:** Therapeutics Data Commons (`Tox(name='hERG_Karim')` and `Tox(name='hERG')`).
- **Nav1.5 / Cav1.2:** ChEMBL targets `CHEMBL1980` (SCN5A) and `CHEMBL1940` (CACNA1C), IC50 measurements.

## Citation

```bibtex
@inproceedings{tursunbadalov2026scarcity,
  title     = {Measurement Scarcity Limits Multi-Channel Cardiac Ion-Channel
               Liability Prediction: A Public-Data Audit},
  author    = {Tursunbadalov, Muhammadjon and Tursunbadalov, Mustafojon},
  booktitle = {IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year      = {2026}
}
```

## License

MIT — see `LICENSE`.
