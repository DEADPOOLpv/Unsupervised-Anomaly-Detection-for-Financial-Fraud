# Financial Statement Fraud Detection in Indian Listed Companies
## An Explainable Unsupervised Anomaly Detection Framework

---

## Project Overview

The project applies unsupervised machine learning — primarily Isolation Forest with SHAP explainability — to detect financial statement anomalies in Indian listed companies, validated against the Beneish M-Score and SEBI enforcement data. 

---

## Research Question

Can unsupervised anomaly detection methods identify financial statement manipulation in Indian listed companies without requiring pre-labeled fraud data, and can SHAP-based explainability make the detections actionable for auditors and regulators?

---

## Methodology Summary

### Approach: Unsupervised Multi-Algorithm Anomaly Detection + Explainable AI

The project deliberately uses **unsupervised learning** rather than supervised classification because India lacks a centralized, machine-readable fraud enforcement database equivalent to the U.S. SEC's AAERs. This makes reliable binary labels (fraud/non-fraud) unavailable at scale.

### Pipeline Architecture

```
Stage 1: Data Collection
    └─ Screener.in Excel exports (free, sourced from Capitaline)
    └─ 94 non-financial Nifty 200 companies, ~10 years each
    └─ 820 firm-year observations

Stage 2: Feature Engineering (28 features)
    ├─ Beneish M-Score 8 variables (DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA)
    ├─ Extended ratios: profitability (ROA, ROE, margins)
    ├─ Cash flow quality (CFO/NI, CFO/Assets, Accrual Ratio)  ← key fraud signals
    ├─ Leverage (D/E, D/A, Interest Coverage)
    ├─ Liquidity (Current Ratio)
    └─ Growth signals (Revenue, NI, Asset growth)

Stage 3: Multi-Algorithm Anomaly Detection
    ├─ Isolation Forest (Liu, Ting & Zhou, 2008) — primary method
    ├─ Local Outlier Factor (Breunig et al., 2000)
    ├─ One-Class SVM (Schölkopf et al., 2001)
    ├─ Elliptic Envelope (robust covariance)
    └─ Consensus score: count of algorithms flagging each observation

Stage 4: SHAP Explainability
    └─ TreeExplainer on Isolation Forest
    └─ Global feature importance ranking
    └─ Local per-firm explanations (top 5 drivers per anomaly)

Stage 5: Validation
    ├─ Convergent validity: IF anomalies vs Beneish M-Score
    ├─ SEBI enforcement order cross-reference (18 cases compiled)
    └─ Known fraud case backtesting (Zee, Adani entities)
```

### Key Methodological Decisions

1. **Excluded financial sector** (banks, NBFCs, insurance) — fundamentally different balance sheet structure makes ratio-based comparison invalid
2. **Consolidated financials preferred** over standalone — captures subsidiary-level manipulation
3. **Winsorization at 1st/99th percentile** — prevents extreme outliers from dominating IF splits
4. **Contamination parameter = 7%** — balances sensitivity with false positive rate
5. **Industry stratification NOT applied** in current version — noted as limitation and future work

---

## Current Results (as of March 2026)

### Dataset
- 94 companies, 820 firm-year observations, 25 features
- Beneish M-Score successfully computed for 608/820 observations (74.1%)
- M-Score classification: 57 Likely Manipulators, 144 Possible, 407 Unlikely

### Anomaly Detection
- 73 firm-years flagged by 2+ algorithms
- 30 firm-years flagged by 3+ algorithms
- 13 firm-years flagged by all 4 algorithms

### Top Flagged Companies
| Company | Avg Consensus | Years Flagged (2+) | Key SHAP Drivers |
|---|---|---|---|
| Adani Green Energy | 3.33 | 9/9 | Extreme leverage (D/E 4.4x), negative profitability, volatile CFO |
| Godrej Properties | 2.89 | 8/9 | Inventory/Sales 3.2x, high DSRI, negative CFO |
| DLF Ltd | 2.56 | 9/9 | Inventory/Sales 2.9x, unusual AQI, NI volatility |
| IndiaMART InterMesh | 1.78 | 6/9 | Asset-light model creates unusual ratio profiles |
| Dixon Technologies | 0.78 | 2/9 | M-Score +0.162 (highest in dataset — strong manipulation signal) |

### Validation
- **Convergent validity confirmed**: IF anomalies have mean M-Score of -1.510 vs normal firms at -2.394 (higher = closer to manipulation)
- **Beneish Likely Manipulators are 8x more likely** to be flagged by unsupervised methods (22.8% vs 2.7%)
- **Zee Entertainment**: Flagged in 2017, before the 2019 fund diversion scandal became public
- **Adani entities**: Consistently flagged across multiple years, consistent with Hindenburg allegations

### Global Feature Importance (SHAP)
Top 5 drivers of anomaly detection:
1. Asset Turnover (0.144)
2. Operating Margin (0.144)
3. Gross Margin Index — GMI (0.137)
4. CFO to Assets (0.136)
5. Asset Quality Index — AQI (0.132)

---

## File Structure

```
fraud_project/
├── screener_data/                    # Raw Screener.in Excel downloads (94 files)
│   ├── TCS.xlsx
│   ├── Infosys.xlsx
│   └── ... (94 company files)
├── screener_batch_processor_v2.py    # Parses Screener.in files → features CSV
├── fsf_detection_pipeline.py         # Original pipeline (Yahoo Finance version)
├── run_anomaly_detection.py          # Anomaly detection + SHAP on parsed data
├── screener_features.csv             # Output: 820 rows × 34 columns (features)
├── anomaly_results_final.csv         # Output: full results with all scores
├── shap_feature_importance.csv       # Output: global SHAP importance ranking
├── shap_values_matrix.csv            # Output: per-observation SHAP values
├── company_risk_scores.csv           # Output: aggregated company-level risk
├── sebi_fraud_database.csv           # SEBI enforcement cases (18 compiled)
└── README.md                         # This file
```

---

## How to Run

### Prerequisites
```bash
pip install pandas numpy openpyxl xlrd scikit-learn shap
```

### Step 1: Data Collection (Manual)
```bash
# Generate download URLs
py screener_batch_processor_v2.py --urls

# Then manually download Excel files from each URL on Screener.in
# Place all .xlsx files in ./screener_data/
```

### Step 2: Parse and Compute Features
```bash
py screener_batch_processor_v2.py --folder ./screener_data/
# Output: screener_features.csv
```

### Step 3: Run Anomaly Detection
```bash
py run_anomaly_detection.py
# Reads screener_features.csv
# Outputs: anomaly_results_final.csv, shap_feature_importance.csv, etc.
```

---

## Data Sources

| Source | What It Provides | Access |
|---|---|---|
| Screener.in | 10-year financial statements (P&L, BS, CF) for all listed companies. Data sourced from Capitaline. | Free (Export to Excel) |
| SEBI (sebi.gov.in) | Enforcement orders, adjudication orders, debarment orders | Free (public records) |
| NSE India (nseindia.com) | Corporate filings, announcements, governance reports | Free |
| BSE India (bseindia.com) | Corporate filings, shareholding patterns | Free |

**No institutional database access used.** The project demonstrates that publishable financial fraud research is feasible with entirely free data sources.

---

## Screener.in Data Sheet Format (Confirmed)

The parser (v2) uses hard-coded row indices based on confirmed Data Sheet structure:

```
Row  0: COMPANY NAME
Row 14: PROFIT & LOSS
Row 15: Report Date          ← year columns start at col 7+ (datetime objects)
Row 16: Sales
Row 17: Raw Material Cost
Row 18: Change in Inventory
Row 21: Employee Cost
Row 22: Selling and admin
Row 25: Depreciation
Row 26: Interest
Row 27: Profit before tax
Row 29: Net profit
Row 54: BALANCE SHEET
Row 55: Report Date
Row 56: Equity Share Capital
Row 57: Reserves
Row 58: Borrowings
Row 59: Other Liabilities
Row 60: Total (liabilities side)
Row 61: Net Block
Row 62: Capital Work in Progress
Row 63: Investments
Row 64: Other Assets
Row 65: Total (assets side)
Row 66: Receivables
Row 67: Inventory
Row 68: Cash & Bank
Row 79: CASH FLOW:
Row 81: Cash from Operating Activity
Row 82: Cash from Investing Activity
Row 83: Cash from Financing Activity
Row 84: Net Cash Flow
```

Year columns: start at column 7+, stored as datetime objects (e.g., 2024-03-31). Most Indian companies use March year-end; some MNCs use December (ABB, Siemens, etc.).

---

## Sample Composition

### 94 non-financial Nifty 200 companies across 10 sectors:
- IT Services (10): TCS, Infosys, Wipro, HCL Tech, Tech Mahindra, LTIMindtree, Persistent, Coforge, Mphasis, LTTS
- Pharma & Healthcare (10): Sun Pharma, Dr. Reddy's, Cipla, Divi's Lab, Aurobindo, Biocon, Lupin, Torrent, Alkem, Apollo Hospitals
- Auto (10): TMPV (Tata Motors PV), M&M, Maruti, Bajaj Auto, Hero Moto, Eicher, Ashok Leyland, TVS Motor, Bosch, Motherson
- FMCG (10): HUL, ITC, Nestle, Britannia, Dabur, Marico, Godrej Consumer, Colgate, Tata Consumer, Varun Beverages
- Energy (10): Reliance, ONGC, IOC, BPCL, GAIL, NTPC, PowerGrid, Adani Green, Tata Power, JSW Energy
- Metals (8): Tata Steel, JSW Steel, Hindalco, Vedanta, Coal India, NMDC, SAIL, Jindal Steel
- Infrastructure & Cement (10): L&T, Adani Enterprises, Adani Ports, DLF, Godrej Properties, UltraTech, Shree Cement, Ambuja, ACC, Grasim
- Telecom/Media/Internet (3): Bharti Airtel, Zee Entertainment, IndiaMART
- Chemicals (5): Pidilite, SRF, Atul, Deepak Nitrite, Navin Fluorine
- Industrials/Consumer (18): Titan, Trent, Havells, Voltas, Crompton, Siemens, ABB, Cummins, Honeywell, Page Industries, Dixon, Polycab, Bata, V-Guard, Delhivery, CONCOR, Blue Star, Relaxo

### Notable symbol changes:
- **TATAMOTORS → TMPV** (demerger Oct 2025; TMPV carries full 10-year history)
- **NAVIN → NAVINFLUOR** on Screener.in
- **JSW ENERGY → JSWENERGY** (no space on Screener)
- Some MNCs (Nestle, Colgate, Siemens, ABB, Honeywell, Page, Bosch, Cummins, Atul) use **standalone** financials only

---

## Known Limitations

1. **Survivorship bias**: Major fraud companies (Satyam, DHFL, IL&FS) are not in the Nifty 200 sample because they were delisted. The model detects anomalies in the surviving population.

2. **Industry effects**: Real estate companies (DLF, Godrej Properties) are flagged primarily due to high inventory ratios inherent to the sector, not necessarily fraud. Industry-stratified models would reduce false positives.

3. **M-Score incompleteness**: 212/820 observations (25.9%) have insufficient data for M-Score computation, mainly IT services and utilities where raw material cost is zero/missing, killing the Gross Margin Index.

4. **SEBI validation is thin for in-sample companies**: Only 2 of our 94 companies have direct SEBI enforcement actions. Most confirmed fraud cases involve smaller or delisted companies outside the Nifty 200.

5. **No textual features**: The model uses only structured financial ratios. MD&A analysis, auditor reports, and corporate announcements could improve detection but require NLP pipelines beyond current scope.

6. **Temporal effects**: The model pools all years without accounting for regime changes (Ind AS transition 2016-17, COVID 2020-21). Rolling-window or time-fixed-effects approaches are recommended for future work.

7. **LVGI (Leverage Index)** shows zero SHAP importance — this Beneish variable may not discriminate well in the Indian context where leverage structures differ from U.S. companies.

---

## SEBI Fraud Database (18 Cases Compiled)

### Major cases (not in sample — delisted/suspended):
- Satyam Computer Services (2003-2008) — Revenue & cash inflation Rs 7,800 Cr
- DHFL (2016-2018) — Loan diversion and fund siphoning
- IL&FS (2017-2018) — Debt concealment and defaults hidden
- Brightcom Group (2014-2022) — Revenue misstatement, SEBI ban Feb 2025
- Bombay Dyeing (2012-2018) — Revenue inflation Rs 2,500 Cr via group entity
- SecureKloud Technologies (2017-2019) — Revenue irregularities, Deloitte resigned
- Seya Industries (2019-2022) — Rs 81.26 Cr fund siphoning
- Manpasand Beverages (2017-2018) — Revenue inflation
- PC Jeweller (2017-2018) — Suspected round-tripping
- Vakrangee (2016-2017) — Circular trading
- Ricoh India (2015-2016) — Revenue/asset inflation
- Gitanjali Gems (2015-2017) — Fund diversion (PNB fraud)
- Cox & Kings (2018-2019) — Financial manipulation
- Kwality Ltd (2017-2018) — Accounting irregularities
- Unitech (2014-2017) — Fund diversion
- Deccan Chronicle (2012-2013) — Financial misrepresentation

### Cases overlapping with sample:
- Zee Entertainment (2018-2019) — Fund diversion allegations → **Model flagged in 2017 (pre-scandal)**
- Adani Group entities (2022-2023) — Hindenburg allegations → **Model flagged consistently across years**

---

## Theoretical Grounding

The project is grounded in:
- **Agency Theory** (Jensen & Meckling, 1976): Information asymmetry between managers and shareholders creates incentives for financial manipulation
- **Signaling Theory** (Spence, 1973): Financial statements serve as signals of corporate integrity
- **Fraud Triangle** (Cressey, 1953) and **Fraud Pentagon** (Marks, 2018): Pressure, opportunity, rationalization, capability, and arrogance as fraud drivers

---

## Key References

### Foundational
- Beneish, M.D. (1999). "The Detection of Earnings Manipulation." *Financial Analysts Journal*, 55(5), 24-36.
- Liu, F.T., Ting, K.M., & Zhou, Z.-H. (2008). "Isolation Forest." *ICDM 2008*.
- Dechow, P.M. et al. (2011). "Predicting Material Accounting Misstatements." *Contemporary Accounting Research*, 28(1), 17-82.

### ML + Financial Fraud
- Bao, Y. et al. (2020). "Detecting Accounting Fraud in Publicly Traded U.S. Firms Using a Machine Learning Approach." *Journal of Accounting Research*, 58(1), 199-235.
- Perols, J. (2011). "Financial Statement Fraud Detection: An Analysis of Statistical and Machine Learning Algorithms." *Auditing: A Journal of Practice & Theory*, 30(2), 19-50.

### Explainable AI
- Sodnomdavaa et al. (2026). "Financial Statement Fraud Detection Through an Integrated ML and Explainable AI Framework." *Journal of Risk and Financial Management*, 19(1), 13.
- Thanathamathee et al. (2024). "SHAP-Instance Weighted and Anchor Explainable AI." *Emerging Science Journal*, 8(6).
- Zhou, Y. et al. (2023). "A User-Centered Explainable AI Approach for Financial Fraud Detection." *Finance Research Letters*.

### Indian Context
- Shah, C., Saraswat, M., & Mehta, A. (2018). "Predicting Earnings Manipulation Using Beneish M-Score of Selected Companies in India." *Indian Journal of Finance*, 12(4), 54-66.
- Kaur, R., Sharma, K., & Khanna, A. (2014). "Detecting Earnings Management in India: A Sector-Wise Study."
- Bhasin, M.L. (2013). "Corporate Accounting Fraud: A Case Study of Satyam Computers Limited." *Open Journal of Accounting*, 2(2).
- Patel, H. et al. (2019). "An Application of Ensemble Random Forest Classifier for Detecting Financial Statement Manipulation of Indian Listed Companies." *Springer*.

### Systematic Reviews
- Gupta, S. & Mehta, S.K. (2024). "Data Mining-Based Financial Statement Fraud Detection: SLR and Meta-Analysis." *Global Business Review*.
- Hernandez Aros et al. (2024). "Financial Fraud Detection Through ML Techniques: A Literature Review." *Humanities and Social Sciences Communications* (Nature).

---

## Assessment Alignment (UGCF Semester VIII)

| Assessment Parameter | Marks | How This Project Addresses It |
|---|---|---|
| Completion of experimentation/data collection/analysis | 40 | 94 companies, 820 observations, 4 ML algorithms, SHAP analysis |
| Final Report (Aim, Literature Review, Methodology, Results, Conclusions) | 80 | Full dissertation structure with all required sections |
| Scholarly Output (one required) | 40 | Target: conference paper or Journal of Financial Crime submission |
| End-Term Presentation and Viva Voce | 80 | PPT with model architecture, results, SHAP visualizations |

**Similarity index**: Must be ≤10%. All code is original; write interpretation sections in own words.  
**AI-generated content**: Must be ≤20%. Code pipeline is tool-assisted; analysis, interpretation, and literature review must be original.

---

## Pending / Future Work

- [ ] Add 5-8 known fraud companies (Vakrangee, PC Jeweller, Manpasand, Kwality) to validate model against confirmed manipulation
- [ ] Industry-stratified anomaly detection (separate models per sector)
- [ ] Visualizations: SHAP summary plots, consensus heatmaps, time-series anomaly charts
- [ ] Rolling-window analysis to account for temporal regime changes
- [ ] Draft conference paper for scholarly output requirement
- [ ] Dissertation chapter writing (per UGCF prescribed format)
- [ ] PowerPoint for viva voce (must include slide showing student conducted the research)

---

*Last updated: March 2026*
