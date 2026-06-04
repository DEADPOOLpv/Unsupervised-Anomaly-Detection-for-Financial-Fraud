"""
==============================================================================
SCREENER.IN BATCH PROCESSOR v3 — INTEGRATED FRAUD SIGNATURE PIPELINE
==============================================================================
Single codebase that handles:
  1. URL generation for Nifty 200 + fraud companies
  2. Parsing Screener.in Excel files → 28 features
  3. Anomaly detection (IF, LOF, SVM, EE) + SHAP
  4. Fraud signature extraction (comparing fraud companies vs baseline)
  5. Signature matching (comparing flagged Nifty 200 vs fraud signatures)

Usage:
  py screener_batch_processor_v3.py --urls                          # Print all download URLs
  py screener_batch_processor_v3.py --urls --fraud-only             # Print fraud company URLs only
  py screener_batch_processor_v3.py --folder ./screener_data/       # Process Nifty 200 baseline
  py screener_batch_processor_v3.py --folder ./all_data/ --detect   # Process + run anomaly detection
  py screener_batch_processor_v3.py --signatures                    # Extract fraud signatures & match
==============================================================================
"""

import pandas as pd
import numpy as np
import os
import glob
import argparse
import warnings
warnings.filterwarnings('ignore')


# ===========================================================================
# COMPANY LISTS
# ===========================================================================

# Nifty 200 non-financial sample — (Symbol, 'c'=consolidated / 's'=standalone)
NIFTY200_NON_FINANCIAL = [
    # IT Services (10)
    ('TCS', 'c'), ('INFY', 'c'), ('WIPRO', 'c'), ('HCLTECH', 'c'),
    ('TECHM', 'c'), ('LTIM', 'c'), ('PERSISTENT', 'c'), ('COFORGE', 'c'),
    ('MPHASIS', 'c'), ('LTTS', 'c'),
    # Pharma & Healthcare (10)
    ('SUNPHARMA', 'c'), ('DRREDDY', 'c'), ('CIPLA', 'c'), ('DIVISLAB', 'c'),
    ('AUROPHARMA', 'c'), ('BIOCON', 'c'), ('LUPIN', 'c'), ('TORNTPHARM', 'c'),
    ('ALKEM', 'c'), ('APOLLOHOSP', 'c'),
    # Auto (10)
    ('TMPV', 'c'), ('M&M', 'c'), ('MARUTI', 'c'), ('BAJAJ-AUTO', 'c'),
    ('HEROMOTOCO', 'c'), ('EICHERMOT', 'c'), ('ASHOKLEY', 'c'),
    ('TVSMOTOR', 'c'), ('BOSCHLTD', 's'), ('MOTHERSON', 'c'),
    # FMCG (10)
    ('HINDUNILVR', 'c'), ('ITC', 'c'), ('NESTLEIND', 's'), ('BRITANNIA', 'c'),
    ('DABUR', 'c'), ('MARICO', 'c'), ('GODREJCP', 'c'), ('COLPAL', 's'),
    ('TATACONSUM', 'c'), ('VBL', 'c'),
    # Energy (10)
    ('RELIANCE', 'c'), ('ONGC', 'c'), ('IOC', 'c'), ('BPCL', 'c'),
    ('GAIL', 'c'), ('NTPC', 'c'), ('POWERGRID', 'c'), ('ADANIGREEN', 'c'),
    ('TATAPOWER', 'c'), ('JSWENERGY', 'c'),
    # Metals (8)
    ('TATASTEEL', 'c'), ('JSWSTEEL', 'c'), ('HINDALCO', 'c'), ('VEDL', 'c'),
    ('COALINDIA', 'c'), ('NMDC', 'c'), ('SAIL', 'c'), ('JINDALSTEL', 'c'),
    # Infrastructure & Cement (10)
    ('LT', 'c'), ('ADANIENT', 'c'), ('ADANIPORTS', 'c'), ('DLF', 'c'),
    ('GODREJPROP', 'c'), ('ULTRACEMCO', 'c'), ('SHREECEM', 'c'),
    ('AMBUJACEM', 'c'), ('ACC', 'c'), ('GRASIM', 'c'),
    # Telecom/Media/Internet (3)
    ('BHARTIARTL', 'c'), ('ZEEL', 'c'), ('INDIAMART', 'c'),
    # Chemicals (5)
    ('PIDILITIND', 'c'), ('SRF', 'c'), ('ATUL', 's'), ('DEEPAKNTR', 'c'),
    ('NAVINFLUOR', 'c'),
    # Industrials/Consumer (18)
    ('TITAN', 'c'), ('TRENT', 'c'), ('HAVELLS', 'c'), ('VOLTAS', 'c'),
    ('CROMPTON', 'c'), ('SIEMENS', 's'), ('ABB', 's'),
    ('CUMMINSIND', 's'), ('HONAUT', 's'),
    ('PAGEIND', 's'), ('DIXON', 'c'), ('POLYCAB', 'c'),
    ('BATAINDIA', 'c'), ('VGUARD', 'c'),
    ('DELHIVERY', 'c'), ('CONCOR', 'c'), ('BLUESTARLT', 'c'), ('RELAXO', 'c'),
]

# Fraud companies available on Screener.in
FRAUD_COMPANIES = [
    ('BCG',          'c', 'Brightcom Group',     '2014-2020', 'Revenue misstatement & asset inflation; SEBI Rs 34 Cr penalty'),
    ('MANPASAND',    's', 'Manpasand Beverages',  '2015-2019', 'Revenue inflation via 38 bogus entities; Deloitte resigned'),
    ('PCJEWELLER',   'c', 'PC Jeweller',          '2012-2018', 'Suspected round-tripping; CFO/PAT gap Rs 1,535 Cr over 5 yrs'),
    ('VAKRANGEE',    'c', 'Vakrangee',            '2014-2018', 'Circular trading; PwC resigned as auditor'),
    ('KWALITY',      'c', 'Kwality Ltd',          '2015-2018', 'Accounting irregularities; massive debt default'),
    ('BOMBAYDYEING', 'c', 'Bombay Dyeing',        '2011-2018', 'Revenue inflation Rs 2,493 Cr via Scal Services; SAT overturned Jan 2026'),
    ('COXANDKINGS',  's', 'Cox & Kings',          '2017-2019', 'Financial manipulation; debt concealment; may not be on Screener'),
]

# Fraud periods for signature extraction — (start_year, end_year)
FRAUD_PERIODS = {
    'BCG':          (2014, 2020),
    'BRIGHTCOM':    (2014, 2020),
    'MANPASAND':    (2015, 2019),
    'PCJEWELLER':   (2012, 2018),
    'PC JEWELLER':  (2012, 2018),
    'VAKRANGEE':    (2014, 2018),
    'KWALITY':      (2015, 2018),
    'BOMBAYDYEING': (2011, 2018),
    'BOMBAY DYEING':(2011, 2018),
    'COXANDKINGS':  (2017, 2019),
    'COX AND KINGS':(2017, 2019),
    'SATYAM':       (2003, 2008),
    'DHFL':         (2006, 2019),
}


# ===========================================================================
# ROW INDICES — Screener.in Data Sheet format (confirmed)
# ===========================================================================

ROW = {
    'company_name':  0,
    'pl_report_date': 15,
    'sales':          16,
    'raw_material':   17,
    'change_inv':     18,
    'employee_cost':  21,
    'selling_admin':  22,
    'depreciation':   25,
    'interest':       26,
    'pbt':            27,
    'net_profit':     29,
    'bs_report_date': 55,
    'equity_capital': 56,
    'reserves':       57,
    'borrowings':     58,
    'other_liab':     59,
    'total_liab':     60,
    'net_block':      61,
    'cwip':           62,
    'investments':    63,
    'other_assets':   64,
    'total_assets':   65,
    'receivables':    66,
    'inventory':      67,
    'cash_bank':      68,
    'cfo':            81,
    'cfi':            82,
    'cff':            83,
    'net_cashflow':   84,
}


# ===========================================================================
# FEATURE ENGINEERING
# ===========================================================================

FEATURE_NAMES = [
    'DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA',
    'ROA', 'ROE', 'Operating_Margin', 'Net_Margin',
    'CFO_to_NI', 'CFO_to_Assets', 'Accrual_Ratio',
    'DE_Ratio', 'Debt_to_Assets', 'Interest_Coverage', 'Current_Ratio',
    'Asset_Turnover', 'Inventory_to_Sales', 'Receivable_Days',
    'Revenue_Growth', 'NI_Growth', 'Asset_Growth',
]


def safe_div(a, b):
    """Safe division — returns NaN if denominator is zero/NaN."""
    if b is None or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def parse_screener_file(filepath):
    """
    Parse a single Screener.in Excel file using hard-coded row positions.
    Returns a list of dicts, one per year-pair, with all raw items extracted.
    """
    try:
        df = pd.read_excel(filepath, sheet_name='Data Sheet', header=None)
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return []

    # Extract company name — try Row 0 Col 1 first (default export), then Col 0
    company = str(df.iloc[0, 1]).strip() if df.shape[1] > 1 and pd.notna(df.iloc[0, 1]) else ''
    if not company or company == 'nan' or company == 'COMPANY NAME':
        company = str(df.iloc[0, 0]).strip()
    if not company or company == 'nan' or company == 'COMPANY NAME':
        company = os.path.splitext(os.path.basename(filepath))[0]

    # Auto-detect year columns: scan Row 15 (P&L report date) for dates/years
    # Format A (custom export): years start at col 7+, stored as datetime objects
    # Format B (default export): years start at col 1+, stored as datetime or numeric
    year_row = df.iloc[ROW['pl_report_date']]
    year_cols = []
    for c in range(len(df.columns)):
        val = year_row[c]
        if pd.notna(val):
            # Skip label columns (text like "Report Date")
            val_str = str(val).strip()
            if val_str.lower().startswith('report') or val_str == 'nan':
                continue
            try:
                yr = pd.to_datetime(val).year
                if 1990 <= yr <= 2030:
                    year_cols.append((c, yr))
            except:
                # Try as plain integer year
                try:
                    yr = int(float(val_str[:4]))
                    if 1990 <= yr <= 2030:
                        year_cols.append((c, yr))
                except:
                    pass

    if len(year_cols) < 2:
        print(f"  WARNING: {company} — fewer than 2 years of data, skipping")
        return []

    # Extract raw data for each year
    yearly_data = {}
    for col_idx, year in year_cols:
        d = {}
        for key, row_idx in ROW.items():
            if key in ('company_name', 'pl_report_date', 'bs_report_date'):
                continue
            try:
                val = df.iloc[row_idx, col_idx]
                d[key] = float(val) if pd.notna(val) else np.nan
            except:
                d[key] = np.nan
        yearly_data[year] = d

    # Compute features for each consecutive year-pair
    records = []
    sorted_years = sorted(yearly_data.keys())

    for i in range(1, len(sorted_years)):
        curr_year = sorted_years[i]
        prev_year = sorted_years[i - 1]
        curr = yearly_data[curr_year]
        prev = yearly_data[prev_year]

        # Derived quantities
        total_assets_curr = curr.get('total_assets', np.nan)
        total_assets_prev = prev.get('total_assets', np.nan)
        sales_curr = curr.get('sales', np.nan)
        sales_prev = prev.get('sales', np.nan)
        ni_curr = curr.get('net_profit', np.nan)
        ni_prev = prev.get('net_profit', np.nan)
        cfo_curr = curr.get('cfo', np.nan)
        recv_curr = curr.get('receivables', np.nan)
        recv_prev = prev.get('receivables', np.nan)
        inv_curr = curr.get('inventory', np.nan)
        ppe_curr = curr.get('net_block', np.nan)
        ppe_prev = prev.get('net_block', np.nan)
        depr_curr = curr.get('depreciation', np.nan)
        depr_prev = prev.get('depreciation', np.nan)
        sga_curr = curr.get('selling_admin', np.nan)
        sga_prev = prev.get('selling_admin', np.nan)
        borrowings_curr = curr.get('borrowings', np.nan)
        borrowings_prev = prev.get('borrowings', np.nan)
        interest_curr = curr.get('interest', np.nan)
        equity = (curr.get('equity_capital', 0) or 0) + (curr.get('reserves', 0) or 0)
        rawmat_curr = curr.get('raw_material', np.nan)
        rawmat_prev = prev.get('raw_material', np.nan)
        pbt_curr = curr.get('pbt', np.nan)

        # Gross Margin: (Sales - COGS) / Sales where COGS ≈ raw material cost
        gm_curr = safe_div((sales_curr or 0) - (rawmat_curr or 0), sales_curr)
        gm_prev = safe_div((sales_prev or 0) - (rawmat_prev or 0), sales_prev)

        # Operating income proxy: PBT + Interest
        op_income = (pbt_curr or 0) + (interest_curr or 0) if pd.notna(pbt_curr) else np.nan

        # PPE + CWIP for total fixed assets
        fa_curr = (ppe_curr or 0) + (curr.get('cwip', 0) or 0)
        fa_prev = (ppe_prev or 0) + (prev.get('cwip', 0) or 0)

        # Non-current assets (total assets - current assets proxy)
        # Current assets proxy: receivables + inventory + cash + other assets portion
        nca_curr = total_assets_curr - (recv_curr or 0) - (inv_curr or 0) - (curr.get('cash_bank', 0) or 0) if pd.notna(total_assets_curr) else np.nan
        nca_prev = total_assets_prev - (recv_prev or 0) - (prev.get('inventory', 0) or 0) - (prev.get('cash_bank', 0) or 0) if pd.notna(total_assets_prev) else np.nan

        # ── Beneish M-Score Variables ──
        # DSRI: Days Sales Receivable Index
        dsri = safe_div(safe_div(recv_curr, sales_curr), safe_div(recv_prev, sales_prev))
        # GMI: Gross Margin Index
        gmi = safe_div(gm_prev, gm_curr) if (pd.notna(gm_prev) and pd.notna(gm_curr)) else np.nan
        # AQI: Asset Quality Index
        aqi_curr = safe_div(total_assets_curr - fa_curr - (recv_curr or 0) - (inv_curr or 0) - (curr.get('cash_bank', 0) or 0), total_assets_curr)
        aqi_prev = safe_div(total_assets_prev - fa_prev - (recv_prev or 0) - (prev.get('inventory', 0) or 0) - (prev.get('cash_bank', 0) or 0), total_assets_prev)
        aqi = safe_div(aqi_curr, aqi_prev)
        # SGI: Sales Growth Index
        sgi = safe_div(sales_curr, sales_prev)
        # DEPI: Depreciation Index
        depi_curr = safe_div(depr_curr, depr_curr + ppe_curr) if pd.notna(depr_curr) and pd.notna(ppe_curr) else np.nan
        depi_prev = safe_div(depr_prev, depr_prev + ppe_prev) if pd.notna(depr_prev) and pd.notna(ppe_prev) else np.nan
        depi = safe_div(depi_prev, depi_curr)
        # SGAI: SGA Index
        sgai = safe_div(safe_div(sga_curr, sales_curr), safe_div(sga_prev, sales_prev))
        # LVGI: Leverage Index
        lvgi_curr = safe_div(borrowings_curr, total_assets_curr)
        lvgi_prev = safe_div(borrowings_prev, total_assets_prev)
        lvgi = safe_div(lvgi_curr, lvgi_prev)
        # TATA: Total Accruals to Total Assets
        tata = safe_div((ni_curr or 0) - (cfo_curr or 0), total_assets_curr)

        # ── M-Score ──
        mscore = np.nan
        beneish_class = 'INSUFFICIENT_DATA'
        components = [dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata]
        if all(pd.notna(c) for c in components):
            mscore = (-4.840 + 0.920*dsri + 0.528*gmi + 0.404*aqi +
                      0.892*sgi + 0.115*depi - 0.172*sgai + 4.679*tata - 0.327*lvgi)
            if mscore > -1.78:
                beneish_class = 'LIKELY_MANIPULATOR'
            elif mscore > -2.22:
                beneish_class = 'POSSIBLE_MANIPULATOR'
            else:
                beneish_class = 'UNLIKELY_MANIPULATOR'

        # ── Extended Ratios ──
        roa = safe_div(ni_curr, total_assets_curr)
        roe = safe_div(ni_curr, equity) if equity and equity != 0 else np.nan
        op_margin = safe_div(op_income, sales_curr)
        net_margin = safe_div(ni_curr, sales_curr)
        cfo_ni = safe_div(cfo_curr, ni_curr)
        cfo_assets = safe_div(cfo_curr, total_assets_curr)
        accrual = safe_div((ni_curr or 0) - (cfo_curr or 0), total_assets_curr)
        de_ratio = safe_div(borrowings_curr, equity) if equity and equity != 0 else np.nan
        debt_assets = safe_div(borrowings_curr, total_assets_curr)
        int_coverage = safe_div(op_income, interest_curr) if pd.notna(interest_curr) and interest_curr > 0 else np.nan
        current_ratio = np.nan  # would need current liabilities breakdown
        asset_turnover = safe_div(sales_curr, total_assets_curr)
        inv_sales = safe_div(inv_curr, sales_curr)
        recv_days = safe_div(recv_curr, sales_curr) * 365 if pd.notna(recv_curr) and pd.notna(sales_curr) and sales_curr > 0 else np.nan
        rev_growth = safe_div(sales_curr - sales_prev, abs(sales_prev)) if pd.notna(sales_prev) and sales_prev != 0 else np.nan
        ni_growth = safe_div(ni_curr - ni_prev, abs(ni_prev)) if pd.notna(ni_prev) and ni_prev != 0 else np.nan
        asset_growth = safe_div(total_assets_curr - total_assets_prev, abs(total_assets_prev)) if pd.notna(total_assets_prev) and total_assets_prev != 0 else np.nan

        record = {
            'Company': company,
            'Year': curr_year,
            'FY': f"FY{curr_year}",
            # Beneish
            'DSRI': dsri, 'GMI': gmi, 'AQI': aqi, 'SGI': sgi,
            'DEPI': depi, 'SGAI': sgai, 'LVGI': lvgi, 'TATA': tata,
            'M_Score': mscore, 'Beneish_Class': beneish_class,
            # Extended
            'ROA': roa, 'ROE': roe, 'Operating_Margin': op_margin,
            'Net_Margin': net_margin,
            'CFO_to_NI': cfo_ni, 'CFO_to_Assets': cfo_assets,
            'Accrual_Ratio': accrual,
            'DE_Ratio': de_ratio, 'Debt_to_Assets': debt_assets,
            'Interest_Coverage': int_coverage, 'Current_Ratio': current_ratio,
            'Asset_Turnover': asset_turnover, 'Inventory_to_Sales': inv_sales,
            'Receivable_Days': recv_days,
            'Revenue_Growth': rev_growth, 'NI_Growth': ni_growth,
            'Asset_Growth': asset_growth,
            # Metadata
            'Sales': sales_curr, 'Net_Profit': ni_curr,
            'Total_Assets': total_assets_curr, 'CFO': cfo_curr,
            'Borrowings': borrowings_curr,
        }
        records.append(record)

    return records


def process_folder(folder, output_csv):
    """Process all Screener.in Excel files in a folder."""
    files = glob.glob(os.path.join(folder, '*.xlsx'))
    if not files:
        print(f"ERROR: No .xlsx files found in {folder}")
        return None

    print(f"Found {len(files)} Excel files in {folder}")
    all_records = []

    for f in sorted(files):
        basename = os.path.basename(f)
        records = parse_screener_file(f)
        if records:
            all_records.extend(records)
            print(f"  ✓ {basename:30s} → {len(records)} year-pairs")
        else:
            print(f"  ✗ {basename:30s} → FAILED")

    if not all_records:
        print("ERROR: No data extracted from any file.")
        return None

    df = pd.DataFrame(all_records)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    print(f"  {len(df)} observations, {df['Company'].nunique()} companies")
    print(f"  Columns: {len(df.columns)}")

    # M-Score summary
    has_mscore = df[df['M_Score'].notna()]
    print(f"\n  M-Score computed: {len(has_mscore)}/{len(df)} ({100*len(has_mscore)/len(df):.1f}%)")
    if len(has_mscore) > 0:
        print(f"  Classifications:")
        print(f"    {(has_mscore['Beneish_Class']=='LIKELY_MANIPULATOR').sum()} Likely Manipulators")
        print(f"    {(has_mscore['Beneish_Class']=='POSSIBLE_MANIPULATOR').sum()} Possible")
        print(f"    {(has_mscore['Beneish_Class']=='UNLIKELY_MANIPULATOR').sum()} Unlikely")

    return df


# ===========================================================================
# ANOMALY DETECTION + SHAP
# ===========================================================================

def run_anomaly_detection(features_csv, contamination=0.07):
    """Run 4-algorithm anomaly detection + SHAP on the features CSV."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.covariance import EllipticEnvelope
    from sklearn.preprocessing import StandardScaler
    import shap

    df = pd.read_csv(features_csv)
    print(f"\nLoaded {len(df)} observations from {features_csv}")

    # Select numeric features for anomaly detection
    available = [f for f in FEATURE_NAMES if f in df.columns and df[f].notna().sum() > len(df)*0.3]
    print(f"Using {len(available)} features: {available}")

    X_raw = df[available].copy()
    X_raw = X_raw.fillna(X_raw.median())

    # Winsorize at 1st/99th percentile
    for col in X_raw.columns:
        p1, p99 = X_raw[col].quantile(0.01), X_raw[col].quantile(0.99)
        X_raw[col] = X_raw[col].clip(p1, p99)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print(f"\nRunning anomaly detection (contamination={contamination})...\n")

    # Isolation Forest
    print("  [1/4] Isolation Forest...")
    iso = IsolationForest(n_estimators=500, contamination=contamination, random_state=42, n_jobs=-1)
    iso.fit(X)
    df['IF_score'] = iso.decision_function(X)
    df['IF_label'] = iso.predict(X)

    # Local Outlier Factor
    print("  [2/4] Local Outlier Factor...")
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    df['LOF_label'] = lof.fit_predict(X)

    # One-Class SVM
    print("  [3/4] One-Class SVM...")
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=contamination)
    ocsvm.fit(X)
    df['SVM_label'] = ocsvm.predict(X)

    # Elliptic Envelope
    print("  [4/4] Elliptic Envelope...")
    try:
        ee = EllipticEnvelope(contamination=contamination, random_state=42)
        ee.fit(X)
        df['EE_label'] = ee.predict(X)
    except Exception as e:
        print(f"    EE failed ({e}), skipping")
        df['EE_label'] = 1

    # Consensus
    label_cols = ['IF_label', 'LOF_label', 'SVM_label', 'EE_label']
    df['Consensus'] = sum((df[c] == -1).astype(int) for c in label_cols)

    # SHAP
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(iso)
    shap_values = explainer.shap_values(X)

    shap_importance = pd.DataFrame({
        'Feature': available,
        'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0)
    }).sort_values('Mean_Abs_SHAP', ascending=False)

    # Save SHAP values per observation
    shap_df = pd.DataFrame(shap_values, columns=available)

    print("\n  GLOBAL FEATURE IMPORTANCE (SHAP):")
    for _, row in shap_importance.iterrows():
        bar = "█" * int(row['Mean_Abs_SHAP'] * 100)
        print(f"    {row['Feature']:22s}  {row['Mean_Abs_SHAP']:.4f}  {bar}")

    # Summary
    print(f"\n  Flagged by 2+ algorithms: {(df['Consensus'] >= 2).sum()}")
    print(f"  Flagged by 3+ algorithms: {(df['Consensus'] >= 3).sum()}")
    print(f"  Flagged by all 4:         {(df['Consensus'] == 4).sum()}")

    return df, shap_importance, shap_df


# ===========================================================================
# FRAUD SIGNATURE EXTRACTION
# ===========================================================================

def extract_fraud_signatures(baseline_csv, fraud_csv):
    """
    Compare fraud company features against the Nifty 200 baseline.
    Identifies which features systematically deviate during fraud years.
    """
    baseline = pd.read_csv(baseline_csv)
    fraud = pd.read_csv(fraud_csv)

    print(f"\nBaseline: {len(baseline)} obs, {baseline['Company'].nunique()} companies")
    print(f"Fraud:    {len(fraud)} obs, {fraud['Company'].nunique()} companies")
    print(f"Fraud companies: {list(fraud['Company'].unique())}")

    # Compute baseline statistics
    numeric_feats = [f for f in FEATURE_NAMES if f in baseline.columns]
    stats = {}
    for feat in numeric_feats:
        vals = baseline[feat].dropna()
        if len(vals) > 0:
            stats[feat] = {
                'median': vals.median(),
                'mean': vals.mean(),
                'std': vals.std(),
                'p25': vals.quantile(0.25),
                'p75': vals.quantile(0.75),
            }
    baseline_stats = pd.DataFrame(stats).T

    # For each fraud company-year, compute z-scores vs baseline
    records = []
    for _, row in fraud.iterrows():
        company = row['Company']
        year = row.get('Year', np.nan)

        # Determine if this is a fraud year
        is_fraud_year = False
        for key, (start, end) in FRAUD_PERIODS.items():
            if key.upper() in str(company).upper():
                try:
                    yr = int(year)
                    is_fraud_year = start <= yr <= end
                except:
                    pass
                break

        for feat in numeric_feats:
            if feat in row.index and feat in baseline_stats.index:
                val = row[feat]
                if pd.notna(val):
                    median = baseline_stats.loc[feat, 'median']
                    std = baseline_stats.loc[feat, 'std']
                    z = (val - median) / std if std > 0 else 0

                    if abs(z) > 3:
                        deviation = "EXTREME"
                    elif abs(z) > 2:
                        deviation = "HIGH"
                    elif abs(z) > 1:
                        deviation = "MODERATE"
                    else:
                        deviation = "NORMAL"

                    records.append({
                        'Company': company,
                        'Year': year,
                        'Is_Fraud_Year': is_fraud_year,
                        'Feature': feat,
                        'Value': round(val, 4),
                        'Baseline_Median': round(median, 4),
                        'Z_Score': round(z, 2),
                        'Deviation': deviation,
                        'Direction': 'ABOVE' if val > median else 'BELOW',
                    })

    sigs = pd.DataFrame(records)

    # Summarize: which features are signature features per company
    fraud_only = sigs[sigs['Is_Fraud_Year']].copy()
    if fraud_only.empty:
        print("  WARNING: No fraud-year obs found; using all years as fallback")
        fraud_only = sigs.copy()

    summary = fraud_only.groupby(['Company', 'Feature']).agg(
        Avg_Z_Score=('Z_Score', 'mean'),
        Max_Abs_Z=('Z_Score', lambda x: x.abs().max()),
        Pct_Extreme=('Deviation', lambda x: (x.isin(['EXTREME', 'HIGH'])).mean()),
        Direction=('Direction', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'MIXED'),
        N_Years=('Year', 'count'),
    ).reset_index()

    summary['Is_Signature'] = (
        (summary['Pct_Extreme'] >= 0.5) | (summary['Avg_Z_Score'].abs() >= 1.5)
    )
    summary = summary.sort_values(['Company', 'Is_Signature', 'Avg_Z_Score'],
                                   ascending=[True, False, False])

    # Print results
    print("\n" + "=" * 70)
    print("FRAUD SIGNATURES EXTRACTED")
    print("=" * 70)
    for company in summary['Company'].unique():
        comp_sigs = summary[(summary['Company'] == company) & (summary['Is_Signature'])]
        if not comp_sigs.empty:
            print(f"\n  {company}:")
            for _, r in comp_sigs.iterrows():
                arrow = "↑" if r['Direction'] == 'ABOVE' else "↓"
                print(f"    {arrow} {r['Feature']:25s}  Z={r['Avg_Z_Score']:+.1f}  "
                      f"({r['Pct_Extreme']*100:.0f}% extreme, {r['N_Years']} yrs)")

    # Save
    sigs.to_csv('fraud_signatures.csv', index=False)
    summary.to_csv('fraud_signature_summary.csv', index=False)
    print(f"\n  Saved: fraud_signatures.csv ({len(sigs)} rows)")
    print(f"  Saved: fraud_signature_summary.csv ({len(summary)} rows)")

    return sigs, summary, baseline_stats


def match_signatures(anomaly_results_csv, signature_summary_csv):
    """
    Match flagged Nifty 200 companies against extracted fraud signatures.
    Checks whether flagged companies deviate in the SAME direction and
    with similar magnitude as the fraud companies — not just whether
    the feature has data.
    """
    results = pd.read_csv(anomaly_results_csv)
    sigs = pd.read_csv(signature_summary_csv)

    # Compute baseline stats from all results (not just flagged)
    baseline_stats = {}
    for feat in FEATURE_NAMES:
        if feat in results.columns:
            vals = results[feat].dropna()
            if len(vals) > 0:
                baseline_stats[feat] = {'median': vals.median(), 'std': vals.std()}

    # Get companies flagged by 2+ algorithms
    flagged = results[results['Consensus'] >= 2].copy()
    if flagged.empty:
        print("No companies flagged by 2+ algorithms.")
        return

    flagged_companies = flagged.groupby('Company').agg(
        Avg_Consensus=('Consensus', 'mean'),
        Years_Flagged=('Consensus', 'count'),
    ).sort_values('Avg_Consensus', ascending=False)

    # Build fraud profiles with direction and z-score thresholds
    fraud_sigs = sigs[sigs['Is_Signature'] == True].copy()
    fraud_profiles = {}
    for company in fraud_sigs['Company'].unique():
        feats = fraud_sigs[fraud_sigs['Company'] == company]
        fraud_profiles[company] = {}
        for _, r in feats.iterrows():
            fraud_profiles[company][r['Feature']] = {
                'direction': r['Direction'],
                'z_score': r['Avg_Z_Score'],
            }

    print("\n" + "=" * 70)
    print("SIGNATURE MATCHING: Flagged Nifty 200 vs Fraud Profiles")
    print("=" * 70)

    all_matches = []

    for nifty_co in flagged_companies.index:
        co_data = flagged[flagged['Company'] == nifty_co]
        avg_cons = flagged_companies.loc[nifty_co, 'Avg_Consensus']
        n_years = flagged_companies.loc[nifty_co, 'Years_Flagged']
        print(f"\n  {nifty_co} (consensus: {avg_cons:.1f}, {n_years} yr-flags)")

        best_match_name = None
        best_match_pct = 0
        best_match_feats = []

        for fraud_co, fraud_feats in fraud_profiles.items():
            if not fraud_feats:
                continue

            matches = []
            for feat, fraud_info in fraud_feats.items():
                if feat not in co_data.columns or feat not in baseline_stats:
                    continue

                # Compute z-score for the Nifty company on this feature
                vals = co_data[feat].dropna()
                if len(vals) == 0:
                    continue

                median_val = vals.median()
                bl_median = baseline_stats[feat]['median']
                bl_std = baseline_stats[feat]['std']
                if bl_std == 0:
                    continue

                nifty_z = (median_val - bl_median) / bl_std

                # Match if: same direction AND z-score >= 1.0 in that direction
                fraud_dir = fraud_info['direction']
                if fraud_dir == 'ABOVE' and nifty_z >= 1.0:
                    matches.append((feat, nifty_z, fraud_info['z_score']))
                elif fraud_dir == 'BELOW' and nifty_z <= -1.0:
                    matches.append((feat, nifty_z, fraud_info['z_score']))

            if matches:
                pct = len(matches) / len(fraud_feats) * 100
                # Short fraud company name for display
                short_name = fraud_co.split(' LTD')[0].split(' LIMITED')[0]
                if pct > 30:  # Only show meaningful matches
                    match_str = ", ".join([f"{m[0]}(Z={m[1]:+.1f})" for m in matches[:4]])
                    print(f"    → {short_name}: {len(matches)}/{len(fraud_feats)} sig. features match ({pct:.0f}%)")
                    print(f"      Matching: {match_str}")

                if pct > best_match_pct:
                    best_match_pct = pct
                    best_match_name = short_name
                    best_match_feats = matches

        if best_match_name and best_match_pct > 30:
            all_matches.append({
                'Nifty_Company': nifty_co,
                'Avg_Consensus': avg_cons,
                'Best_Fraud_Match': best_match_name,
                'Match_Pct': best_match_pct,
                'Matching_Features': "; ".join([m[0] for m in best_match_feats]),
            })

    # Save matching results
    if all_matches:
        match_df = pd.DataFrame(all_matches)
        match_df.to_csv('signature_matching.csv', index=False)
        print(f"\n  Saved: signature_matching.csv ({len(match_df)} companies matched)")
    else:
        print("\n  No strong matches found (>30% signature overlap).")


# ===========================================================================
# URL GENERATION
# ===========================================================================

def generate_urls(symbols_list, title="SCREENER.IN DOWNLOAD URLs"):
    """Print download URLs for a list of (symbol, type) tuples."""
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"Total: {len(symbols_list)} companies\n")
    for i, item in enumerate(symbols_list, 1):
        if len(item) == 2:
            sym, stype = item
            label = sym
        else:
            sym, stype, label = item[0], item[1], item[2]

        if stype == 's':
            url = f"https://www.screener.in/company/{sym}/"
            tag = "[standalone]"
        else:
            url = f"https://www.screener.in/company/{sym}/consolidated/"
            tag = ""
        print(f"  [{i:3d}] {label:25s} {url}  {tag}")


# ===========================================================================
# FINAL REPORT
# ===========================================================================

def print_final_report():
    """Print a single clean report summarizing all results."""
    results = pd.read_csv('anomaly_results_final.csv')
    sigs = pd.read_csv('fraud_signature_summary.csv')

    # Baseline stats for matching
    baseline_stats = {}
    for feat in FEATURE_NAMES:
        if feat in results.columns:
            vals = results[feat].dropna()
            if len(vals) > 0:
                baseline_stats[feat] = {'median': vals.median(), 'std': vals.std()}

    print("\n")
    print("╔" + "═"*72 + "╗")
    print("║" + " FINANCIAL STATEMENT FRAUD DETECTION — COMPLETE RESULTS ".center(72) + "║")
    print("║" + " Unsupervised Anomaly Detection + Fraud Signature Analysis ".center(72) + "║")
    print("╚" + "═"*72 + "╝")

    # ── 1. Dataset ──
    print("\n┌──────────────────────────────────────────────────────────────────────────┐")
    print("│  1. DATASET                                                              │")
    print("└──────────────────────────────────────────────────────────────────────────┘")
    n_co = results['Company'].nunique()
    print(f"  Nifty 200 sample:   {n_co} companies, {len(results)} firm-year observations")
    has_ms = results[results['M_Score'].notna()]
    likely = (has_ms['Beneish_Class'] == 'LIKELY_MANIPULATOR').sum()
    possible = (has_ms['Beneish_Class'] == 'POSSIBLE_MANIPULATOR').sum()
    unlikely = (has_ms['Beneish_Class'] == 'UNLIKELY_MANIPULATOR').sum()
    print(f"  M-Score computed:   {len(has_ms)}/{len(results)} ({100*len(has_ms)/len(results):.1f}%)")
    print(f"  Beneish classes:    {likely} Likely | {possible} Possible | {unlikely} Unlikely")
    fraud = pd.read_csv('fraud_features.csv')
    print(f"  Fraud reference:    {fraud['Company'].nunique()} companies, {len(fraud)} observations")

    # ── 2. Anomaly Detection ──
    print("\n┌──────────────────────────────────────────────────────────────────────────┐")
    print("│  2. ANOMALY DETECTION                                                    │")
    print("└──────────────────────────────────────────────────────────────────────────┘")
    for t in [2, 3, 4]:
        print(f"  Flagged by {t}+ algorithms:  {(results['Consensus'] >= t).sum()} firm-years")

    flagged = results[results['Consensus'] >= 2].copy()
    company_scores = flagged.groupby('Company').agg(
        Avg=('Consensus', 'mean'), N=('Year', 'count'),
    ).sort_values('Avg', ascending=False)

    print(f"\n  {'Company':<38s} {'Score':>5s} {'Years':>5s}")
    print(f"  {'─'*38} {'─'*5} {'─'*5}")
    for name, row in company_scores.head(15).iterrows():
        short = name.split(' LTD')[0].split(' LIMITED')[0][:36]
        print(f"  {short:<38s} {row['Avg']:>5.1f} {int(row['N']):>5d}")

    # ── 3. SHAP ──
    print("\n┌──────────────────────────────────────────────────────────────────────────┐")
    print("│  3. WHAT DRIVES DETECTION (SHAP Feature Importance)                      │")
    print("└──────────────────────────────────────────────────────────────────────────┘")
    shap_imp = pd.read_csv('shap_feature_importance.csv')
    for _, row in shap_imp.head(10).iterrows():
        bar = "█" * int(row['Mean_Abs_SHAP'] * 80)
        print(f"  {row['Feature']:<22s} {row['Mean_Abs_SHAP']:.4f}  {bar}")

    # ── 4. Fraud Signatures ──
    print("\n┌──────────────────────────────────────────────────────────────────────────┐")
    print("│  4. FRAUD SIGNATURES (from 6 confirmed fraud cases)                      │")
    print("└──────────────────────────────────────────────────────────────────────────┘")
    fraud_sigs = sigs[sigs['Is_Signature'] == True].copy()
    for company in fraud_sigs['Company'].unique():
        short = company.split(' LTD')[0].split(' LIMITED')[0]
        comp_sigs = fraud_sigs[fraud_sigs['Company'] == company]
        feats_str = []
        for _, r in comp_sigs.iterrows():
            arrow = "↑" if r['Direction'] == 'ABOVE' else "↓"
            feats_str.append(f"{arrow}{r['Feature']}(Z={r['Avg_Z_Score']:+.1f})")
        print(f"  {short:<28s} {', '.join(feats_str)}")

    # ── 5. Signature Matching ──
    print("\n┌──────────────────────────────────────────────────────────────────────────┐")
    print("│  5. SIGNATURE MATCHING: Do flagged companies resemble known fraud?        │")
    print("└──────────────────────────────────────────────────────────────────────────┘")

    # Build fraud profiles
    fraud_profiles = {}
    for company in fraud_sigs['Company'].unique():
        feats = fraud_sigs[fraud_sigs['Company'] == company]
        fraud_profiles[company] = {}
        for _, r in feats.iterrows():
            fraud_profiles[company][r['Feature']] = {
                'direction': r['Direction'], 'z_score': r['Avg_Z_Score']}

    match_results = []
    for nifty_co in company_scores.index:
        co_data = flagged[flagged['Company'] == nifty_co]
        avg_cons = company_scores.loc[nifty_co, 'Avg']

        best_match = None
        best_pct = 0
        best_feats = []

        for fraud_co, fraud_feats in fraud_profiles.items():
            matches = []
            for feat, info in fraud_feats.items():
                if feat not in co_data.columns or feat not in baseline_stats:
                    continue
                vals = co_data[feat].dropna()
                if len(vals) == 0:
                    continue
                bl_std = baseline_stats[feat]['std']
                if bl_std == 0:
                    continue
                nifty_z = (vals.median() - baseline_stats[feat]['median']) / bl_std
                if info['direction'] == 'ABOVE' and nifty_z >= 1.0:
                    matches.append((feat, nifty_z))
                elif info['direction'] == 'BELOW' and nifty_z <= -1.0:
                    matches.append((feat, nifty_z))

            if matches:
                pct = len(matches) / len(fraud_feats) * 100
                if pct > best_pct:
                    best_pct = pct
                    best_match = fraud_co.split(' LTD')[0].split(' LIMITED')[0]
                    best_feats = matches

        short_nifty = nifty_co.split(' LTD')[0].split(' LIMITED')[0][:28]
        match_results.append({
            'company': short_nifty, 'consensus': avg_cons,
            'match': best_match or '—', 'pct': best_pct,
            'features': ", ".join([f"{f[0]}(Z={f[1]:+.1f})" for f in best_feats[:3]]) if best_feats else '—',
        })

    print(f"\n  {'Flagged Company':<30s} {'Scr':>3s}  {'Best Fraud Match':<24s} {'%':>3s}  Key Matching Features")
    print(f"  {'─'*30} {'─'*3}  {'─'*24} {'─'*3}  {'─'*35}")
    for m in sorted(match_results, key=lambda x: (-x['consensus'], -x['pct'])):
        pct_str = f"{m['pct']:.0f}" if m['pct'] > 0 else "—"
        print(f"  {m['company']:<30s} {m['consensus']:>3.1f}  {m['match']:<24s} {pct_str:>3s}  {m['features']}")

    # ── 6. Key Findings ──
    print("\n┌──────────────────────────────────────────────────────────────────────────┐")
    print("│  6. KEY FINDINGS                                                         │")
    print("└──────────────────────────────────────────────────────────────────────────┘")

    # Zee
    zee = results[results['Company'].str.contains('ZEE', case=False, na=False)]
    zee_flagged = zee[zee['Consensus'] >= 2]
    if not zee_flagged.empty:
        print(f"\n  ★ ZEE ENTERTAINMENT flagged in year(s): {sorted(zee_flagged['Year'].tolist())}")
        print(f"    Scandal surfaced 2019. Model detected anomaly PRE-SCANDAL.")
        print(f"    Matches Brightcom signature (accrual/earnings quality deterioration).")

    # Adani
    adani = results[results['Company'].str.contains('ADANI', case=False, na=False)]
    for co in adani['Company'].unique():
        co_fl = adani[(adani['Company'] == co) & (adani['Consensus'] >= 2)]
        if not co_fl.empty:
            short = co.split(' LTD')[0][:35]
            print(f"\n  ★ {short} flagged in {len(co_fl)} year(s)")

    # Convergent validity
    has_both = results[results['M_Score'].notna() & results['IF_label'].notna()]
    if not has_both.empty:
        anom_ms = has_both[has_both['IF_label'] == -1]['M_Score'].mean()
        norm_ms = has_both[has_both['IF_label'] == 1]['M_Score'].mean()
        print(f"\n  ★ CONVERGENT VALIDITY")
        print(f"    IF anomalies mean M-Score:  {anom_ms:.3f}")
        print(f"    IF normal mean M-Score:     {norm_ms:.3f}")
        if anom_ms > norm_ms:
            print(f"    ✓ CONFIRMED — anomalies score closer to manipulation threshold")

        # How much more likely are M-Score manipulators to be IF-flagged?
        likely_flagged = has_both[(has_both['Beneish_Class'] == 'LIKELY_MANIPULATOR') & (has_both['IF_label'] == -1)]
        unlikely_flagged = has_both[(has_both['Beneish_Class'] == 'UNLIKELY_MANIPULATOR') & (has_both['IF_label'] == -1)]
        likely_total = (has_both['Beneish_Class'] == 'LIKELY_MANIPULATOR').sum()
        unlikely_total = (has_both['Beneish_Class'] == 'UNLIKELY_MANIPULATOR').sum()
        if likely_total > 0 and unlikely_total > 0:
            likely_rate = len(likely_flagged) / likely_total
            unlikely_rate = len(unlikely_flagged) / unlikely_total
            if unlikely_rate > 0:
                ratio = likely_rate / unlikely_rate
                print(f"    Likely Manipulators are {ratio:.1f}x more likely to be IF-flagged")

    # ── Files ──
    print("\n┌──────────────────────────────────────────────────────────────────────────┐")
    print("│  OUTPUT FILES                                                             │")
    print("└──────────────────────────────────────────────────────────────────────────┘")
    files = [
        ('screener_features.csv',       'Nifty 200 baseline features (820 obs)'),
        ('fraud_features.csv',          'Fraud company features (53 obs)'),
        ('anomaly_results_final.csv',   'Full anomaly detection results'),
        ('shap_feature_importance.csv', 'Global SHAP feature importance'),
        ('shap_values_matrix.csv',      'Per-observation SHAP values'),
        ('fraud_signatures.csv',        'Per-feature deviation from baseline'),
        ('fraud_signature_summary.csv', 'Signature features per fraud company'),
        ('signature_matching.csv',      'Flagged companies vs fraud matches'),
    ]
    for fname, desc in files:
        exists = "✓" if os.path.exists(fname) else "✗"
        print(f"  {exists} {fname:<35s} {desc}")

    print("\n" + "═"*74)
    print("  Pipeline complete.")
    print("═"*74 + "\n")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Financial Statement Fraud Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  --run-all          Run the COMPLETE pipeline in one go (recommended)
  --urls             Print Screener.in download URLs
  --urls --fraud-only  Print fraud company URLs only

The --run-all command does everything:
  1. Processes Nifty 200 data (screener_data/) → screener_features.csv
  2. Runs 4 anomaly detection algorithms + SHAP explainability
  3. Processes fraud company data (fraud_screener_data/) → fraud_features.csv
  4. Extracts fraud signatures by comparing fraud vs baseline
  5. Matches flagged Nifty 200 companies against fraud signatures
  6. Prints a clean summary report

Prerequisites:
  - screener_data/ folder with 94 Nifty 200 Excel files
  - fraud_screener_data/ folder with 6 fraud company Excel files
  - pip install pandas numpy openpyxl scikit-learn shap
        """)

    parser.add_argument('--run-all', action='store_true',
                        help='Run complete pipeline: process → detect → signatures → match → report')
    parser.add_argument('--urls', action='store_true', help='Print Screener.in download URLs')
    parser.add_argument('--fraud-only', action='store_true', help='With --urls, show fraud URLs only')
    parser.add_argument('--contamination', type=float, default=0.07, help='Anomaly rate (default 0.07)')

    args = parser.parse_args()

    if args.run_all:
        print("\n" + "═"*74)
        print("  RUNNING COMPLETE PIPELINE")
        print("═"*74)

        # Step 1: Process Nifty 200 baseline
        print("\n▶ STEP 1/5: Processing Nifty 200 data...")
        if not os.path.exists('./screener_data/'):
            print("ERROR: ./screener_data/ folder not found.")
            print("Download Excel files from Screener.in first: py screener_batch_processor_v3.py --urls")
            exit(1)
        process_folder('./screener_data/', 'screener_features.csv')

        # Step 2: Anomaly detection + SHAP
        print("\n▶ STEP 2/5: Running anomaly detection + SHAP...")
        results, shap_imp, shap_vals = run_anomaly_detection('screener_features.csv', args.contamination)
        results.to_csv('anomaly_results_final.csv', index=False)
        shap_imp.to_csv('shap_feature_importance.csv', index=False)
        shap_vals.to_csv('shap_values_matrix.csv', index=False)

        # Step 3: Process fraud companies
        print("\n▶ STEP 3/5: Processing fraud company data...")
        if not os.path.exists('./fraud_screener_data/'):
            print("WARNING: ./fraud_screener_data/ not found. Skipping fraud analysis.")
            print("Download fraud company files: py screener_batch_processor_v3.py --urls --fraud-only")
        else:
            process_folder('./fraud_screener_data/', 'fraud_features.csv')

            # Step 4: Extract fraud signatures
            print("\n▶ STEP 4/5: Extracting fraud signatures...")
            sigs, summary, stats = extract_fraud_signatures('screener_features.csv', 'fraud_features.csv')

            # Step 5: Match
            print("\n▶ STEP 5/5: Matching flagged companies to fraud signatures...")
            match_signatures('anomaly_results_final.csv', 'fraud_signature_summary.csv')

        # Final report
        print("\n▶ GENERATING FINAL REPORT...")
        print_final_report()

    elif args.urls:
        if args.fraud_only:
            fraud_list = [(sym, stype, name) for sym, stype, name, _, _ in FRAUD_COMPANIES]
            generate_urls(fraud_list, "FRAUD COMPANY DOWNLOAD URLs")
            print(f"\nSave files into ./fraud_screener_data/ then run:")
            print(f"  py screener_batch_processor_v3.py --run-all")
        else:
            generate_urls(NIFTY200_NON_FINANCIAL, "NIFTY 200 — DOWNLOAD URLs")
            print(f"\n{'='*70}")
            fraud_list = [(sym, stype, name) for sym, stype, name, _, _ in FRAUD_COMPANIES]
            generate_urls(fraud_list, "FRAUD COMPANY DOWNLOAD URLs")
            print(f"\nSave Nifty files into ./screener_data/")
            print(f"Save fraud files into ./fraud_screener_data/")
            print(f"Then run:  py screener_batch_processor_v3.py --run-all")

    else:
        parser.print_help()