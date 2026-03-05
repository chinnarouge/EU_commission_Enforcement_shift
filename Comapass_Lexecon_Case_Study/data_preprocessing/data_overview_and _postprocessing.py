import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from scipy import stats

DATA_PATH = r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\processesd_data\structured_data_competition_cases.csv"
OUTPUT_DIR = r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\processesd_data\final_competition_cases.csv"

def plot_cases_per_year(df, out_path=None):
    # Normalize year column to numeric and drop missing
    df = df.copy()
    df['year'] = pd.to_numeric(df.get('year'), errors='coerce')
    counts = df.dropna(subset=['year']).groupby('year').size().sort_index()
    if counts.empty:
        print('No year data available to plot.')
        return

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind='bar', ax=ax, color='C0')
    ax.set_title('Competition Cases per Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of cases')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if out_path:
        out_dir = os.path.dirname(out_path) or '.'
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f'Plot saved to: {out_path}')

    plt.show()

def main():
    # Load the classified data
    df = pd.read_csv(DATA_PATH)

    print(f"Total rows: {len(df)}")
    print(f"Competition rows: {len(df)}")
    if 'case_type' in df.columns:
        print(f"Case type distribution:\n{df['case_type'].value_counts()}")
    if 'year' in df.columns:
        print(f"Year range: {df['year'].min()} - {df['year'].max()}")


    df['text'] = df.get('title', '').fillna('') + ' ' + df.get('summary', '').fillna('')

    sector_keywords = {
    "Agriculture, Food & Fisheries": 
        ["agriculture", "food", "fish", "fisheries", "farm"],

    "Arts, Recreation, Education, Tourism & Sports":
        ["tourism", "education", "sport", "culture", "university"],

    "Energy & Environment":
        ["energy", "electricity", "gas", "renewable", "climate", "carbon", "power"],

    "Financial Services":
        ["bank", "insurance", "financial", "credit", "loan", "capital", "guarantee"],

    "Digital, Media & Electronic Communications":
        ["digital", "platform", "data", "telecom", "broadband", "online", "software"],

    "Manufacturing & Basic Industries":
        ["steel", "manufacturing", "industry", "cement", "chemical"],

    "Pharmaceuticals & Health Services":
        ["pharma", "hospital", "medicine", "vaccine", "health"],
    
    "Transport":
        ["airline", "airport", "railway", "rail", "shipping", "port",
         "aviation", "logistics", "freight", "road", "motorway",
         "carrier", "ferry", "transport"]
        }


    def assign_sector(text):
        if not isinstance(text, str):
            return "Professional & Other Services"
        text_lower = text.lower()
        scores = {}
        for sector, keywords in sector_keywords.items():
            scores[sector] = sum(1 for kw in keywords if kw in text_lower)
        best_sector = max(scores, key=scores.get)
        return best_sector if scores[best_sector] > 0 else "Professional & Other Services"


    df['sector'] = df['text'].apply(assign_sector)


    # cases per year (simple plot)
    if 'year' in df.columns:
        cases_year = df.groupby('year').size()
        cases_year.plot(title="Total Enforcement Cases per Year")
        # Also call the improved plot function
        plot_cases_per_year(df, out_path=r'C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\plots\cases_per_year.png')


    # sector evolution over time
    if 'year' in df.columns:
        sector_year = df.groupby(['year', 'sector']).size().unstack(fill_value=0)
        sector_year.plot(figsize=(12, 6))


    ##############################
    # Decision stage keyword mapping — structured like sector_keywords
    decision_stage_keywords = {
        "Approval": [
            "approves", "approved", "approval", "clears", "cleared", "clearance",
            "authorises", "authorised", "authorizes", "authorized", "authorisation",
            "endorses", "endorsed", "endorsement", "green light","go ahead",
            "allows", "allowed", "permits", "permitted", "unconditional",
            "compatible", "no objections", "accepts", "accepted",
        ],
        "Conditional Approval": [
            "conditional", "subject to conditions", "commitments",
            "subject to commitments", "with conditions", "divest", "divestiture",
            "undertakings", "remedies", "remedy",
        ],
        "Investigation": [
            "opens", "opened", "investigation", "in-depth", "inquiry",
            "probe", "probes", "launches", "launched", "initiates", "initiated",
            "examines", "examining", "review", "reviews", "phase ii", "phase 2",
            "preliminary", "dawn raid", "raid", "raids", "inspect", "inspects",
            "inspection", "searches", "refers", "referral", "formal procedure",
        ],
        "Objection": [
            "sends statement of objections", "statement of objections",
            "objections", "objection", "warns", "warning", "concerns",
            "charges", "charged",
        ],
        "Fine": [
            "fines", "fined", "fine", "penalty", "penalises", "penalizes",
            "sanction", "sanctions", "sanctioned",
            "million euro fine", "billion euro fine",
            "cartel", "infringement", "infringed", "breach", "breached",
            "violates", "violated", "violation",
        ],
        "Prohibition": [
            "prohibits", "prohibited", "prohibition",
            "blocks", "blocked", "blocking",
            "rejects", "rejected", "rejection",
            "refuses", "refused", "refusal",
            "opposes", "opposed", "vetoes", "veto",
            "incompatible", "illegal",
        ],
        "Settlement": [
            "settles", "settled", "settlement",
            "accepts commitments", "commitment decision",
        ],
        "Recovery": [
            "recover", "recovery", "orders recovery",
            "repay", "repayment", "pay back",
        ],
        "Closure": [
            "closes", "closed", "closure",
            "concludes", "concluded", "terminates", "terminated",
            "dismisses", "dismissed", "drops", "dropped",
            "withdraws", "withdrawn",
        ],
    }

    binary_map = {
    "Approval":             "Approved",
    "Conditional Approval": "Approved",   
    "Settlement":           "Approved",   
    "Closure":              "Approved",   
    "Prohibition":          "Rejected",
    "Fine":                 "Rejected",   
    "Objection":            "Rejected",    
    "Recovery":             "Rejected",
        }

    def decision_stage(text):
        if not text:
            return "Other"
        text_lower = text.lower()
        scores = {stage: sum(1 for kw in keywords if kw in text_lower) for stage, keywords in decision_stage_keywords.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "Other"

    df['decision_stage'] = df['text'].apply(decision_stage)
    df['outcome_binary'] = df['decision_stage'].map(binary_map)


    year_stage = df.groupby(['year', 'decision_stage']).size().unstack(fill_value=0) if 'year' in df.columns else pd.DataFrame()

    if not year_stage.empty:
        year_stage['strictness'] = (
            year_stage.get('Fine', 0) +
            year_stage.get('Prohibition', 0)
        ) / year_stage.sum(axis=1)
        year_stage['strictness'].plot(title="Strictness Index")

    #####################################################

    print("\n--- DATA OVERVIEW ---")
    print(f"Total cases: {len(df)}")
    if 'year' in df.columns:
        print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print("\nCase type distribution:")
    if 'case_type' in df.columns:
        print(df['case_type'].value_counts())
    print("\nSector distribution:")
    print(df['sector'].value_counts())
    print("\nDecision stage distribution:")
    print(df['decision_stage'].value_counts())
    print("\nData completeness (% non-null):")
    key_cols = ['title', 'summary', 'date', 'year', 'case_type', 'sector', 'decision_stage', 'country', 'company']
    for col in key_cols:
        if col in df.columns:
            non_null_pct = df[col].notna().mean() * 100
            print(f"  {col:15s}: {non_null_pct:6.2f}%")

    print("\nUniqueness overview:")
    for col in ['case_type', 'sector', 'decision_stage', 'country', 'company']:
        if col in df.columns:
            print(f"  Unique {col:12s}: {df[col].nunique(dropna=True)}")

    print("\nText length overview:")
    title_len = df.get('title', '').fillna('').astype(str).str.split().str.len()
    summary_len = df.get('summary', '').fillna('').astype(str).str.split().str.len()
    print(f"  Avg title words   : {title_len.mean():.1f}")
    print(f"  Avg summary words : {summary_len.mean():.1f}")
    print(f"  Median summary words: {summary_len.median():.1f}")

    print("\nTop countries (non-null):")
    if 'country' in df.columns:
        print(df['country'].dropna().value_counts().head(10))

    print("\nTop companies (non-null):")
    if 'company' in df.columns:
        print(df['company'].dropna().value_counts().head(10))

    if not year_stage.empty and 'strictness' in year_stage.columns:
        strictness_series = year_stage['strictness'].dropna()
        if len(strictness_series) > 0:
            print("\nStrictness index overview:")
            print(f"  Average strictness: {strictness_series.mean():.3f}")
            print(f"  Max strictness year: {strictness_series.idxmax()} ({strictness_series.max():.3f})")
            print(f"  Min strictness year: {strictness_series.idxmin()} ({strictness_series.min():.3f})")

    print("\nTop 10 year x decision_stage combinations:")
    top_year_stage = (
        df.groupby(['year', 'decision_stage'])
          .size()
          .reset_index(name='count')
          .sort_values('count', ascending=False)
          .head(10)
    )
    print(top_year_stage)
    print("\nSample rows:")
    sample_cols = [c for c in ['title', 'sector', 'decision_stage', 'year', 'case_type'] if c in df.columns]
    print(df[sample_cols].head(10))
    print("\n--- END OVERVIEW ---\n")


    # ensure output directory exists and save
    out_dir = os.path.dirname(OUTPUT_DIR)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(OUTPUT_DIR, index=False)


    ######################################lineaer regresion for enforcement

    cases_per_year = df.groupby("year").size().reset_index(name="cases")

    X = cases_per_year["year"].values.reshape(-1,1)
    y = cases_per_year["cases"].values

    model = LinearRegression()
    model.fit(X,y)

    print("Trend slope:", model.coef_[0])

    #####################################correlation between year and cases
    corr, p_value = stats.pearsonr(cases_per_year["year"], cases_per_year["cases"])
    print(f"Correlation between year and cases: {corr:.3f}, p-value: {p_value:.3f}")

    #############################################HHi
    if 'sector' in df.columns:
        sector_counts = df['sector'].value_counts()
        hhi = (sector_counts**2).sum() / (sector_counts.sum()**2)
        print(f"\nMarket concentration (HHI) based on sector distribution: {hhi:.4f}")
    
    hhi_year = df.groupby('year')['sector'].value_counts().unstack(fill_value=0).apply(lambda x: (x**2).sum() / (x.sum()**2), axis=1)
    print("\nMarket concentration (HHI) by year:")
    print(hhi_year)



if __name__ == "__main__":
    main()