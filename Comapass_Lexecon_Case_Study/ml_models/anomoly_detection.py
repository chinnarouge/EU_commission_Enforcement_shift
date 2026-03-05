import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack

DATA_PATH = r'C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\processesd_data\final_competition_cases.csv'
PLOTS_DIR = r'C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_data(path=DATA_PATH):
    """Load CSV and print basic info."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} cases from {path}")
    return df


def build_features(df, max_tfidf=300):
    """Add text_length, encode categoricals, vectorise text.
    Returns (df, X_combined) where X_combined is a sparse matrix."""
    df['text_length'] = df['text'].fillna('').apply(len)

    le = LabelEncoder()
    df['case_type_encoded'] = le.fit_transform(df['case_type'].fillna('Unknown'))
    df['sector_encoded'] = le.fit_transform(df['sector'].fillna('Unknown'))
    df['outcome_encoded'] = le.fit_transform(df['decision_stage'].fillna('Unknown'))

    vectorizer = TfidfVectorizer(max_features=max_tfidf, stop_words='english')
    X_text = vectorizer.fit_transform(df['text'].astype(str))

    X_structured = df[['year', 'text_length',
                        'case_type_encoded', 'sector_encoded',
                        'outcome_encoded']].fillna(0)
    X_combined = hstack([csr_matrix(X_structured), X_text])
    return df, X_combined


def run_isolation_forest(df, X, contamination=0.02):
    """Fit Isolation Forest, add anomaly_score & is_anomaly columns."""
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X)
    df['anomaly_score'] = iso.score_samples(X)   # lower = more anomalous
    df['is_anomaly'] = iso.predict(X)             # -1 = anomaly, 1 = normal
    return df


def print_anomaly_summary(df, top_n=15):
    """Print the most anomalous cases and their breakdowns."""
    anomalies = df[df['is_anomaly'] == -1].sort_values('anomaly_score')
    print(f"\nFlagged {len(anomalies)} cases as anomalous out of {len(df):,}")
    print(f"\nTop {top_n} most anomalous cases:")
    cols = ['article_id', 'year', 'case_type', 'sector',
            'decision_stage', 'text_length', 'anomaly_score']
    cols = [c for c in cols if c in df.columns]
    print(anomalies[cols].head(top_n).to_string(index=False))

    print("\nWhich sectors appear most in anomalies?")
    print(anomalies['sector'].value_counts())
    print("\nWhich decision outcomes appear most in anomalies?")
    print(anomalies['decision_stage'].value_counts())


def plot_anomaly_scores(df):
    """Histogram of anomaly scores split by normal / anomaly."""
    fig, ax = plt.subplots(figsize=(10, 5))
    normal = df[df['is_anomaly'] == 1]['anomaly_score']
    anomalous = df[df['is_anomaly'] == -1]['anomaly_score']
    ax.hist(normal, bins=50, alpha=0.7, color='steelblue', label='Normal')
    ax.hist(anomalous, bins=20, alpha=0.7, color='red', label='Anomaly')
    ax.set_title('Isolation Forest Anomaly Scores', fontweight='bold')
    ax.set_xlabel('Anomaly Score (lower = more anomalous)')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_sector_anomaly_concentration(df):
    stats = df.groupby('sector').agg(
        total=('is_anomaly', 'count'),
        anomaly_count=('is_anomaly', lambda x: (x == -1).sum()),
        mean_score=('anomaly_score', 'mean')
    ).reset_index()
    stats = stats[stats['total'] >= 5]
    stats['anomaly_rate'] = stats['anomaly_count'] / stats['total'] * 100

    print("\nSECTOR ANOMALY CONCENTRATION (Isolation Forest output)")
    print("=" * 65)
    print("(Sectors where enforcement cases are most structurally atypical)\n")
    print(stats[['sector', 'total', 'anomaly_count', 'anomaly_rate', 'mean_score']].round(2).to_string(index=False))

    # ── Plot 1: Anomaly Rate per Sector ──────────────────────────────────
    rate_sorted = stats.sort_values('anomaly_rate', ascending=True)
    bar_height = max(0.4, min(0.7, 6 / len(rate_sorted)))
    fig_h = max(5, len(rate_sorted) * 0.55 + 1.5)

    fig1, ax1 = plt.subplots(figsize=(10, fig_h))
    norm = rate_sorted['anomaly_rate'] / rate_sorted['anomaly_rate'].max()
    colors1 = plt.cm.Reds(0.3 + 0.65 * norm)
    bars = ax1.barh(rate_sorted['sector'], rate_sorted['anomaly_rate'],
                    height=bar_height, color=colors1, edgecolor='white')
    for bar, n, rate in zip(bars, rate_sorted['total'], rate_sorted['anomaly_rate']):
        ax1.text(rate + 0.1, bar.get_y() + bar.get_height() / 2,
                 f'  n={n}', va='center', fontsize=9)
    ax1.set_xlabel('Anomaly Rate (%)', fontsize=11)
    ax1.set_title('% of Cases Flagged as Anomalous by Isolation Forest\nper Sector',
                  fontsize=12, fontweight='bold', pad=12)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.set_xlim(0, rate_sorted['anomaly_rate'].max() * 1.25)
    ax1.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'sector_shifts.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # ── Plot 2: Mean Anomaly Score per Sector ────────────────────────────
    score_sorted = stats.sort_values('mean_score', ascending=False)
    median_score = stats['mean_score'].median()

    fig2, ax2 = plt.subplots(figsize=(10, fig_h))
    colors2 = ['#d73027' if s < median_score else '#4575b4'
               for s in score_sorted['mean_score']]
    ax2.barh(score_sorted['sector'], score_sorted['mean_score'],
             height=bar_height, color=colors2, edgecolor='white')
    ax2.axvline(median_score, color='black', linewidth=1.2,
                linestyle='--', label=f'Median ({median_score:.3f})')
    for i, (score, n) in enumerate(zip(score_sorted['mean_score'], score_sorted['total'])):
        ax2.text(score + 0.001, i,
                 f'  n={n}', va='center', fontsize=9)
    ax2.set_xlabel('Mean Anomaly Score  (more negative = more atypical)', fontsize=11)
    ax2.set_title('Mean Isolation Forest Score per Sector\nRed = below median (structurally irregular)',
                  fontsize=12, fontweight='bold', pad=12)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'sector_anomaly_scores.png'), dpi=300, bbox_inches='tight')
    plt.show()

    return stats


def find_surprising_outcomes(df, top_n=30, save_csv=os.path.join(PLOTS_DIR, '..', 'processesd_data', 'surprising_cases.csv')):
    """Flag cases whose outcome is unusual for their sector+case_type."""
    expected = df.groupby(['sector', 'case_type'])['decision_stage'].apply(
        lambda x: (x == 'Approval').mean()
    ).reset_index(name='approval_rate')

    df = df.merge(expected, on=['sector', 'case_type'], how='left',
                  suffixes=('', '_dup'))
    # drop duplicate column if re-run
    for c in [col for col in df.columns if col.endswith('_dup')]:
        df.drop(columns=c, inplace=True)

    df['was_approved'] = (df['decision_stage'] == 'Approval').astype(int)
    df['surprise_score'] = abs(df['was_approved'] - df['approval_rate'])

    surprising = df.sort_values('surprise_score', ascending=False).head(top_n)

    print("MOST SURPRISING CASE OUTCOMES")
    print("=" * 70)
    print("(Cases where the outcome was unusual for their sector + case type)\n")
    cols = ['article_id', 'year', 'sector', 'case_type',
            'decision_stage', 'approval_rate', 'surprise_score']
    cols = [c for c in cols if c in df.columns]
    print(surprising[cols].to_string(index=False))

    if save_csv:
        surprising[cols].to_csv(save_csv, index=False)
        print(f"\nSaved: {save_csv}")
    return df

def main():
    df = load_data()
    df, X = build_features(df)
    df = run_isolation_forest(df, X)
    print_anomaly_summary(df)
    plot_anomaly_scores(df)
    plot_sector_anomaly_concentration(df)
    df = find_surprising_outcomes(df)

if __name__ == '__main__':
    main()