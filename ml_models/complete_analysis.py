import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import zscore
from collections import defaultdict
import warnings
import os
import seaborn as sns

FILE_PATH = r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\processesd_data\final_competition_cases.csv"
PLOTS_DIR = r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv(FILE_PATH)

###############################################################################
#basic stats
print("\nTotal cases:", len(df))
print("\nYears covered:", df["year"].min(), "-", df["year"].max())
print("\nCase types:", df["case_type"].nunique())
print("\nSectors:", df["sector"].nunique())
print("\nDecision_stages:", df["decision_stage"].nunique())
print("\nOutcomes:", df["outcome_binary"].nunique())


##############################################################################
# Correlations — Cramér's V (proper categorical association measure)
from scipy.stats import chi2_contingency

def cramers_v(col1, col2, data):
    ct = pd.crosstab(data[col1], data[col2])
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    r, k = ct.shape
    v = np.sqrt(chi2 / (n * (min(r, k) - 1)))
    return v

df_labeled = df[df['outcome_binary'].isin(['Approved', 'Rejected'])].copy()

cv_sector  = cramers_v('sector',    'outcome_binary', df_labeled)
cv_casetype = cramers_v('case_type', 'outcome_binary', df_labeled)

print(f"\nCramér's V (sector  → outcome): {cv_sector:.3f}   "
      f"{'strong' if cv_sector > 0.3 else 'moderate' if cv_sector > 0.1 else 'weak'} association")
print(f"Cramér's V (case type → outcome): {cv_casetype:.3f}   "
      f"{'strong' if cv_casetype > 0.3 else 'moderate' if cv_casetype > 0.1 else 'weak'} association")

# Approval rate per sector
print("\nApproval rate by sector:")
sector_rates = (df_labeled.groupby('sector')['outcome_binary']
                .value_counts(normalize=True)
                .mul(100).round(1)
                .unstack(fill_value=0))
print(sector_rates.to_string())

print("\nApproval rate by case type:")
casetype_rates = (df_labeled.groupby('case_type')['outcome_binary']
                  .value_counts(normalize=True)
                  .mul(100).round(1)
                  .unstack(fill_value=0))
print(casetype_rates.to_string())

###############################################################################
# Plots — Approved vs Rejected counts per sector and case type (grouped bars)
OUTCOME_COLORS = {'Approved': '#2ca02c', 'Rejected': '#d62728'}

#Approved vs Rejected by Sector
sector_counts = (df_labeled.groupby(['sector', 'outcome_binary'])
                 .size().unstack(fill_value=0)
                 .reindex(columns=['Approved', 'Rejected'], fill_value=0))

fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(sector_counts))
width = 0.35
bars_a = ax.bar(x - width/2, sector_counts['Approved'], width,
                label='Approved', color=OUTCOME_COLORS['Approved'], edgecolor='white')
bars_r = ax.bar(x + width/2, sector_counts['Rejected'], width,
                label='Rejected', color=OUTCOME_COLORS['Rejected'], edgecolor='white')
# Count labels on bars
for bar in bars_a:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2, str(int(h)), ha='center', va='bottom', fontsize=7)
for bar in bars_r:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2, str(int(h)), ha='center', va='bottom', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(sector_counts.index, rotation=30, ha='right', fontsize=9)
ax.set_title('Approved vs Rejected Cases by Sector', fontsize=14, fontweight='bold')
ax.set_xlabel('Sector')
ax.set_ylabel('Number of Cases')
ax.legend(title='Outcome')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'sector_outcome_bars.png'), dpi=300)
plt.show()

#Approved vs Rejected by Case Type
casetype_counts = (df_labeled.groupby(['case_type', 'outcome_binary'])
                   .size().unstack(fill_value=0)
                   .reindex(columns=['Approved', 'Rejected'], fill_value=0))

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(casetype_counts))
bars_a2 = ax.bar(x - width/2, casetype_counts['Approved'], width,
                 label='Approved', color=OUTCOME_COLORS['Approved'], edgecolor='white')
bars_r2 = ax.bar(x + width/2, casetype_counts['Rejected'], width,
                 label='Rejected', color=OUTCOME_COLORS['Rejected'], edgecolor='white')
for bar in bars_a2:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2, str(int(h)), ha='center', va='bottom', fontsize=9)
for bar in bars_r2:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2, str(int(h)), ha='center', va='bottom', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(casetype_counts.index, rotation=15, ha='right', fontsize=10)
ax.set_title('Approved vs Rejected Cases by Case Type', fontsize=14, fontweight='bold')
ax.set_xlabel('Case Type')
ax.set_ylabel('Number of Cases')
ax.legend(title='Outcome')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'casetype_outcome_bars.png'), dpi=300)
plt.show()

# Approval rate over time by sector
sector_year_rate = (df_labeled.groupby(['year', 'sector'])['outcome_binary']
                    .agg(lambda s: (s == 'Approved').mean())
                    .reset_index(name='approval_rate'))

plt.figure(figsize=(13, 6))
for sector in sorted(sector_year_rate['sector'].unique()):
    data = sector_year_rate[sector_year_rate['sector'] == sector]
    plt.plot(data['year'], data['approval_rate'] * 100, label=sector, marker='o', markersize=3)
plt.axhline(50, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='50% line')
plt.ylim(0, 105)
plt.title('Approval Rate Over Time by Sector', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Approval Rate (%)')
plt.legend(title='Sector', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'sector_approval_rate_time.png'), dpi=300)
plt.show()

# Approval rate over time by case type
casetype_year_rate = (df_labeled.groupby(['year', 'case_type'])['outcome_binary']
                      .agg(lambda s: (s == 'Approved').mean())
                      .reset_index(name='approval_rate'))

plt.figure(figsize=(12, 5))
for ct in sorted(casetype_year_rate['case_type'].unique()):
    data = casetype_year_rate[casetype_year_rate['case_type'] == ct]
    plt.plot(data['year'], data['approval_rate'] * 100, label=ct, marker='o', markersize=3)
plt.axhline(50, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='50% line')
plt.ylim(0, 105)
plt.title('Approval Rate Over Time by Case Type', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Approval Rate (%)')
plt.legend(title='Case Type')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'casetype_approval_rate_time.png'), dpi=300)
plt.show()

# Stacked bar: number of cases by year, coloured by sector
yearly_sector = df.groupby(['year', 'sector']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(14, 5))
bottom = np.zeros(len(yearly_sector))
colors = plt.colormaps['tab10'].resampled(len(yearly_sector.columns)).colors
for col, color in zip(yearly_sector.columns, colors):
    ax.bar(yearly_sector.index, yearly_sector[col], bottom=bottom, label=col, color=color, edgecolor='none')
    bottom += yearly_sector[col].values
# Show every 2nd year to avoid label overlap
years = yearly_sector.index.tolist()
ax.set_xticks([y for y in years if y % 2 == 0])
ax.set_xticklabels([y for y in years if y % 2 == 0], rotation=45, ha='right', fontsize=8)
ax.set_title('Number of Cases by Year and Sector (Stacked)', fontsize=14, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Cases')
ax.legend(title='Sector', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'cases_by_year_sector.png'), dpi=300)
plt.show()

#Stacked-area by case type + line chart by sector.
yearly_casetype = df.groupby(['year', 'case_type']).size().unstack(fill_value=0)
yearly_sector = df.groupby(['year', 'sector']).size().unstack(fill_value=0)
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
yearly_casetype.plot(kind='area', stacked=True, ax=axes[0],
                        alpha=0.7, colormap='tab10')
axes[0].set_title('Cases by Type Over Time', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Number of Cases')
axes[0].legend(title='Case Type', bbox_to_anchor=(1.01, 1), loc='upper left')
axes[0].grid(alpha=0.3)
yearly_sector.plot(kind='line', ax=axes[1], linewidth=2,
                        marker='o', markersize=3, colormap='tab10')
axes[1].set_title('Cases by Sector Over Time', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Number of Cases')
axes[1].legend(title='Sector', bbox_to_anchor=(1.01, 1), loc='upper left')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'casetype_and_sector_vs_time.png'), dpi=300)
plt.show()

####################################################################################
#heatmap of case type vs sector
pivot = df.pivot_table(index='case_type', columns='sector', values='article_id', aggfunc='count', fill_value=0)
plt.figure(figsize=(14, 8))
sns.heatmap(pivot, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap of Case Type vs Sector', fontsize=14, fontweight='bold')
plt.xlabel('Sector')
plt.ylabel('Case Type')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'casetype_sector_heatmap.png'), dpi=300)
plt.show()

# Box plot: distribution of approval rate per sector (across all years)
fig, ax = plt.subplots(figsize=(13, 6))
sector_order = (sector_year_rate.groupby('sector')['approval_rate']
                .median().sort_values(ascending=False).index.tolist())
plot_data = [sector_year_rate.loc[sector_year_rate['sector'] == s, 'approval_rate'].values * 100
             for s in sector_order]
bp = ax.boxplot(plot_data, patch_artist=True, medianprops=dict(color='black', linewidth=2))
colors_box = plt.colormaps['tab10'].resampled(len(sector_order)).colors
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(50, color='red', linestyle='--', linewidth=0.8, alpha=0.6, label='50% line')
ax.set_xticks(range(1, len(sector_order) + 1))
ax.set_xticklabels(sector_order, rotation=30, ha='right', fontsize=9)
ax.set_title('Approval Rate Distribution by Sector (across years)', fontsize=14, fontweight='bold')
ax.set_xlabel('Sector (sorted by median approval rate)')
ax.set_ylabel('Approval Rate (%)')
ax.set_ylim(0, 105)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'sector_approval_rate_boxplot.png'), dpi=300)
plt.show()


#Zscore anomalies in sector-year approval rates plot
sector_year_counts = (
    df.groupby(["year", "sector"])
    .size()
    .reset_index(name="count")
)

sector_year_approval_rates = sector_year_counts.merge(sector_year_rate, on=["year", "sector"], how="left")
sector_year_approval_rates["approval_rate_zscore"] = (
    sector_year_approval_rates.groupby("sector")["approval_rate"].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
)
anomalies = sector_year_approval_rates[abs(sector_year_approval_rates["approval_rate_zscore"]) > 2]
print("Anomalies in sector-year approval rates:")
print(anomalies[["year", "sector", "approval_rate", "approval_rate_zscore"]])
# Plot anomalies
fig, ax = plt.subplots(figsize=(14, 6))
for sector in anomalies['sector'].unique():
    sector_data = anomalies[anomalies['sector'] == sector]
    ax.scatter(sector_data['year'], sector_data['approval_rate'], label=sector, s=100)
ax.set_xlabel('Year')
ax.set_ylabel('Approval Rate (%)')
ax.set_title('Anomalies in Sector-Year Approval Rates (Z-score > 2)')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'anomalies_sector_year_approval_rates.png'), dpi=300)
plt.show()


#visualise cramers v associations in a heatmap
assoc_matrix = pd.DataFrame({
    'sector': [cv_sector],
    'case_type': [cv_casetype]
}, index=['outcome_binary'])
plt.figure(figsize=(6, 4))
sns.heatmap(assoc_matrix, annot=True, cmap='Reds', vmin=0, vmax=1)
plt.title("Cramér's V Association with Outcome", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'cramers_v_heatmap.png'), dpi=300)
plt.show()
#########################################

CUTOFF = 2010

df['period'] = df['year'].apply(lambda x: f'Before {CUTOFF}' if x < CUTOFF else f'After {CUTOFF}')
before_label, after_label = f'Before {CUTOFF}', f'After {CUTOFF}'

shift = (
    df.groupby(['sector', 'period'])
    .apply(lambda x: (x['decision_stage'] == 'Investigation').mean() * 100)
    .unstack(fill_value=0)
)
# Ensure both columns exist even if one period has no data
for col in [before_label, after_label]:
    if col not in shift.columns:
        shift[col] = 0.0
shift['change_pp'] = shift[after_label] - shift[before_label]
shift = shift.sort_values('change_pp', ascending=True)

print("\nSECTORS WITH BIGGEST SHIFT IN INVESTIGATION RATE")
print("=" * 60)
print(f"(Positive = more investigations after {CUTOFF})\n")
print(shift[[before_label, after_label, 'change_pp']].round(1).to_string())

fig, ax = plt.subplots(figsize=(10, max(5, len(shift) * 0.6 + 1.5)))
colors = ['#d73027' if v > 0 else '#4575b4' for v in shift['change_pp']]
bars = ax.barh(shift.index, shift['change_pp'], color=colors, edgecolor='white', height=0.6)
ax.axvline(0, color='black', linewidth=1.0)
for bar, val in zip(bars, shift['change_pp']):
    sign = '+' if val >= 0 else ''
    ax.text(val + (0.3 if val >= 0 else -0.3),
            bar.get_y() + bar.get_height() / 2,
            f'{sign}{val:.1f} pp', va='center',
            ha='left' if val >= 0 else 'right', fontsize=9)
ax.set_xlabel('Change in Investigation Rate (percentage points)', fontsize=11)
ax.set_title(f'Shift in Investigation Rate per Sector\nBefore vs After {CUTOFF}  '
             f'(Red = stricter enforcement post-{CUTOFF})',
             fontsize=12, fontweight='bold', pad=12)
ax.tick_params(axis='y', labelsize=10)
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'sector_investigation_shift.png'), dpi=300, bbox_inches='tight')
plt.show()
########################################
#top companies by number of cases
top_companies = df['company'].value_counts().head(7)
#plot
plt.figure(figsize=(12, 6))
sns.barplot(x=top_companies.values, y=top_companies.index, palette='viridis')
plt.title('Top 7 Companies by Number of Cases', fontsize=14, fontweight='bold')
plt.xlabel('Number of Cases')
plt.ylabel('Company')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'top_companies_cases.png'), dpi=300)
plt.show()