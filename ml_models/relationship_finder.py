import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv(r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\processesd_data\final_competition_cases.csv")
df = df[df['outcome_binary'].isin(['Approved', 'Rejected'])]

'''
def era(year):
    return 'pre2000' if year < 2000 else '2000s' if year < 2010 else '2010s' if year < 2020 else '2020s'

df['era'] = df['year'].apply(era)

def build_transaction(row):
    return [
        f"sector_{row['sector']}",
        f"type_{row['case_type']}",
        f"outcome_{row['outcome_binary']}"
    ]

def get_rules(subset):
    trans = subset.apply(build_transaction, axis=1).tolist()
    te = TransactionEncoder()
    enc = pd.DataFrame(te.fit_transform(trans), columns=te.columns_)
    fi  = apriori(enc, min_support=0.05, use_colnames=True, max_len=3)
    r   = association_rules(fi, metric="confidence", min_threshold=0.5)
    r   = r.sort_values('lift', ascending=False)

    clean = r[
        r['consequents'].apply(lambda x: len(x) == 1 and all('outcome_' in i for i in x)) &
        r['antecedents'].apply(lambda x: not any('outcome_' in i for i in x))
    ].copy()

    clean['IF']   = clean['antecedents'].apply(lambda x: ' + '.join(sorted(x)))
    clean['THEN'] = clean['consequents'].apply(lambda x: list(x)[0].replace('outcome_', ''))
    return clean

eras = ['pre2000', '2000s', '2010s', '2020s']

# One row of 2 plots per era: scatter + bar
fig, axes = plt.subplots(len(eras), 2, figsize=(18, len(eras) * 5))
fig.suptitle('Association Rules by Era — Sector + Case Type → Outcome',
             fontsize=16, fontweight='bold', y=1.01)

for row_idx, period in enumerate(eras):
    subset = df[df['era'] == period]

    ax_scatter = axes[row_idx, 0]
    ax_bar     = axes[row_idx, 1]

    if len(subset) < 50:
        ax_scatter.text(0.5, 0.5, f'{period}: not enough data',
                        ha='center', va='center', transform=ax_scatter.transAxes)
        ax_bar.axis('off')
        continue

    try:
        rules = get_rules(subset)
    except Exception:
        ax_scatter.text(0.5, 0.5, f'{period}: no rules found',
                        ha='center', va='center', transform=ax_scatter.transAxes)
        ax_bar.axis('off')
        continue

    approved = rules[rules['THEN'] == 'Approved']
    rejected = rules[rules['THEN'] == 'Rejected']

    # --- Scatter ---
    ax_scatter.scatter(approved['support'], approved['confidence'],
                       s=approved['lift'] * 40, alpha=0.7,
                       color='steelblue', label='→ Approved')
    ax_scatter.scatter(rejected['support'], rejected['confidence'],
                       s=rejected['lift'] * 40, alpha=0.7,
                       color='crimson', label='→ Rejected')
    ax_scatter.set_title(f'{period}  ({len(subset)} cases)\nSupport vs Confidence',
                         fontweight='bold')
    ax_scatter.set_xlabel('Support')
    ax_scatter.set_ylabel('Confidence')
    ax_scatter.legend(fontsize=8)
    ax_scatter.grid(alpha=0.3)

    # --- Bar: top 10 rules ---
    top10  = rules.head(10)
    colors = ['steelblue' if t == 'Approved' else 'crimson' for t in top10['THEN']]
    ax_bar.barh(range(len(top10)), top10['confidence'], color=colors, alpha=0.8)
    ax_bar.set_yticks(range(len(top10)))
    ax_bar.set_yticklabels([r[:55] for r in top10['IF']], fontsize=8)
    ax_bar.set_xlabel('Confidence')
    ax_bar.set_title(f'{period} — Top Rules\nBlue=Approved  Red=Rejected',
                     fontweight='bold')
    ax_bar.invert_yaxis()
    ax_bar.axvline(0.5, color='black', linestyle='--', alpha=0.4)
    ax_bar.grid(alpha=0.3, axis='x')

    # Annotate confidence values on bars
    for i, (_, r) in enumerate(top10.iterrows()):
        ax_bar.text(r['confidence'] + 0.005, i,
                    f"{r['confidence']:.0%}  lift {r['lift']:.1f}x",
                    va='center', fontsize=7)

plt.tight_layout()
plt.savefig('apriori_by_era.png', dpi=300, bbox_inches='tight')
plt.show()'''

def era(year):
    return 'pre2000' if year < 2000 else '2000s' if year < 2010 else '2010s' if year < 2020 else '2020s'

df['era'] = df['year'].apply(era)

# Get top 3 TF-IDF words per document
vectorizer = TfidfVectorizer(max_features=300, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df['text'])
feature_names = vectorizer.get_feature_names_out()

def top3_words(row_vec):
    scores = zip(feature_names, row_vec.toarray()[0])
    top = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
    return [word for word, score in top if score > 0]

print("Extracting top 3 TF-IDF words per case...")
df['top_words'] = [top3_words(tfidf_matrix[i]) for i in range(len(df))]

# Build transactions from top words + outcome only
def build_transaction(row):
    items = [f"word_{w}" for w in row['top_words']]
    items.append(f"outcome_{row['outcome_binary']}")
    return items

def get_rules(subset):
    trans = subset.apply(build_transaction, axis=1).tolist()
    te  = TransactionEncoder()
    enc = pd.DataFrame(te.fit_transform(trans), columns=te.columns_)
    fi  = apriori(enc, min_support=0.03, use_colnames=True, max_len=4)
    r   = association_rules(fi, metric="confidence", min_threshold=0.5)
    r   = r.sort_values('lift', ascending=False)

    clean = r[
        r['consequents'].apply(lambda x: len(x) == 1 and all('outcome_' in i for i in x)) &
        r['antecedents'].apply(lambda x: not any('outcome_' in i for i in x))
    ].copy()

    clean['IF']   = clean['antecedents'].apply(lambda x: ' + '.join(sorted(i.replace('word_', '') for i in x)))
    clean['THEN'] = clean['consequents'].apply(lambda x: list(x)[0].replace('outcome_', ''))
    return clean

# Plot
eras = ['pre2000', '2000s', '2010s', '2020s']
fig, axes = plt.subplots(len(eras), 2, figsize=(18, len(eras) * 5))
fig.suptitle('Text-Driven Association Rules by Era — Top TF-IDF Words → Outcome',
             fontsize=15, fontweight='bold', y=1.01)

for row_idx, period in enumerate(eras):
    subset   = df[df['era'] == period]
    ax_scatter = axes[row_idx, 0]
    ax_bar     = axes[row_idx, 1]

    if len(subset) < 50:
        ax_scatter.text(0.5, 0.5, f'{period}: not enough data', ha='center', va='center', transform=ax_scatter.transAxes)
        ax_bar.axis('off')
        continue

    try:
        rules = get_rules(subset)
    except Exception:
        ax_scatter.text(0.5, 0.5, f'{period}: no rules found', ha='center', va='center', transform=ax_scatter.transAxes)
        ax_bar.axis('off')
        continue

    if rules.empty:
        ax_scatter.text(0.5, 0.5, f'{period}: no rules passed threshold', ha='center', va='center', transform=ax_scatter.transAxes)
        ax_bar.axis('off')
        continue

    approved = rules[rules['THEN'] == 'Approved']
    rejected = rules[rules['THEN'] == 'Rejected']

    # Scatter
    ax_scatter.scatter(approved['support'], approved['confidence'],
                       s=approved['lift'] * 40, alpha=0.7, color='steelblue', label='→ Approved')
    ax_scatter.scatter(rejected['support'], rejected['confidence'],
                       s=rejected['lift'] * 40, alpha=0.7, color='crimson', label='→ Rejected')
    ax_scatter.set_title(f'{period}  ({len(subset)} cases)', fontweight='bold')
    ax_scatter.set_xlabel('Support')
    ax_scatter.set_ylabel('Confidence')
    ax_scatter.legend(fontsize=8)
    ax_scatter.grid(alpha=0.3)

    # Bar
    top10  = rules.head(10)
    colors = ['steelblue' if t == 'Approved' else 'crimson' for t in top10['THEN']]
    ax_bar.barh(range(len(top10)), top10['confidence'], color=colors, alpha=0.8)
    ax_bar.set_yticks(range(len(top10)))
    ax_bar.set_yticklabels([r[:55] for r in top10['IF']], fontsize=8)
    ax_bar.set_xlabel('Confidence')
    ax_bar.set_title(f'{period} — Top Word Combination Rules\nBlue=Approved  Red=Rejected', fontweight='bold')
    ax_bar.invert_yaxis()
    ax_bar.axvline(0.5, color='black', linestyle='--', alpha=0.4)
    ax_bar.grid(alpha=0.3, axis='x')

    for i, (_, r) in enumerate(top10.iterrows()):
        ax_bar.text(r['confidence'] + 0.005, i,
                    f"{r['confidence']:.0%}  lift {r['lift']:.1f}x",
                    va='center', fontsize=7)

plt.tight_layout()
import os as _os
_plots_dir = r'C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\plots'
_os.makedirs(_plots_dir, exist_ok=True)
plt.savefig(_os.path.join(_plots_dir, 'apriori_text_driven_by_era.png'), dpi=300, bbox_inches='tight')
plt.show()