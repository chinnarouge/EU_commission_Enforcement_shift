import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
from matplotlib.gridspec import GridSpec

DATA_PATH = r"C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\processesd_data\final_competition_cases.csv"

DOMAIN_STOPWORDS = {
    # EC / legal boilerplate
    'commission', 'european', 'market', 'markets', 'competition', 'company',
    'companies', 'case', 'cases', 'decision', 'decisions', 'article', 'articles',
    'ec', 'eu', 'eea', 'treaty', 'treaties', 'regulation', 'regulations',
    'directive', 'directives', 'member', 'states', 'state', 'union',
    'council', 'parliament', 'brussels', 'luxembourg',
    # Merger / procedure terms
    'aid', 'aids', 'position', 'relevant', 'concerned', 'operation', 'operations',
    'transaction', 'transactions', 'proposed', 'notified', 'notification',
    'pursuant', 'compatible', 'incompatible', 'common', 'internal',
    'acquisition', 'acquisitions', 'control', 'sole', 'joint', 'venture',
    'concentration', 'concentrations', 'merger', 'mergers',
    # Generic business
    'activities', 'activity', 'products', 'product', 'services', 'service',
    'based', 'active', 'given', 'lead', 'leads', 'creation', 'created',
    'strengthening', 'dominant', 'effective', 'effectively', 'significantly',
    'impeded', 'substantial', 'declaration', 'declared', 'approved', 'approval',
    'parties', 'party', 'undertaking', 'undertakings', 'group', 'groups',
    'new', 'share', 'shares', 'holding', 'holdings', 'subsidiary', 'subsidiaries',
    'sector', 'sectors', 'area', 'areas', 'particular', 'particularly',
    'order', 'respect', 'account', 'view', 'fact', 'basis', 'related',
    'regard', 'regards', 'according', 'therefore', 'however', 'following',
    'provided', 'provide', 'provides', 'including', 'included', 'include',
    'certain', 'also', 'well', 'two', 'three', 'one', 'first', 'second',
    'may', 'would', 'could', 'shall', 'must', 'made', 'make', 'makes',
    'subject', 'period', 'date', 'year', 'years', 'number', 'total',
    'million', 'billion', 'eur', 'euro', 'euros', 'percent', 'percentage',
    'approximately', 'respectively', 'addition', 'additional', 'already',
    'since', 'entire', 'specific', 'general', 'mainly', 'overall',
    'existing', 'present', 'currently', 'future', 'previous', 'previously',
    'level', 'terms', 'form', 'part', 'whole', 'range', 'type', 'types',
    'way', 'set', 'rights', 'right', 'measure', 'measures', 'scheme',
    'schemes', 'procedure', 'procedures', 'assessment', 'analysis',
    'information', 'conditions', 'condition', 'commitments', 'commitment',
    'investigation', 'investigations', 'proceedings', 'proceeding',
    'objections', 'objection', 'statement', 'conclusion', 'concluded',
    'considers', 'considered', 'consideration', 'effect', 'effects',
    'impact', 'supply', 'demand', 'price', 'prices', 'costs', 'cost',
    'national', 'international', 'world', 'worldwide', 'global',
    'country', 'countries', 'region', 'regions', 'regional',
    'ip', 'memo', 'press', 'release', 'ref', 'page',
}

ALL_STOPWORDS = list(ENGLISH_STOP_WORDS | DOMAIN_STOPWORDS)


def clean_for_lda(text):
    """Lowercase, strip numbers / short tokens / punctuation."""
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)          
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2]       
    return ' '.join(tokens)


def era(year):
    if year < 2000:
        return 'pre2000'
    if year < 2016:
        return '2000-2015'
    return '2016-now'


def build_lda(texts, n_topics=5, max_features=3000, min_df=10, max_df=0.25):
    """Fit LDA and return (vectorizer, lda_model, dtm, vocab)."""
    vec = CountVectorizer(
        max_features=max_features,
        stop_words=ALL_STOPWORDS,
        ngram_range=(1, 1),      # unigrams only — avoids overlap from shared bigram parts
        min_df=min_df,           # must appear in ≥10 docs
        max_df=max_df,           # must appear in <25 % of docs
        token_pattern=r'\b[a-z]{3,}\b',  # letters only, ≥3 chars
    )
    dtm = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=300,            # more iterations for convergence
        learning_decay=0.7,
        doc_topic_prior=0.1,     # sparser doc-topic → each doc belongs to fewer topics
        topic_word_prior=0.01,   # sparser topic-word → each topic uses fewer words
        n_jobs=-1,
    )
    lda.fit(dtm)
    return vec, lda, dtm, vocab


def get_topic_words(lda, vocab, n_words=10):
    """Return list of dicts with 'words' and 'weights' for each topic."""
    topics = []
    for comp in lda.components_:
        top_idx = comp.argsort()[-n_words:][::-1]
        topics.append({
            'words':   [vocab[i] for i in top_idx],
            'weights': [comp[i]  for i in top_idx],
        })
    return topics

def compute_era_shares(df, vec, lda, eras, n_topics):
    """% of docs whose dominant topic is t, per era."""
    era_shares = {}
    for period in eras:
        subset = df[df['era'] == period]['clean_text'].tolist()
        if not subset:
            era_shares[period] = [0.0] * n_topics
            continue
        dtm_era = vec.transform(subset)
        doc_topics = lda.transform(dtm_era)
        dominant = doc_topics.argmax(axis=1)
        era_shares[period] = [
            (dominant == t).sum() / len(dominant) * 100
            for t in range(n_topics)
        ]
    return era_shares


def plot_topics(global_topics, era_shares, eras, n_topics):
    """Top-word bars + era-share bars."""
    fig = plt.figure(figsize=(24, 14))
    fig.suptitle('LDA Topic Model — Composition & Era Distribution',
                 fontsize=15, fontweight='bold')

    gs     = GridSpec(2, 1, height_ratios=[1.4, 1], hspace=0.5)
    gs_top = gs[0].subgridspec(1, n_topics, wspace=0.4)
    gs_bot = gs[1].subgridspec(1, len(eras), wspace=0.35)

    colors = ['#2196F3', '#FF5722', '#4CAF50', '#FF9800', '#9C27B0',
              '#00BCD4', '#E91E63', '#8BC34A'][:n_topics]


    for t in range(n_topics):
        ax = fig.add_subplot(gs_top[0, t])
        topic = global_topics[t]
        ax.barh(topic['words'][::-1], topic['weights'][::-1],
                color=colors[t], alpha=0.8)
        ax.set_title(f'Topic {t + 1}', fontsize=11, fontweight='bold',
                     color=colors[t])
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=7)
        ax.set_xlabel('Weight', fontsize=8)
        ax.grid(alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    topic_labels = [f'T{t+1}' for t in range(n_topics)]
    y_max = max(max(era_shares[p]) for p in eras) + 8

    for col_idx, period in enumerate(eras):
        ax = fig.add_subplot(gs_bot[0, col_idx])
        shares = era_shares[period]
        bars = ax.bar(topic_labels, shares, color=colors,
                      alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.set_title(period, fontsize=11, fontweight='bold')
        ax.set_ylabel('% of cases', fontsize=9)
        ax.set_ylim(0, y_max)
        ax.grid(alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, share in zip(bars, shares):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{share:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    plots_dir = r'C:\Users\z004xh1j\OneDrive - Innomotics\Desktop\Comapass_Lexecon_Case_Study\data\plots'
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'lda_topics_and_era_share.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    n_topics = 5
    n_words  = 10
    eras     = ['pre2000', '2000-2015', '2016-now']

    df = pd.read_csv(DATA_PATH)
    df['era'] = df['year'].apply(era)

    # Pre-clean text for LDA
    df['clean_text'] = df['summary'].fillna('').apply(clean_for_lda)

    texts = df['clean_text'].tolist()

    vec, lda, dtm, vocab = build_lda(texts, n_topics=n_topics)

    global_topics = get_topic_words(lda, vocab, n_words=n_words)

    # Print topics to console for quick inspection
    print("\n" + "=" * 60)
    for i, t in enumerate(global_topics):
        print(f"  Topic {i+1}: {', '.join(t['words'])}")
    print("=" * 60 + "\n")

    era_shares = compute_era_shares(df, vec, lda, eras, n_topics)

    plot_topics(global_topics, era_shares, eras, n_topics)


if __name__ == '__main__':
    main()